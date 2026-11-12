"""Fig 3: forecast archive availability in time AND by site.

Replaced the submitted single-panel Fig 3 in response to Reviewer 2 comment 2
(link Table 2 and Fig 3; show availability across methods *and* basins).

  A) Same as the submitted Fig 3: fraction of the 26 sites with an April 1st f50
     forecast each water year (1990-2023), by method.
  B) Site x method class: fraction of the 20 evaluation years (WY2004-2023)
     with an April 1st f50 forecast at each site, with products grouped by
     class -- statistical (NRCS or CA DWR, i.e. the paper's primary benchmark),
     RFC ESP (Official or Hindcast), and WSF Rodeo ML (any of 1st-5th). A
     site-year counts if any product in the group has a forecast. Panel A keeps
     the products separate, so Official vs. Hindcast stays visible there.
     Sites are numbered as in Fig 1 / Table S.1 and grouped by RFC.

Row labels carry the number of sites with any evaluation-period coverage.
Both panels read the archive directly (April 1st f50 non-NaN). The RFC
seasonal-target exclusions flagged in Fig S.1 are already NaN in the archive,
so panel B's per-site coverage agrees with Fig S.1's pair counts.

Hatched cells in panel B: the RFC issued forecasts at that site during the
evaluation period, but only for a different season than the site's Rodeo
predictand (e.g. MBRFC/ABRFC's April-September volumes vs. the Rodeo's
April-July), so none enter the archive. Read from the per-RFC source CSVs
under source_data/rfc/, before concat_rfc_wsf.py drops mismatched seasons.
Detroit (14181500) is added by hand (OTHER_SEASON_NOT_ARCHIVED): NWRFC's
natural-flow product has Apr-Jul and Apr-Sep forecasts there from ~2020, but
download_nwrfc.py only requests Apr-Jun, so none are in source_data.
Only the RFC row needs this: the statistical row's one empty cell (Ross) has no
NRCS forecast point for any season.
"""

from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

from wsf_plot_common import (
    archive_nc_path,
    archive_root,
    site_meta_file,
    OTHER_METHODS as BASE_METHODS,
    WSF_RODEO_METHODS,
    METHOD_LABELS,
)

OUT_DIR = Path("./figures")
TABLE_DIR = Path("./tables")
OUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

EVAL_START = 2004
EVAL_END = 2023
TARGET_YEARS = np.arange(1990, 2024)
EVAL_YEARS = np.arange(EVAL_START, EVAL_END + 1)
RFC_ORDER = ["NWRFC", "CNRFC", "CBRFC", "MBRFC", "ABRFC", "WGRFC"]

LABEL_MAP = {**{m: METHOD_LABELS[m] for m in BASE_METHODS}, "wsf_rodeo": "WSF Rodeo (1st-5th)"}
row_keys = BASE_METHODS + ["wsf_rodeo"]

# Panel B rows: method classes (label -> archive products pooled into that row)
GROUPS = {
    "Statistical\n(NRCS or CA DWR)": ["nrcs", "ca_dwr"],
    "RFC ESP\n(Official or Hindcast)": ["rfc_official", "rfc_hcst"],
    "WSF Rodeo ML\n(1st-5th)": WSF_RODEO_METHODS,
}

# ---- Load: April 1st f50 for every site/method, as a (method, site, year) availability cube ----
ds = xr.open_dataset(archive_nc_path())
sites = np.array([str(s).zfill(8) for s in ds["wsf"].coords["site"].values])

init_dates = pd.to_datetime(ds["init_date"].values)
apr1 = (init_dates.month == 4) & (init_dates.day == 1)
apr1_dates = ds["init_date"].values[apr1]
apr1_years = init_dates[apr1].year.values

f50 = ds["wsf"].sel(percentile="f50", init_date=apr1_dates)


def avail_by_year(methods):
    """(site, year) bool: any of `methods` has an April 1st f50 at that site/year."""
    arr = ~np.isnan(f50.sel(fcst_method=methods).transpose("site", "fcst_method", "init_date").values)
    arr = arr.any(axis=1)  # any method in the group
    return np.stack([arr[:, apr1_years == y].any(axis=1) for y in TARGET_YEARS], axis=1)


avail = {m: avail_by_year([m]) for m in BASE_METHODS}
avail["wsf_rodeo"] = avail_by_year(WSF_RODEO_METHODS)
group_avail = {g: avail_by_year(ms) for g, ms in GROUPS.items()}

# ---- Site order: Fig 1 / Table S.1 numbering (metadata row order), grouped by RFC ----
meta = pd.read_csv(site_meta_file())
meta["USGS ID"] = meta["USGS ID"].astype(str).str.zfill(8)
meta["site_num"] = np.arange(1, len(meta) + 1)
meta = meta.set_index("USGS ID").loc[sites].reset_index()
meta["rfc_rank"] = meta["RFC"].map({r: i for i, r in enumerate(RFC_ORDER)})
if meta["rfc_rank"].isna().any():
    raise SystemExit(f"Unmapped RFC(s): {sorted(meta.loc[meta['rfc_rank'].isna(), 'RFC'].unique())}")
meta = meta.sort_values(["rfc_rank", "site_num"])
site_order = [int(np.where(sites == s)[0][0]) for s in meta["USGS ID"]]

# ---- Matrices ----
eval_cols = np.isin(TARGET_YEARS, EVAL_YEARS)
time_matrix = np.array([avail[k].mean(axis=0) for k in row_keys])  # (method, year): fraction of sites
n_sites_served = np.array([(avail[k][:, eval_cols].any(axis=1)).sum() for k in row_keys])
group_keys = list(GROUPS)
site_matrix = np.array([group_avail[g][site_order][:, eval_cols].mean(axis=1) for g in group_keys])  # (group, site)
group_n_served = (site_matrix > 0).sum(axis=1)

# ---- RFC forecasts issued only for a different season (hatched in panel B) ----
RFC_GROUP = "RFC ESP\n(Official or Hindcast)"
rodeo_season = {s: (int(a), int(b)) for s, a, b in
                zip(sites, ds["fcst_start_month"].values, ds["fcst_end_month"].values)}
rfc_id_to_usgs = dict(zip(meta["RFC Site ID"].astype(str), meta["USGS ID"]))
rfc_src = pd.concat([pd.read_csv(f) for f in sorted((archive_root() / "source_data" / "rfc").glob("*.csv"))
                     if f.name != "all_rfc_wsf.csv"], ignore_index=True)
rfc_src["usgs"] = rfc_src["site"].astype(str).map(
    lambda c: rfc_id_to_usgs.get(c, rfc_id_to_usgs.get(c[:-1]) if c.endswith("N") else None))
pub = pd.to_datetime(rfc_src["publicationDate"])
rfc_src = rfc_src[pub.dt.month.between(1, 4) & pub.dt.year.between(EVAL_START, EVAL_END)
                  & rfc_src["usgs"].notna() & rfc_src["forecast_50"].notna()]
fs = pd.to_datetime(rfc_src["forecastPeriodStart"]).dt.month
fe = pd.to_datetime(rfc_src["forecastPeriodEnd"]).dt.month
other = rfc_src[[(a, b) != rodeo_season[u] for a, b, u in zip(fs, fe, rfc_src["usgs"])]]
gi = group_keys.index(RFC_GROUP)
# Other-season RFC forecasts verified on the issuing RFC's site but never downloaded into source_data.
OTHER_SEASON_NOT_ARCHIVED = {
    # nwrfc.noaa.gov/natural/plot/nat_text.cgi?id=DETO3&per=APR-JUL (and APR-SEP): Jan-Apr 1 issuances
    # present for WY2020 and WY2023, absent for WY2004-2018 (checked 2026-09-26). Rodeo season is Apr-Jun.
    "14181500": [("Apr", "Jul"), ("Apr", "Sep")],
}
other_season_only = np.zeros_like(site_matrix, dtype=bool)
for si, u in enumerate(meta["USGS ID"]):
    if site_matrix[gi, si] == 0 and ((other["usgs"] == u).any() or u in OTHER_SEASON_NOT_ARCHIVED):
        other_season_only[gi, si] = True
hatch_info = {u: sorted(set(zip(pd.to_datetime(g["forecastPeriodStart"]).dt.strftime("%b"),
                                pd.to_datetime(g["forecastPeriodEnd"]).dt.strftime("%b"))))
              for u, g in other.groupby("usgs")}
hatch_info.update(OTHER_SEASON_NOT_ARCHIVED)
for si, (_, r) in enumerate(meta.iterrows()):
    if other_season_only[gi, si]:
        print(f"Hatched (RFC, other season only): #{r['site_num']} {r['name']} -> issued {hatch_info[r['USGS ID']]}, "
              f"Rodeo season months {rodeo_season[r['USGS ID']]}")

# ---- Tables ----
site_rows = []
for mi, g in enumerate(group_keys):
    for si, (_, r) in enumerate(meta.iterrows()):
        site_rows.append({
            "method_group": g.replace("\n", " "), "site_num": r["site_num"], "usgs_id": r["USGS ID"], "site_name": r["name"],
            "rfc": r["RFC"], "n_eval_years_available": int(round(site_matrix[mi, si] * len(EVAL_YEARS))),
            "frac_eval_years_available": round(float(site_matrix[mi, si]), 3),
            "other_season_only": bool(other_season_only[mi, si]),
        })
pd.DataFrame(site_rows).to_csv(TABLE_DIR / "fig3.fcst_avail_2panel.site_coverage.csv", index=False)

summary = pd.DataFrame({
    "method": [LABEL_MAP[k] for k in row_keys],
    "n_sites_any_eval_coverage": n_sites_served,
    "n_sites_full_eval_coverage": [int(avail[k][:, eval_cols].all(axis=1).sum()) for k in row_keys],
    "mean_site_fraction_eval_period": np.round(time_matrix[:, eval_cols].mean(axis=1), 3),
})
summary_groups = pd.DataFrame({
    "method": [g.replace("\n", " ") for g in group_keys],
    "n_sites_any_eval_coverage": group_n_served,
    "n_sites_full_eval_coverage": (site_matrix >= 1.0).sum(axis=1),
    "n_sites_ge10_eval_years": (np.round(site_matrix * len(EVAL_YEARS)) >= 10).sum(axis=1),
    "mean_site_fraction_eval_period": np.round(site_matrix.mean(axis=1), 3),
})
summary = pd.concat([summary.assign(panel="A"), summary_groups.assign(panel="B")], ignore_index=True)
summary.to_csv(TABLE_DIR / "fig3.fcst_avail_2panel.headline_stats.csv", index=False)
print(summary.to_string(index=False))

# ---- Colormap: 0 = gray, >0 = single-hue blues (starting mid-light so partial coverage still reads as blue, not as a gap) ----
N_COLORS = 256
cmap = mcolors.ListedColormap(np.vstack([[0.78, 0.78, 0.78, 1.0], plt.cm.Blues(np.linspace(0.28, 1, N_COLORS - 1))]))


class ZeroGrayNorm(mcolors.Normalize):
    """Maps 0 to gray and values >0 to the blues ramp."""

    def __init__(self, n=256):
        super().__init__(vmin=0, vmax=1, clip=True)
        self.n = n

    def __call__(self, value, clip=None):
        result = super().__call__(value, clip)
        nonzero = np.asarray(result) > 0
        return np.ma.array(np.where(nonzero, 1 / self.n + np.asarray(result) * (1 - 1 / self.n), 0))


norm = ZeroGrayNorm(N_COLORS)
row_labels = [f"{LABEL_MAP[k]}\n(n = {n}/{len(sites)})" for k, n in zip(row_keys, n_sites_served)]
n_rows = len(row_keys)
group_labels = [f"{g}\n(n = {n}/{len(sites)})" for g, n in zip(group_keys, group_n_served)]
n_grp = len(group_keys)

fig, (axA, axB) = plt.subplots(2, 1, figsize=(14, 9.6), gridspec_kw={"hspace": 0.42, "height_ratios": [5, 4.4]})

# Panel A: through time
axA.imshow(time_matrix[::-1], aspect="auto", cmap=cmap, norm=norm, origin="lower",
           extent=[TARGET_YEARS[0] - 0.5, TARGET_YEARS[-1] + 0.5, -0.5, n_rows - 0.5])
axA.add_patch(patches.Rectangle((EVAL_START - 0.5, -0.5), EVAL_END - EVAL_START + 1, n_rows,
                                linewidth=3.0, edgecolor="red", facecolor="none", zorder=5))
# Inside the bottom edge of the box, as in the original Fig 3.
axA.text((EVAL_START + EVAL_END) / 2, -0.38, "Evaluation Period", color="red", ha="center", va="bottom",
         zorder=6,
         fontsize=15, fontweight="bold")
axA.set_yticks(np.arange(n_rows), row_labels[::-1], fontsize=15)
axA.set_xticks(TARGET_YEARS[::5], TARGET_YEARS[::5], fontsize=15)
axA.set_xlabel("Water Year", fontsize=17, fontweight="bold")
axA.set_title("A) Fraction of sites with an April 1 forecast, by water year", fontsize=16, fontweight="bold", loc="left",
              pad=8)

# Panel B: by site
n_s = len(site_order)
im = axB.imshow(site_matrix[::-1], aspect="auto", cmap=cmap, norm=norm, origin="lower",
                extent=[-0.5, n_s - 0.5, -0.5, n_grp - 0.5])
axB.set_yticks(np.arange(n_grp), group_labels[::-1], fontsize=15)
axB.set_xticks(np.arange(n_s), meta["site_num"].astype(str), fontsize=14)
axB.tick_params(axis="x", length=0)
for x in np.arange(n_s + 1) - 0.5:
    axB.axvline(x, color="white", lw=0.8)
for y in np.arange(n_grp + 1) - 0.5:
    axB.axhline(y, color="white", lw=0.8)
HATCH = "///"
plt.rcParams["hatch.color"] = "#555555"
for gi_, si in zip(*np.where(other_season_only)):
    axB.add_patch(patches.Rectangle((si - 0.5, (n_grp - 1 - gi_) - 0.5), 1, 1, facecolor="none",
                                    edgecolor="#555555", hatch=HATCH, linewidth=0, zorder=3))
# RFC group brackets below the site numbers
rfc_seq = meta["RFC"].values
starts = [0] + [i for i in range(1, n_s) if rfc_seq[i] != rfc_seq[i - 1]]
ends = starts[1:] + [n_s]
for s, e in zip(starts, ends):
    axB.annotate("", xy=(s - 0.4, -0.155), xytext=(e - 0.6, -0.155), xycoords=("data", "axes fraction"),
                 arrowprops=dict(arrowstyle="-", lw=1.5, color="black"), annotation_clip=False)
    # Short RFC codes ("NW" for NWRFC): single-site groups (AB, WG) are one cell wide.
    axB.text((s + e - 1) / 2, -0.185, rfc_seq[s].replace("RFC", ""), transform=axB.get_xaxis_transform(),
             ha="center", va="top", fontsize=14, fontweight="bold")
axB.set_xlabel("Site #, grouped by River Forecast Center", fontsize=17, fontweight="bold",
               labelpad=40)
axB.set_title(f"B) Fraction of WY{EVAL_START}-{EVAL_END} with an April 1 forecast, by site and method class",
              fontsize=16, fontweight="bold", loc="left")

# Colorbar + gray keys in a right-hand column placed by hand: the swatches sit at the
# bottom of panel B (labels beside them), and the colorbar runs from just above them
# to the top of panel A.
posA, posB = axA.get_position(), axB.get_position()
KEY_X = posA.x1 + 0.03
KEY_W = 0.018
SW_H = 0.028
SW_GAP = 0.018
key_rows = [("no forecasts", {"facecolor": cmap(0), "edgecolor": "black"})]
if other_season_only.any():
    key_rows.append(("other forecast\ntarget season only", {"facecolor": cmap(0), "edgecolor": "#555555", "hatch": HATCH}))
for i, (label, style) in enumerate(reversed(key_rows)):  # bottom-up, last swatch flush with panel B's bottom
    y = posB.y0 + i * (SW_H + SW_GAP)
    fig.patches.append(patches.Rectangle((KEY_X, y), KEY_W, SW_H, transform=fig.transFigure, linewidth=0.8, **style))
    fig.text(KEY_X + KEY_W + 0.008, y + SW_H / 2, label, ha="left", va="center", fontsize=15, linespacing=1.05)
cbar_y0 = posB.y0 + len(key_rows) * (SW_H + SW_GAP) + 0.03
cax = fig.add_axes([KEY_X, cbar_y0, KEY_W, posA.y1 - cbar_y0])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Fraction available", fontsize=17, fontweight="bold")
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0", "0.25", "0.5", "0.75", "1.0"])
cbar.ax.tick_params(labelsize=15)

fig_path = OUT_DIR / "fig3.fcst_avail_2panel.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved: {fig_path}")
