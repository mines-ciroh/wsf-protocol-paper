"""Fig S.1: forecast-observation pair counts by site and method (2004-2023).

Site metadata (name, RFC) is local-only by design, read from
WSF_Rodeo_ForecastLocations.csv via WSF_LOCAL_DATA_ROOT, separate from the
published archive.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from wsf_plot_common import (
    site_meta_file,
    MIN_N_PAIRS,
    METHOD_COLORS as method_colors,
    PLOT_ORDER as plot_methods,
)

# ---- CONFIG ----
SKILL_FILE = "../calc_stats/skill_stats/wsf.determ_stats.by_init_method.csv"
CRPS_FILE = "../calc_stats/skill_stats/wsf.prob_stats.by_init_method.csv"
SITE_META_FILE = site_meta_file()

OUT_DIR = Path("./figures")
TABLE_DIR = Path("./tables")
OUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

# Sites with different seasonal definition
NO_AMJJ_RFC_FCST = {"06054500", "06259000", "07099400"}
NO_AMJ_RFC_FCST = {"14181500"}
NO_RFC_FCST = {"06639000"}
NO_NRCS_FCST = {"12175000", "11251000", "11446220"}

# Short x-tick labels; the abbreviations are keyed in the text below the panels.
TICK_LABELS = {
    "nrcs": "NRCS",
    "rfc_official": "RFC Off.",
    "rfc_hcst": "RFC Hcst.",
    "ca_dwr": "CA DWR",
    **{f"wsf_rodeo_{i}place": f"Rodeo {i}" for i in range(1, 6)},
}

# ---- LOAD DATA ----
nrmse_df = pd.read_csv(SKILL_FILE)
crpss_df = pd.read_csv(CRPS_FILE)
sites_meta = pd.read_csv(SITE_META_FILE)

nrmse_df = nrmse_df[nrmse_df["method"].isin(plot_methods)]
crpss_df = crpss_df[crpss_df["method"].isin(plot_methods)]

sites = sorted(nrmse_df["site"].astype(str).str.zfill(8).unique())
n_sites = len(sites)

sites_meta["USGS ID"] = sites_meta["USGS ID"].astype(str).str.zfill(8)
site_name_lookup = dict(zip(sites_meta["USGS ID"], sites_meta["name"]))
site_rfc_lookup = dict(zip(sites_meta["USGS ID"], sites_meta["RFC"]))

# ---- Headline-stats table: total pairs by site and method ----
rows = []
for site in sites:
    nrmse_site = nrmse_df[nrmse_df["site"].astype(str).str.zfill(8) == site].groupby("method")["n_pairs"].sum().reindex(plot_methods, fill_value=0)
    crpss_site = crpss_df[crpss_df["site"].astype(str).str.zfill(8) == site].groupby("method")["n_pairs"].sum().reindex(plot_methods, fill_value=0)
    for method in plot_methods:
        rows.append(
            {
                "site": site,
                "site_name": site_name_lookup.get(site, site),
                "method": method,
                "n_pairs_deterministic": int(nrmse_site[method]),
                "n_pairs_probabilistic": int(crpss_site[method]),
            }
        )
table_df = pd.DataFrame(rows)
table_path = TABLE_DIR / "fig.s1.headline_stats.csv"
table_df.to_csv(table_path, index=False)
print(f"Saved: {table_path}")
print(f"Sites with 0 deterministic pairs for any method: "
      f"{sorted(table_df.loc[table_df['n_pairs_deterministic'] == 0, 'site'].unique())}")

# ---- PLOT: 26 subplots, one per site ----
n_cols = 4
n_rows = int(np.ceil(n_sites / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 20))
axes = axes.flatten()

for idx, site in enumerate(sites):
    ax = axes[idx]
    nrmse_site = nrmse_df[nrmse_df["site"].astype(str).str.zfill(8) == site].groupby("method")["n_pairs"].sum().reindex(plot_methods, fill_value=0)
    crpss_site = crpss_df[crpss_df["site"].astype(str).str.zfill(8) == site].groupby("method")["n_pairs"].sum().reindex(plot_methods, fill_value=0)

    x = np.arange(len(plot_methods))
    width = 0.4

    for i, method in enumerate(plot_methods):
        base_color = method_colors[method]
        rgb = tuple(int(base_color.lstrip("#")[j : j + 2], 16) / 255.0 for j in (0, 2, 4))
        lighter = tuple(min(1.0, c + 0.3) for c in rgb)
        ax.bar(i - width / 2, nrmse_site[method], width, color=lighter, edgecolor="black", linewidth=0.5)
        ax.bar(i + width / 2, crpss_site[method], width, color=rgb, edgecolor="black", linewidth=0.5)

    site_name = site_name_lookup.get(site, site)
    rfc = site_rfc_lookup.get(site, "")

    if site in NO_AMJJ_RFC_FCST:
        site_name = f"{site_name}*"
    elif site in NO_AMJ_RFC_FCST:
        site_name = f"{site_name}**"
    elif site in NO_RFC_FCST:
        site_name = f"{site_name}+"
    elif site in NO_NRCS_FCST:
        site_name = f"{site_name}++"

    title_str = f"{site_name}\n({site}, {rfc})" if pd.notna(rfc) and rfc != "" else f"{site_name}\n({site})"

    ax.set_title(title_str, fontsize=10, pad=3)
    ax.set_ylim(0, max(20, nrmse_site.max(), crpss_site.max()) + 2)
    ax.set_xticks(x)
    ax.set_xticklabels([TICK_LABELS[m] for m in plot_methods], rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("N Pairs", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.tick_params(axis="y", labelsize=7)

for idx in range(n_sites, len(axes)):
    axes[idx].axis("off")

legend_elements = [
    mpatches.Patch(facecolor="whitesmoke", edgecolor="black", label="Deterministic (50% exceedance)"),
    mpatches.Patch(facecolor="darkgray", edgecolor="black", label="Probabilistic (10-50-90% exceedance)"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=2, fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.98))

# Abbreviation key + site footnotes in one boxed block in the empty bottom-right
# panels (Reviewer 1 comment 13: "R1" was undefined in the submitted figure).
key_lines = [
    "RFC Off. / RFC Hcst. = RFC Official / RFC Hindcast",
    "Rodeo 1-5 = WSF Rodeo 1st-5th place solutions (Table S.3)",
    "",
    "* RFC forecasts excluded (RFC Target: AMJJAS, Rodeo Target: AMJJ)",
    "** RFC forecasts excluded (RFC Target: AMJJ, Rodeo Target: AMJ)",
    "+ Not an NWS RFC WSF site",
    "++ Not an NRCS WSF site",
]
fig.text(0.535, 0.045, "\n".join(key_lines), ha="left", va="bottom", fontsize=11, linespacing=1.6,
         bbox=dict(boxstyle="square,pad=0.7", facecolor="white", edgecolor="black", linewidth=0.8))

plt.suptitle("Forecast-Observation Pairs by Site and Method (2004-2023)", fontsize=16, y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.97))
fig_path = OUT_DIR / "fig.s1.fcst_count.png"
plt.savefig(fig_path, dpi=300)
print(f"Saved: {fig_path}")
plt.close()
