"""Fig S.3: sharpness (normalized f10-f90 spread) distributions by method/month.

Also writes a headline-stats table of median normalized spread by method and
initialization month.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from wsf_plot_common import (
    archive_nc_path,
    START_YEAR as start_year,
    END_YEAR as end_year,
    MIN_N_PAIRS,
    INIT_MMDD_LIST as MMDD_list,
    INIT_LABELS as hcst_titles,
    PLOT_ORDER as plot_order,
    WSF_RODEO_METHODS,
    METHOD_COLORS as method_colors,
    METHOD_LABELS as method_labels,
    assert_percentile_order,
)

# ---- SETTINGS ----
nc_path = archive_nc_path()

OUT_DIR = Path("./figures")
TABLE_DIR = Path("./tables")
OUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)
out_fig = OUT_DIR / "fig.s3.sharp_diagram.png"

NORMALIZE_YEARS = np.arange(2004, 2024)

METHOD_STYLES = {method: {"color": method_colors[method], "label": method_labels[method]} for method in method_colors}

# ---- LOAD DATA ----
print("Loading data...")
ds = xr.open_dataset(nc_path)
site_names = ds["site"].values
init_dates = ds["init_date"].values
method_names = ds["fcst_method"].values
percentile_labels = ds["percentile"].values
assert_percentile_order(percentile_labels)
year_vals = ds["year"].values

init_dates_pd = pd.to_datetime(init_dates, errors="coerce")
mmdd_strs = init_dates_pd.strftime("%m%d")
year_strs = init_dates_pd.year

# --- Compute climatological mean obs for normalization ---
print("Computing site climatology...")
site_obs_means = {}
for site_idx, site in enumerate(site_names):
    obs_vals = []
    for year in NORMALIZE_YEARS:
        yidxs = np.where(year_vals == year)[0]
        if len(yidxs) == 0:
            continue
        obs_val = ds["obs"][site_idx, yidxs[0]].values
        if np.isfinite(obs_val):
            obs_vals.append(obs_val)
    site_obs_means[site] = np.mean(obs_vals) if obs_vals else np.nan

# ---- EXTRACT SHARPNESS DATA ----
print("Extracting sharpness data...")
sharpness_rows = []
n_skipped_unexpected = 0

for mmdd in MMDD_list:
    mmdd_mask = mmdd_strs == mmdd
    init_date_idxs = np.where(mmdd_mask)[0]
    years_for_this_mmdd = year_strs[init_date_idxs]

    for method_idx, method in enumerate(method_names):
        fcst_quantiles_all = ds["wsf"].isel(fcst_method=method_idx).values[:, init_date_idxs, :]

        for site_idx, site in enumerate(site_names):
            climo_mean = site_obs_means.get(site, np.nan)
            if np.isnan(climo_mean) or climo_mean == 0:
                continue

            for date_idx, init_date_idx in enumerate(init_date_idxs):
                fcst_year = years_for_this_mmdd[date_idx]
                if fcst_year < start_year or fcst_year > end_year:
                    continue
                obs_year_idx = np.where(year_vals == fcst_year)[0]
                if len(obs_year_idx) == 0:
                    continue
                try:
                    obs_val = ds["obs"][site_idx, obs_year_idx[0]].values
                    fcst_quantiles = fcst_quantiles_all[site_idx, date_idx, :]
                except (IndexError, KeyError):
                    n_skipped_unexpected += 1
                    continue

                required_quantiles = fcst_quantiles[[0, 2, 4]]  # f10, f50, f90
                if np.isfinite(obs_val) and np.all(np.isfinite(required_quantiles)):
                    f10 = fcst_quantiles[0]
                    f90 = fcst_quantiles[4]
                    spread = f10 - f90  # exceedance convention: f10 > f90
                    sharpness_rows.append(
                        {
                            "site": site, "method": method, "mmdd": mmdd, "year": fcst_year,
                            "normalized_spread": spread / climo_mean, "spread": spread, "climo_mean": climo_mean,
                        }
                    )

if n_skipped_unexpected:
    print(f"WARNING: skipped {n_skipped_unexpected} records due to indexing errors")

sharpness_df = pd.DataFrame(sharpness_rows)
print(f"Collected {len(sharpness_df)} sharpness measurements")

# ---- Headline-stats table: median normalized spread by method/month ----
#
# median_normalized_spread is MARGINAL: each method over its own record. That is
# what the figure plots, and it is the right descriptive question -- but the
# methods do not cover the same basins or years (Rodeo has all 26 sites x 20
# years; RFC Official has 21 sites and only 233 of 520 April site-years), so
# comparing two of these medians directly mixes a method difference with a
# basin-composition difference.
#
# A single shared sample is not available: ZERO April site-years are common to
# all nine methods (CA DWR covers only its 2 California sites, which RFC
# Hindcast's 11 sites exclude). Dropping CA DWR leaves 24 site-years at 6 sites,
# which would gut the figure. So the figure stays marginal, and the like-for-like
# comparison is reported PAIRWISE here instead -- each row against its own
# overlap, which is large (233 site-years vs RFC Official, 428 vs NRCS).
#
# rodeo_* columns give the Rodeo spread restricted to THIS row's own site-years.
# Only the Rodeo-vs-RFC-Official contrast moves materially (April +0.242 ->
# +0.139); NRCS and RFC Hindcast barely change and CA DWR's gap widens, which is
# what you would expect if this is sampling rather than something systemic.
RODEO_POOLED = [m for m in WSF_RODEO_METHODS if m in method_names]
R3 = "wsf_rodeo_3place"


def _paired_median(reference_rows, candidate_methods, mmdd):
    """Median Rodeo spread over exactly the site-years `reference_rows` covers."""
    keys = set(zip(reference_rows["site"], reference_rows["year"]))
    cand = sharpness_df[(sharpness_df["method"].isin(candidate_methods)) & (sharpness_df["mmdd"] == mmdd)]
    if not len(cand):
        return np.nan, 0
    match = cand[[k in keys for k in zip(cand["site"], cand["year"])]]
    return (round(match["normalized_spread"].median(), 3) if len(match) else np.nan), len(match)


rows = []
for method in plot_order:
    if method not in method_names:
        continue
    for mmdd in MMDD_list:
        sub = sharpness_df[(sharpness_df["method"] == method) & (sharpness_df["mmdd"] == mmdd)]
        marginal = round(sub["normalized_spread"].median(), 3) if len(sub) >= MIN_N_PAIRS else np.nan
        row = {
            "method": method,
            "init_date": mmdd,
            "median_normalized_spread": marginal,
            "n_pairs": len(sub),
        }
        # Pairwise columns are meaningless for the Rodeo methods themselves.
        if method in RODEO_POOLED or len(sub) < MIN_N_PAIRS:
            row.update({"rodeo_pooled_median_same_site_years": np.nan, "rodeo_pooled_paired_gap": np.nan,
                        "rodeo_3rd_median_same_site_years": np.nan, "rodeo_3rd_paired_gap": np.nan,
                        "n_site_years_paired": np.nan})
        else:
            pooled, _ = _paired_median(sub, RODEO_POOLED, mmdd)
            r3, n_pair = _paired_median(sub, [R3], mmdd)
            row.update({
                "rodeo_pooled_median_same_site_years": pooled,
                "rodeo_pooled_paired_gap": round(pooled - marginal, 3) if np.isfinite(pooled) else np.nan,
                "rodeo_3rd_median_same_site_years": r3,
                "rodeo_3rd_paired_gap": round(r3 - marginal, 3) if np.isfinite(r3) else np.nan,
                "n_site_years_paired": n_pair,
            })
        rows.append(row)
table_df = pd.DataFrame(rows)
table_path = TABLE_DIR / "fig.s3.headline_stats.csv"
table_df.to_csv(table_path, index=False)
print(f"Saved: {table_path}")
print(table_df.to_string(index=False))

# ---- CREATE 2x2 SHARPNESS FIGURE ----
print("Creating sharpness KDE plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={"hspace": 0.20, "wspace": 0.15})
axes_flat = axes.flatten()

bins_sharp = np.linspace(0, 3, 31)
x_kde = np.linspace(0, 3, 200)

for col_idx, mmdd in enumerate(MMDD_list):
    ax = axes_flat[col_idx]
    mmdd_sharp = sharpness_df[sharpness_df["mmdd"] == mmdd]

    for method in plot_order:
        if method not in method_names:
            continue
        style = METHOD_STYLES.get(method, {"color": "gray", "label": method})
        method_sharp = mmdd_sharp[mmdd_sharp["method"] == method]
        if len(method_sharp) < MIN_N_PAIRS:
            continue

        normalized_spreads = method_sharp["normalized_spread"].values
        median_spread = np.median(normalized_spreads)

        ax.hist(normalized_spreads, bins=bins_sharp, color=style["color"], alpha=0.2,
                edgecolor="black", linewidth=0.5, density=True)

        try:
            kde = gaussian_kde(normalized_spreads)
            ax.plot(x_kde, kde(x_kde), color=style["color"], linewidth=2.5,
                    label=fr"{style['label']} ($\tilde{{x}}$ = {median_spread:.2f})")
        except np.linalg.LinAlgError:
            ax.plot([], [], color=style["color"], linewidth=2.5,
                    label=fr"{style['label']} ($\tilde{{x}}$ = {median_spread:.2f})")

        ax.axvline(median_spread, color=style["color"], linestyle="--", linewidth=2.5, alpha=0.8)

    ax.set_ylim(0, 5)
    ax.set_xlim(0, 3)

    row, col = divmod(col_idx, 2)
    if row == 1:
        ax.set_xlabel("Normalized Spread (f10-f90) / Climatology", fontsize=11, fontweight="bold")
    else:
        ax.set_xlabel("")
    if col == 0:
        ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.set_title(f"{hcst_titles[mmdd]}", fontsize=13, fontweight="bold", pad=0)
    ax.grid(True, alpha=0.3, linewidth=0.5, axis="y")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

plt.suptitle("Sharpness by Forecast Initialization Date (2004-2023)", fontsize=16, fontweight="bold", y=0.95)
plt.tight_layout()
plt.savefig(out_fig, dpi=300, bbox_inches="tight")
print(f"Saved: {out_fig}")
plt.close()
