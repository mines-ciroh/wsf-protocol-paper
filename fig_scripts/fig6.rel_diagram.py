"""Fig 6: reliability diagrams by method/initialization month.

The figure shows reliability curves only. A Brier score is still written to the
headline table (on the f10/f50/f90 levels common to every method; see
COMMON_LEVELS), but it is NOT shown in the figure: with forecast probability
fixed at the quantile level it reduces to BS = (1.07 - 0.8*C)/3, where C is the
fraction of obs inside [f90, f10]. It therefore ranks methods by 80%-interval
coverage, rewards over-dispersion (a calibrated method scores 0.1433; lower =
too wide), and carries no information from f50. Do not read it as "reliability".

Also writes a headline-stats table (Brier score, median/mean sharpness, and
a simple reliability-error category) per method and month, plus a pooled
overall-rankings table -- the numbers behind the manuscript's Section 3.2
reliability/calibration discussion.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

from wsf_plot_common import (
    archive_nc_path,
    START_YEAR as start_year,
    END_YEAR as end_year,
    INIT_MMDD_LIST as MMDD_list,
    INIT_LABELS as hcst_titles,
    PLOT_ORDER as plot_order,
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
out_fig = OUT_DIR / "fig6.rel_diagram.png"

NORMALIZE_YEARS = np.arange(2004, 2024)
RELIABILITY_MIN_N = 50  # min forecast-obs pairs to include a method/month in reliability stats (distinct from MIN_N_PAIRS elsewhere)

METHOD_STYLES = {method: {"color": method_colors[method], "label": method_labels[method]} for method in method_colors}

N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95
N_BINS = 5

# ---- LOAD DATA ----
print("Loading data...")
ds = xr.open_dataset(nc_path)
site_names = ds["site"].values
init_dates = ds["init_date"].values
method_names = ds["fcst_method"].values
percentile_labels = ds["percentile"].values
assert_percentile_order(percentile_labels)
year_vals = ds["year"].values
quantile_levels = np.array([int(p[1:]) / 100 for p in percentile_labels])

# Brier is scored ONLY on the three levels every method provides (f10/f50/f90).
# Here the forecast probability is fixed at the quantile level, so a perfectly
# calibrated method scores exactly mean p(1-p) -- a value set entirely by WHICH
# levels it contributes, not by its skill: 0.1433 over {.1,.5,.9} but 0.1700 over
# {.1,.3,.5,.7,.9}. NRCS/RFC carry all 5 levels while CA DWR and every Rodeo entry
# carry only 3, so averaging over each method's native level set made the scores
# differ by 0.027 before any skill entered, flattering the 3-quantile methods.
# The reliability CURVES still plot every level a method has -- each point is
# independently interpretable against the 1:1 line -- it is only the scalar
# summary that needs a common basis.
COMMON_LEVELS = np.array([0.1, 0.5, 0.9])


def brier_on_common_levels(y_prob, y_true):
    """Brier restricted to f10/f50/f90, the levels shared by all methods."""
    m = np.isin(y_prob, COMMON_LEVELS)
    return (np.mean((y_prob[m] - y_true[m]) ** 2) if m.any() else np.nan), int(m.sum())

init_dates_pd = pd.to_datetime(init_dates, errors="coerce")
mmdd_strs = init_dates_pd.strftime("%m%d")
year_strs = init_dates_pd.year

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

# ---- EXTRACT RELIABILITY DATA (one row per quantile level per fcst-obs pair) ----
print("Extracting reliability data...")
reliability_rows = []
# ---- EXTRACT SHARPNESS DATA (one row per fcst-obs pair; feeds the headline table only) ----
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
                if not (np.isfinite(obs_val) and np.all(np.isfinite(required_quantiles))):
                    continue

                for q_idx, q_level in enumerate(quantile_levels):
                    fcst_threshold = fcst_quantiles[q_idx]
                    if np.isfinite(fcst_threshold):
                        reliability_rows.append(
                            {
                                "site": site, "method": method, "mmdd": mmdd, "year": fcst_year,
                                "forecast_prob": q_level,
                                "obs_exceeded_threshold": 1 if obs_val > fcst_threshold else 0,
                                "obs_value": obs_val, "fcst_threshold": fcst_threshold, "climo_mean": climo_mean,
                            }
                        )

                f10, f90 = fcst_quantiles[0], fcst_quantiles[4]
                spread = f10 - f90  # exceedance convention: f10 > f90
                sharpness_rows.append(
                    {"site": site, "method": method, "mmdd": mmdd, "year": fcst_year, "normalized_spread": spread / climo_mean}
                )

if n_skipped_unexpected:
    print(f"WARNING: skipped {n_skipped_unexpected} records due to indexing errors")

reliability_df = pd.DataFrame(reliability_rows)
sharpness_df = pd.DataFrame(sharpness_rows)
print(f"Collected {len(reliability_df)} reliability pairs, {len(sharpness_df)} sharpness measurements")

print("Calculating climatological exceedance frequencies...")
climo_exceedance_freq = {}
for mmdd in MMDD_list:
    mmdd_data = reliability_df[reliability_df["mmdd"] == mmdd]
    obs_exceed_climo = (mmdd_data["obs_value"] > mmdd_data["climo_mean"]).astype(int)
    climo_exceedance_freq[mmdd] = obs_exceed_climo.mean() if len(obs_exceed_climo) else 0.5
    print(f"  {mmdd}: Climatological exceedance frequency = {climo_exceedance_freq[mmdd]:.3f}")


def compute_reliability_curve_sklearn(y_true, y_prob, n_bins=20, apply_isotonic=False):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    if apply_isotonic and len(prob_pred) > 3:
        prob_true_smooth = IsotonicRegression(out_of_bounds="clip").fit_transform(prob_pred, prob_true)
    else:
        prob_true_smooth = prob_true
    return prob_pred, prob_true, prob_true_smooth


def bootstrap_reliability_sklearn(y_true, y_prob, n_bootstrap=1000, n_bins=20):
    n = len(y_true)
    boot_curves = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        try:
            _, _, smooth_freq = compute_reliability_curve_sklearn(y_true[idx], y_prob[idx], n_bins)
            boot_curves.append(smooth_freq)
        except ValueError:
            continue
    if not boot_curves:
        return None, None
    max_len = max(len(c) for c in boot_curves)
    padded = np.array([np.pad(c, (0, max_len - len(c)), mode="edge") for c in boot_curves])
    lower = np.percentile(padded, (1 - CONFIDENCE_LEVEL) / 2 * 100, axis=0)
    upper = np.percentile(padded, (1 + CONFIDENCE_LEVEL) / 2 * 100, axis=0)
    return lower, upper


# ---- CREATE 2x2 RELIABILITY FIGURE ----
print("Creating reliability diagrams...")
fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={"hspace": 0.15, "wspace": 0.1})
axes_flat = axes.flatten()

for col_idx, mmdd in enumerate(MMDD_list):
    ax = axes_flat[col_idx]
    mmdd_data = reliability_df[reliability_df["mmdd"] == mmdd]
    climo_freq = climo_exceedance_freq[mmdd]

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, zorder=1, label="Perfect reliability")
    ax.axhline(climo_freq, color="black", linestyle=":", linewidth=2, alpha=0.7, zorder=20, label="No resolution")
    ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color="black", alpha=0.1, linewidth=0, zorder=0)

    for method in reversed(plot_order):
        if method not in method_names:
            continue
        style = METHOD_STYLES.get(method, {"color": "gray", "label": method})
        method_data = mmdd_data[mmdd_data["method"] == method]
        if len(method_data) < RELIABILITY_MIN_N:
            continue

        y_prob = method_data["forecast_prob"].values
        y_true = method_data["obs_exceeded_threshold"].values
        prob_pred, prob_true, prob_true_smooth = compute_reliability_curve_sklearn(y_true, y_prob, n_bins=N_BINS, apply_isotonic=False)
        lower, upper = bootstrap_reliability_sklearn(y_true, y_prob, N_BOOTSTRAP, n_bins=N_BINS)

        ax.plot(prob_pred, prob_true_smooth, "-o", color=style["color"], linewidth=2.5, markersize=6,
                label=style["label"], zorder=3)
        if lower is not None and upper is not None:
            ax.fill_between(prob_pred, lower[: len(prob_pred)], upper[: len(prob_pred)], color=style["color"], alpha=0.2, zorder=2)

    row, col = divmod(col_idx, 2)
    if row == 1:
        ax.set_xlabel("Forecast Exceedance Probability", fontsize=14, fontweight="bold")
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("Observed Exceedance Frequency", fontsize=14, fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    ax.set_title(f"{hcst_titles[mmdd]}", fontsize=15, fontweight="bold", pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(axis="both", labelsize=13)

# Single shared legend outside the 2x2, to the right of the top-right panel.
# Collected across all panels (CA DWR is absent in January), ordered reference
# lines first, then methods in the canonical plot order.
legend_entries = {}
for ax in axes_flat:
    for h, l in zip(*ax.get_legend_handles_labels()):
        legend_entries.setdefault(l, h)
legend_order = ["Perfect reliability", "No resolution"] + [METHOD_STYLES[m]["label"] for m in plot_order]
legend_labels = [l for l in legend_order if l in legend_entries]
legend_handles = [legend_entries[l] for l in legend_labels]
legend_handles.append(Patch(facecolor="gray", alpha=0.3, edgecolor="none"))
legend_labels.append("95% bootstrap CI")
axes_flat[1].legend(legend_handles, legend_labels, loc="upper left", bbox_to_anchor=(1.04, 1.0),
                    borderaxespad=0, fontsize=12, frameon=True)

plt.savefig(out_fig, dpi=300, bbox_inches="tight")
print(f"Saved: {out_fig}")
plt.close()

# ==================================================================
# ---- HEADLINE-STATS TABLE: Brier score + sharpness by method x month ----
# ==================================================================
summary_stats = []
for mmdd in MMDD_list:
    mmdd_data = reliability_df[reliability_df["mmdd"] == mmdd]
    mmdd_sharp = sharpness_df[sharpness_df["mmdd"] == mmdd]
    climo_freq = climo_exceedance_freq[mmdd]

    for method in plot_order:
        if method not in method_names:
            continue
        method_data = mmdd_data[mmdd_data["method"] == method]
        n_samples = len(method_data)
        if n_samples < RELIABILITY_MIN_N:
            continue

        y_prob = method_data["forecast_prob"].values
        y_true = method_data["obs_exceeded_threshold"].values
        brier, n_common = brier_on_common_levels(y_prob, y_true)

        method_sharp = mmdd_sharp[mmdd_sharp["method"] == method]
        median_spread = np.median(method_sharp["normalized_spread"].values) if len(method_sharp) else np.nan
        mean_spread = np.mean(method_sharp["normalized_spread"].values) if len(method_sharp) else np.nan

        prob_pred, prob_true, prob_true_smooth = compute_reliability_curve_sklearn(y_true, y_prob, n_bins=N_BINS, apply_isotonic=False)
        reliability_error = np.mean(np.abs(prob_pred - prob_true_smooth))
        reliability_cat = (
            "Excellent" if reliability_error < 0.05 else
            "Good" if reliability_error < 0.10 else
            "Fair" if reliability_error < 0.15 else
            "Poor"
        )

        summary_stats.append(
            {
                "init_date": mmdd, "init_label": hcst_titles[mmdd], "method": method, "method_label": method_labels[method],
                "n_samples": n_samples, "n_samples_common_levels": n_common,
                "brier_score": round(brier, 4),
                "median_sharpness": round(median_spread, 3) if not np.isnan(median_spread) else np.nan,
                "mean_sharpness": round(mean_spread, 3) if not np.isnan(mean_spread) else np.nan,
                "reliability_category": reliability_cat, "climo_exceedance_freq": round(climo_freq, 3),
            }
        )

summary_df = pd.DataFrame(summary_stats)
table_path = TABLE_DIR / "fig6.headline_stats.csv"
summary_df.to_csv(table_path, index=False)
print(f"Saved: {table_path}")
print(summary_df.to_string(index=False))

# ---- Overall rankings (mean across all init dates), console only ----
print("\nOVERALL RANKINGS (averaged across all initialization dates)")
overall_summary = (
    summary_df.groupby(["method", "method_label"])
    .agg(brier_score=("brier_score", "mean"), median_sharpness=("median_sharpness", "mean"), n_samples=("n_samples", "sum"))
    .reset_index()
    .sort_values("brier_score")
)
for rank, (_, row) in enumerate(overall_summary.iterrows(), 1):
    print(f"{rank:<3} {row['method_label']:<20} Avg Brier={row['brier_score']:.4f}  Avg Sharpness={row['median_sharpness']:.3f}  Total N={row['n_samples']}")

# ---- Best/worst performers by month, console only ----
print("\nBEST AND WORST PERFORMERS BY MONTH")
for mmdd in MMDD_list:
    mmdd_summary = summary_df[summary_df["init_date"] == mmdd]
    if len(mmdd_summary) == 0:
        continue
    best_brier = mmdd_summary.loc[mmdd_summary["brier_score"].idxmin()]
    worst_brier = mmdd_summary.loc[mmdd_summary["brier_score"].idxmax()]
    print(f"{hcst_titles[mmdd]}: best Brier = {best_brier['method_label']} ({best_brier['brier_score']:.4f}), "
          f"worst Brier = {worst_brier['method_label']} ({worst_brier['brier_score']:.4f})")
