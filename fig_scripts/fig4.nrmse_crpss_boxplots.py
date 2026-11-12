"""Fig 4: cross-site nRMSE and CRPSS by initialization month, plus site counts.

Triple-panel boxplot (nRMSE, CRPSS, site counts) across all study sites and
methods, filtered to a minimum sample size. Also writes a consolidated
headline-stats table (median/mean nRMSE and CRPSS by method and month, plus
an "All" pooled-months row) with the numbers that feed the manuscript prose.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
from pathlib import Path

from wsf_plot_common import (
    START_YEAR as start_year,
    END_YEAR as end_year,
    MIN_N_PAIRS,
    INIT_MMDD_LIST as MMDD_list,
    INIT_LABELS as hcst_titles,
    OTHER_METHODS,
    WSF_RODEO_METHODS,
    METHOD_COLORS as method_colors,
    METHOD_LABELS as method_labels,
)

# ---- SETTINGS ----
SKILL_FILE = "../calc_stats/skill_stats/wsf.determ_stats.by_init_method.csv"
CRPS_FILE = "../calc_stats/skill_stats/wsf.prob_stats.by_init_method.csv"
CLIMO_FILE = "../calc_stats/skill_stats/wsf.climo_stats.by_site.csv"
OUT_DIR = Path("./figures")
TABLE_DIR = Path("./tables")
OUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

plot_methods = WSF_RODEO_METHODS + OTHER_METHODS  # Rodeos first, then others (will plot left to right)

# int-keyed (matches the numeric init_date column) -- distinct from hcst_titles' string keys, not shared
months = {101: "Jan", 201: "Feb", 301: "Mar", 401: "Apr"}

# ---- LOAD PRECOMPUTED STATS ----
stats_df_full = pd.read_csv(SKILL_FILE)
crps_df_full = pd.read_csv(CRPS_FILE)
climo_df = pd.read_csv(CLIMO_FILE)

stats_df = stats_df_full[(stats_df_full["method"].isin(plot_methods)) & (stats_df_full["n_pairs"] >= MIN_N_PAIRS)]
crps_df = crps_df_full[(crps_df_full["method"].isin(plot_methods)) & (crps_df_full["n_pairs"] >= MIN_N_PAIRS)]
methods_plot = plot_methods

# ---- Climatology IQR bands (from site-level climatology stats) ----
climo_nrmse = climo_df["climo_nrmse"].values
climo_median = np.median(climo_nrmse)
climo_25 = np.percentile(climo_nrmse, 25)
climo_75 = np.percentile(climo_nrmse, 75)
iqr = climo_75 - climo_25
low_fence = climo_25 - 1.5 * iqr
high_fence = climo_75 + 1.5 * iqr
actual_low_whisker = np.min(climo_nrmse[climo_nrmse >= low_fence])
actual_high_whisker = np.max(climo_nrmse[climo_nrmse <= high_fence])

# ---- Prepare boxplot data for nRMSE, CRPSS, and site counts ----
box_data_nrmse, box_data_crpss, positions, colors, xtick_labels, xtick_positions = [], [], [], [], [], []
site_counts = []
width = 0.13
sep = 0.07

for i, mmdd in enumerate(MMDD_list):
    base_pos = i * (len(methods_plot) * width + sep)
    for j, method in enumerate(methods_plot):
        pos = base_pos + j * width

        vals_det = stats_df[(stats_df["method"] == method) & (stats_df["init_date"] == int(mmdd))]["nRMSE"].values
        if len(vals_det) == 0:
            vals_det = [np.nan]
        box_data_nrmse.append(vals_det)

        vals_prob = crps_df[(crps_df["method"] == method) & (crps_df["init_date"] == int(mmdd))]["CRPSS"].values
        if len(vals_prob) == 0:
            vals_prob = [np.nan]
        box_data_crpss.append(vals_prob)

        n_prob = np.sum(~np.isnan(vals_prob))
        n_det = np.sum(~np.isnan(vals_det))
        site_counts.append(max(n_det, n_prob))

        positions.append(pos)
        colors.append(method_colors[method])

    xtick_labels.append(hcst_titles[mmdd].split()[0])
    xtick_positions.append(base_pos + (len(methods_plot) - 1) * width / 2)

# ---- PLOT ----
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.4], hspace=0.1)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)

# ----- TOP PANEL: nRMSE -----
ax1.axhspan(actual_low_whisker, actual_high_whisker, xmin=0, xmax=1, color="gray", alpha=0.35, zorder=-2)
ax1.axhspan(climo_25, climo_75, xmin=0, xmax=1, color="gray", alpha=0.60, zorder=-1)
ax1.axhline(climo_median, color="black", linestyle="--", linewidth=2, zorder=0)
bps1 = ax1.boxplot(
    box_data_nrmse, positions=positions, widths=width * 0.85, patch_artist=True,
    showfliers=True, medianprops=dict(color="black", linewidth=2),
)
for patch, color in zip(bps1["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_linewidth(1.2)

ax1.set_xlim(positions[0] - 0.25, positions[-1] + 0.3)
ax1.set_ylim(0, 90)
ax1.set_ylabel("nRMSE (%)", fontsize=17, fontweight="bold")
ax1.set_title("A) Deterministic (f50) forecast skill", fontsize=17, fontweight="bold", loc="left")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
plt.setp(ax1.get_xticklabels(), visible=False)

legend_elements = [
    Patch(facecolor=method_colors[m], edgecolor="black", label=method_labels.get(m, m)) for m in plot_methods
] + [
    Patch(facecolor="gray", edgecolor="none", alpha=0.60, label="Climatology IQR"),
    Patch(facecolor="gray", edgecolor="none", alpha=0.35, label="Climatology IQR × 1.5"),
]
ax1.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, ncol=1, fontsize=13, frameon=True)

# ----- MIDDLE PANEL: CRPSS -----
bps2 = ax2.boxplot(
    box_data_crpss, positions=positions, widths=width * 0.85, patch_artist=True,
    showfliers=True, medianprops=dict(color="black", linewidth=2),
)
for patch, color in zip(bps2["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_linewidth(1.2)

ax2.set_xlim(positions[0] - 0.25, positions[-1] + 0.3)
ax2.axhline(y=0, color="black")
ax2.set_ylabel("CRPSS", fontsize=17, fontweight="bold")
ax2.set_title("B) Probabilistic (f10-f50-f90) forecast skill", fontsize=17, fontweight="bold", loc="left")
ax2.grid(axis="y", linestyle="--", alpha=0.4)
plt.setp(ax2.get_xticklabels(), visible=False)

# ----- BOTTOM PANEL: SITE COUNTS -----
ax3.bar(positions, site_counts, width=width * 0.85, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
ax3.set_xlim(positions[0] - 0.25, positions[-1] + 0.3)
ax3.set_ylim(0, 27)
hline = ax3.axhline(y=26, color="grey", linestyle="--", label="All Sites (n = 26)")
ax3.set_ylabel("Site Count", fontsize=17, fontweight="bold")
ax3.set_title("C) Site inventory", fontsize=17, fontweight="bold", loc="left")
ax3.set_xlabel("Forecast Initialization Date", fontsize=17, fontweight="bold")
ax3.set_xticks(xtick_positions)
ax3.set_xticklabels(xtick_labels, fontsize=16)
ax3.grid(axis="y", linestyle="--", alpha=0.4)
ax3.legend(handles=[hline], loc="lower left", fontsize=11, frameon=True)
for ax in (ax1, ax2, ax3):
    ax.tick_params(axis="y", labelsize=14)

fig_path = OUT_DIR / "fig4.nrmse_crpss_boxplots.png"
plt.savefig(fig_path, dpi=250, bbox_inches="tight")
print(f"Saved: {fig_path}")
plt.close()

# ==================================================================
# ---- HEADLINE-STATS TABLE: median/mean nRMSE & CRPSS by method x month, plus pooled "All" ----
# ==================================================================
all_methods = ["wsf_rodeo_1place", "wsf_rodeo_2place", "wsf_rodeo_3place", "wsf_rodeo_4place", "wsf_rodeo_5place",
               "nrcs", "rfc_hcst", "rfc_official", "ca_dwr"]

rows = []
for method in all_methods:
    for init_date, month_name in list(months.items()) + [(None, "All")]:
        if init_date is None:
            det_sub = stats_df[stats_df["method"] == method]
            crps_sub = crps_df[crps_df["method"] == method]
        else:
            det_sub = stats_df[(stats_df["method"] == method) & (stats_df["init_date"] == init_date)]
            crps_sub = crps_df[(crps_df["method"] == method) & (crps_df["init_date"] == init_date)]
        rows.append(
            {
                "method": method,
                "month": month_name,
                "median_nRMSE": round(det_sub["nRMSE"].median(), 3) if len(det_sub) else np.nan,
                "mean_nRMSE": round(det_sub["nRMSE"].mean(), 3) if len(det_sub) else np.nan,
                "n_pairs_nRMSE": int(det_sub["n_pairs"].sum()) if len(det_sub) else 0,
                "median_CRPSS": round(crps_sub["CRPSS"].median(), 3) if len(crps_sub) else np.nan,
                "mean_CRPSS": round(crps_sub["CRPSS"].mean(), 3) if len(crps_sub) else np.nan,
                "n_pairs_CRPSS": int(crps_sub["n_pairs"].sum()) if len(crps_sub) else 0,
            }
        )

table_df = pd.DataFrame(rows)
table_path = TABLE_DIR / "fig4.headline_stats.csv"
table_df.to_csv(table_path, index=False)
print(f"Saved: {table_path}")
print(table_df.to_string(index=False))

# ---- Best performing method by month (console only, derived from table above) ----
print()
print(f"BEST PERFORMING METHOD BY MONTH (min {MIN_N_PAIRS} pairs)")
for init_date, month_name in months.items():
    nrmse_best = stats_df[stats_df["init_date"] == init_date].groupby("method")["nRMSE"].median().idxmin()
    crpss_best = crps_df[crps_df["init_date"] == init_date].groupby("method")["CRPSS"].median().idxmax()
    print(f"{month_name}: Best nRMSE = {nrmse_best}, Best CRPSS = {crpss_best}")

# ---- Prose-ready April 1st summary (matches manuscript phrasing) ----
april = table_df[table_df["month"] == "Apr"].set_index("method")
rodeo_methods = [m for m in all_methods if m.startswith("wsf_rodeo")]

print()
print("FORMATTED TEXT FOR MANUSCRIPT (APRIL 1ST)")
ref_methods = ["nrcs", "rfc_hcst", "rfc_official", "ca_dwr"]
ref_nrmse_range = (april.loc[ref_methods, "median_nRMSE"].min(), april.loc[ref_methods, "median_nRMSE"].max())
print(f"By April, the reference forecasts have median nRMSE values of {ref_nrmse_range[0]:.1f}-{ref_nrmse_range[1]:.1f}%")
for m in ref_methods:
    print(f"  {method_labels[m]}: {april.loc[m, 'median_nRMSE']:.1f}% ({april.loc[m, 'n_pairs_nRMSE']} pairs)")
rodeo_nrmse_range = (april.loc[rodeo_methods, "median_nRMSE"].min(), april.loc[rodeo_methods, "median_nRMSE"].max())
rodeo_pairs = april.loc[rodeo_methods, "n_pairs_nRMSE"].iloc[0]
print(f"while the top WSF Rodeo solutions range from {rodeo_nrmse_range[0]:.1f}-{rodeo_nrmse_range[1]:.1f}% ({rodeo_pairs} pairs).")
print()
print("April 1st operational forecasts also have higher median CRPSS")
for m in ref_methods:
    print(f"  {method_labels[m]}: {april.loc[m, 'median_CRPSS']:.2f} ({april.loc[m, 'n_pairs_CRPSS']} pairs)")
rodeo_crpss_range = (april.loc[rodeo_methods, "median_CRPSS"].min(), april.loc[rodeo_methods, "median_CRPSS"].max())
rodeo_crpss_pairs = april.loc[rodeo_methods, "n_pairs_CRPSS"].iloc[0]
print(f"compared to Rodeo solutions ({rodeo_crpss_range[0]:.2f}-{rodeo_crpss_range[1]:.2f}, {rodeo_crpss_pairs} pairs).")
