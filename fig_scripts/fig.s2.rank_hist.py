"""Fig S.2: rank (Talagrand) histograms by method and initialization month.

Each observation is classified directly against its own f10/f50/f90 forecast
thresholds (exceedance convention: f10 > f50 > f90 in value) into one of 4
canonical exceedance-probability bins (<10%, 10-50%, 50-90%, >90%). f30/f70
are deliberately ignored even when present, so every method -- 3-quantile
natively (CA DWR, WSF Rodeo) or 5-quantile (NRCS, RFC) -- is classified the
same apples-to-apples way. Also writes a headline-stats table of observed
vs. expected (perfect-reliability) counts per method/month/bin.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from wsf_plot_common import (
    archive_nc_path,
    START_YEAR as start_year,
    END_YEAR as end_year,
    INIT_MMDD_LIST as MMDD_list,
    PLOT_ORDER as plot_order,
    METHOD_LABELS as method_labels,
    assert_percentile_order,
)

nc_path = archive_nc_path()

OUT_DIR = Path("./figures")
TABLE_DIR = Path("./tables")
OUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

ds = xr.open_dataset(nc_path)
site_names = ds["site"].values
init_dates = ds["init_date"].values
method_names = ds["fcst_method"].values
percentile_labels = ds["percentile"].values
assert_percentile_order(percentile_labels)
year_vals = ds["year"].values

# Positions of f10/f50/f90 in the 5-slot percentile axis; fixed by
# assert_percentile_order's [f10, f30, f50, f70, f90] guarantee above.
F10_IDX, F50_IDX, F90_IDX = 0, 2, 4

init_labels = {"0101": "Jan 1", "0201": "Feb 1", "0301": "Mar 1", "0401": "Apr 1"}

# 4 canonical bins: <10%, 10-50%, 50-90%, >90%
BIN_LABELS = ["<10%", "10-50%", "50-90%", ">90%"]
N_CANONICAL = 4
BIN_COLORS = ["#deebf7", "#9ecae1", "#4292c6", "#08519c"]
EXPECTED_PROPS = np.array([0.10, 0.40, 0.40, 0.10])

init_dates_pd = pd.to_datetime(init_dates, errors="coerce")
mmdd_strs = init_dates_pd.strftime("%m%d")
year_strs = init_dates_pd.year
site_idxs = range(len(site_names))


def classify_canonical_bin(fcst_quantiles, obs_val):
    """Classify one obs against its own f10/f50/f90 into a canonical bin
    index (0=<10%, 1=10-50%, 2=50-90%, 3=>90%), exceedance convention
    (f10 > f50 > f90 in value)."""
    f10 = fcst_quantiles[F10_IDX]
    f50 = fcst_quantiles[F50_IDX]
    f90 = fcst_quantiles[F90_IDX]
    if obs_val > f10:
        return 0
    elif obs_val > f50:
        return 1
    elif obs_val > f90:
        return 2
    return 3


# ---- Collect data ----
all_method_data = {}

for method in plot_order:
    if method not in method_names:
        continue
    method_idx = np.where(method_names == method)[0][0]

    counts_by_init = {}
    total_by_init = {}

    for mmdd in MMDD_list:
        canonical = np.zeros(N_CANONICAL, dtype=float)

        for site_idx in site_idxs:
            for date_idx, init_date in enumerate(init_dates):
                if mmdd_strs[date_idx] != mmdd:
                    continue
                year = year_strs[date_idx]
                if year < start_year or year > end_year:
                    continue
                fcst_quantiles = ds["wsf"][site_idx, date_idx, :, method_idx].values
                required = fcst_quantiles[[F10_IDX, F50_IDX, F90_IDX]]
                if not np.all(np.isfinite(required)):
                    continue
                year_idx = np.where(year_vals == year)[0]
                if len(year_idx) == 0:
                    continue
                obs_val = ds["obs"][site_idx, year_idx[0]].values
                if not np.isfinite(obs_val):
                    continue
                bin_idx = classify_canonical_bin(fcst_quantiles, obs_val)
                canonical[bin_idx] += 1

        if canonical.sum() == 0:
            continue

        counts_by_init[mmdd] = canonical
        total_by_init[mmdd] = int(canonical.sum())

    all_method_data[method] = {"counts_by_init": counts_by_init, "total_by_init": total_by_init}

# ---- Headline-stats table: observed vs expected counts by method/month/bin ----
rows = []
for method in plot_order:
    if method not in all_method_data:
        continue
    data = all_method_data[method]
    for mmdd in MMDD_list:
        if mmdd not in data["counts_by_init"]:
            continue
        total = data["total_by_init"][mmdd]
        counts = data["counts_by_init"][mmdd]
        for b, bin_label in enumerate(BIN_LABELS):
            rows.append(
                {
                    "method": method,
                    "init_date": mmdd,
                    "bin": bin_label,
                    "observed_count": int(counts[b]),
                    "observed_frac": round(counts[b] / total, 3) if total else np.nan,
                    "expected_frac": EXPECTED_PROPS[b],
                    "total_n": total,
                }
            )
table_df = pd.DataFrame(rows)
table_path = TABLE_DIR / "fig.s2.headline_stats.csv"
table_df.to_csv(table_path, index=False)
print(f"Saved: {table_path}")
print(table_df.to_string(index=False))

# ---- Layout ----
n_methods = len(plot_order)
n_cols = 3
n_rows = int(np.ceil(n_methods / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.8, n_rows * 3.4), squeeze=False)
axes_flat = axes.flatten()

n_inits = len(MMDD_list)
group_width = 0.75
bar_width = group_width / N_CANONICAL
offsets = np.linspace(-(group_width / 2) + bar_width / 2, (group_width / 2) - bar_width / 2, N_CANONICAL)


def int_fmt(x, pos):
    return f"{int(x)}"


for m, method in enumerate(plot_order):
    if method not in method_names or method not in all_method_data:
        axes_flat[m].axis("off")
        continue

    ax = axes_flat[m]
    row_idx = m // n_cols
    col_idx = m % n_cols
    data = all_method_data[method]
    counts_by_init = data["counts_by_init"]
    total_by_init = data["total_by_init"]

    x_positions = np.arange(n_inits)

    for b, (bin_label, color) in enumerate(zip(BIN_LABELS, BIN_COLORS)):
        bar_vals = [counts_by_init[mmdd][b] if mmdd in counts_by_init else 0 for mmdd in MMDD_list]
        ax.bar(x_positions + offsets[b], bar_vals, width=bar_width, color=color, label=bin_label,
               alpha=0.92, edgecolor="k", linewidth=0.6)

    for i, mmdd in enumerate(MMDD_list):
        if mmdd not in total_by_init:
            continue
        total = total_by_init[mmdd]
        for b, prop in enumerate(EXPECTED_PROPS):
            exp = prop * total
            x_left = x_positions[i] + offsets[b] - bar_width / 2
            x_right = x_positions[i] + offsets[b] + bar_width / 2
            ax.hlines(exp, x_left, x_right, colors="k", linewidth=1.8, linestyles="--", alpha=0.85)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([init_labels[mmdd] for mmdd in MMDD_list], fontsize=11)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_title(method_labels[method], fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(int_fmt))

    if col_idx == 0:
        ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    if row_idx == n_rows - 1:
        ax.set_xlabel("Initialization Date", fontsize=12, fontweight="bold")

for idx in range(n_methods, len(axes_flat)):
    axes_flat[idx].axis("off")

bin_handles = [plt.Rectangle((0, 0), 1, 1, color=BIN_COLORS[b], alpha=0.92, ec="k", lw=0.6) for b in range(N_CANONICAL)]
exp_handle = Line2D([0], [0], color="k", linewidth=1.8, linestyle="--", alpha=0.85)

fig.legend(
    bin_handles + [exp_handle], BIN_LABELS + ["Expected"],
    title="Exceedance Probability Bin", fontsize=12, title_fontsize=14,
    loc="lower center", ncol=N_CANONICAL + 1, bbox_to_anchor=(0.5, 0.01),
    frameon=True, framealpha=0.95,
)

fig.suptitle("Rank (Talagrand) Histograms", fontsize=18, fontweight="bold", y=0.98)

plt.tight_layout(rect=[0, 0.07, 1.0, 0.97])
fig_path = OUT_DIR / "fig.s2.rank_hist.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved: {fig_path}")
plt.close()
