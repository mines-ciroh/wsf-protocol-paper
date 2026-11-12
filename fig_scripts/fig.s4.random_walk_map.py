"""Fig S.4: spatial map of random-walk cumulative skill wins, by Rodeo method
and initialization month.

Site metadata (lat/lon) is local-only by design; basin/RFC boundary GIS
files live in the published archive. Also writes a headline-stats table of
pooled better/worse/equal site counts by method and month.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

from wsf_plot_common import (
    archive_root,
    site_meta_file,
    MIN_N_PAIRS,
    WSF_RODEO_METHODS as rodeo_methods,
    METHOD_LABELS as method_labels,
    INIT_LABELS_SHORT,
)

# ---- CONFIG ----
WSF_ARCHIVE_ROOT = archive_root()

WIN_CSV = Path("../calc_stats/skill_stats/random_walk_sign_test.results.csv")
SITE_META_FILE = site_meta_file()
BASIN_GPKG = WSF_ARCHIVE_ROOT / "other_data" / "wsf_rodeo_forecast_with_ha.gpkg"
RFC_SHP = WSF_ARCHIVE_ROOT / "other_data" / "rfc_us.shp"

OUT_DIR = Path("./figures")
TABLE_DIR = Path("./tables")
OUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

win_max = 20
size_min, size_max = 10, 500

init_order = [INIT_LABELS_SHORT[mmdd] for mmdd in ["0101", "0201", "0301", "0401"]]  # individual months only, no pooled "All"


def win_color(w, sig):
    if w > 0:
        return "#f16913" if sig else "#ffe0c1"
    elif w < 0:
        return "#2171b5" if sig else "#d5e8fa"
    return "#d9d9d9"


# ---- Load and merge data ----
df = pd.read_csv(WIN_CSV)
df = df[df["method"].isin(rodeo_methods) & df["init"].isin(init_order)]
# Table 1's minimum sample, the same floor figs7_8.random_walks.py applies before
# its significance counts. Without it 40 of 520 panel cells with N = 6, 7 or 9 are
# plotted and counted, contributing one significant-favors-Rodeo and one
# significant-favors-benchmark cell to this figure's headline table.
df = df[df["N"] >= MIN_N_PAIRS]
df["site"] = df["site"].astype(str).str.zfill(8)

site_meta = pd.read_csv(SITE_META_FILE)
site_meta["USGS ID"] = site_meta["USGS ID"].astype(str).str.zfill(8)
# First-occurrence Latitude/Longitude = USGS gauge location (see WSF_Rodeo_ForecastLocations.csv's
# duplicate-column note in fig1.site_map.py).
site_meta = site_meta.rename(columns={"USGS ID": "site", "Latitude": "lat", "Longitude": "lon"})
df = df.merge(site_meta[["site", "lat", "lon"]], on="site", how="left")
n_before = len(df)
df = df.dropna(subset=["lat", "lon"])
if len(df) < n_before:
    print(f"WARNING: dropped {n_before - len(df)} rows with no site lat/lon match")

# Significance: the +/-2 sqrt(N) random-walk envelope, the same rule as Fig 7 and the
# figs7_8 tables (|final net wins| > 2 sqrt(N), N = pairs in that comparison).
df["sig"] = df["skill_wins_final"].abs() > 2 * np.sqrt(df["N"])
df["dot_color"] = df.apply(lambda row: win_color(row["skill_wins_final"], row["sig"]), axis=1)
df["dot_size"] = size_min + (np.clip(np.abs(df["skill_wins_final"]), 0, win_max) / win_max) * (size_max - size_min)

# ---- Headline-stats table: pooled better/worse/equal counts by method x month ----
rows = []
for method in rodeo_methods:
    for init in init_order:
        d = df[(df["method"] == method) & (df["init"] == init)]
        n_sig_better = int(((d["skill_wins_final"] > 0) & d["sig"]).sum())
        n_sig_worse = int(((d["skill_wins_final"] < 0) & d["sig"]).sum())
        rows.append(
            {
                "method": method, "init": init,
                "n_sites": len(d),
                "n_better": int((d["skill_wins_final"] > 0).sum()),
                "n_worse": int((d["skill_wins_final"] < 0).sum()),
                "n_equal": int((d["skill_wins_final"] == 0).sum()),
                "n_significant_better": n_sig_better,
                "n_significant_worse": n_sig_worse,
            }
        )
table_df = pd.DataFrame(rows)
table_path = TABLE_DIR / "fig.s4.headline_stats.csv"
table_df.to_csv(table_path, index=False)
print(f"Saved: {table_path}")
print(table_df.to_string(index=False))

# ---- Load spatial data ----
print("Loading spatial data...")
basins = gpd.read_file(BASIN_GPKG)
rfc_basins = gpd.read_file(RFC_SHP)
states = cfeature.STATES
borders = cfeature.BORDERS

# ---- FIGURE: 4 rows (init) x 5 cols (methods) ----
print("Creating maps figure...")
fig = plt.figure(figsize=(17, 18))
gs = fig.add_gridspec(len(init_order), len(rodeo_methods), hspace=0.03, wspace=0.01, left=0.03, right=0.99, top=0.92, bottom=0.11)

for row, init in enumerate(init_order):
    for col, method in enumerate(rodeo_methods):
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        ax.set_extent([-126, -103, 29, 50])

        basins.plot(ax=ax, facecolor="none", edgecolor="gray", linewidth=0.4, zorder=1, rasterized=True)
        rfc_basins.plot(ax=ax, facecolor="none", edgecolor="gray", linewidth=0.4, zorder=1, rasterized=True)
        ax.add_feature(states, linewidth=0.6, zorder=2)
        ax.add_feature(borders, linewidth=0.6, zorder=2)

        df_panel = df[(df["method"] == method) & (df["init"] == init)]
        ax.scatter(df_panel["lon"], df_panel["lat"], s=df_panel["dot_size"], c=df_panel["dot_color"],
                   edgecolor="k", linewidth=0.5, alpha=0.89, zorder=5, rasterized=True)

        if row == 0:
            ax.set_title(method_labels[method], fontsize=16, pad=8, fontweight="bold")
        if col == 0:
            ax.set_ylabel(init, fontsize=16, fontweight="bold", labelpad=12, rotation=0, ha="right", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

        counts = {
            "Better": int((df_panel["skill_wins_final"] > 0).sum()),
            "Worse": int((df_panel["skill_wins_final"] < 0).sum()),
            "Equal": int((df_panel["skill_wins_final"] == 0).sum()),
        }
        # Per-panel site counts as large colored text in the empty lower-left
        # (Pacific) corner; replaces the earlier inset bar charts, which were
        # unreadable at print size.
        for k, (lbl, n, col) in enumerate([("Rodeo", counts["Better"], "#e6550d"),
                                           ("Bench", counts["Worse"], "#2171b5"),
                                           ("Equal", counts["Equal"], "#636363")]):
            ax.text(0.04, 0.20 - 0.075 * k, f"{lbl} {n}", transform=ax.transAxes, fontsize=15,
                    fontweight="bold", color=col, ha="left", va="bottom", zorder=6,
                    path_effects=[pe.withStroke(linewidth=4, foreground="white")])

sign_handles = [
    Patch(facecolor="#f16913", edgecolor="k", label="Rodeo better, outside ±2√N"),
    Patch(facecolor="#ffe0c1", edgecolor="k", label="Rodeo better, within ±2√N"),
    Patch(facecolor="#2171b5", edgecolor="k", label="Benchmark better, outside ±2√N"),
    Patch(facecolor="#d5e8fa", edgecolor="k", label="Benchmark better, within ±2√N"),
    Patch(facecolor="#d9d9d9", edgecolor="k", label="Equal"),
]
# Both legends span the full width of the map grid (left=0.03 .. right=0.99):
# each is anchored to an explicit (x0, y0, width, height) box with mode="expand".
LEG_TOP, LEG_FS, LEG_TITLE_FS = 0.095, 16, 18  # legend tops sit just under the map grid (bottom=0.11)
fig.legend(handles=sign_handles, title="Performance & Significance", loc="upper left", mode="expand",
           bbox_to_anchor=(0.03, 0.0, 0.615, LEG_TOP), fontsize=LEG_FS, title_fontsize=LEG_TITLE_FS,
           frameon=True, ncol=3, handlelength=2.2, handleheight=1.2, handletextpad=0.6, borderaxespad=0)

size_examples = [5, 10, 20]
size_example_pts = [size_min + (abs(s) / win_max) * (size_max - size_min) for s in size_examples]
size_handles = [
    Line2D([0], [0], marker="o", color="w", label=f"{s} wins", markerfacecolor="gray", markeredgecolor="k",
           markersize=np.sqrt(sz), alpha=0.7, lw=0)
    for s, sz in zip(size_examples, size_example_pts)
]
fig.legend(handles=size_handles, title="Dot Size: Abs. # Wins", loc="upper left", mode="expand",
           bbox_to_anchor=(0.66, 0.0, 0.33, LEG_TOP), fontsize=LEG_FS, title_fontsize=LEG_TITLE_FS,
           frameon=True, ncol=3, handletextpad=0.8, borderaxespad=0)

fig.suptitle("Random Walk Cumulative Skill Wins per Site\nWSF Rodeo Methods (2004-2023 Evaluation Period)", fontsize=24, y=1.02)

print("Saving maps figure...")
fig_path = OUT_DIR / "fig.s4.random_walk_map.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved: {fig_path}")
plt.close()
