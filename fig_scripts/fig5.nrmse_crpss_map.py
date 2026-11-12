"""Fig 5: spatial map of the top-ranked method by nRMSE skill score and CRPSS,
January 1 and April 1.

nRMSE skill score = 1 - nRMSE_method / nRMSE_climatology (manuscript Sec 7.1).
Site metadata (lat/lon) is local-only by design; basin/RFC boundary GIS files
live in the published archive. Also writes a headline-stats table of win
counts (how many sites each method is top-ranked at) by metric and month.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

from wsf_plot_common import (
    archive_root,
    site_meta_file,
    MIN_N_PAIRS,
    METHOD_COLORS as method_colors,
    METHOD_LABELS as method_labels,
    PLOT_ORDER as plot_methods,
)

# ---- CONFIG ----
WSF_ARCHIVE_ROOT = archive_root()

SKILL_FILE = "../calc_stats/skill_stats/wsf.determ_stats.by_init_method.csv"
CRPS_FILE = "../calc_stats/skill_stats/wsf.prob_stats.by_init_method.csv"
CLIMO_FILE = "../calc_stats/skill_stats/wsf.climo_stats.by_site.csv"
SITE_META_FILE = site_meta_file()
BASIN_GPKG = WSF_ARCHIVE_ROOT / "other_data" / "wsf_rodeo_forecast_with_ha.gpkg"
RFC_SHP = WSF_ARCHIVE_ROOT / "other_data" / "rfc_us.shp"

OUT_DIR = Path("./figures")
TABLE_DIR = Path("./tables")
OUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

# ---- LOAD DATA ----
nrmse_df = pd.read_csv(SKILL_FILE)
crpss_df = pd.read_csv(CRPS_FILE)
climo_df = pd.read_csv(CLIMO_FILE)

nrmse_df = nrmse_df[(nrmse_df["method"].isin(plot_methods)) & (nrmse_df["n_pairs"] >= MIN_N_PAIRS)]
crpss_df = crpss_df[(crpss_df["method"].isin(plot_methods)) & (crpss_df["n_pairs"] >= MIN_N_PAIRS)]
plot_methods_with_climo = ["climatology"] + plot_methods

basins = gpd.read_file(BASIN_GPKG)
rfc_basins = gpd.read_file(RFC_SHP)
sites_meta = pd.read_csv(SITE_META_FILE)
sites_meta["USGS ID"] = sites_meta["USGS ID"].astype(str).str.zfill(8)

sites_gdf = gpd.GeoDataFrame(sites_meta, geometry=gpd.points_from_xy(sites_meta["Longitude"], sites_meta["Latitude"]), crs="EPSG:4326")
rfc_basins_reproj = rfc_basins.to_crs("EPSG:4326")
sites_with_rfc = gpd.sjoin(sites_gdf, rfc_basins_reproj, how="left", predicate="within")

climo_nrmse_lookup = climo_df.set_index("site")["climo_nrmse"].to_dict()

# ---- PLOT MAPS: 2 rows x 2 columns ----
fig = plt.figure(figsize=(16.2, 14))
axes = [fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree()) for i in range(4)]
plt.subplots_adjust(left=0.05, right=0.79, top=0.95, bottom=0.02, wspace=0.03, hspace=0.06)

init_dates = [("0101", "January 1 Forecasts"), ("0401", "April 1 Forecasts")]
size_min, size_max = 10, 1500
ss_vmin, ss_vmax = 0.0, 1.0  # skill-score dot-size scale, shared by both rows

headline_rows = []  # (metric, init_title, method, n_sites)


def plot_panel(ax, best_method_per_site, init_title):
    ax.set_extent([-125, -103, 31, 50])
    basins.plot(ax=ax, facecolor="none", edgecolor="gray", linewidth=0.7, zorder=1)
    rfc_basins.plot(ax=ax, facecolor="none", edgecolor="gray", linewidth=0.7, zorder=1)
    ax.add_feature(cfeature.STATES, linewidth=0.7, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.7, zorder=2)
    for m in best_method_per_site["method"].unique():
        df_m = best_method_per_site[best_method_per_site["method"] == m]
        ax.scatter(df_m["Longitude"], df_m["Latitude"], s=df_m["size"], c=[method_colors[m]] * len(df_m),
                   alpha=0.87, edgecolor="k", linewidth=0.7, label=method_labels.get(m, m), zorder=5)
    if init_title:  # column headers on the top row only
        ax.set_title(init_title, fontsize=22, fontweight="bold", pad=10)


# ---- ROW 1: nRMSE Skill Score (Jan, Apr) ----
for col_idx, (init_code, init_title) in enumerate(init_dates):
    plot_sub = nrmse_df[nrmse_df["init_date"].astype(str).str.zfill(4) == init_code]
    site_method_skill = plot_sub.groupby(["site", "method"])["nRMSE"].mean().reset_index()
    best_method_per_site = site_method_skill.loc[site_method_skill.groupby("site")["nRMSE"].idxmin()].reset_index(drop=True)

    # nRMSE_SS comes from calc_stats, where the climatology reference is rebuilt
    # over exactly this site/method/init's own paired years. Computing it here as
    # 1 - nRMSE/climo_nrmse instead would divide a method's RMSE (its own years,
    # as few as 8 for sparse RFC records) by a climatology RMSE over all 20 eval
    # years -- different year sets, shifting the score by up to 0.24 and changing
    # the plotted winner at 3 of 26 sites in the January panel.
    best_method_per_site = best_method_per_site.merge(
        plot_sub[["site", "method", "nRMSE_SS"]], on=["site", "method"], how="left"
    )

    best_method_per_site["site"] = best_method_per_site["site"].astype(str).str.zfill(8)
    best_method_per_site = best_method_per_site.merge(
        sites_with_rfc[["USGS ID", "Latitude", "Longitude", "RFC"]], left_on="site", right_on="USGS ID", how="left"
    )
    best_method_per_site = best_method_per_site.dropna(subset=["Latitude", "Longitude", "nRMSE_SS"])

    ss_clipped = np.clip(best_method_per_site["nRMSE_SS"], ss_vmin, ss_vmax)
    best_method_per_site["size"] = size_min + (ss_clipped - ss_vmin) / (ss_vmax - ss_vmin) * (size_max - size_min)

    plot_panel(axes[col_idx], best_method_per_site, init_title)
    for method, n in best_method_per_site["method"].value_counts().items():
        headline_rows.append({"metric": "nRMSE_skill_score", "init_date": init_code, "method": method, "n_sites_best": int(n)})

# ---- ROW 2: CRPSS (Jan, Apr) ----
for col_idx, (init_code, init_title) in enumerate(init_dates):
    plot_sub = crpss_df[crpss_df["init_date"].astype(str).str.zfill(4) == init_code]
    site_method_skill = plot_sub.groupby(["site", "method"])["CRPSS"].mean().reset_index()
    best_method_per_site = site_method_skill.loc[site_method_skill.groupby("site")["CRPSS"].idxmax()].reset_index(drop=True)

    best_method_per_site["site"] = best_method_per_site["site"].astype(str).str.zfill(8)
    best_method_per_site = best_method_per_site.merge(
        sites_with_rfc[["USGS ID", "Latitude", "Longitude", "RFC"]], left_on="site", right_on="USGS ID", how="left"
    )
    best_method_per_site = best_method_per_site.dropna(subset=["Latitude", "Longitude"])

    crpss_clipped = np.clip(best_method_per_site["CRPSS"], ss_vmin, ss_vmax)
    best_method_per_site["size"] = size_min + (crpss_clipped - ss_vmin) / (ss_vmax - ss_vmin) * (size_max - size_min)

    plot_panel(axes[col_idx + 2], best_method_per_site, None)
    for method, n in best_method_per_site["method"].value_counts().items():
        headline_rows.append({"metric": "CRPSS", "init_date": init_code, "method": method, "n_sites_best": int(n)})

axes[0].text(-0.06, 0.5, "nRMSE Skill Score", transform=axes[0].transAxes, rotation=90, va="center", ha="center", fontsize=22, fontweight="bold")
axes[2].text(-0.06, 0.5, "CRPSS", transform=axes[2].transAxes, rotation=90, va="center", ha="center", fontsize=22, fontweight="bold")

legend_order = ["climatology", "nrcs", "rfc_official", "rfc_hcst", "ca_dwr",
                "wsf_rodeo_1place", "wsf_rodeo_2place", "wsf_rodeo_3place", "wsf_rodeo_4place", "wsf_rodeo_5place"]
present_methods = [m for m in legend_order if m in pd.concat([nrmse_df, crpss_df])["method"].unique()]
method_handles = [Patch(facecolor=method_colors[m], edgecolor="k", label=method_labels[m]) for m in present_methods]

example_ss = [0.0, 0.4, 0.8]
example_sizes_ss = [size_min + (ss - ss_vmin) / (ss_vmax - ss_vmin) * (size_max - size_min) for ss in example_ss]
dot_handles_ss = [
    Line2D([0], [0], marker="o", color="w", label=f"Skill Score {s:.1f}", markerfacecolor="gray", markeredgecolor="k",
           markersize=np.sqrt(sz), alpha=0.65, lw=0)
    for s, sz in zip(example_ss, example_sizes_ss)
]

leg1 = axes[1].legend(handles=method_handles, title="Top-Ranked Method", loc="upper left", bbox_to_anchor=(1.02, 1.01), fontsize=14, title_fontsize=15, frameon=True)
axes[1].legend(handles=dot_handles_ss, title="Dot Size\n(Skill vs. Climatology)", loc="upper left", bbox_to_anchor=(1.02, 0.47), fontsize=13, title_fontsize=14, labelspacing=1.6)
axes[1].add_artist(leg1)

fig_path = OUT_DIR / "fig5.nrmse_crpss_map.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved: {fig_path}")
plt.close()

# ---- Headline-stats table: win counts by metric x month x method ----
table_df = pd.DataFrame(headline_rows).sort_values(["metric", "init_date", "n_sites_best"], ascending=[True, True, False])
table_path = TABLE_DIR / "fig5.headline_stats.csv"
table_df.to_csv(table_path, index=False)
print(f"Saved: {table_path}")
print(table_df.to_string(index=False))
