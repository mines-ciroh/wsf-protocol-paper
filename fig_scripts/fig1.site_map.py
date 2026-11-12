"""Fig 1: map of the 26 study basins and NWS RFC boundaries.

Site metadata (lat/lon, RFC assignment) is local-only by design -- it is not
part of the published benchmark archive -- so it's read from
WSF_Rodeo_ForecastLocations.csv via WSF_LOCAL_DATA_ROOT, separate from the
published-archive GIS files (basin polygons, RFC boundaries) which do live
in the archive under WSF_ARCHIVE_ROOT.
"""

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patheffects as path_effects
import numpy as np
from pyproj import Geod

from wsf_plot_common import archive_root, site_meta_file

# ---- FILEPATHS ----
WSF_ARCHIVE_ROOT = archive_root()

SITE_META_FILE = site_meta_file()
BASIN_GPKG = WSF_ARCHIVE_ROOT / "other_data" / "wsf_rodeo_forecast_with_ha.gpkg"
RFC_SHP = WSF_ARCHIVE_ROOT / "other_data" / "rfc_us.shp"
# Natural Earth 1:50m (public domain), used only for the locator inset's country
# outlines. Defaults to cartopy's local download cache; override with
# NATURAL_EARTH_DIR if your copy lives elsewhere.
NE_DIR = Path(os.environ.get(
    "NATURAL_EARTH_DIR",
    Path.home() / ".local/share/cartopy/shapefiles/natural_earth/cultural",
))
NE_COAST_SHP = NE_DIR.parent / "physical" / "ne_50m_coastline.shp"
NE_BORDERS_SHP = NE_DIR / "ne_50m_admin_0_boundary_lines_land.shp"
NE_STATES_SHP = NE_DIR / "ne_50m_admin_1_states_provinces_lakes.shp"

OUT_DIR = Path("./figures")
TABLE_DIR = Path("./tables")
OUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

for p in (SITE_META_FILE, BASIN_GPKG, RFC_SHP, NE_COAST_SHP, NE_BORDERS_SHP, NE_STATES_SHP):
    if not p.exists():
        raise SystemExit(f"Required file not found: {p}")

# ---- LOAD DATA ----
# WSF_Rodeo_ForecastLocations.csv has duplicate Latitude/Longitude columns
# (USGS gauge, NRCS site, RFC site); pandas suffixes duplicates as
# Latitude/Latitude.1/Latitude.2 on read. We want the USGS gauge location
# (the first occurrence), which is what the outlet marker represents.
sites_meta = pd.read_csv(SITE_META_FILE)
sites_meta["USGS ID"] = sites_meta["USGS ID"].astype(str).str.zfill(8)
gdf_sites = gpd.GeoDataFrame(
    sites_meta,
    geometry=gpd.points_from_xy(sites_meta["Longitude"], sites_meta["Latitude"]),
    crs="EPSG:4326",
)

gdf_basins = gpd.read_file(BASIN_GPKG)
if gdf_basins.crs != "EPSG:4326":
    gdf_basins = gdf_basins.to_crs("EPSG:4326")
gdf_sites = gdf_sites.reset_index(drop=True)
gdf_sites["basin_idx"] = gdf_sites.index + 1

gdf_rfc = gpd.read_file(RFC_SHP)
if gdf_rfc.crs != "EPSG:4326":
    gdf_rfc = gdf_rfc.to_crs("EPSG:4326")

# ---- Headline-stats table: site count by RFC ----
table_df = (
    gdf_sites.groupby("RFC")["USGS ID"]
    .count()
    .rename("n_sites")
    .reset_index()
    .sort_values("n_sites", ascending=False)
)
table_path = TABLE_DIR / "fig1.headline_stats.csv"
table_df.to_csv(table_path, index=False)
print(f"Saved: {table_path}")
print(table_df.to_string(index=False))
print(f"Total sites: {len(gdf_sites)}, states: {sites_meta['State'].nunique()}")

# ---- PLOT ----
fig, ax = plt.subplots(figsize=(14, 10))

gdf_rfc.boundary.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=1, zorder=10, alpha=0.5, label="RFC Boundaries")
gdf_basins.plot(ax=ax, color="#90c3d4", edgecolor="steelblue", linewidth=0.6, zorder=2, alpha=0.8, label="WSF Rodeo Basins")
gdf_sites.plot(ax=ax, color="red", markersize=20, edgecolor="black", zorder=3, label="Basin Outlet/Gauge", marker="o")

for _, row in gdf_sites.iterrows():
    ax.text(
        row.geometry.x, row.geometry.y, str(int(row["basin_idx"])),
        fontsize=16, fontweight="bold", color="black", ha="left", va="bottom",
        bbox=dict(facecolor="none", edgecolor="none", alpha=0.5, pad=0.5),
    )

xmin, ymin, xmax, ymax = gdf_basins.total_bounds
x_buffer = (xmax - xmin) * 0.14
y_buffer = (ymax - ymin) * 0.1
ax.set_xlim(xmin - x_buffer, xmax + x_buffer)
ax.set_ylim(ymin - y_buffer, ymax + y_buffer)

cx.add_basemap(ax, crs=gdf_basins.crs, source=cx.providers.Esri.WorldTopoMap, alpha=1.0, zoom=6, attribution=False)

keep_rfc = ["NWRFC", "CNRFC", "CBRFC", "MBRFC", "WGRFC", "ABRFC"]
gdf_rfc_sel = gdf_rfc[gdf_rfc["NAME"].isin(keep_rfc)]

manual_label_positions = {
    "MBRFC": (-108.0, 47.0),
    "WGRFC": (-106.2, 34.8),
    "ABRFC": (-103.5, 37.0),
    "CNRFC": (-119.0, 40.0),
    "CBRFC": (-111.0, 38.5),
}

for _, row in gdf_rfc_sel.iterrows():
    name = row["NAME"]
    if row.geometry.is_empty:
        continue
    if name in manual_label_positions:
        x, y = manual_label_positions[name]
    else:
        point = row.geometry.representative_point()
        x, y = point.x, point.y
    text = ax.text(x, y, name, fontsize=16, color="black", ha="center", va="center", fontweight="bold", alpha=0.9, zorder=20)
    text.set_path_effects([path_effects.Stroke(linewidth=2.5, foreground="white"), path_effects.Normal()])

basin_patch = mpatches.Patch(facecolor="#90c3d4", edgecolor="steelblue", label="Study Basins", alpha=0.6)
rfc_line = mlines.Line2D([], [], color="black", label="NWS-RFC Boundaries", linewidth=1, alpha=0.5)
site_marker = mlines.Line2D([], [], color="red", marker="o", linestyle="None", markersize=5, markeredgecolor="black", label="Basin Outlet/Gauge")
ax.legend(handles=[basin_patch, rfc_line, site_marker], loc="upper left", fontsize=15, markerscale=1.5,
          facecolor="white", framealpha=1.0).set_zorder(50)

# ---- Scale bar (Reviewer 1 comment 4) ----
# The map is drawn in geographic lat/lon (EPSG:4326), so east-west scale varies
# with latitude. Size the bar geodesically at the latitude it's drawn at and
# report that latitude, rather than implying a single map-wide scale.
SCALEBAR_KM = 400
SCALEBAR_TICK_KM = 100
SCALEBAR_FONTSIZE = 20
map_x0, map_x1 = ax.get_xlim()
map_y0, map_y1 = ax.get_ylim()
sb_lon0 = map_x0 + 0.04 * (map_x1 - map_x0)
sb_lat = map_y0 + 0.015 * (map_y1 - map_y0)
km_per_deg_lon = Geod(ellps="WGS84").inv(sb_lon0, sb_lat, sb_lon0 + 1.0, sb_lat)[2] / 1000.0
sb_deg = SCALEBAR_KM / km_per_deg_lon
sb_h = 0.012 * (map_y1 - map_y0)
n_seg = SCALEBAR_KM // SCALEBAR_TICK_KM
for i in range(n_seg):
    ax.add_patch(mpatches.Rectangle(
        (sb_lon0 + i * sb_deg / n_seg, sb_lat), sb_deg / n_seg, sb_h,
        facecolor="black" if i % 2 == 0 else "white", edgecolor="black", linewidth=1.5, zorder=30,
    ))
for i in range(0, n_seg + 1, 2):
    ax.text(sb_lon0 + i * sb_deg / n_seg, sb_lat + 1.4 * sb_h, f"{i * SCALEBAR_TICK_KM}",
            fontsize=SCALEBAR_FONTSIZE, ha="center", va="bottom", zorder=31)
ax.text(sb_lon0 + sb_deg + 0.35, sb_lat + 0.5 * sb_h, "km", fontsize=SCALEBAR_FONTSIZE, ha="left", va="center", zorder=31)
print(f"Scale bar: {SCALEBAR_KM} km drawn at {sb_lat:.2f} N ({km_per_deg_lon:.1f} km/deg lon)")

ax.set_axis_off()
plt.tight_layout()

# ---- U.S. locator inset (Reviewer 1 comment 4) ----
# Country outlines only (Natural Earth coastlines + land borders), with the U.S.
# lightly filled; red box = main-map extent. Pinned to the map's top-right
# corner; the fixed aspect shrinks the box toward that corner.
INSET_PAD = 0.01
INSET_W = 0.31
INSET_H = 0.25
us_states = gpd.read_file(NE_STATES_SHP)
us_fill = us_states[us_states["admin"] == "United States of America"]
ax_in = ax.inset_axes([1 - INSET_PAD - INSET_W, 1 - INSET_PAD - INSET_H, INSET_W, INSET_H])
us_fill.plot(ax=ax_in, facecolor="#e6e6e6", edgecolor="none")
gpd.read_file(NE_COAST_SHP).plot(ax=ax_in, color="black", linewidth=0.5)
gpd.read_file(NE_BORDERS_SHP).plot(ax=ax_in, color="black", linewidth=0.5)
ax_in.add_patch(mpatches.Rectangle(
    (map_x0, map_y0), map_x1 - map_x0, map_y1 - map_y0,
    facecolor="none", edgecolor="red", linewidth=1.5, zorder=5,
))
ax_in.set_xlim(-128.0, -64.0)
ax_in.set_ylim(22.5, max(51.0, map_y1 + 1.0))  # main map reaches into Canada
ax_in.set_aspect(1 / np.cos(np.deg2rad(37.0)))
ax_in.set_anchor("NE")
ax_in.set_xticks([])
ax_in.set_yticks([])
ax_in.set_facecolor("white")
for spine in ax_in.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1)
ax_in.set_zorder(40)

fig_path = OUT_DIR / "fig1.site_map.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
print(f"Saved: {fig_path}")
plt.close()
