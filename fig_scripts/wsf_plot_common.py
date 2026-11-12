"""Shared constants and archive-loading helpers for fig_scripts/*.py.

Single source of truth for method colors/labels/order, the study window,
minimum-sample-size filtering, and the archive/local-data path resolution
repeated across figure scripts -- see CLAUDE.md's "Data conventions" section
for the protocol these encode (exceedance-probability convention, WY2004-2023
evaluation window, min-N floor). Not used by calc_stats/calc_bulk_fcst_stats.py,
which resolves paths differently by design (see that script's own comments).
"""

import os
from pathlib import Path

import numpy as np

# ---- Study period / sample-size floor ----
START_YEAR = 2004
END_YEAR = 2023
MIN_N_PAIRS = 10  # Table 1: minimum forecast-obs pairs for any cross-site plotted statistic

# ---- Initialization dates ----
INIT_MMDD_LIST = ["0101", "0201", "0301", "0401"]
INIT_LABELS = {"0101": "January 1st", "0201": "February 1st", "0301": "March 1st", "0401": "April 1st"}
INIT_LABELS_SHORT = {"0101": "Jan", "0201": "Feb", "0301": "Mar", "0401": "Apr"}

# ---- Methods: colors, labels, canonical plot order ----
OTHER_METHODS = ["nrcs", "rfc_official", "rfc_hcst", "ca_dwr"]
WSF_RODEO_METHODS = [f"wsf_rodeo_{i}place" for i in range(1, 6)]
PLOT_ORDER = OTHER_METHODS + WSF_RODEO_METHODS

METHOD_COLORS = {
    "climatology": "#dcdcdc",
    "nrcs": "#2171b5",
    "rfc_official": "#56B870",
    "rfc_hcst": "#b6e0c3",
    "ca_dwr": "#ffc107",
    "wsf_rodeo_1place": "#fc8d59",
    "wsf_rodeo_2place": "#f03b20",
    "wsf_rodeo_3place": "#bd0026",
    "wsf_rodeo_4place": "#800026",
    "wsf_rodeo_5place": "#490006",
}
METHOD_LABELS = {
    "climatology": "Climatology",
    "nrcs": "NRCS",
    "rfc_official": "RFC Official",
    "rfc_hcst": "RFC Hindcast",
    "ca_dwr": "CA DWR",
    "wsf_rodeo_1place": "WSF Rodeo 1st",
    "wsf_rodeo_2place": "WSF Rodeo 2nd",
    "wsf_rodeo_3place": "WSF Rodeo 3rd",
    "wsf_rodeo_4place": "WSF Rodeo 4th",
    "wsf_rodeo_5place": "WSF Rodeo 5th",
}

# ---- Percentile ordering (CRPS/sharpness code hardcodes f10/f50/f90 at indices [0, 2, 4]) ----
EXPECTED_PERCENTILES = np.array(["f10", "f30", "f50", "f70", "f90"])


def assert_percentile_order(percentile_labels):
    """Raise if the archive's percentile coord isn't exactly f10/f30/f50/f70/f90.

    Guards the hardcoded [0, 2, 4] (f10/f50/f90) indexing used wherever this
    is called -- without it, a reordered archive would silently score the
    wrong quantile instead of failing loudly.
    """
    if not np.array_equal(np.asarray(percentile_labels), EXPECTED_PERCENTILES):
        raise SystemExit(
            f"Archive percentile order {list(percentile_labels)} does not match "
            f"the expected order {list(EXPECTED_PERCENTILES)}; f10/f90 indices are hardcoded to it."
        )


# ---- Path resolution ----
def archive_root():
    return Path(os.environ.get("WSF_ARCHIVE_ROOT", str(Path.home() / "wsf_archive_v1.0")))


def archive_nc_path():
    return os.environ.get("WSF_ARCHIVE_NC", str(archive_root() / "proc_data" / "wsf_obs_with_all_forecasts.nc"))


def local_data_root():
    return Path(os.environ.get("WSF_LOCAL_DATA_ROOT", str(Path.home() / "projects" / "get_data" / "wsf")))


def site_meta_file():
    return local_data_root() / "WSF_Rodeo_ForecastLocations.csv"
