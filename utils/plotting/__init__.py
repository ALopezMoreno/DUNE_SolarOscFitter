"""SolarOscFitter plotting package.

Split out of the former monolithic ``plotting.py``. The public API is re-exported
here so existing ``import plotting; plotting.X(...)`` call sites keep working
unchanged. Importing this package applies the shared CMS/LaTeX style (via
``style``) as a side effect, matching the old module's behaviour.
"""
import style  # noqa: F401  applies the shared CMS/LaTeX rcParams on import
from style import save_figure, ensure_parent_dir

from .colormaps import (parula_map, parula_map_r, scaled_cmap,
                        create_sequential_colormap, fade_color_to_white)
from .formatting import format_func, two_sig_figs_math, gaussian, sigma_to_prob
from .contours import get_contours, overlay_contours, add_external_solar_data
from .cornerplot import plot_corner, plot_default_corner
from .covariance import plot_covariance
from .predictive import plot_binmap, plot_predictive_spectrum, plot_binseries
from .layout import (GroupSpec, GridLayout, build_grouped_layout,
                     add_group_colorbar, add_sidebar_totals_in_margin)
