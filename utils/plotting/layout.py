"""Grouped grid-layout engine and grouped colorbars / sidebar totals."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import colors
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Literal


@dataclass(frozen=True)
class GroupSpec:
    """Defines one group of panels (data columns)."""
    name: str
    ncols: int
    wspace: float = 0.02  # currently unused in this interleaved approach


CBarMode = Literal["none", "global", "per_group", "per_plot"]


@dataclass
class GridLayout:
    """
    Holds axes for data panels and any colorbar axes.

    - axs: shape (nrows, n_data_cols)
    - cax_global: one Axes if cbar_mode == "global"
    - cax_group: group name -> Axes if cbar_mode == "per_group"
    - cax_plot: shape (nrows, n_data_cols) if cbar_mode == "per_plot"
    """
    axs: np.ndarray
    cax_global: Optional[plt.Axes] = None
    cax_group: Dict[str, plt.Axes] = field(default_factory=dict)
    cax_plot: Optional[np.ndarray] = None


def build_grouped_layout(
    fig: plt.Figure,
    nrows: int,
    groups: Tuple[GroupSpec, ...],
    *,
    panel_ratio: float = 1.0,
    cbar_ratio: float = 0.06,
    cbar_mode: CBarMode = "per_group",
    square: bool = True,
    sharey_within_group: bool = True,
    hide_inner_ylabel: bool = True,
) -> GridLayout:
    """
    Build an interleaved outer GridSpec.

    Colorbar modes:
      - "none":     data panels only
      - "global":   data panels + ONE global colorbar column at the far right (spans all rows)
      - "per_group":each group gets ONE colorbar column after its data columns (spans all rows)
      - "per_plot": each individual subplot gets its own colorbar column immediately to its right
                   (so columns are [panel][cbar][panel][cbar]...)

    Returns a GridLayout with the appropriate cax populated.
    """

    n_data_cols = sum(g.ncols for g in groups)

    # --- determine total outer columns and width ratios ---
    width_ratios_outer: list[float] = []
    if cbar_mode == "per_plot":
        # For each data column: [panel][cbar]
        ncols_outer = n_data_cols * 2
        for _ in range(n_data_cols):
            width_ratios_outer.extend([panel_ratio, cbar_ratio])
    else:
        # Data columns first, optionally with group/global cbar columns interleaved/added
        if cbar_mode == "none":
            ncols_outer = n_data_cols
            width_ratios_outer = [panel_ratio] * n_data_cols

        elif cbar_mode == "global":
            ncols_outer = n_data_cols + 1
            width_ratios_outer = [panel_ratio] * n_data_cols + [cbar_ratio]

        elif cbar_mode == "per_group":
            ncols_outer = n_data_cols + len(groups)
            for g in groups:
                width_ratios_outer.extend([panel_ratio] * g.ncols)
                width_ratios_outer.append(cbar_ratio)

        else:
            raise ValueError(f"Unknown cbar_mode: {cbar_mode!r}")

    gs_outer = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols_outer,
        width_ratios=width_ratios_outer,
    )

    axs = np.empty((nrows, n_data_cols), dtype=object)

    cax_global: Optional[plt.Axes] = None
    cax_group: Dict[str, plt.Axes] = {}
    cax_plot: Optional[np.ndarray] = None
    if cbar_mode == "per_plot":
        cax_plot = np.empty((nrows, n_data_cols), dtype=object)

    # --- build axes ---
    outer_col = 0     # column cursor into gs_outer
    data_col = 0      # column cursor into axs

    for g in groups:
        for j in range(g.ncols):
            # Create data axes for this data column (all rows)
            for r in range(nrows):
                ax = fig.add_subplot(gs_outer[r, outer_col])
                axs[r, data_col] = ax

            # Share y within group across its columns (for each row)
            # We share to the first column of the group for that row.
            if sharey_within_group and g.ncols > 1 and j > 0:
                for r in range(nrows):
                    axs[r, data_col].sharey(axs[r, data_col - j])  # group's first col for that row
                    if hide_inner_ylabel:
                        axs[r, data_col].tick_params(labelleft=False)

            if square:
                for r in range(nrows):
                    axs[r, data_col].set_box_aspect(1)

            # If per-plot cbar, allocate a cbar axis right next to this panel
            if cbar_mode == "per_plot":
                for r in range(nrows):
                    cax_plot[r, data_col] = fig.add_subplot(gs_outer[r, outer_col + 1])
                outer_col += 2
            else:
                outer_col += 1

            data_col += 1

        # If per-group cbar, allocate one cbar column spanning all rows after group's data
        if cbar_mode == "per_group":
            cax_group[g.name] = fig.add_subplot(gs_outer[:, outer_col])
            outer_col += 1

    # If global cbar, allocate it at the far right spanning all rows
    if cbar_mode == "global":
        cax_global = fig.add_subplot(gs_outer[:, -1])

    return GridLayout(axs=axs, cax_global=cax_global, cax_group=cax_group, cax_plot=cax_plot)


def add_group_colorbar(
    fig: plt.Figure,
    mappable,
    cax: plt.Axes,
    label: str,
    *,
    ticks_right: bool = True,
    labelpad: float = 2,
    labelsize: float = 20,
    ticksize: float = 20,
    tickpad: float = 2,
    shift_left: float = 0.0,
    sci_range: tuple[float, float] = (0.1, 100.0),
    powerlimits: tuple[int, int] = (-2, 3),
    exponent_size: float | None = None,   # NEW
):
    # --- physically move the cax left ---
    if shift_left != 0.0:
        pos = cax.get_position()
        cax.set_position([pos.x0 - shift_left, pos.y0, pos.width, pos.height])

    cbar = fig.colorbar(mappable, cax=cax)

    # --- force initial tick creation ---
    cbar.update_ticks()

    # --- inspect tick locations ---
    ticks = np.asarray(cbar.get_ticks(), dtype=float)
    ticks = ticks[np.isfinite(ticks)]

    lo, hi = sci_range
    abs_ticks = np.abs(ticks[ticks != 0])

    use_sci = True
    if abs_ticks.size > 0:
        use_sci = not (abs_ticks.min() >= lo and abs_ticks.max() <= hi)

    # --- formatter ---
    formatter = ScalarFormatter(useMathText=True)
    if use_sci:
        formatter.set_scientific(True)
        formatter.set_powerlimits(powerlimits)
    else:
        formatter.set_scientific(False)

    cbar.formatter = formatter
    cbar.update_ticks()

    # --- offset text styling (×10^n) ---
    offset = cbar.ax.yaxis.get_offset_text()
    if exponent_size is not None:
        offset.set_fontsize(exponent_size)
    else:
        offset.set_fontsize(ticksize)

    # --- labels & ticks ---
    cbar.set_label(label, rotation=90, fontsize=labelsize)
    cbar.ax.yaxis.labelpad = labelpad
    cbar.ax.tick_params(labelsize=ticksize, pad=tickpad)

    if ticks_right:
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.yaxis.set_label_position("right")
    else:
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")

    return cbar


def add_sidebar_totals_in_margin(fig, totals, which="dm2",
                                order=("CCnight", "CCday", "ESnight", "ESday"),
                                rect=(0.945, 0.12, 0.045, 0.76),
                                fontsize=8):
    """
    Two stacked sidebar bar plots (dm2 on top, sin2 on bottom)
    - x tick labels shown
    - NO tick marks
    - visible separation between panels
    - axes box preserved
    """

    left, bottom, width, height = rect
    gap = 0.05 * height
    h = (height - gap) / 2.0

    ax_sin2  = fig.add_axes((left, bottom + h + gap, width, h))
    ax_dm2 = fig.add_axes((left, bottom,             width, h),
                           sharex=ax_sin2)

    vals_dm2  = [totals[s]["dm2"]  for s in order]
    vals_sin2 = [totals[s]["sin2"] for s in order]

    colors = ["darkred", "lightsalmon", "midnightblue", "dodgerblue"]

    ax_dm2.bar(order,  vals_dm2,  color=colors[:len(order)], width=0.8)
    ax_sin2.bar(order, vals_sin2, color=colors[:len(order)], width=0.8)

    ax_dm2.set_title(r"$\Delta m^2_{21}$", fontsize=fontsize*1.3, pad=6)
    ax_sin2.set_title(r"$\sin^2\theta_{12}$", fontsize=fontsize*1.3, pad=5)

    for ax in (ax_dm2, ax_sin2):
        # keep ticks for labels, but make them invisible
        ax.tick_params(axis="x",
                       length=0,        # <<< NO TICK LINES
                       labelrotation=90,
                       labelsize=fontsize)
        ax.tick_params(axis="y",
                       length=0,
                       labelleft=False)

        ax.grid(True, axis="y", alpha=0.2)

    # show x labels only on bottom panel
    ax_sin2.tick_params(labelbottom=False)

    _fmt = {"CCnight": "CC-night", "CCday": "CC-day", "ESnight": "ES-night", "ESday": "ES-day"}
    labels = [_fmt.get(s, s) for s in order]

    for ax in (ax_dm2, ax_sin2):
        ax.minorticks_off()
        ax.tick_params(axis="x", length=0, labelrotation=90, labelsize=fontsize)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels)

    return ax_dm2, ax_sin2
