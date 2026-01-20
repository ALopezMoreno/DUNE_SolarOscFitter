import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import networkx as nx
import mplhep as hep
import time
import sys
import plotting
import posteriorHelpers
import binImportanceHelpers
from termcolor import colored


parser = argparse.ArgumentParser(description="Process bin-importance diagnostics from a derived chain.")

parser.add_argument('chains', nargs='+', help="Input MCMC derived chain files.")
parser.add_argument('-o', '--output', type=str, help="Output file (optional).")

args = parser.parse_args()

mcmc_chains = args.chains 
output_name = args.output

#########################################################
if output_name:                                         #
    print(f"saving output as {output_name}.pdf")        #
    out_pdf = f"{output_name}.pdf"                      #
else:                                                   #
    out_pdf = "images/per-bin_likelihood_report.pdf"    #
#########################################################

diagnostics = posteriorHelpers.load_bin_diagnostics(mcmc_chains)

# Helper: if a key differs across chains, take the first
A = lambda x: x[0] if isinstance(x, list) else x

# Helper: top K bins
def topk_indices(arr, k=20, mask=None):
    a = np.array(arr, copy=False)
    if mask is not None:
        a = np.where(mask, a, -np.inf)

    flat = a.ravel()
    k = min(k, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]

    if a.ndim == 1:
        return idx
    rows, cols = np.unravel_index(idx, a.shape)
    return rows, cols

VAR_EPS = 1e-12

###################################
### ENERGY THRESHOLD AND LIMITS ###
threshold = 5

limits_CC = [4, 20]
limits_ES = [2, 20]
###################################

# =========================
# Night-time binning shape
# =========================
ccnight_shape = [int(i) for i in diagnostics["derived_CCnight_shape"]]
esnight_shape = [int(i) for i in diagnostics["derived_ESnight_shape"]]

xedges_cc = np.linspace(limits_CC[0], limits_CC[1], ccnight_shape[1])
xedges_es = np.linspace(limits_ES[0], limits_ES[1], esnight_shape[1])
yedges = np.linspace(-1, 0, ccnight_shape[0])

##################################################################################
##################################################################################

bin_maps = binImportanceHelpers.build_llh_bin_maps(diagnostics, var_eps=1e-12, transpose_night=False)
pp_maps  = binImportanceHelpers.build_posterior_predictive_maps(diagnostics, transpose_night=False)
totals   = binImportanceHelpers.build_sample_total_scores(diagnostics)

print(totals)

lims = binImportanceHelpers.compute_shared_color_limits(bin_maps)

# ----------------------------
# USER SETTINGS
# ----------------------------
TRANSPOSE_NIGHT = False  # set True if your 2D arrays are (E, cosz) instead of (cosz, E)
K_TOP =  0               # top bins to overlay on score maps (optional)

def Znight(Z):
    return Z.T if TRANSPOSE_NIGHT else Z


# ----------------------------
# Pull maps from bin_maps -- We work with a -llh so the correlations flip sign
# ----------------------------
ccnight_var        = bin_maps["CCnight"]["var_llh"]
ccnight_corr_sin2  = -bin_maps["CCnight"]["corr_sin2"]
ccnight_corr_dm2   = -bin_maps["CCnight"]["corr_dm2"]
ccnight_score_sin2 = bin_maps["CCnight"]["score_sin2_norm"]   # within-sample normalized
ccnight_score_dm2  = bin_maps["CCnight"]["score_dm2_norm"]

esnight_var        = bin_maps["ESnight"]["var_llh"]
esnight_corr_sin2  = -bin_maps["ESnight"]["corr_sin2"]
esnight_corr_dm2   = -bin_maps["ESnight"]["corr_dm2"]
esnight_score_sin2 = bin_maps["ESnight"]["score_sin2_norm"]
esnight_score_dm2  = bin_maps["ESnight"]["score_dm2_norm"]

ccday_var        = bin_maps["CCday"]["var_llh"]
ccday_corr_sin2  = -bin_maps["CCday"]["corr_sin2"]
ccday_corr_dm2   = -bin_maps["CCday"]["corr_dm2"]
ccday_score_sin2   = bin_maps["CCday"]["score_sin2_norm"]
ccday_score_dm2    = bin_maps["CCday"]["score_dm2_norm"]

esday_var        = bin_maps["ESday"]["var_llh"]
esday_corr_sin2  = -bin_maps["ESday"]["corr_sin2"]
esday_corr_dm2   = -bin_maps["ESday"]["corr_dm2"]
esday_score_sin2   = bin_maps["ESday"]["score_sin2_norm"]
esday_score_dm2    = bin_maps["ESday"]["score_dm2_norm"]

# ----------------------------
# Color limits
# ----------------------------
vmin_var, vmax_var = lims["vmin_var"], lims["vmax_var"]
corr_lim           = lims["corr_lim"]

# normalized within-sample maps have a meaningful shared scale
vmin_score_sin2, vmax_score_sin2 = lims["vmin_score"], lims["vmax_score_sin2_norm"]
vmin_score_dm2,  vmax_score_dm2  = lims["vmin_score"], lims["vmax_score_dm2_norm"]

# ----------------------------
# OPTIONAL TOP-K OVERLAYS
# ----------------------------
mask_cc = Znight(ccnight_var) > 1e-12
mask_es = Znight(esnight_var) > 1e-12

top_cc_s2 = topk_indices(Znight(ccnight_score_sin2), K_TOP, mask=mask_cc) if K_TOP else None
top_cc_dm = topk_indices(Znight(ccnight_score_dm2),  K_TOP, mask=mask_cc) if K_TOP else None
top_es_s2 = topk_indices(Znight(esnight_score_sin2), K_TOP, mask=mask_es) if K_TOP else None
top_es_dm = topk_indices(Znight(esnight_score_dm2),  K_TOP, mask=mask_es) if K_TOP else None

# ----------------------------
# FIGURE / GRID
# ----------------------------
PAGE_RECT    = (0.04, 0.08, 0.85, 0.88)   # reserves right margin
SIDEBAR_RECT = (0.915, 0.16, 0.06, 0.77)  # fixed sidebar in right margin

with PdfPages(out_pdf) as pdf:
    # ============================================================
    # PAGE 1: VARIANCE (night 2D + day 1D) + narrow totals sidebar
    #   - CC night var (2D, square) + its own colorbar
    #   - CC day  var  (1D histogram)
    #   - ES night var (2D, square) + its own colorbar
    #   - ES day  var  (1D histogram)
    #   - Sidebar: 4 bars (dtotals), each bar different color, vertical labels
    # ============================================================
    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)
    fig.get_layout_engine().set(rect=PAGE_RECT)

    # Two data columns total:
    #   col0 = night variance (square 2D map)
    #   col1 = day variance  (1D histogram)
    # BUT: per-plot colorbars means we add a cbar column next to EACH subplot.
    groups = (
        plotting.GroupSpec(name="var_night", ncols=1, wspace=0.03),
        plotting.GroupSpec(name="var_day",   ncols=1, wspace=0.03),
    )

    layout = plotting.build_grouped_layout(
        fig,
        nrows=2,
        groups=groups,
        panel_ratio=1.0,
        cbar_ratio=0.06,          # width of each per-plot cbar
        cbar_mode="per_plot",     # <-- key change: one cbar per subplot
        square=True,              # keeps the NIGHT panels square
        sharey_within_group=True,
        hide_inner_ylabel=True,
    )

    axs = layout.axs          # shape (2, 2): [night | day]
    cax = layout.cax_plot     # shape (2, 2): cbar axis for each subplot


    m_cc_var = plotting.plot_binmap(
        axs[0, 0], Znight(ccnight_var),
        title=r"CC night $\mathrm{Var}(\log\mathcal{L}_{\mathrm{bin}})$",
        vmin=0.0, vmax=float(np.nanmax(Znight(ccnight_var))), cmap="cividis",
        add_colorbar=False, x_edges=xedges_cc, y_edges=yedges,
    )
    plotting.add_group_colorbar(   # if this helper is just a wrapper around fig.colorbar, it's fine to reuse
        fig, m_cc_var, cax[0, 0],
        r"$\mathrm{Var}(\log\mathcal{L}_{\mathrm{bin}})$",
        ticks_right=True
    )

    plotting.plot_binseries(
        axs[0, 1], ccday_var,
        title=r"CC day $\mathrm{Var}(\log\mathcal{L}_{\mathrm{bin}})$",
        kind="bar",
        x_edges=xedges_cc,
    )


    m_es_var = plotting.plot_binmap(
        axs[1, 0], Znight(esnight_var),
        title=r"ES night $\mathrm{Var}(\log\mathcal{L}_{\mathrm{bin}})$",
        vmin=0.0, vmax=float(np.nanmax(Znight(esnight_var))), cmap="cividis",
        add_colorbar=False, x_edges=xedges_es, y_edges=yedges,
    )
    plotting.add_group_colorbar(
        fig, m_es_var, cax[1, 0],
        r"$\mathrm{Var}(\log\mathcal{L}_{\mathrm{bin}})$",
        ticks_right=True
    )

    plotting.plot_binseries(
        axs[1, 1], esday_var,
        title=r"ES day $\mathrm{Var}(\log\mathcal{L}_{\mathrm{bin}})$",
        kind="bar",
        x_edges=xedges_es,
    )

    # We ONLY want cbars for the two 2D maps, so delete/hide the histogram cbar axes.
    for r in (0, 1):
        fig.delaxes(cax[r, 1])        # remove the (empty) histogram cbar slot

    # Skinny totals sidebar (in reserved right margin)
    plotting.add_sidebar_totals_in_margin(
        fig, totals,
        rect=SIDEBAR_RECT,
        fontsize=12
    )

    fig.suptitle(r"Per-bin likelihood diagnostics: variance (night + day)", y=.995)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)

    # ============================================================
    # PAGE 2: NIGHT CORRELATIONS (2D, ONE shared colorbar)
    # ============================================================
    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)
    fig.get_layout_engine().set(rect=PAGE_RECT)

    # Two columns: sin2 | dm2
    groups = (
        plotting.GroupSpec(name="corr", ncols=2, wspace=0.03),
    )

    layout = plotting.build_grouped_layout(
        fig,
        nrows=2,
        groups=groups,
        panel_ratio=1.0,
        cbar_ratio=0.06,
        cbar_mode="global",          # <-- SINGLE shared colorbar
        square=True,
        sharey_within_group=True,
        hide_inner_ylabel=True,
    )

    axs = layout.axs
    cax = layout.cax_global         # <-- the ONE colorbar axis

    m_corr = plotting.plot_binmap(
        axs[0, 0], Znight(ccnight_corr_sin2),
        title=r"CC night $\mathrm{Corr}(\log\mathcal{L}, \sin^2\theta_{12})$",
        vmin=-corr_lim, vmax=corr_lim,
        symmetric=False, cmap="RdBu_r",
        add_colorbar=False,
        x_edges=xedges_cc, y_edges=yedges,
    )
    plotting.plot_binmap(
        axs[1, 0], Znight(ccnight_corr_dm2),
        title=r"CC night $\mathrm{Corr}(\log\mathcal{L}, \Delta m^2_{21})$",
        vmin=-corr_lim, vmax=corr_lim,
        symmetric=False, cmap="RdBu_r",
        add_colorbar=False,
        x_edges=xedges_cc, y_edges=yedges,
    )

    plotting.plot_binmap(
        axs[0, 1], Znight(esnight_corr_sin2),
        title=r"ES night $\mathrm{Corr}(\log\mathcal{L}, \sin^2\theta_{12})$",
        vmin=-corr_lim, vmax=corr_lim,
        symmetric=False, cmap="RdBu_r",
        add_colorbar=False,
        x_edges=xedges_es, y_edges=yedges,
    )
    plotting.plot_binmap(
        axs[1, 1], Znight(esnight_corr_dm2),
        title=r"ES night $\mathrm{Corr}(\log\mathcal{L}, \Delta m^2_{21})$",
        vmin=-corr_lim, vmax=corr_lim,
        symmetric=False, cmap="RdBu_r",
        add_colorbar=False,
        x_edges=xedges_es, y_edges=yedges,
    )

    # --- single shared colorbar ---
    plotting.add_group_colorbar(
        fig, m_corr, cax,
        r"$\mathrm{Corr}(\log\mathcal{L}_{\mathrm{bin}}, \theta)$",
        ticks_right=True,
    )

    # Skinny totals sidebar (in reserved right margin)
    plotting.add_sidebar_totals_in_margin(
        fig, totals,
        rect=SIDEBAR_RECT,
        fontsize=12
    )

    fig.suptitle(
        r"Per-bin likelihood diagnostics (night): correlations",
        y=.995
    )

    pdf.savefig(fig, dpi=300)
    plt.close(fig)

    # ============================================================
    # PAGE 3: NIGHT DRIVER SCORES (2D, within-sample normalized)
    #   - CC: score(sin2), score(dm2)
    #   - ES: score(sin2), score(dm2)
    #   - colorbar max set to the max bin value (no shared vmax cap)
    # ============================================================
    vmax_score_sin2 = float(np.nanmax([np.nanmax(Znight(ccnight_score_sin2)),
                                    np.nanmax(Znight(esnight_score_sin2))]))
    vmax_score_dm2  = float(np.nanmax([np.nanmax(Znight(ccnight_score_dm2)),
                                    np.nanmax(Znight(esnight_score_dm2))]))
    vmin_score_sin2 = 0.0
    vmin_score_dm2  = 0.0

    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)
    fig.get_layout_engine().set(rect=PAGE_RECT)

    # Two columns (sin2, dm2). Per-plot colorbars so every 2D plot gets its own.
    groups = (plotting.GroupSpec(name="scores", ncols=2, wspace=0.02),)
    layout = plotting.build_grouped_layout(
        fig, nrows=2, groups=groups,
        panel_ratio=1.0, cbar_ratio=0.06,
        cbar_mode="per_plot",      # <-- per-plot colorbars
        square=True,
        sharey_within_group=True, hide_inner_ylabel=True,
    )
    axs = layout.axs
    cax = layout.cax_plot  # shape (2, 2)

    m_cc_s2 = plotting.plot_binmap(
        axs[0, 0], Znight(ccnight_score_sin2),
        title=r"CC night $(\sin^2\theta_{12})$",
        vmin=vmin_score_sin2, vmax=vmax_score_sin2, cmap="cividis",
        add_colorbar=False, x_edges=xedges_cc, y_edges=yedges, topk=top_cc_s2
    )
    m_cc_dm = plotting.plot_binmap(
        axs[1, 0], Znight(ccnight_score_dm2),
        title=r"CC night $(\Delta m^2_{21})$",
        vmin=vmin_score_dm2, vmax=vmax_score_dm2, cmap="cividis",
        add_colorbar=False, x_edges=xedges_cc, y_edges=yedges, topk=top_cc_dm
    )
    m_es_s2 = plotting.plot_binmap(
        axs[0, 1], Znight(esnight_score_sin2),
        title=r"ES night $(\sin^2\theta_{12})$",
        vmin=vmin_score_sin2, vmax=vmax_score_sin2, cmap="cividis",
        add_colorbar=False, x_edges=xedges_es, y_edges=yedges, topk=top_es_s2
    )
    m_es_dm = plotting.plot_binmap(
        axs[1, 1], Znight(esnight_score_dm2),
        title=r"ES night $(\Delta m^2_{21})$",
        vmin=vmin_score_dm2, vmax=vmax_score_dm2, cmap="cividis",
        add_colorbar=False, x_edges=xedges_es, y_edges=yedges, topk=top_es_dm
    )

    # One colorbar per 2D plot (labels can be per-parameter if you prefer)
    plotting.add_group_colorbar(fig, m_cc_s2, cax[0, 0], r"$D(\sin^2\theta_{12})$", ticks_right=True)
    plotting.add_group_colorbar(fig, m_es_s2, cax[0, 1], r"$D(\sin^2\theta_{12})$", ticks_right=True)
    plotting.add_group_colorbar(fig, m_cc_dm, cax[1, 0], r"$D(\Delta m^2_{21})$", ticks_right=True)
    plotting.add_group_colorbar(fig, m_es_dm, cax[1, 1], r"$D(\Delta m^2_{21})$", ticks_right=True)

    # Skinny totals sidebar (in reserved right margin)
    plotting.add_sidebar_totals_in_margin(
        fig, totals,
        rect=SIDEBAR_RECT,
        fontsize=12
    )

    fig.suptitle(r"Per-bin likelihood diagnostics (night): driver scores (within-sample)", y=.995)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


    # ============================================================
    # PAGE 4: DAY DRIVER SCORES (1D, within-sample normalized)
    #   - rewrite to use the new grouped layout (NO colorbars)
    #   - keep square-ish panels via set_box_aspect(1)
    # ============================================================
    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)
    fig.get_layout_engine().set(rect=PAGE_RECT)

    groups = (plotting.GroupSpec(name="day_scores", ncols=2, wspace=0.02),)
    layout = plotting.build_grouped_layout(
        fig, nrows=2, groups=groups,
        panel_ratio=1.0,
        cbar_ratio=0.06,        # ignored when cbar_mode="none"
        cbar_mode="none",       # <-- no colorbars on this page
        square=True,            # keep the same visual style (box aspect = 1)
        sharey_within_group=True, hide_inner_ylabel=True,
    )
    axs = layout.axs

    plotting.plot_binseries(
        axs[0, 0], ccday_score_sin2,
        title=r"CC day Driver score  $(\sin^2\theta_{12})$",
        kind="bar", x_edges=xedges_cc,
        #ylim=[vmin_score_sin2, vmax_score_sin2],
    )
    plotting.plot_binseries(
        axs[1, 0], ccday_score_dm2,
        title=r"CC day Driver score  $(\Delta m^2_{21})$",
        kind="bar", x_edges=xedges_cc,
        #ylim=[vmin_score_dm2, vmax_score_dm2],
    )
    plotting.plot_binseries(
        axs[0, 1], esday_score_sin2,
        title=r"ES day Driver score  $(\sin^2\theta_{12})$",
        kind="bar", x_edges=xedges_es,
        #ylim=[vmin_score_sin2, vmax_score_sin2],
    )
    plotting.plot_binseries(
        axs[1, 1], esday_score_dm2,
        title=r"ES day Driver score  $(\Delta m^2_{21})$",
        kind="bar", x_edges=xedges_es,
        #ylim=[vmin_score_dm2, vmax_score_dm2],
    )

    # Skinny totals sidebar (in reserved right margin)
    plotting.add_sidebar_totals_in_margin(
        fig, totals,
        rect=SIDEBAR_RECT,
        fontsize=12
    )

    fig.suptitle(r"Per-bin likelihood diagnostics (day): driver scores (within-sample)", y=.995)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)

    # ============================================================
    # PAGE 5: POSTERIOR PREDICTIVES (rates) + SAMPLE-TOTAL SCORES
    #   - 2D: night PP mean/var (shared limits)
    #   - 1D: day PP mean/var
    #   - bars: totals (dm2 on top row, sin2 on bottom row)
    # ============================================================
    ccday_pp_mean   = pp_maps["CCday"]["pp_mean"]
    ccday_pp_var    = pp_maps["CCday"]["pp_var"]
    esday_pp_mean   = pp_maps["ESday"]["pp_mean"]
    esday_pp_var    = pp_maps["ESday"]["pp_var"]

    ccnight_pp_mean = pp_maps["CCnight"]["pp_mean"]
    ccnight_pp_var  = pp_maps["CCnight"]["pp_var"]
    esnight_pp_mean = pp_maps["ESnight"]["pp_mean"]
    esnight_pp_var  = pp_maps["ESnight"]["pp_var"]

    vmax_pp_mean = float(np.nanmax([
        np.nanmax(ccday_pp_mean), np.nanmax(esday_pp_mean),
        np.nanmax(Znight(ccnight_pp_mean)), np.nanmax(Znight(esnight_pp_mean))
    ]))
    vmax_pp_var = float(np.nanmax([
        np.nanmax(ccday_pp_var), np.nanmax(esday_pp_var),
        np.nanmax(Znight(ccnight_pp_var)), np.nanmax(Znight(esnight_pp_var))
    ]))

    fig = plt.figure(figsize=(25, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2, ncols=7,
        width_ratios=[1, 1, 0.06, 1, 1, 0.06, 1.2],
        wspace=0.25, hspace=0.25
    )

    # Row 1 (CC)
    ax_ccn_mean = fig.add_subplot(gs[0, 0])
    ax_ccn_var  = fig.add_subplot(gs[0, 1])
    cax_ccn     = fig.add_subplot(gs[0, 2])
    ax_ccd_mean = fig.add_subplot(gs[0, 3])
    ax_ccd_var  = fig.add_subplot(gs[0, 4])
    ax_bar_top  = fig.add_subplot(gs[0, 6])

    # Row 2 (ES)
    ax_esn_mean = fig.add_subplot(gs[1, 0])
    ax_esn_var  = fig.add_subplot(gs[1, 1])
    cax_esn     = fig.add_subplot(gs[1, 2])
    ax_esd_mean = fig.add_subplot(gs[1, 3])
    ax_esd_var  = fig.add_subplot(gs[1, 4])
    ax_bar_bot  = fig.add_subplot(gs[1, 6])

    # CC night PP (share same cmap/limits for mean and var within this page)
    m_ccn_mean = plotting.plot_binmap(
        ax_ccn_mean, Znight(ccnight_pp_mean),
        title=r"CC night PP mean rate",
        vmin=0.0, vmax=vmax_pp_mean, cmap="cividis",
        add_colorbar=False, x_edges=xedges_cc, y_edges=yedges,
    )
    plotting.plot_binmap(
        ax_ccn_var, Znight(ccnight_pp_var),
        title=r"CC night PP var rate",
        vmin=0.0, vmax=vmax_pp_var, cmap="cividis",
        add_colorbar=False, x_edges=xedges_cc, y_edges=yedges,
    )
    plotting.add_group_colorbar(fig, m_ccn_mean, cax_ccn, r"PP mean (shared scale)", ticks_right=True)

    # CC day PP
    plotting.plot_binseries(
        ax_ccd_mean, ccday_pp_mean,
        title=r"CC day PP mean rate",
        kind="bar", ylabel="rate", x_edges=xedges_cc,
    )
    plotting.plot_binseries(
        ax_ccd_var, ccday_pp_var,
        title=r"CC day PP var rate",
        kind="bar", ylabel="rate", x_edges=xedges_cc,
    )

    # ES night PP
    m_esn_mean = plotting.plot_binmap(
        ax_esn_mean, Znight(esnight_pp_mean),
        title=r"ES night PP mean rate",
        vmin=0.0, vmax=vmax_pp_mean, cmap="cividis",
        add_colorbar=False, x_edges=xedges_es, y_edges=yedges,
    )
    plotting.plot_binmap(
        ax_esn_var, Znight(esnight_pp_var),
        title=r"ES night PP var rate",
        vmin=0.0, vmax=vmax_pp_var, cmap="cividis",
        add_colorbar=False, x_edges=xedges_es, y_edges=yedges,
    )
    plotting.add_group_colorbar(fig, m_esn_mean, cax_esn, r"PP mean (shared scale)", ticks_right=True)

    # ES day PP
    plotting.plot_binseries(
        ax_esd_mean, esday_pp_mean,
        title=r"ES day PP mean rate",
        kind="bar", ylabel="rate", x_edges=xedges_es,
    )
    plotting.plot_binseries(
        ax_esd_var, esday_pp_var,
        title=r"ES day PP var rate",
        kind="bar", ylabel="rate", x_edges=xedges_es,
    )

    # Sample-total dominance bars (no normalization; raw proxy)
    order = ["CCnight", "CCday", "ESnight", "ESday"]
    dm2_vals  = [totals[s]["dm2"]  for s in order]
    sin2_vals = [totals[s]["sin2"] for s in order]

    ax_bar_top.bar(order, dm2_vals)
    ax_bar_top.set_title(r"Total sample score $(\Delta m^2_{21})$")
    ax_bar_top.set_ylabel("score")
    ax_bar_top.tick_params(axis="x", rotation=45)

    ax_bar_bot.bar(order, sin2_vals)
    ax_bar_bot.set_title(r"Total sample score $(\sin^2\theta_{12})$")
    ax_bar_bot.set_ylabel("score")
    ax_bar_bot.tick_params(axis="x", rotation=45)

    fig.suptitle(r"Posterior predictives (rates) and sample-total scores", y=.995)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)

print(f"Wrote multi-page PDF: {out_pdf}")
