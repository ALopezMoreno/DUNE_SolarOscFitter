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

# Detect inclusive mode: CC diagnostic keys were not computed for this chain
inclusive_mode = diagnostics.get("inclusive_mode", False)
if inclusive_mode:
    print("Inclusive mode: CC panels will be omitted.")

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
esnight_shape = [int(i) for i in diagnostics["derived_ESnight_shape"]]
xedges_es = np.linspace(limits_ES[0], limits_ES[1], esnight_shape[1])
yedges = np.linspace(-1, 0, esnight_shape[0])

if not inclusive_mode:
    ccnight_shape = [int(i) for i in diagnostics["derived_CCnight_shape"]]
    xedges_cc = np.linspace(limits_CC[0], limits_CC[1], ccnight_shape[1])

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
esnight_var        = bin_maps["ESnight"]["var_llh"]
esnight_corr_sin2  = -bin_maps["ESnight"]["corr_sin2"]
esnight_corr_dm2   = -bin_maps["ESnight"]["corr_dm2"]
esnight_score_sin2 = bin_maps["ESnight"]["score_sin2_norm"]
esnight_score_dm2  = bin_maps["ESnight"]["score_dm2_norm"]

esday_var        = bin_maps["ESday"]["var_llh"]
esday_corr_sin2  = -bin_maps["ESday"]["corr_sin2"]
esday_corr_dm2   = -bin_maps["ESday"]["corr_dm2"]
esday_score_sin2   = bin_maps["ESday"]["score_sin2_norm"]
esday_score_dm2    = bin_maps["ESday"]["score_dm2_norm"]

if not inclusive_mode:
    ccnight_var        = bin_maps["CCnight"]["var_llh"]
    ccnight_corr_sin2  = -bin_maps["CCnight"]["corr_sin2"]
    ccnight_corr_dm2   = -bin_maps["CCnight"]["corr_dm2"]
    ccnight_score_sin2 = bin_maps["CCnight"]["score_sin2_norm"]   # within-sample normalized
    ccnight_score_dm2  = bin_maps["CCnight"]["score_dm2_norm"]

    ccday_var        = bin_maps["CCday"]["var_llh"]
    ccday_corr_sin2  = -bin_maps["CCday"]["corr_sin2"]
    ccday_corr_dm2   = -bin_maps["CCday"]["corr_dm2"]
    ccday_score_sin2   = bin_maps["CCday"]["score_sin2_norm"]
    ccday_score_dm2    = bin_maps["CCday"]["score_dm2_norm"]

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
mask_es = Znight(esnight_var) > 1e-12
top_es_s2 = topk_indices(Znight(esnight_score_sin2), K_TOP, mask=mask_es) if K_TOP else None
top_es_dm = topk_indices(Znight(esnight_score_dm2),  K_TOP, mask=mask_es) if K_TOP else None

if not inclusive_mode:
    mask_cc = Znight(ccnight_var) > 1e-12
    top_cc_s2 = topk_indices(Znight(ccnight_score_sin2), K_TOP, mask=mask_cc) if K_TOP else None
    top_cc_dm = topk_indices(Znight(ccnight_score_dm2),  K_TOP, mask=mask_cc) if K_TOP else None

# ----------------------------
# FIGURE / GRID
# ----------------------------
PAGE_RECT    = (0.04, 0.08, 0.85, 0.88)   # reserves right margin
SIDEBAR_RECT = (0.915, 0.16, 0.06, 0.77)  # fixed sidebar in right margin

# Sidebar order: omit CC samples in inclusive mode
sidebar_order = ("ESnight", "ESday") if inclusive_mode else ("CCnight", "CCday", "ESnight", "ESday")

with PdfPages(out_pdf) as pdf:
    # ============================================================
    # PAGE 1: VARIANCE (night 2D + day 1D) + narrow totals sidebar
    #   Non-inclusive: CC row + ES row
    #   Inclusive:     ES row only
    # ============================================================
    nrows_p1 = 1 if inclusive_mode else 2

    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)
    fig.get_layout_engine().set(rect=PAGE_RECT)

    groups = (
        plotting.GroupSpec(name="var_night", ncols=1, wspace=0.03),
        plotting.GroupSpec(name="var_day",   ncols=1, wspace=0.03),
    )

    layout = plotting.build_grouped_layout(
        fig,
        nrows=nrows_p1,
        groups=groups,
        panel_ratio=1.0,
        cbar_ratio=0.06,
        cbar_mode="per_plot",
        square=True,
        sharey_within_group=True,
        hide_inner_ylabel=True,
    )

    axs = layout.axs
    cax = layout.cax_plot

    if not inclusive_mode:
        m_cc_var = plotting.plot_binmap(
            axs[0, 0], Znight(ccnight_var),
            title=r"CC night $\mathrm{Var}(\log\mathcal{L}_{\mathrm{bin}})$",
            vmin=0.0, vmax=float(np.nanmax(Znight(ccnight_var))), cmap="cividis",
            add_colorbar=False, x_edges=xedges_cc, y_edges=yedges,
        )
        plotting.add_group_colorbar(
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
        es_row = 1
    else:
        es_row = 0

    m_es_var = plotting.plot_binmap(
        axs[es_row, 0], Znight(esnight_var),
        title=r"ES night $\mathrm{Var}(\log\mathcal{L}_{\mathrm{bin}})$",
        vmin=0.0, vmax=float(np.nanmax(Znight(esnight_var))), cmap="cividis",
        add_colorbar=False, x_edges=xedges_es, y_edges=yedges,
    )
    plotting.add_group_colorbar(
        fig, m_es_var, cax[es_row, 0],
        r"$\mathrm{Var}(\log\mathcal{L}_{\mathrm{bin}})$",
        ticks_right=True
    )
    plotting.plot_binseries(
        axs[es_row, 1], esday_var,
        title=r"ES day $\mathrm{Var}(\log\mathcal{L}_{\mathrm{bin}})$",
        kind="bar",
        x_edges=xedges_es,
    )

    # Remove empty histogram cbar slots
    for r in range(nrows_p1):
        fig.delaxes(cax[r, 1])

    plotting.add_sidebar_totals_in_margin(
        fig, totals,
        rect=SIDEBAR_RECT,
        fontsize=12,
        order=sidebar_order,
    )

    title_suffix = " (ES only — inclusive mode)" if inclusive_mode else ""
    fig.suptitle(r"Per-bin likelihood diagnostics: variance (night + day)" + title_suffix, y=.995)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)

    # ============================================================
    # PAGE 2: NIGHT CORRELATIONS (2D, ONE shared colorbar)
    #   Non-inclusive: 2 columns (CC, ES) × 2 rows (sin2, dm2)
    #   Inclusive:     1 column (ES)       × 2 rows
    # ============================================================
    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)
    fig.get_layout_engine().set(rect=PAGE_RECT)

    ncols_p2 = 1 if inclusive_mode else 2
    groups = (
        plotting.GroupSpec(name="corr", ncols=ncols_p2, wspace=0.03),
    )

    layout = plotting.build_grouped_layout(
        fig,
        nrows=2,
        groups=groups,
        panel_ratio=1.0,
        cbar_ratio=0.06,
        cbar_mode="global",
        square=True,
        sharey_within_group=True,
        hide_inner_ylabel=True,
    )

    axs = layout.axs
    cax = layout.cax_global

    if not inclusive_mode:
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
        es_col = 1
    else:
        es_col = 0

    m_corr_es = plotting.plot_binmap(
        axs[0, es_col], Znight(esnight_corr_sin2),
        title=r"ES night $\mathrm{Corr}(\log\mathcal{L}, \sin^2\theta_{12})$",
        vmin=-corr_lim, vmax=corr_lim,
        symmetric=False, cmap="RdBu_r",
        add_colorbar=False,
        x_edges=xedges_es, y_edges=yedges,
    )
    plotting.plot_binmap(
        axs[1, es_col], Znight(esnight_corr_dm2),
        title=r"ES night $\mathrm{Corr}(\log\mathcal{L}, \Delta m^2_{21})$",
        vmin=-corr_lim, vmax=corr_lim,
        symmetric=False, cmap="RdBu_r",
        add_colorbar=False,
        x_edges=xedges_es, y_edges=yedges,
    )

    # Use whichever mappable was plotted first for the shared colorbar
    m_corr_ref = m_corr if not inclusive_mode else m_corr_es
    plotting.add_group_colorbar(
        fig, m_corr_ref, cax,
        r"$\mathrm{Corr}(\log\mathcal{L}_{\mathrm{bin}}, \theta)$",
        ticks_right=True,
    )

    plotting.add_sidebar_totals_in_margin(
        fig, totals,
        rect=SIDEBAR_RECT,
        fontsize=12,
        order=sidebar_order,
    )

    fig.suptitle(
        r"Per-bin likelihood diagnostics (night): correlations" + title_suffix,
        y=.995
    )

    pdf.savefig(fig, dpi=300)
    plt.close(fig)

    # ============================================================
    # PAGE 3: NIGHT DRIVER SCORES (2D, within-sample normalized)
    #   Non-inclusive: CC col + ES col, 2 rows (sin2, dm2)
    #   Inclusive:     ES col only
    # ============================================================
    if not inclusive_mode:
        vmax_score_sin2 = float(np.nanmax([np.nanmax(Znight(ccnight_score_sin2)),
                                           np.nanmax(Znight(esnight_score_sin2))]))
        vmax_score_dm2  = float(np.nanmax([np.nanmax(Znight(ccnight_score_dm2)),
                                           np.nanmax(Znight(esnight_score_dm2))]))
    else:
        vmax_score_sin2 = float(np.nanmax(Znight(esnight_score_sin2)))
        vmax_score_dm2  = float(np.nanmax(Znight(esnight_score_dm2)))
    vmin_score_sin2 = 0.0
    vmin_score_dm2  = 0.0

    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)
    fig.get_layout_engine().set(rect=PAGE_RECT)

    ncols_p3 = 1 if inclusive_mode else 2
    groups = (plotting.GroupSpec(name="scores", ncols=ncols_p3, wspace=0.02),)
    layout = plotting.build_grouped_layout(
        fig, nrows=2, groups=groups,
        panel_ratio=1.0, cbar_ratio=0.06,
        cbar_mode="per_plot",
        square=True,
        sharey_within_group=True, hide_inner_ylabel=True,
    )
    axs = layout.axs
    cax = layout.cax_plot

    if not inclusive_mode:
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
        plotting.add_group_colorbar(fig, m_cc_s2, cax[0, 0], r"$D(\sin^2\theta_{12})$", ticks_right=True)
        plotting.add_group_colorbar(fig, m_cc_dm, cax[1, 0], r"$D(\Delta m^2_{21})$", ticks_right=True)
        es_col = 1
    else:
        es_col = 0

    m_es_s2 = plotting.plot_binmap(
        axs[0, es_col], Znight(esnight_score_sin2),
        title=r"ES night $(\sin^2\theta_{12})$",
        vmin=vmin_score_sin2, vmax=vmax_score_sin2, cmap="cividis",
        add_colorbar=False, x_edges=xedges_es, y_edges=yedges, topk=top_es_s2
    )
    m_es_dm = plotting.plot_binmap(
        axs[1, es_col], Znight(esnight_score_dm2),
        title=r"ES night $(\Delta m^2_{21})$",
        vmin=vmin_score_dm2, vmax=vmax_score_dm2, cmap="cividis",
        add_colorbar=False, x_edges=xedges_es, y_edges=yedges, topk=top_es_dm
    )
    plotting.add_group_colorbar(fig, m_es_s2, cax[0, es_col], r"$D(\sin^2\theta_{12})$", ticks_right=True)
    plotting.add_group_colorbar(fig, m_es_dm, cax[1, es_col], r"$D(\Delta m^2_{21})$", ticks_right=True)

    plotting.add_sidebar_totals_in_margin(
        fig, totals,
        rect=SIDEBAR_RECT,
        fontsize=12,
        order=sidebar_order,
    )

    fig.suptitle(r"Per-bin likelihood diagnostics (night): driver scores (within-sample)" + title_suffix, y=.995)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


    # ============================================================
    # PAGE 4: DAY DRIVER SCORES (1D, within-sample normalized)
    #   Non-inclusive: CC col + ES col, 2 rows
    #   Inclusive:     ES col only
    # ============================================================
    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)
    fig.get_layout_engine().set(rect=PAGE_RECT)

    ncols_p4 = 1 if inclusive_mode else 2
    groups = (plotting.GroupSpec(name="day_scores", ncols=ncols_p4, wspace=0.02),)
    layout = plotting.build_grouped_layout(
        fig, nrows=2, groups=groups,
        panel_ratio=1.0,
        cbar_ratio=0.06,
        cbar_mode="none",
        square=True,
        sharey_within_group=True, hide_inner_ylabel=True,
    )
    axs = layout.axs

    if not inclusive_mode:
        plotting.plot_binseries(
            axs[0, 0], ccday_score_sin2,
            title=r"CC day Driver score  $(\sin^2\theta_{12})$",
            kind="bar", x_edges=xedges_cc,
        )
        plotting.plot_binseries(
            axs[1, 0], ccday_score_dm2,
            title=r"CC day Driver score  $(\Delta m^2_{21})$",
            kind="bar", x_edges=xedges_cc,
        )
        es_col = 1
    else:
        es_col = 0

    plotting.plot_binseries(
        axs[0, es_col], esday_score_sin2,
        title=r"ES day Driver score  $(\sin^2\theta_{12})$",
        kind="bar", x_edges=xedges_es,
    )
    plotting.plot_binseries(
        axs[1, es_col], esday_score_dm2,
        title=r"ES day Driver score  $(\Delta m^2_{21})$",
        kind="bar", x_edges=xedges_es,
    )

    plotting.add_sidebar_totals_in_margin(
        fig, totals,
        rect=SIDEBAR_RECT,
        fontsize=12,
        order=sidebar_order,
    )

    fig.suptitle(r"Per-bin likelihood diagnostics (day): driver scores (within-sample)" + title_suffix, y=.995)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)

    # ============================================================
    # PAGES 5–6: P-PREDICTIVE RATES (mean on p5, variance on p6)
    #   Each page: night 2D + day 1D, matching pages 1–4 layout
    #   Non-inclusive: CC row + ES row; Inclusive: ES row only
    # ============================================================
    esday_pp_mean   = pp_maps["ESday"]["pp_mean"]
    esday_pp_var    = pp_maps["ESday"]["pp_var"]
    esday_pp_lo     = pp_maps["ESday"].get("pp_lo")
    esday_pp_hi     = pp_maps["ESday"].get("pp_hi")
    esday_signal_mean = pp_maps["ESday"].get("signal_pp_mean") if pp_maps["ESday"].get("signal_pp_mean") is not None else esday_pp_mean
    bg_esday_mean   = pp_maps["ESday"].get("bg_mean")
    bg_esday_var    = pp_maps["ESday"].get("bg_var")
    esnight_pp_mean = pp_maps["ESnight"]["pp_mean"]
    esnight_pp_var  = pp_maps["ESnight"]["pp_var"]

    # Statistical (Poisson) error bars on data; systematic uncertainty lives in the PPD bands.
    _data_es = pp_maps["ESday"].get("data")
    data_esday_signal = (_data_es - bg_esday_mean) if (_data_es is not None and bg_esday_mean is not None) else _data_es
    err_esday = np.sqrt(np.maximum(_data_es, 0.0)) if _data_es is not None else None

    # Build nested credible bands (3σ outermost → 1σ innermost)
    if esday_pp_lo is not None and pp_maps["ESday"].get("pp_lo3") is not None:
        esday_pp_bands = [
            (pp_maps["ESday"]["pp_lo3"], pp_maps["ESday"]["pp_hi3"]),
            (pp_maps["ESday"]["pp_lo2"], pp_maps["ESday"]["pp_hi2"]),
            (esday_pp_lo, esday_pp_hi),
        ]
    elif esday_pp_lo is not None:
        esday_pp_bands = [(esday_pp_lo, esday_pp_hi)]
    else:
        esday_pp_bands = None

    if not inclusive_mode:
        ccday_pp_mean   = pp_maps["CCday"]["pp_mean"]
        ccday_pp_var    = pp_maps["CCday"]["pp_var"]
        ccday_pp_lo     = pp_maps["CCday"].get("pp_lo")
        ccday_pp_hi     = pp_maps["CCday"].get("pp_hi")
        ccday_signal_mean = pp_maps["CCday"].get("signal_pp_mean") if pp_maps["CCday"].get("signal_pp_mean") is not None else ccday_pp_mean
        bg_ccday_mean   = pp_maps["CCday"].get("bg_mean")
        bg_ccday_var    = pp_maps["CCday"].get("bg_var")
        ccnight_pp_mean = pp_maps["CCnight"]["pp_mean"]
        ccnight_pp_var  = pp_maps["CCnight"]["pp_var"]

        _data_cc = pp_maps["CCday"].get("data")
        data_ccday_signal = (_data_cc - bg_ccday_mean) if (_data_cc is not None and bg_ccday_mean is not None) else _data_cc
        err_ccday = np.sqrt(np.maximum(_data_cc, 0.0)) if _data_cc is not None else None

        if ccday_pp_lo is not None and pp_maps["CCday"].get("pp_lo3") is not None:
            ccday_pp_bands = [
                (pp_maps["CCday"]["pp_lo3"], pp_maps["CCday"]["pp_hi3"]),
                (pp_maps["CCday"]["pp_lo2"], pp_maps["CCday"]["pp_hi2"]),
                (ccday_pp_lo, ccday_pp_hi),
            ]
        elif ccday_pp_lo is not None:
            ccday_pp_bands = [(ccday_pp_lo, ccday_pp_hi)]
        else:
            ccday_pp_bands = None

    nrows_pp = 1 if inclusive_mode else 2

    # Tuple fields: (esnight, esday_mean, esday_bands,
    #                ccnight, ccday_mean, ccday_bands,
    #                esday_data, esday_err, ccday_data, ccday_err,
    #                label, title, use_t2k_day)
    pp_pages = [
        (esnight_pp_mean, esday_signal_mean, esday_pp_bands,
         ccnight_pp_mean        if not inclusive_mode else None,
         ccday_signal_mean      if not inclusive_mode else None,
         ccday_pp_bands         if not inclusive_mode else None,
         data_esday_signal, err_esday,
         data_ccday_signal      if not inclusive_mode else None,
         err_ccday              if not inclusive_mode else None,
         "Signal rate", "P-predictive signal rates (background subtracted)", True),
        (esnight_pp_var, esday_pp_var, None,
         ccnight_pp_var  if not inclusive_mode else None,
         ccday_pp_var    if not inclusive_mode else None,
         None,
         _data_es, None,
         pp_maps["CCday"].get("data") if not inclusive_mode else None, None,
         "Rate variance", "P-predictive rate variance", False),
    ]

    for (esnight_pp, esday_pp_mean_pg, esday_bands_pg,
         ccnight_pp, ccday_pp_mean_pg, ccday_bands_pg,
         esday_data_pg, esday_err_pg, ccday_data_pg, ccday_err_pg,
         pp_label, page_title, use_t2k_day) in pp_pages:

        fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)
        fig.get_layout_engine().set(rect=PAGE_RECT)

        groups = (
            plotting.GroupSpec(name="pp_night", ncols=1, wspace=0.03),
            plotting.GroupSpec(name="pp_day",   ncols=1, wspace=0.03),
        )
        layout = plotting.build_grouped_layout(
            fig, nrows=nrows_pp, groups=groups,
            panel_ratio=1.0, cbar_ratio=0.06,
            cbar_mode="per_plot",
            square=True,
            sharey_within_group=True, hide_inner_ylabel=True,
        )
        axs = layout.axs
        cax = layout.cax_plot

        if not inclusive_mode:
            m_ccn = plotting.plot_binmap(
                axs[0, 0], Znight(ccnight_pp),
                title=r"CC night — " + pp_label,
                vmin=0.0, vmax=float(np.nanmax(Znight(ccnight_pp))), cmap="cividis",
                add_colorbar=False, x_edges=xedges_cc, y_edges=yedges,
            )
            plotting.add_group_colorbar(fig, m_ccn, cax[0, 0], pp_label, ticks_right=True)
            if use_t2k_day:
                plotting.plot_predictive_spectrum(
                    axs[0, 1], ccday_pp_mean_pg, bands=ccday_bands_pg, x_centers=xedges_cc,
                    data=ccday_data_pg, data_err=ccday_err_pg,
                    total_rate=ccday_pp_mean,
                    bg_var=bg_ccday_var,
                    title=r"CC day — P-predictive (signal only)",
                    ylabel="Events",
                )
            else:
                plotting.plot_binseries(
                    axs[0, 1], ccday_pp_mean_pg,
                    title=r"CC day — " + pp_label,
                    kind="bar", ylabel=pp_label, x_edges=xedges_cc,
                )
            fig.delaxes(cax[0, 1])
            es_row = 1
        else:
            es_row = 0

        m_esn = plotting.plot_binmap(
            axs[es_row, 0], Znight(esnight_pp),
            title=r"ES night — " + pp_label,
            vmin=0.0, vmax=float(np.nanmax(Znight(esnight_pp))), cmap="cividis",
            add_colorbar=False, x_edges=xedges_es, y_edges=yedges,
        )
        plotting.add_group_colorbar(fig, m_esn, cax[es_row, 0], pp_label, ticks_right=True)
        if use_t2k_day:
            plotting.plot_predictive_spectrum(
                axs[es_row, 1], esday_pp_mean_pg, bands=esday_bands_pg, x_centers=xedges_es,
                data=esday_data_pg, data_err=esday_err_pg,
                total_rate=esday_pp_mean,
                bg_var=bg_esday_var,
                title=r"ES day — P-predictive (signal only)",
                ylabel="Events",
            )
        else:
            plotting.plot_binseries(
                axs[es_row, 1], esday_pp_mean_pg,
                title=r"ES day — " + pp_label,
                kind="bar", ylabel=pp_label, x_edges=xedges_es,
            )
        fig.delaxes(cax[es_row, 1])

        plotting.add_sidebar_totals_in_margin(
            fig, totals, rect=SIDEBAR_RECT, fontsize=12, order=sidebar_order,
        )
        fig.suptitle(page_title + title_suffix, y=.995)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

print(f"Wrote multi-page PDF: {out_pdf}")
