import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import LogLocator, NullFormatter
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

# Energy x-positions: bin centres of length n_E (used as centres for the 1D day
# series/spectra; plot_binmap only uses first/last for its extent). Prefer the
# real reco edges saved by `RunMode: derived`; fall back to hard-coded ranges.
def _energy_centres(key, fb_lo, fb_hi, n_E):
    if key in diagnostics:
        e = np.asarray(diagnostics[key], dtype=float)        # n_E+1 edges, MeV
        if e.size == n_E + 1:
            return 0.5 * (e[:-1] + e[1:])
    return np.linspace(fb_lo, fb_hi, n_E)

def _energy_edges(key, fb_lo, fb_hi, n_E):
    """Full n_E+1 reco energy bin edges (MeV) for pcolormesh / threshold lines."""
    if key in diagnostics:
        e = np.asarray(diagnostics[key], dtype=float)
        if e.size == n_E + 1:
            return e
    return np.linspace(fb_lo, fb_hi, n_E + 1)
###################################

# =========================
# Night-time binning shape   (saved shape vector is Julia (n_cosz, n_Ereco))
# =========================
esnight_shape = [int(i) for i in diagnostics["derived_ESnight_shape"]]
xedges_es = _energy_centres("derived_Ereco_edges_ES", 2, 20, esnight_shape[1])
es_E_edges = _energy_edges("derived_Ereco_edges_ES", 2, 20, esnight_shape[1])

def _underlay_day_events(ax, E_edges, ch, fade=0.30, span=0.96, floor=1e-3):
    """Translucent copy of the page-1 '{ch} day — signal+background' panel (plot 2,1),
    drawn DIRECTLY on the host axis (no twin): the same stacked log-y spectrum, rescaled
    into the panel's own y-range so it reads as a faint backdrop. Avoiding a twin axis
    means no second y-axis AND no constrained-layout size mismatch — so the background
    can never spill past the frame. The host's own x/y ranges are restored afterwards."""
    key = f"derived_signal_{ch}day"
    if key not in diagnostics:
        return
    sig = np.nan_to_num(np.asarray(diagnostics[key], dtype=float))
    if not np.any(sig > 0):
        return
    by    = dict(_bg_components(ch) or [])
    order = [n for n in _BG_ORDER if n in by] + [n for n in by if n not in _BG_ORDER]
    cum, stacks = np.zeros(len(E_edges) - 1), []
    for n in order:
        new = cum + np.nan_to_num(by[n])
        stacks.append((cum.copy(), new.copy(), _BG_COLOR.get(n, "grey"))); cum = new
    total = cum
    # Map log10(events) into the panel's current y-range (clipped to the frame by xlim).
    xlim, (ylo, yhi) = ax.get_xlim(), ax.get_ylim()
    vmax = max(np.nanmax(np.maximum(total, floor)), np.nanmax(np.maximum(sig, floor)), floor * 10)
    lmin, lmax = np.log10(floor), np.log10(vmax)
    mp = lambda v: ylo + span * (yhi - ylo) * (np.log10(np.maximum(np.asarray(v, float), floor)) - lmin) / (lmax - lmin)
    for lo, hi, col in stacks:                                       # stacked fills
        ax.fill_between(E_edges, _estep(mp(np.maximum(lo, floor))), _estep(mp(hi)),
                        step="post", color=col, alpha=0.25 * fade, lw=0, zorder=0.0)
    for lo, hi, _c in stacks[:-1]:                                   # thin dividers
        ax.step(E_edges, _estep(mp(hi)), where="post", color="black", lw=1.0, alpha=0.8 * fade, zorder=0.1)
    if stacks:                                                       # total outline
        ax.step(E_edges, _estep(mp(total)), where="post", color="black", lw=2.5, alpha=fade, zorder=0.1)
    ax.step(E_edges, _estep(mp(np.where(sig > 0, sig, floor))),      # ν signal line
            where="post", color="blue", lw=2.5, alpha=0.9 * fade, zorder=0.1)
    ax.set_xlim(xlim); ax.set_ylim(ylo, yhi)        # underlay must not alter the panel's own ranges

# Diagonal-hatch style for the day diagnostic bars (variance AND driver-score panels) so
# the translucent underlay shows through them. Hatch line thickness is the global
# hatch.linewidth rcParam; the patch linewidth controls the bar outline.
plt.rcParams["hatch.linewidth"] = 2.4
_DIAG_BAR = dict(facecolor="none", edgecolor="C0", hatch="///", linewidth=1.2)

# cosz y-edges for the 2D night maps (piecewise-uniform over the exposure
# support, NOT a uniform [-1, 0] grid). imshow uses only first/last for extent.
if "derived_cosz_edges" in diagnostics:
    yedges = np.asarray(diagnostics["derived_cosz_edges"], dtype=float)
else:
    yedges = np.linspace(-1, 0, esnight_shape[0] + 1)

if not inclusive_mode:
    ccnight_shape = [int(i) for i in diagnostics["derived_CCnight_shape"]]
    xedges_cc = _energy_centres("derived_Ereco_edges_CC", 4, 20, ccnight_shape[1])
    cc_E_edges = _energy_edges("derived_Ereco_edges_CC", 4, 20, ccnight_shape[1])

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


def _cosz_density(Z, cosz_edges, power=1):
    """Night maps (n_cosz, n_E) of EXTENSIVE quantities scale with the cos z bin width, so
    with non-uniform bins the per-bin colour dips in the narrow (fine) bins purely from
    geometry. Divide each row by Δcos z**power to show a DENSITY that is continuous across
    the binning change: rates ∝ Δcz (power=1), per-bin Var(logL) ∝ Δcz² (power=2), driver
    score = √Var·|corr| ∝ Δcz (power=1). power=0 is a no-op for INTENSIVE quantities
    (correlations and probabilities — already binning-independent)."""
    if power == 0:
        return np.asarray(Z, dtype=float)
    w = np.diff(np.asarray(cosz_edges, dtype=float))      # (n_cosz,)
    return np.asarray(Z, dtype=float) / (w[:, None] ** power)


# ----------------------------
# Pull maps from bin_maps -- We work with a -llh so the correlations flip sign
# Night variance/score scale with cos z bin width (Var ∝ Δcz², score ∝ Δcz); divide it out
# so the colour is a density, continuous across the non-uniform bins. Correlations are
# intensive (ratio) and the day panels use uniform E, so neither is rescaled.
# ----------------------------
esnight_var        = _cosz_density(bin_maps["ESnight"]["var_llh"], yedges, 2)
esnight_corr_sin2  = -bin_maps["ESnight"]["corr_sin2"]
esnight_corr_dm2   = -bin_maps["ESnight"]["corr_dm2"]
esnight_score_sin2 = _cosz_density(bin_maps["ESnight"]["score_sin2_norm"], yedges, 1)
esnight_score_dm2  = _cosz_density(bin_maps["ESnight"]["score_dm2_norm"], yedges, 1)

esday_var        = bin_maps["ESday"]["var_llh"]
esday_corr_sin2  = -bin_maps["ESday"]["corr_sin2"]
esday_corr_dm2   = -bin_maps["ESday"]["corr_dm2"]
esday_score_sin2   = bin_maps["ESday"]["score_sin2_norm"]
esday_score_dm2    = bin_maps["ESday"]["score_dm2_norm"]

if not inclusive_mode:
    ccnight_var        = _cosz_density(bin_maps["CCnight"]["var_llh"], yedges, 2)
    ccnight_corr_sin2  = -bin_maps["CCnight"]["corr_sin2"]
    ccnight_corr_dm2   = -bin_maps["CCnight"]["corr_dm2"]
    ccnight_score_sin2 = _cosz_density(bin_maps["CCnight"]["score_sin2_norm"], yedges, 1)   # within-sample normalized
    ccnight_score_dm2  = _cosz_density(bin_maps["CCnight"]["score_dm2_norm"], yedges, 1)

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


# ============================================================
# OSCILLATED-SAMPLE PAGE(S) (rendered first), one per channel
#   2 rows  : signal-only (oscillated, no bkg)  /  signal + background
#   2 cols  : day = 1D rate vs reconstructed energy
#             night = 2D map (E_reco on x, cos θ_z solar angle on y, events in colour)
#   Full reco energy range; red vertical line at the analysis energy threshold.
#   "signal" = Asimov sample − posterior-mean background (the oscillated+reco signal);
#   "signal+bkg" = the full Asimov sample.
# ============================================================
def _logsafe(arr):
    """Copy with non-positive entries set to NaN (blank on log scale / pcolormesh)."""
    a = np.array(arr, dtype=float)
    a[~(a > 0)] = np.nan
    return a

def _bg_components(ch):
    """Per-component background day spectra (list of (name, spectrum)) if saved, else None.
    Saved as Julia (n_Ereco, n_comp) → h5py (n_comp, n_Ereco); names are a comma string."""
    key = f"derived_{ch}bg_comp_day"
    if key not in diagnostics:
        return None
    comp = np.atleast_2d(np.asarray(diagnostics[key], dtype=float))   # (n_comp, n_Ereco)
    nm = diagnostics.get(f"derived_{ch}bg_comp_names", "")
    nm = nm.item() if hasattr(nm, "item") else nm
    nm = nm.decode() if isinstance(nm, (bytes, bytearray)) else str(nm)
    names = nm.split(",") if nm else [f"bkg {i+1}" for i in range(comp.shape[0])]
    return list(zip(names, comp))

# temp_5.py-style background colours/labels and bottom→top stack order
_BG_COLOR = {"gamma": "orange", "neutron": "green", "radiological": "red", "alpha": "purple"}
_BG_LABEL = {"gamma": r"$\gamma$", "neutron": r"$n$", "radiological": "radiological", "alpha": r"$\alpha$"}
_BG_ORDER = ["gamma", "neutron", "radiological", "alpha"]

def _estep(v):
    """Append last bin value so a step='post' fill/line covers the final bin edge."""
    v = np.asarray(v, dtype=float)
    return np.append(v, v[-1])

def _day_panel(ax, E_edges, signal, comps, floor=1e-3):
    """temp_5.py log-y style: stacked-fill backgrounds (γ→n→radiological), thin black
    dividers + thick total outline, ν signal as a thick blue line (not stacked)."""
    xx = E_edges
    if comps:
        by = dict(comps)
        order = [n for n in _BG_ORDER if n in by] + [n for n in by if n not in _BG_ORDER]
        cum, tops = np.zeros(len(E_edges) - 1), []
        for n in order:
            new = cum + np.nan_to_num(by[n])
            ax.fill_between(xx, np.maximum(_estep(cum), floor), np.maximum(_estep(new), floor),
                            step="post", color=_BG_COLOR.get(n, "grey"), alpha=0.25,
                            label=_BG_LABEL.get(n, n))
            tops.append(new); cum = new
        for t in tops[:-1]:                                   # thin dividers between layers
            ax.step(xx, np.maximum(_estep(t), floor), where="post", color="black", lw=1, alpha=0.8)
        ax.step(xx, np.maximum(_estep(tops[-1]), floor), where="post", color="black", lw=2.5)  # total
    sig = np.where(np.asarray(signal, dtype=float) > 0, signal, np.nan)
    ax.step(xx, _estep(sig), where="post", color="blue", lw=2.5, alpha=0.9, label=r"$\nu$ signal")
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    ax.yaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10), numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=4)

def add_sample_page(pdf, ch, E_edges, idx):
    # Full Asimov sample (signal+bkg) and the clean signal-only sample (no bkg),
    # both at the Asimov truth and over the full reco range (saved by derive mode).
    data_day  = np.asarray(diagnostics[f"derived_data_{ch}day"],   dtype=float)
    sig_day   = np.asarray(diagnostics[f"derived_signal_{ch}day"], dtype=float)
    data_night = binImportanceHelpers._night_to_plot_layout(diagnostics[f"derived_data_{ch}night"])
    sig_night  = binImportanceHelpers._night_to_plot_layout(diagnostics[f"derived_signal_{ch}night"])
    bg_comps   = _bg_components(ch)

    # Skip a channel with no sample (e.g. ES in a CC-only run)
    if np.nansum(np.abs(data_day)) == 0 and np.nansum(np.abs(data_night)) == 0:
        return
    E_thr  = E_edges[idx]                      # lower edge of first above-threshold bin
    cosz_e = yedges
    cz_cen = 0.5 * (cosz_e[:-1] + cosz_e[1:])

    # Same grouped/square layout engine as the per-bin diagnostic pages.
    fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.02)
    fig.get_layout_engine().set(rect=PAGE_RECT)
    groups = (plotting.GroupSpec(name="day",   ncols=1, wspace=0.03),
              plotting.GroupSpec(name="night", ncols=1, wspace=0.03))
    layout = plotting.build_grouped_layout(
        fig, nrows=2, groups=groups, panel_ratio=1.0, cbar_ratio=0.06,
        cbar_mode="per_plot", square=True, sharey_within_group=False, hide_inner_ylabel=True,
    )
    axs, cax = layout.axs, layout.cax_plot

    rows = [("signal (oscillated, no bkg)", sig_night,  None),
            ("signal + background",         data_night, bg_comps)]

    for r, (label, dnight, comps) in enumerate(rows):
        # --- day 1D (left column), temp_5.py log-y style; ν signal line is always
        #     the no-bkg signal, backgrounds are stacked only on the signal+bkg row ---
        ax = axs[r, 0]
        _day_panel(ax, E_edges, sig_day, comps)
        ax.axvline(E_thr, color="red", lw=1.3)
        ax.set_xlim(E_edges[0], E_edges[-1])
        ax.set_xlabel(r"$E_{reco}(MeV)$", fontsize=20)
        ax.set_ylabel("Events", fontsize=20)
        ax.tick_params(axis="both", labelsize=18)
        ax.legend(fontsize=11, loc="upper right", ncol=1, frameon=False)
        ax.set_title(f"{ch} day — {label}", fontsize=12)
        fig.delaxes(cax[r, 0])                  # day has no colorbar

        # --- night 2D map (right column), log colour, matched to plot_binmap style ---
        ax = axs[r, 1]
        Z = _logsafe(_cosz_density(dnight, cosz_e))   # events / Δcos z (continuous across bin widths)
        if np.any(Z > 0):
            vmax = np.nanmax(Z)
            vmin = max(np.nanmin(Z[Z > 0]), vmax / 1e4)   # 4-decade clip → exposure band visible
            # pcolormesh with the real (non-uniform) cos z edges so the horizon-refined
            # fine bins display at their true heights, not squashed into equal pixels.
            m = ax.pcolormesh(E_edges, cosz_e, Z, cmap="viridis",
                              norm=LogNorm(vmin=vmin, vmax=vmax), shading="flat")
            ax.set_xlim(E_edges[0], E_edges[-1]); ax.set_ylim(cosz_e[0], cosz_e[-1])
            cb = fig.colorbar(m, cax=cax[r, 1]); cb.ax.yaxis.set_ticks_position("right")
            cb.set_label(r"Events / $\Delta\cos\theta_z$", rotation=90, fontsize=20); cb.ax.tick_params(labelsize=16)
            proj = np.nansum(np.maximum(dnight[:, idx:], 0.0), axis=1)   # above-thr cosz profile
            if np.any(proj > 0):                                         # mark exposure peak
                ax.axhline(cz_cen[np.nanargmax(proj)], color="w", ls=":", lw=0.9)
        else:
            fig.delaxes(cax[r, 1])
        ax.axvline(E_thr, color="red", lw=1.3)
        ax.set_xlabel(r"$E_{reco}(MeV)$", fontsize=20)
        ax.set_ylabel(r"$\cos\theta_z$", fontsize=20)
        ax.tick_params(axis="both", labelsize=18)
        ax.set_title(f"{ch} night — {label}", fontsize=12)

    fig.suptitle(f"{ch} oscillated sample: day (energy) and night (energy × solar angle)", y=0.995)
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def add_oscmap_page(pdf):
    """Best-fit (posterior-mean) oscillation map P(νe→νe) for 8B: the day survival curve
    (1D) and the night regeneration map (2D, true energy × cos z). Computed in derive mode
    at the posterior-mean parameters; skipped silently if those keys aren't present."""
    if "derived_oscmap_8B_night" not in diagnostics:
        return
    night = np.asarray(diagnostics["derived_oscmap_8B_night"], dtype=float).T   # (n_cosz, n_Etrue)
    day   = np.asarray(diagnostics["derived_oscmap_8B_day"],   dtype=float)      # (n_Etrue,)
    if "derived_oscmap_Etrue_edges" in diagnostics:
        Et = np.asarray(diagnostics["derived_oscmap_Etrue_edges"], dtype=float)
    else:
        Et = np.linspace(1.0, 21.0, day.size + 1)

    fig, (axd, axn) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True,
                                   gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    # day survival probability P_ee(E_true)
    axd.step(Et, np.append(day, day[-1]), where="post", color="C0", lw=2.5)
    axd.set_xlim(Et[0], Et[-1]); axd.set_ylim(bottom=0.0)
    axd.set_xlabel(r"$E_{true}$ (MeV)", fontsize=18)
    axd.set_ylabel(r"$P(\nu_e\!\to\!\nu_e)$", fontsize=18)
    axd.set_title("day", fontsize=13)
    axd.tick_params(labelsize=14)
    # night regeneration map — plot_binmap uses pcolormesh, so the graded cos z bins show
    plotting.plot_binmap(axn, night, x_edges=0.5 * (Et[:-1] + Et[1:]), y_edges=yedges,
                         title="night (regeneration)", cmap="viridis",
                         add_colorbar=True, cbar_kw=dict(label=r"$P(\nu_e\!\to\!\nu_e)$"))
    axn.set_xlabel(r"$E_{true}$ (MeV)", fontsize=18)   # override plot_binmap's E_reco label
    axn.set_ylabel(r"$\cos\theta_z$", fontsize=18)
    fig.suptitle(r"Best-fit oscillation map  $P(\nu_e\!\to\!\nu_e)$  ($^8$B)", y=1.06)
    pdf.savefig(fig, dpi=300); plt.close(fig)


def _recover_exposure_weights(ch):
    """Per-cos z exposure fraction. Prefer the saved array; else recover it from the saved
    background, since the derive builds BG_night(c,e) = exposure_weights(c) · BG_day(e) exactly,
    so exposure_weights = BG_night[:, e0] / BG_day[e0] for any e0 with appreciable background."""
    if "derived_exposure_weights" in diagnostics:
        return np.asarray(diagnostics["derived_exposure_weights"], dtype=float)
    bn = diagnostics.get(f"derived_BG{ch}night_pp_mean")
    bd = diagnostics.get(f"derived_BG{ch}day_pp_mean")
    if bn is None or bd is None:
        return None
    bn = np.asarray(bn, dtype=float)   # h5py (n_Ereco, n_cosz)
    bd = np.asarray(bd, dtype=float)   # (n_Ereco,)
    e0 = int(np.nanargmax(bd))
    if not np.isfinite(bd[e0]) or bd[e0] <= 0:
        return None
    return bn[e0, :] / bd[e0]


def add_dnratio_page(pdf):
    """Day-night asymmetry A = 2(P_day − P_night)/(P_day + P_night): truth (vs E_true, from the
    best-fit osc map) beside the reconstructed signal (vs E_reco). Forming the asymmetry cancels
    flux × cross-section (and the smooth solar MSW), and the per-cos z exposure cancels too — what
    remains is the Earth day-night structure, so the two panels show directly how much of it
    survives the energy reconstruction. Skipped if the required keys are absent."""
    need = ["derived_oscmap_8B_night", "derived_oscmap_8B_day"]
    if any(k not in diagnostics for k in need):
        return
    ch, E_edges = (("CC", cc_E_edges) if (not inclusive_mode and "derived_signal_CCnight" in diagnostics)
                   else ("ES", es_E_edges))
    if f"derived_signal_{ch}night" not in diagnostics or f"derived_signal_{ch}day" not in diagnostics:
        return
    expw = _recover_exposure_weights(ch)
    if expw is None:
        return

    # truth asymmetry over E_true:  A = 2(P_day − P_night)/(P_day + P_night)
    on = np.asarray(diagnostics["derived_oscmap_8B_night"], dtype=float).T    # (n_cosz, n_Etrue)
    od = np.asarray(diagnostics["derived_oscmap_8B_day"],   dtype=float)       # (n_Etrue,)
    true_asym = 2.0 * (od[None, :] - on) / (od[None, :] + on)
    Et = (np.asarray(diagnostics["derived_oscmap_Etrue_edges"], dtype=float)
          if "derived_oscmap_Etrue_edges" in diagnostics else np.linspace(1.0, 21.0, on.shape[1] + 1))

    # reco asymmetry over E_reco. The no-regeneration "day equivalent" at each cos z is
    # exposure × signal_day; the exposure cancels in the asymmetry, leaving A in terms of the
    # response-averaged ⟨P_day⟩, ⟨P_night⟩ — i.e. the reco-smeared day-night asymmetry.
    sn = binImportanceHelpers._night_to_plot_layout(diagnostics[f"derived_signal_{ch}night"])  # (n_cosz, n_Ereco)
    sd = np.asarray(diagnostics[f"derived_signal_{ch}day"], dtype=float)                        # (n_Ereco,)
    with np.errstate(divide="ignore", invalid="ignore"):
        day_eq    = expw[:, None] * sd[None, :]               # night signal expected with no regeneration
        reco_asym = 2.0 * (day_eq - sn) / (day_eq + sn)
    reco_asym[~np.isfinite(reco_asym)] = np.nan
    reco_asym[:, sd < 1e-2 * np.nanmax(sd)] = np.nan          # blank bins with negligible signal

    # shared diverging scale centred at 0
    allv = np.concatenate([true_asym[np.isfinite(true_asym)].ravel(),
                           reco_asym[np.isfinite(reco_asym)].ravel()])
    d = float(np.nanpercentile(np.abs(allv), 99)) if allv.size else 0.05
    d = max(d, 1e-3)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-d, vmax=d)

    fig, (axt, axr) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axt.pcolormesh(Et, yedges, true_asym, cmap="RdBu_r", norm=norm, shading="flat")
    axt.set_title("truth  ($E_{true}$)", fontsize=12)
    axt.set_xlabel(r"$E_{true}$ (MeV)", fontsize=16); axt.set_ylabel(r"$\cos\theta_z$", fontsize=16)
    m = axr.pcolormesh(E_edges, yedges, reco_asym, cmap="RdBu_r", norm=norm, shading="flat")
    axr.set_title(f"{ch} reconstructed  ($E_{{reco}}$)", fontsize=12)
    axr.set_xlabel(r"$E_{reco}$ (MeV)", fontsize=16)
    fig.colorbar(m, ax=[axt, axr], fraction=0.046, pad=0.02,
                 label=r"$2(P_{\nu_e}^{\,\mathrm{day}}-P_{\nu_e}^{\,\mathrm{night}})/(P_{\nu_e}^{\,\mathrm{day}}+P_{\nu_e}^{\,\mathrm{night}})$")
    fig.suptitle("Day-night asymmetry — oscillation structure surviving reconstruction")
    pdf.savefig(fig, dpi=300, bbox_inches="tight"); plt.close(fig)


with PdfPages(out_pdf) as pdf:
    idx_es = int(np.asarray(diagnostics["derived_index_ES"])) - 1
    if not inclusive_mode:
        idx_cc = int(np.asarray(diagnostics["derived_index_CC"])) - 1
        add_sample_page(pdf, "CC", cc_E_edges, idx_cc)
    add_sample_page(pdf, "ES", es_E_edges, idx_es)
    add_oscmap_page(pdf)   # best-fit P(νe→νe) day curve + night regeneration map
    add_dnratio_page(pdf)  # day-night survival ratio: truth (E_true) vs reconstructed (E_reco)

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
            r"$\mathrm{Var}(\log\mathcal{L}_{\mathrm{bin}})\,/\,\Delta\cos\theta_z^{2}$",
            ticks_right=True
        )
        plotting.plot_binseries(
            axs[0, 1], ccday_var,
            title=r"CC day $\mathrm{Var}(\log\mathcal{L}_{\mathrm{bin}})$",
            kind="bar", bar_kwargs=_DIAG_BAR,
            x_edges=xedges_cc,
        )
        _underlay_day_events(axs[0, 1], cc_E_edges, "CC")
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
        r"$\mathrm{Var}(\log\mathcal{L}_{\mathrm{bin}})\,/\,\Delta\cos\theta_z^{2}$",
        ticks_right=True
    )
    plotting.plot_binseries(
        axs[es_row, 1], esday_var,
        title=r"ES day $\mathrm{Var}(\log\mathcal{L}_{\mathrm{bin}})$",
        kind="bar", bar_kwargs=_DIAG_BAR,
        x_edges=xedges_es,
    )
    _underlay_day_events(axs[es_row, 1], es_E_edges, "ES")

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
        plotting.add_group_colorbar(fig, m_cc_s2, cax[0, 0], r"$D(\sin^2\theta_{12})\,/\,\Delta\cos\theta_z$", ticks_right=True)
        plotting.add_group_colorbar(fig, m_cc_dm, cax[1, 0], r"$D(\Delta m^2_{21})\,/\,\Delta\cos\theta_z$", ticks_right=True)
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
    plotting.add_group_colorbar(fig, m_es_s2, cax[0, es_col], r"$D(\sin^2\theta_{12})\,/\,\Delta\cos\theta_z$", ticks_right=True)
    plotting.add_group_colorbar(fig, m_es_dm, cax[1, es_col], r"$D(\Delta m^2_{21})\,/\,\Delta\cos\theta_z$", ticks_right=True)

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
            kind="bar", x_edges=xedges_cc, bar_kwargs=_DIAG_BAR,
        )
        _underlay_day_events(axs[0, 0], cc_E_edges, "CC")
        plotting.plot_binseries(
            axs[1, 0], ccday_score_dm2,
            title=r"CC day Driver score  $(\Delta m^2_{21})$",
            kind="bar", x_edges=xedges_cc, bar_kwargs=_DIAG_BAR,
        )
        _underlay_day_events(axs[1, 0], cc_E_edges, "CC")
        es_col = 1
    else:
        es_col = 0

    plotting.plot_binseries(
        axs[0, es_col], esday_score_sin2,
        title=r"ES day Driver score  $(\sin^2\theta_{12})$",
        kind="bar", x_edges=xedges_es, bar_kwargs=_DIAG_BAR,
    )
    _underlay_day_events(axs[0, es_col], es_E_edges, "ES")
    plotting.plot_binseries(
        axs[1, es_col], esday_score_dm2,
        title=r"ES day Driver score  $(\Delta m^2_{21})$",
        kind="bar", x_edges=xedges_es, bar_kwargs=_DIAG_BAR,
    )
    _underlay_day_events(axs[1, es_col], es_E_edges, "ES")

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

        # Night rate maps scale with cos z bin width (rate ∝ Δcz, its variance ∝ Δcz²);
        # show as a DENSITY so the colour is continuous across the non-uniform bins.
        cz_pow = 1 if pp_label == "Signal rate" else 2
        cz_lab = pp_label + (r" / $\Delta\cos\theta_z$" if cz_pow == 1 else r" / $\Delta\cos\theta_z^2$")
        if not inclusive_mode:
            ccn = _cosz_density(Znight(ccnight_pp), yedges, cz_pow)
            m_ccn = plotting.plot_binmap(
                axs[0, 0], ccn,
                title=r"CC night — " + pp_label,
                vmin=0.0, vmax=float(np.nanmax(ccn)), cmap="cividis",
                add_colorbar=False, x_edges=xedges_cc, y_edges=yedges,
            )
            plotting.add_group_colorbar(fig, m_ccn, cax[0, 0], cz_lab, ticks_right=True)
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

        esn = _cosz_density(Znight(esnight_pp), yedges, cz_pow)
        m_esn = plotting.plot_binmap(
            axs[es_row, 0], esn,
            title=r"ES night — " + pp_label,
            vmin=0.0, vmax=float(np.nanmax(esn)), cmap="cividis",
            add_colorbar=False, x_edges=xedges_es, y_edges=yedges,
        )
        plotting.add_group_colorbar(fig, m_esn, cax[es_row, 0], cz_lab, ticks_right=True)
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
