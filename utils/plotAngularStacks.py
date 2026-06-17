import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import os
import plotting   # sets CMS/LaTeX rcParams

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Multi-stage pipeline diagnostic plots from *_angular_stacks.jld2."
)
parser.add_argument("infile", help="Path to *_angular_stacks.jld2 written by the fitter.")
parser.add_argument("-o", "--output", type=str,
                    help="Output file path (no extension). Default: images/<stem>_debug.pdf")
parser.add_argument("--n-panels", type=int, default=6,
                    help="Angular-slice panels on the combined-stacks page (default: 6).")
parser.add_argument("--n-cosz-slices", type=int, default=6,
                    help="Night zenith slices shown on the angular-night page (default: 6).")
parser.add_argument("--stages", choices=["all", "angular"], default="all",
                    help="'all' = full 7-page report (default); 'angular' = combined-stacks page only.")
args = parser.parse_args()

stem = os.path.splitext(os.path.basename(args.infile))[0]
if args.output:
    print(f"Saving output as {args.output}.pdf")
    out_pdf = f"{args.output}.pdf"
elif args.stages == "angular":
    out_pdf = f"images/{stem}_angular.pdf"
else:
    out_pdf = f"images/{stem}_debug.pdf"

os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
ES_FILL  = "#ff6b6b";  ES_EDGE  = "#b33939"
CC_FILL  = "#70a1ff";  CC_EDGE  = "#1e3799"
BG_FILL  = "#888888";  BG_EDGE  = "#333333"
NUE_COL  = "#0894bc";  NUOTHER_COL = "#a41007"
LW      = 1.5
LW_HIST = 3.5   # thicker lines for dedicated histogram page

# ── Load data ─────────────────────────────────────────────────────────────────
# JLD2 is HDF5-compatible.  Julia stores arrays in column-major order; HDF5 +
# h5py reverse the dimension order, so a Julia (m, n) matrix arrives in Python
# as (n, m).  We call .T on all multi-dimensional arrays after loading.
with h5py.File(args.infile, "r") as f:

    # backward-compat top-level keys
    ES_angular = f["ES_angular"][:].T   # (n_cos, n_E)
    CC_angular = f["CC_angular"][:].T
    BG_angular = f["BG_angular"][:].T
    Ereco_min  = float(f["Ereco_min"][()])
    Ereco_max  = float(f["Ereco_max"][()])
    cos_min    = float(f["cos_min"][()])
    cos_max    = float(f["cos_max"][()])

    # axis metadata (only present in files written by save_debug_data)
    full_data = "meta/Etrue_n" in f

    # defaults for legacy files that predate the mode-flag meta keys
    has_angular     = True   # old files always had angular distributions
    has_ES_mode     = True
    has_CC_separate = False
    has_CC          = False
    has_CC_incl     = False
    cosz_edges_real = None   # real (possibly non-uniform) night bin edges, if saved

    if full_data:
        Etrue_min       = float(f["meta/Etrue_min"][()])
        Etrue_max       = float(f["meta/Etrue_max"][()])
        Etrue_n         = int(f["meta/Etrue_n"][()])
        Ereco_ES_min    = float(f["meta/Ereco_ES_min"][()])
        Ereco_ES_max    = float(f["meta/Ereco_ES_max"][()])
        Ereco_ES_n      = int(f["meta/Ereco_ES_n"][()])
        cos_scatter_min = float(f["meta/cos_scatter_min"][()])
        cos_scatter_max = float(f["meta/cos_scatter_max"][()])
        cos_scatter_n   = int(f["meta/cos_scatter_n"][()])
        cosz_min        = float(f["meta/cosz_min"][()])
        cosz_max        = float(f["meta/cosz_max"][()])
        cosz_n          = int(f["meta/cosz_n"][()])
        cosz_edges_real = f["meta/cosz_edges"][:] if "meta/cosz_edges" in f else None
        has_CC          = bool(f["meta/has_CC"][()])
        has_CC_incl     = bool(f["meta/has_CC_inclusive"][()])
        has_angular     = bool(f["meta/has_angular"][()]) if "meta/has_angular" in f else False
        has_ES_mode     = bool(f["meta/has_ES_mode"][()]) if "meta/has_ES_mode" in f else True
        has_CC_separate = bool(f["meta/has_CC_separate"][()]) if "meta/has_CC_separate" in f else False

        # unoscillated
        u_ES_nue_8B     = f["unosc/ES_nue_8B"][:]
        u_ES_nuother_8B = f["unosc/ES_nuother_8B"][:]
        u_CC_8B         = f["unosc/CC_8B"][:]
        u_ES_nue_hep    = f["unosc/ES_nue_hep"][:]
        u_ES_nuother_hep= f["unosc/ES_nuother_hep"][:]
        u_CC_hep        = f["unosc/CC_hep"][:]

        # response matrices — guard on mode; Julia (n_Etrue, n_Ereco) → h5py → .T
        resp_ES_nue     = f["resp/ES_nue"][:].T     if has_ES_mode and "resp/ES_nue"     in f else None
        resp_ES_nuother = f["resp/ES_nuother"][:].T if has_ES_mode and "resp/ES_nuother" in f else None
        resp_ES_angular = f["resp/ES_angular"][:].T if has_angular and "resp/ES_angular" in f else None
        resp_CC         = f["resp/CC"][:].T          if has_CC      else None
        resp_CC_incl    = f["resp/CC_inclusive"][:].T if has_CC_incl else None
        Ereco_CC_min    = float(f["meta/Ereco_CC_min"][()]) if has_CC else Ereco_ES_min
        Ereco_CC_max    = float(f["meta/Ereco_CC_max"][()]) if has_CC else Ereco_ES_max
        Ereco_CC_n      = int(f["meta/Ereco_CC_n"][()])     if has_CC else 0

        # oscillation probabilities
        op_nue_8B_day    = f["osc_probs/nue_8B_day"][:]
        op_nue_8B_night  = f["osc_probs/nue_8B_night"][:].T   # (n_cosz, n_Etrue)
        op_nue_hep_day   = f["osc_probs/nue_hep_day"][:]
        op_nue_hep_night = f["osc_probs/nue_hep_night"][:].T

        # oscillated spectra (Etrue space)
        osc_ES_nue_day      = f["oscillated/ES_nue_day"][:]     if "oscillated/ES_nue_day"      in f else None
        osc_ES_nuother_day  = f["oscillated/ES_nuother_day"][:] if "oscillated/ES_nuother_day"  in f else None
        osc_ES_nue_night    = f["oscillated/ES_nue_night"][:].T if "oscillated/ES_nue_night"    in f else None
        osc_ES_nuother_night= f["oscillated/ES_nuother_night"][:].T if "oscillated/ES_nuother_night" in f else None
        osc_CC_day          = f["oscillated/CC_day"][:]         if "oscillated/CC_day"          in f else None
        osc_CC_night        = f["oscillated/CC_night"][:].T     if "oscillated/CC_night"        in f else None

        # reco-space spectra (always saved in this format)
        ereco_ES_day    = f["ereco/ES_day"][:]
        ereco_ES_night  = f["ereco/ES_night"][:].T    # (n_cosz, n_Ereco_ES)
        ereco_CC_day    = f["ereco/CC_day"][:]
        ereco_CC_night  = f["ereco/CC_night"][:].T    # (n_cosz, n_Ereco_CC)
        ereco_BG_ES_day = f["ereco/BG_ES_day"][:]

        # angular distributions — only present when angular_reco=true
        if has_angular and "angular/ES_day" in f:
            ES_angular_day      = f["angular/ES_day"][:].T   # (n_cos, n_Ereco_ES)
            # Julia (n_cos, n_Ereco, n_cosz) → h5py (n_cosz, n_Ereco, n_cos) → .T → (n_cos, n_Ereco, n_cosz)
            ES_angular_night_3d = f["angular/ES_night_3d"][:].T
            CC_angular_night_3d = f["angular/CC_night_3d"][:].T if "angular/CC_night_3d" in f else None
            BG_angular_night_3d = f["angular/BG_night_3d"][:].T if "angular/BG_night_3d" in f else None
        else:
            ES_angular_day      = None
            ES_angular_night_3d = None
            CC_angular_night_3d = None
            BG_angular_night_3d = None

if full_data:
    n_E   = Ereco_ES_n
    n_cos = cos_scatter_n
    E_edges_reco   = np.linspace(Ereco_ES_min, Ereco_ES_max, n_E + 1)
    E_centers_reco = 0.5 * (E_edges_reco[:-1] + E_edges_reco[1:])
    cos_edges      = np.linspace(cos_scatter_min, cos_scatter_max, n_cos + 1)
else:
    n_cos, n_E = ES_angular.shape
    E_edges_reco   = np.linspace(Ereco_min, Ereco_max, n_E + 1)
    E_centers_reco = 0.5 * (E_edges_reco[:-1] + E_edges_reco[1:])
    cos_edges      = np.linspace(cos_min, cos_max, n_cos + 1)

# Mask for energy bins above the analysis threshold (Ereco_min+1 = 3 MeV for default config).
# Bins below this threshold contain huge backgrounds not used in the fit, which swamp
# the angular plots when summed over energy.
e_thresh_mask = E_centers_reco >= (Ereco_min + 1)
e_5mev_mask   = E_centers_reco >= 5.0
e_10mev_mask  = E_centers_reco >= 10.0

# ── Helpers ───────────────────────────────────────────────────────────────────
def filled_stairs(ax, edges, y, fill_color, edge_color, label=None, lw=LW):
    ax.stairs(y, edges, fill=True, color=fill_color, edgecolor=edge_color,
              linewidth=lw, label=label, baseline=0)

def line_stairs(ax, edges, y, color, label=None, lw=LW):
    ax.stairs(y, edges, color=color, linewidth=lw, label=label, baseline=None)

def heatmap(ax, Z, x_edges, y_edges, xlabel, ylabel, title, cmap="cividis", density=False):
    # density=True: divide by the y-bin width (Δcos z) so EXTENSIVE rate maps show a density
    # that is continuous across non-uniform cos z bins (per-bin events ∝ Δcos z otherwise).
    Z = np.asarray(Z, dtype=float)
    if density:
        Z = Z / np.diff(np.asarray(y_edges, dtype=float))[:, None]
    im = ax.pcolormesh(x_edges, y_edges, Z, cmap=cmap, shading="flat")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im

# ── Main PDF ──────────────────────────────────────────────────────────────────
with PdfPages(out_pdf) as pdf:

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1: Unoscillated spectra (true energy, flux × cross-section)
    # ══════════════════════════════════════════════════════════════════════════
    if full_data and args.stages == "all":
        E_edges_true   = np.linspace(Etrue_min, Etrue_max, Etrue_n + 1)

        specs = [
            (u_ES_nue_8B,     r"$\nu_e$ ES  (8B)",      NUE_COL),
            (u_ES_nuother_8B, r"$\nu_x$ ES  (8B)",      NUOTHER_COL),
            (u_CC_8B,         r"CC  (8B)",               CC_EDGE),
            (u_ES_nue_hep,    r"$\nu_e$ ES  (HEP)",     NUE_COL),
            (u_ES_nuother_hep,r"$\nu_x$ ES  (HEP)",     NUOTHER_COL),
            (u_CC_hep,        r"CC  (HEP)",              CC_EDGE),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
        for (ax, (y, title, col)) in zip(axes.ravel(), specs):
            line_stairs(ax, E_edges_true, y, col)
            ax.set_xlabel(r"$E_{\rm true}$ [MeV]")
            ax.set_ylabel("Events / bin")
            ax.set_title(title)
        fig.suptitle("Unoscillated spectra (flux $\\times$ cross-section $\\times$ detector)", y=1.001)
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2: Response matrices
    # ══════════════════════════════════════════════════════════════════════════
    if full_data and args.stages == "all":
        E_edges_true  = np.linspace(Etrue_min, Etrue_max, Etrue_n + 1)
        E_edges_ES    = np.linspace(Ereco_ES_min, Ereco_ES_max, Ereco_ES_n + 1)
        cos_sc_edges  = np.linspace(cos_scatter_min, cos_scatter_max, cos_scatter_n + 1)

        def col_norm(Z):
            col_sums = Z.sum(axis=0, keepdims=True)
            return np.where(col_sums > 0, Z / col_sums, 0.0)

        panels = []
        if resp_ES_nue is not None:
            panels.append((col_norm(resp_ES_nue.T),     E_edges_true, E_edges_ES,   r"$E_{true}$ [MeV]", r"$E_{reco}$ ES [MeV]", r"ES $\nu_e$ response"))
        if resp_ES_nuother is not None:
            panels.append((col_norm(resp_ES_nuother.T), E_edges_true, E_edges_ES,   r"$E_{true}$ [MeV]", r"$E_{reco}$ ES [MeV]", r"ES $\nu_x$ response"))
        if resp_ES_angular is not None:
            panels.append((resp_ES_angular,              E_edges_ES,   cos_sc_edges, r"$E_{reco}$ ES [MeV]", r"$\cos\theta_s$",   r"ES angular response"))
        if resp_CC is not None:
            E_edges_CC = np.linspace(Ereco_CC_min, Ereco_CC_max, resp_CC.shape[1] + 1)
            panels.append((col_norm(resp_CC.T), E_edges_true, E_edges_CC, r"$E_{true}$ [MeV]", r"$E_{reco}$ CC [MeV]", r"CC response"))
        if resp_CC_incl is not None:
            panels.append((col_norm(resp_CC_incl.T), E_edges_true, E_edges_ES, r"$E_{true}$ [MeV]", r"$E_{reco}$ ES [MeV]", r"CC$\to$ES inclusive response"))

        ncols = min(len(panels), 3)
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6.5 * nrows), constrained_layout=True)
        axes_flat = np.atleast_1d(axes).ravel()
        for ax, (Z, xe, ye, xl, yl, title) in zip(axes_flat, panels):
            heatmap(ax, Z, xe, ye, xl, yl, title)
        for ax in axes_flat[len(panels):]:
            ax.set_visible(False)
        fig.suptitle("Response matrices", y=1.001)
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3: Oscillation probabilities
    # ══════════════════════════════════════════════════════════════════════════
    if full_data and args.stages == "all":
        E_edges_true = np.linspace(Etrue_min, Etrue_max, Etrue_n + 1)
        cosz_edges   = cosz_edges_real if cosz_edges_real is not None else np.linspace(cosz_min, cosz_max, cosz_n + 1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

        # day
        for ax, y, label in [
            (axes[0, 0], op_nue_8B_day,  r"$P(\nu_e\to\nu_e)$ 8B  day"),
            (axes[0, 1], op_nue_hep_day, r"$P(\nu_e\to\nu_e)$ HEP day"),
        ]:
            line_stairs(ax, E_edges_true, y, NUE_COL)
            ax.set_xlabel(r"$E_{\rm true}$ [MeV]")
            ax.set_ylabel(r"$P(\nu_e \to \nu_e)$")
            ax.set_ylim(0, 1)
            ax.set_title(label)

        # night heatmaps  (n_cosz, n_Etrue)
        for ax, Z, title in [
            (axes[1, 0], op_nue_8B_night,  r"$P(\nu_e\to\nu_e)$ 8B  night"),
            (axes[1, 1], op_nue_hep_night, r"$P(\nu_e\to\nu_e)$ HEP night"),
        ]:
            heatmap(ax, Z, E_edges_true, cosz_edges,
                    r"$E_{\rm true}$ [MeV]", r"$\cos\theta_z$", title,
                    cmap="viridis")

        fig.suptitle("Oscillation probabilities", y=1.001)
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4: Oscillated spectra (true energy, before detector response)
    # ══════════════════════════════════════════════════════════════════════════
    if full_data and args.stages == "all":
        E_edges_true = np.linspace(Etrue_min, Etrue_max, Etrue_n + 1)
        cosz_edges   = cosz_edges_real if cosz_edges_real is not None else np.linspace(cosz_min, cosz_max, cosz_n + 1)

        fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)

        # day (top row)
        day_specs = [
            (osc_ES_nue_day,     NUE_COL,     r"ES $\nu_e$ day (oscillated)"),
            (osc_ES_nuother_day, NUOTHER_COL, r"ES $\nu_x$ day (oscillated)"),
            (osc_CC_day,         CC_EDGE,      r"CC day (oscillated)"),
        ]
        for ax, (y, col, title) in zip(axes[0], day_specs):
            if y is not None:
                line_stairs(ax, E_edges_true, y, col)
            ax.set_xlabel(r"$E_{\rm true}$ [MeV]")
            ax.set_ylabel("Events / bin")
            ax.set_title(title)

        # night heatmaps (bottom row)  arrays are (n_cosz, n_Etrue)
        night_specs = [
            (osc_ES_nue_night,     r"ES $\nu_e$ night (oscillated)"),
            (osc_ES_nuother_night, r"ES $\nu_x$ night (oscillated)"),
            (osc_CC_night,         r"CC night (oscillated)"),
        ]
        for ax, (Z, title) in zip(axes[1], night_specs):
            if Z is not None:
                heatmap(ax, Z, E_edges_true, cosz_edges,
                        r"$E_{\rm true}$ [MeV]", r"$\cos\theta_z$", title,
                        cmap="cividis", density=True)
            else:
                ax.set_visible(False)

        fig.suptitle("Oscillated spectra ($E_{\\rm true}$ space, before detector response)", y=1.001)
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 5a: Reco-space spectra (day step + night heatmap, ES and CC)
    # ══════════════════════════════════════════════════════════════════════════
    if full_data and args.stages == "all":
        cosz_edges   = cosz_edges_real if cosz_edges_real is not None else np.linspace(cosz_min, cosz_max, cosz_n + 1)
        E_edges_ES   = np.linspace(Ereco_ES_min, Ereco_ES_max, Ereco_ES_n + 1)
        E_edges_CC   = np.linspace(Ereco_CC_min, Ereco_CC_max, Ereco_CC_n + 1) if Ereco_CC_n > 0 else E_edges_ES

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

        # ES day (top-left)
        ax = axes[0, 0]
        if has_ES_mode:
            filled_stairs(ax, E_edges_ES, ereco_BG_ES_day + ereco_ES_day, ES_FILL, ES_EDGE, label="ES signal")
            filled_stairs(ax, E_edges_ES, ereco_BG_ES_day,                BG_FILL, BG_EDGE, label="BG")
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "N/A (no ES channel)", ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_xlabel(r"$E_{reco}^{ES}$ [MeV]")
        ax.set_ylabel("Events / bin")
        ax.set_title("ES day spectrum")

        # ES night heatmap (bottom-left)  ereco_ES_night is (n_cosz, n_Ereco_ES)
        ax = axes[1, 0]
        if has_ES_mode and ereco_ES_night.shape[0] > 0:
            heatmap(ax, ereco_ES_night, E_edges_ES, cosz_edges,
                    r"$E_{reco}^{ES}$ [MeV]", r"$\cos\theta_z$", "ES night spectrum", cmap="cividis", density=True)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("ES night spectrum")

        # CC day (top-right)
        ax = axes[0, 1]
        if has_CC_separate and len(ereco_CC_day) > 0:
            filled_stairs(ax, E_edges_CC, ereco_CC_day, CC_FILL, CC_EDGE, label="CC (signal+BG)")
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=9)
        else:
            label = "N/A (CC in ES channel)" if has_CC and not has_CC_separate else "N/A (no CC channel)"
            ax.text(0.5, 0.5, label, ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_xlabel(r"$E_{reco}^{CC}$ [MeV]")
        ax.set_ylabel("Events / bin")
        ax.set_title("CC day spectrum")

        # CC night heatmap (bottom-right)
        ax = axes[1, 1]
        if has_CC_separate and ereco_CC_night.shape[0] > 0 and ereco_CC_night.shape[1] > 0:
            heatmap(ax, ereco_CC_night, E_edges_CC, cosz_edges,
                    r"$E_{reco}^{CC}$ [MeV]", r"$\cos\theta_z$", "CC night spectrum", cmap="cividis", density=True)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title("CC night spectrum")

        fig.suptitle("Reco-space spectra (after detector response)", y=1.001)
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 5: Angular distribution — day (cos_scatter × Ereco heatmap + 1D)
    # ══════════════════════════════════════════════════════════════════════════
    if full_data and has_angular and args.stages == "all":
        E_edges_ES   = np.linspace(Ereco_ES_min, Ereco_ES_max, Ereco_ES_n + 1)
        cos_sc_edges = np.linspace(cos_scatter_min, cos_scatter_max, cos_scatter_n + 1)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

        # 2D heatmap  (n_cos_scatter, n_Ereco)
        heatmap(axes[0], ES_angular_day,
                E_edges_ES, cos_sc_edges,
                r"$E_{reco}$ [MeV]", r"$\cos\theta_s$",
                r"ES day angular distribution",
                cmap="cividis")

        # 1D stacked spectrum (day only, above-threshold energy bins).
        # Zero out sub-threshold bins to match x-axis clip on PAGE 7 (Ereco_min+1).
        # CC_angular and BG_angular are day+night totals; multiply by 0.5 for day-only.
        es_day_1d = (ES_angular_day * e_thresh_mask).sum(axis=0)
        cc_day_1d = (CC_angular * e_thresh_mask).sum(axis=0) * 0.5
        bg_day_1d = (BG_angular * e_thresh_mask).sum(axis=0) * 0.5
        ax = axes[1]
        filled_stairs(ax, E_edges_reco, bg_day_1d + cc_day_1d + es_day_1d, ES_FILL, ES_EDGE, label="ES day")
        filled_stairs(ax, E_edges_reco, bg_day_1d + cc_day_1d,              CC_FILL, CC_EDGE, label="CC")
        filled_stairs(ax, E_edges_reco, bg_day_1d,                          BG_FILL, BG_EDGE, label="BG")
        ax.set_xlabel(r"$E_{reco}$ [MeV]")
        ax.set_ylabel("Events / bin")
        ax.set_title("Day stacked spectrum (integrated over $\\cos\\theta_s$)")
        ax.set_xlim(Ereco_ES_min, Ereco_ES_max)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)

        fig.suptitle("Angular distribution — day", y=1.001)
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 6: Angular distribution — night by zenith slice
    #   For each selected cosz bin: 1D cos_scatter distribution (summed over E)
    # ══════════════════════════════════════════════════════════════════════════
    if full_data and has_angular and args.stages == "all":
        n_z     = ES_angular_night_3d.shape[2]
        n_slices = args.n_cosz_slices
        step_z   = max(1, n_z // n_slices)
        z_indices = list(range(step_z - 1, n_z, step_z))[:n_slices]

        cosz_centers = np.linspace(cosz_min, cosz_max, n_z, endpoint=False) + \
                       (cosz_max - cosz_min) / (2 * n_z)

        ncols_z = 3
        nrows_z = int(np.ceil(len(z_indices) / ncols_z))
        fig, axes = plt.subplots(nrows_z, ncols_z,
                                 figsize=(7 * ncols_z, 6 * nrows_z),
                                 constrained_layout=True)
        axes_flat = np.atleast_1d(axes).ravel()

        for idx, iz in enumerate(z_indices):
            ax = axes_flat[idx]
            # Sum over above-threshold energy bins only (e_thresh_mask excludes the
            # sub-threshold 2–3 MeV bin whose 12 M BG events swamp the angular plot).
            es_z = ES_angular_night_3d[:, e_thresh_mask, iz].sum(axis=1)   # (n_cos_scatter,)
            cc_z = CC_angular_night_3d[:, e_thresh_mask, iz].sum(axis=1) if CC_angular_night_3d is not None \
                   else (CC_angular * e_thresh_mask).sum(axis=1) / n_z
            bg_z = BG_angular_night_3d[:, e_thresh_mask, iz].sum(axis=1) if BG_angular_night_3d is not None \
                   else (BG_angular * e_thresh_mask).sum(axis=1) / n_z
            total_z = es_z + cc_z + bg_z
            y_max = max(10.0, float(total_z.max()) * 1.5)

            filled_stairs(ax, cos_edges, bg_z + cc_z + es_z, ES_FILL, ES_EDGE, label="ES night")
            filled_stairs(ax, cos_edges, bg_z + cc_z,        CC_FILL, CC_EDGE, label="CC")
            filled_stairs(ax, cos_edges, bg_z,               BG_FILL, BG_EDGE, label="BG")
            ax.set_xlim(cos_min, cos_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel(r"$\cos\theta_s$")
            ax.set_ylabel("Events / bin")
            ax.set_title(rf"$\cos\theta_z \approx {cosz_centers[iz]:.2f}$")

        for ax in axes_flat[len(z_indices):]:
            ax.set_visible(False)

        fig.suptitle("Angular distribution — night, by zenith slice", y=1.001)
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 7 (or only page when --stages angular): Combined stacks
    #   Stacked spectrum (lin + log) + N angular-slice panels + >5 MeV panel
    # Emitted only when angular_reco=true (angular arrays are meaningful)
    # ══════════════════════════════════════════════════════════════════════════
    if has_angular or (args.stages == "angular"):
        n_panels  = args.n_panels
        step_e    = max(1, n_E // n_panels)
        e_slices  = list(range(step_e - 1, n_E, step_e))[:n_panels]

        es_1d = ES_angular.sum(axis=0)
        cc_1d = CC_angular.sum(axis=0)
        bg_1d = BG_angular.sum(axis=0)

        panel_data = []
        panel_data.append(("spec_lin",))
        panel_data.append(("spec_log",))
        for i_E in e_slices:
            panel_data.append(("ang", i_E))
        panel_data.append(("ang_above5",))
        panel_data.append(("ang_above10",))

        n_total = len(panel_data)
        ncols   = 3
        nrows   = int(np.ceil(n_total / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(7 * ncols, 6 * nrows),
                                 constrained_layout=True)
        axes_flat = np.atleast_1d(axes).ravel()

        for idx, spec in enumerate(panel_data):
            ax = axes_flat[idx]

            if spec[0] == "spec_lin":
                filled_stairs(ax, E_edges_reco, bg_1d + cc_1d + es_1d, ES_FILL, ES_EDGE, label=r"ES")
                filled_stairs(ax, E_edges_reco, bg_1d + cc_1d,          CC_FILL, CC_EDGE, label=r"CC")
                filled_stairs(ax, E_edges_reco, bg_1d,                  BG_FILL, BG_EDGE, label=r"BG")
                ax.set_xlabel(r"$E_{\rm reco}$ [MeV]")
                ax.set_ylabel("Events")
                ax.set_title(r"Stacked Spectrum (linear)")
                ax.set_xlim(Ereco_min + 1, Ereco_max)
                ax.set_ylim(bottom=0)
                ax.legend(fontsize=10, loc="upper right")

            elif spec[0] == "spec_log":
                _clamp = lambda v: np.where(v > 0, v, 1e-2)
                line_stairs(ax, E_edges_reco, _clamp(es_1d), ES_EDGE, label=r"ES")
                line_stairs(ax, E_edges_reco, _clamp(cc_1d), CC_EDGE, label=r"CC")
                line_stairs(ax, E_edges_reco, _clamp(bg_1d), BG_EDGE, label=r"BG")
                ax.set_yscale("log")
                ax.set_xlabel(r"$E_{\rm reco}$ [MeV]")
                ax.set_ylabel("Events")
                ax.set_title(r"Components (log scale)")
                ax.set_xlim(Ereco_min + 1, Ereco_max)
                ax.set_ylim(1e-1, 1e6)
                ax.legend(fontsize=10, loc="upper right")

            elif spec[0] == "ang":  # angular slice at fixed energy
                i_E = spec[1]
                es  = ES_angular[:, i_E]
                cc  = CC_angular[:, i_E]
                bg  = BG_angular[:, i_E]
                y_max = max(10.0, float((bg + cc + es).max()) * 1.5)
                filled_stairs(ax, cos_edges, bg + cc + es, ES_FILL, ES_EDGE)
                filled_stairs(ax, cos_edges, bg + cc,      CC_FILL, CC_EDGE)
                filled_stairs(ax, cos_edges, bg,           BG_FILL, BG_EDGE)
                ax.set_xlim(cos_min, cos_max)
                ax.set_ylim(0, y_max)
                ax.set_xlabel(r"$\cos\theta_s$")
                ax.set_ylabel("Events")
                ax.set_title(rf"$E_{{\rm reco}} \approx {E_centers_reco[i_E]:.1f}\ \mathrm{{MeV}}$")

            elif spec[0] == "ang_above5":
                es_5 = ES_angular[:, e_5mev_mask].sum(axis=1)
                cc_5 = CC_angular[:, e_5mev_mask].sum(axis=1)
                bg_5 = BG_angular[:, e_5mev_mask].sum(axis=1)
                y_max = max(10.0, float((bg_5 + cc_5 + es_5).max()) * 1.5)
                filled_stairs(ax, cos_edges, bg_5 + cc_5 + es_5, ES_FILL, ES_EDGE, label=r"ES")
                filled_stairs(ax, cos_edges, bg_5 + cc_5,        CC_FILL, CC_EDGE, label=r"CC")
                filled_stairs(ax, cos_edges, bg_5,               BG_FILL, BG_EDGE, label=r"BG")
                ax.set_xlim(cos_min, cos_max)
                ax.set_ylim(0, y_max)
                ax.set_xlabel(r"$\cos\theta_s$")
                ax.set_ylabel("Events")
                ax.set_title(r"$E_{\rm reco} > 5\ \mathrm{MeV}$")
                ax.legend(fontsize=10)

            else:  # ang_above10 — angular distribution integrated over E > 10 MeV
                es_10 = ES_angular[:, e_10mev_mask].sum(axis=1)
                cc_10 = CC_angular[:, e_10mev_mask].sum(axis=1)
                bg_10 = BG_angular[:, e_10mev_mask].sum(axis=1)
                y_max = max(10.0, float((bg_10 + cc_10 + es_10).max()) * 1.5)
                filled_stairs(ax, cos_edges, bg_10 + cc_10 + es_10, ES_FILL, ES_EDGE, label=r"ES")
                filled_stairs(ax, cos_edges, bg_10 + cc_10,         CC_FILL, CC_EDGE, label=r"CC")
                filled_stairs(ax, cos_edges, bg_10,                 BG_FILL, BG_EDGE, label=r"BG")
                ax.set_xlim(cos_min, cos_max)
                ax.set_ylim(0, y_max)
                ax.set_xlabel(r"$\cos\theta_s$")
                ax.set_ylabel("Events")
                ax.set_title(r"$E_{\rm reco} > 10\ \mathrm{MeV}$")
                ax.legend(fontsize=10)

        for ax in axes_flat[n_total:]:
            ax.set_visible(False)

        fig.suptitle("Inclusive angular stacks (combined day$+$night)", y=1.001)
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 8: Dedicated energy histograms — thicker lines, two panels
    #   Left: stacked linear  |  Right: components log scale
    # ══════════════════════════════════════════════════════════════════════════
    if has_angular or (args.stages == "angular"):
        es_1d = ES_angular.sum(axis=0)
        cc_1d = CC_angular.sum(axis=0)
        bg_1d = BG_angular.sum(axis=0)

        fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(15, 7),
                                              constrained_layout=True)

        # linear stacked
        filled_stairs(ax_lin, E_edges_reco, bg_1d + cc_1d + es_1d, ES_FILL, ES_EDGE,
                      label=r"ES", lw=LW_HIST)
        filled_stairs(ax_lin, E_edges_reco, bg_1d + cc_1d,          CC_FILL, CC_EDGE,
                      label=r"CC", lw=LW_HIST)
        filled_stairs(ax_lin, E_edges_reco, bg_1d,                  BG_FILL, BG_EDGE,
                      label=r"BG", lw=LW_HIST)
        ax_lin.set_xlabel(r"$E_{\rm reco}\ [\mathrm{MeV}]$")
        ax_lin.set_ylabel(r"Events / bin")
        ax_lin.set_title(r"Stacked spectrum (linear)")
        ax_lin.set_xlim(Ereco_min + 1, Ereco_max)
        ax_lin.set_ylim(bottom=0)
        ax_lin.legend(loc="upper right")

        # log components
        _clamp = lambda v: np.where(v > 0, v, 1e-2)
        line_stairs(ax_log, E_edges_reco, _clamp(es_1d), ES_EDGE, label=r"ES", lw=LW_HIST)
        line_stairs(ax_log, E_edges_reco, _clamp(cc_1d), CC_EDGE, label=r"CC", lw=LW_HIST)
        line_stairs(ax_log, E_edges_reco, _clamp(bg_1d), BG_EDGE, label=r"BG", lw=LW_HIST)
        ax_log.set_yscale("log")
        ax_log.set_xlabel(r"$E_{\rm reco}\ [\mathrm{MeV}]$")
        ax_log.set_ylabel(r"Events / bin")
        ax_log.set_title(r"Components (log scale)")
        ax_log.set_xlim(Ereco_min + 1, Ereco_max)
        ax_log.set_ylim(1e-1, 1e6)
        ax_log.legend(loc="upper right")

        fig.suptitle(r"Energy spectra — inclusive angular stacks (day$+$night)", y=1.001)
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        plt.close(fig)

print(f"Wrote {out_pdf}")
