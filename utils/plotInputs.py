"""Plot what the fitter builds from its inputs, for one detector of a SETUP (or any) run.

Reads the files a run writes next to its ``outFile`` prefix:

- ``<prefix>_<det>_{ES,CC}_breakdown.csv``: signal and backgrounds per E_reco bin at the truth;
- ``<prefix>_<det>_angular_stacks.jld2``: fluxes, cross sections, response matrices, true-energy
  spectra, P_ee;
- ``<prefix>_config.yaml`` (run record): the event-level input files, for the raw-sample page.

Pages: (1) solar fluxes and cross sections; (2) true-energy spectra, P_ee and the regeneration;
(3) signal and background per E_reco bin with S/B; (4) each background spectrum; (5) response
matrices P(E_reco | E_true) as the fitter bins them; (6) the night P_ee map; (7) the raw
event-level samples. 1D quantities are filled histograms, 2D ones filled contours. Run from the
repository root (input paths in the run record are relative to it)::

    julia --project src/readConfig.jl tutorial_example/01_setup.yaml
    python3 utils/plotInputs.py outputs/tutorial/setup -o outputs/tutorial/inputs
"""
import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator

import plotting  # noqa: F401  applies the shared CMS/LaTeX style

ES_FILL, ES_EDGE = "#ff6b6b", "#b33939"
CC_FILL, CC_EDGE = "#70a1ff", "#1e3799"
BG_FILL, BG_EDGE = "#888888", "#333333"
NUE_COL, NUOTHER_COL = "#0894bc", "#a41007"
DAY_COL, NIGHT_COL = "#E69F00", "#2f4f8f"
B8_COL, HEP_COL = "#E69F00", "#009E73"
LW = 1.5
CMAP = "viridis"

EREC = r"$E_{\rm reco}$ [MeV]"
ETRUE = r"$E_{\rm true}$ [MeV]"
CHANNELS = (("ES_nue", r"ES $\nu_e$", NUE_COL), ("ES_nuother", r"ES $\nu_{\mu,\tau}$", NUOTHER_COL),
            ("CC", "CC", CC_EDGE))


def parse_args():
    p = argparse.ArgumentParser(description="Plot fluxes, cross sections, spectra, backgrounds and response matrices of a run.")
    p.add_argument("prefix", help="the run's outFile, e.g. outputs/tutorial/setup")
    p.add_argument("--detector", help="detector name (default: the only one the run wrote)")
    p.add_argument("-o", "--output", help="output file without extension (default: images/<stem>_inputs)")
    p.add_argument("--no-raw", action="store_true", help="skip the raw event-level sample page")
    return p.parse_args()


def find_detector(prefix, name):
    if name:
        return name
    found = sorted(os.path.basename(f)[len(os.path.basename(prefix)) + 1:-len("_angular_stacks.jld2")]
                   for f in glob.glob(f"{prefix}_*_angular_stacks.jld2"))
    if len(found) != 1:
        raise SystemExit(f"found detectors {found} for {prefix}; pick one with --detector")
    return found[0]


def load_jld2(path):
    """Datasets of the debug file as numpy arrays; Julia (a, b) arrays arrive as (b, a), so 2D ones are transposed back."""
    out = {}
    with h5py.File(path, "r") as f:
        def walk(group, base=""):
            for k, v in group.items():
                if isinstance(v, h5py.Group):
                    walk(v, base + k + "/")
                else:
                    a = v[()]
                    out[base + k] = a.T if getattr(a, "ndim", 0) == 2 else a
        walk(f)
    return out


# ── drawing helpers (filled style of utils/plotAngularStacks.py and the corner plots) ──
def filled_stairs(ax, edges, y, fill, edge, label=None, alpha=0.75, baseline=0, lw=LW):
    ax.stairs(y, edges, baseline=baseline, fill=True, color=fill, alpha=alpha, label=label)
    ax.stairs(y, edges, baseline=baseline, color=edge, lw=lw)


def filled_curve(ax, x, y, fill, edge, label=None, alpha=0.35, lw=LW):
    ax.fill_between(x, y, color=fill, alpha=alpha, lw=0)
    ax.plot(x, y, color=edge, lw=lw, label=label)


def filled_map(ax, x_edges, y_edges, Z, label, log=True, cmap=CMAP, nlev=15, vmin=None, vmax=None):
    """Filled contours of a binned map Z (n_y, n_x) at its bin centres."""
    xc, yc = 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])
    Z = np.asarray(Z, float)
    if log:
        pos = Z[Z > 0]
        lo = vmin or max(pos.min(), pos.max() * 1e-5)
        levels = np.logspace(np.log10(lo), np.log10(vmax or pos.max()), nlev)
        m = ax.contourf(xc, yc, np.ma.masked_less(Z, lo), levels=levels, norm=LogNorm(), cmap=cmap)
        cb = ax.figure.colorbar(m, ax=ax, label=label, fraction=0.046, pad=0.04,
                                ticks=LogLocator(), format=LogFormatterMathtext())
        cb.minorticks_off()
    else:
        m = ax.contourf(xc, yc, Z, levels=np.linspace(vmin, vmax, nlev), cmap=cmap, extend="both")
        ax.figure.colorbar(m, ax=ax, label=label, fraction=0.046, pad=0.04)
    return m


def square_one_to_one(ax, lo, hi):
    """Same range on both axes, equal scaling and the E_reco = E_true line."""
    ax.plot([lo, hi], [lo, hi], color="white", lw=1.2, ls="--", label=r"$E_{\rm reco} = E_{\rm true}$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")


def reco_edges(df):
    """Finite E_reco edges for drawing: the under- and overflow bins are drawn one bin width wide."""
    lo, hi = df.Ereco_lo_MeV.to_numpy(float), df.Ereco_hi_MeV.to_numpy(float)
    w = hi[1] - lo[1]
    lo[0] = hi[0] - w
    hi[-1] = lo[-1] + w
    return np.append(lo, hi[-1])


def fit_window(df):
    f = df.in_fit.to_numpy(bool)
    idx = np.flatnonzero(f)
    return f, reco_edges(df)[np.r_[idx, idx[-1] + 1]]


def etrue_edges(d):
    return np.linspace(d["meta/Etrue_min"], d["meta/Etrue_max"], int(d["meta/Etrue_n"]) + 1)


def savepage(pdf, fig):
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ── pages ──
def page_model(pdf, d):
    if "model/E_MeV" not in d:
        print("flux/cross-section page skipped: rerun SETUP with the current fitter to store them")
        return
    E = d["model/E_MeV"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax = axes[0]
    filled_curve(ax, E, d["model/flux_8B"], B8_COL, B8_COL, r"$^8$B")
    filled_curve(ax, E, d["model/flux_hep"], HEP_COL, HEP_COL, "hep")
    ax.set_yscale("log")
    ax.set_ylim(d["model/flux_hep"].max() * 1e-3, d["model/flux_8B"].max() * 3)
    ax.set_xlabel(r"$E_\nu$ [MeV]")
    ax.set_ylabel(r"$d\Phi/dE$ [cm$^{-2}$ s$^{-1}$ MeV$^{-1}$]")
    ax.set_title("Solar fluxes at the truth")
    ax.legend()
    ax = axes[1]
    for key, lab, col in CHANNELS:
        lab = r"CC $\nu_e\,^{40}$Ar (per nucleus)" if key == "CC" else lab + " (per electron)"
        filled_curve(ax, E, d[f"model/xsec_{key}"], col, col, lab)
    ax.set_yscale("log")
    ax.set_ylim(1e-46, None)
    ax.set_xlabel(r"$E_\nu$ [MeV]")
    ax.set_ylabel(r"$\sigma$ [cm$^2$]")
    ax.set_title(r"Cross sections (ES: $T_e >$ TPC threshold)")
    ax.legend(fontsize=10)
    ax = axes[2]
    for key, lab, col in CHANNELS:
        r = d[f"model/xsec_{key}"] * (d["model/flux_8B"] + d["model/flux_hep"])
        if r.max() > 0:
            filled_curve(ax, E, r / np.trapezoid(r, E), col, col, lab)
    ax.set_xlabel(r"$E_\nu$ [MeV]")
    ax.set_ylabel(r"$\Phi\,\sigma$, normalised [MeV$^{-1}$]")
    ax.set_title("Interaction spectrum, no oscillation")
    ax.legend()
    savepage(pdf, fig)


def page_true(pdf, d):
    et = etrue_edges(d)
    fD, fN = d["meta/signal_day_fraction"], d["meta/signal_night_fraction"]
    present = [(k, lab, col) for k, lab, col in CHANNELS if f"oscillated/{k}_day" in d]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for j, (k, lab, col) in enumerate(present):
        ax = axes[0, j]
        total = d[f"oscillated/{k}_day"] + d[f"oscillated/{k}_night"].sum(axis=0)
        filled_stairs(ax, et, total, col, col, "all interactions", alpha=0.3)
        resp = d.get(f"resp/{k}")
        if resp is not None:
            filled_stairs(ax, et, total * resp[:, 1:].sum(axis=1), col, col, r"$E_{\rm reco}$ above threshold", alpha=0.8)
        ax.set_yscale("log")
        ax.set_ylim(total.max() * 1e-4, total.max() * 3)
        ax.set_xlabel(ETRUE)
        ax.set_ylabel("Events / bin (day + night)")
        ax.set_title(f"{lab}: oscillated spectrum at the truth")
        ax.legend(fontsize=10)
    for j in range(len(present), 3):
        axes[0, j].axis("off")
    ax = axes[1, 0]
    pn = d["osc_probs/nue_8B_night"]                  # (cos z, E_true)
    w = d["oscillated/CC_night"] if "oscillated/CC_night" in d else d["oscillated/ES_nue_night"]
    wz = w.sum(axis=1) / w.sum()                      # night weight per cos z bin (exposure x rate)
    ax.stairs(d["osc_probs/nue_8B_day"], et, color=DAY_COL, lw=LW, label=r"$^8$B day")
    ax.stairs(wz @ pn, et, color=NIGHT_COL, lw=LW, label=r"$^8$B night (average)")
    ax.stairs(d["osc_probs/nue_hep_day"], et, color=HEP_COL, lw=LW, ls="--", label="hep day")
    ax.set_xlabel(ETRUE)
    ax.set_ylabel(r"$P_{ee}$")
    ax.set_title(r"$\nu_e$ survival probability")
    ax.legend()
    ax = axes[1, 1]
    for k, lab, col in present:
        if k == "ES_nuother":
            continue
        day, night = d[f"oscillated/{k}_day"] / fD, d[f"oscillated/{k}_night"].sum(axis=0) / fN
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.stairs(np.where(day > 0, night / day - 1, np.nan), et, color=col, lw=LW,
                      ls="--" if k == "ES_nue" else "-", label=lab)
    ax.axhline(0, color=BG_EDGE, lw=0.8)
    ax.set_xlabel(ETRUE)
    ax.set_ylabel(r"night/day $- 1$")
    ax.set_title("Earth regeneration")
    ax.legend()
    axes[1, 2].axis("off")
    savepage(pdf, fig)


def page_rates(pdf, breakdown, det):
    chans = list(breakdown)
    fig, axes = plt.subplots(2, len(chans), figsize=(max(7.5 * len(chans), 10), 8.5), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1.2]}, sharex="col")
    for j, ch in enumerate(chans):
        df = breakdown[ch]
        f, e = fit_window(df)
        sig, bg = df.signal.to_numpy()[f], df.background.to_numpy()[f]
        comps = [c for c in df.columns if c.startswith("bg_")]
        ax = axes[0, j]
        bottom = np.zeros_like(bg)
        greys = plt.cm.Greys(np.linspace(0.35, 0.7, max(len(comps), 1)))
        for c, col in zip(comps, greys):
            y = bottom + df[c].to_numpy()[f]
            filled_stairs(ax, e, y, col, BG_EDGE, c[3:].replace("_", r"\_"), alpha=0.9, baseline=bottom, lw=0.8)
            bottom = y
        fill, edge = (ES_FILL, ES_EDGE) if ch == "ES" else (CC_FILL, CC_EDGE)
        filled_stairs(ax, e, bottom + sig, fill, edge, f"{ch} signal", alpha=0.9, baseline=bottom)
        S, B = sig.sum(), bg.sum()
        ax.text(0.97, 0.95, rf"$S = {S:.4g}$, $B = {B:.4g}$" + "\n" + rf"$S/B = {S / B if B else np.inf:.3g}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=12)
        ax.set_ylabel("Events / bin (day + night)")
        ax.set_title(f"{det} {ch}: fit window, at the truth")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.8)
        ax.set_yscale("log")
        ax.set_ylim(max(1e-1, 0.5 * np.min(np.r_[sig[sig > 0], bg[bg > 0]], initial=1)), None)
        r = axes[1, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            filled_stairs(r, e, np.where(bg > 0, sig / bg, 0), fill, edge, alpha=0.6)
        r.axhline(S / B if B else 0, color=BG_EDGE, ls="--", lw=1, label="window total")
        r.set_ylabel(r"$S/B$ per bin")
        r.set_xlabel(EREC)
        r.legend(fontsize=10)
    savepage(pdf, fig)


def page_backgrounds(pdf, breakdown, det):
    chans = [ch for ch, df in breakdown.items() if any(c.startswith("bg_") for c in df.columns)]
    if not chans:
        return
    fig, axes = plt.subplots(1, len(chans), figsize=(max(7.5 * len(chans), 10), 6.5), squeeze=False)
    for j, ch in enumerate(chans):
        df = breakdown[ch]
        f, e = fit_window(df)
        ax = axes[0, j]
        comps = [c for c in df.columns if c.startswith("bg_")]
        cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(comps)))
        for c, col in zip(comps, cols):
            y = df[c].to_numpy()[f]
            if y.max() > 0:
                filled_stairs(ax, e, y, col, col, c[3:].replace("_", r"\_"), alpha=0.45, lw=1.2)
        ax.stairs(df.signal.to_numpy()[f], e, color=ES_EDGE if ch == "ES" else CC_EDGE, lw=LW, ls="--",
                  label=f"{ch} signal (for scale)")
        ax.set_yscale("log")
        top = max(df[c].to_numpy()[f].max() for c in comps)
        ax.set_ylim(top * 1e-4, top * 5)
        ax.set_xlabel(EREC)
        ax.set_ylabel("Events / bin (day + night)")
        ax.set_title(f"{det} {ch} backgrounds, fit window")
        ax.legend(fontsize=8, framealpha=0.8)
    savepage(pdf, fig)


def page_response(pdf, d, breakdown):
    mats = [(k, lab) for k, lab, _ in CHANNELS if f"resp/{k}" in d]
    et = etrue_edges(d)
    fig, axes = plt.subplots(2, len(mats), figsize=(6.5 * len(mats), 9.5), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1.2]})
    for j, (k, lab) in enumerate(mats):
        ch = "CC" if k == "CC" else "ES"
        R = d[f"resp/{k}"]                             # (E_true, E_reco), rows sum to 1
        er = reco_edges(breakdown[ch]) if ch in breakdown else \
            np.linspace(d[f"meta/Ereco_{ch}_min"], d[f"meta/Ereco_{ch}_max"], int(d[f"meta/Ereco_{ch}_n"]) + 1)
        ax = axes[0, j]
        filled_map(ax, et, er[1:], R[:, 1:].T, r"$P(E_{\rm reco}\,|\,E_{\rm true})$")   # below-threshold bin left out
        lo, hi = 0.0, max(et[-1], er[-1])
        square_one_to_one(ax, lo, hi)
        ax.axhline(er[1], color="white", lw=0.8, ls=":")       # the threshold
        ax.set_title(f"{lab} response (fit bins)")
        ax.set_xlabel(ETRUE)
        ax.set_ylabel(EREC)
        a = axes[1, j]
        fill, edge = (CC_FILL, CC_EDGE) if ch == "CC" else (ES_FILL, ES_EDGE)
        filled_stairs(a, et, R[:, 1:].sum(axis=1), fill, edge, alpha=0.6)
        a.set_ylabel("above threshold")
        a.set_xlabel(ETRUE)
        a.set_xlim(lo, hi)
        a.set_ylim(0, 1.05)
    savepage(pdf, fig)


def page_night(pdf, d):
    et = etrue_edges(d)
    cz = d["meta/cosz_edges"]
    diff = d["osc_probs/nue_8B_night"] - d["osc_probs/nue_8B_day"][None, :]
    v = np.nanmax(np.abs(diff))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    filled_map(ax, et, cz, diff, r"$P_{ee}^{\rm night} - P_{ee}^{\rm day}$ ($^8$B)", log=False,
               cmap="RdBu_r", vmin=-v, vmax=v, nlev=21)
    ax.set_xlabel(ETRUE)
    ax.set_ylabel(r"$\cos\theta_z$")
    ax.set_title(r"Night $P_{ee}$ by zenith bin (analysis grid)")
    savepage(pdf, fig)


def page_raw(pdf, block, thr):
    samples = [(k, lab, col) for k, lab, col in CHANNELS
               if os.path.isfile(block.get(f"reconstruction_sample_{k}", ""))]
    bgs = [(ch, p) for ch in ("ES", "CC") for p in block.get(f"{ch}_background_files", []) if os.path.isfile(p)]
    if not samples and not bgs:
        print("raw-sample page skipped: no input files found from the working directory")
        return
    ncol = max(len(samples), 3)
    fig, axes = plt.subplots(2, ncol, figsize=(6.5 * ncol, 11), squeeze=False)
    for j, (k, lab, _) in enumerate(samples):
        df = pd.read_csv(block[f"reconstruction_sample_{k}"])
        x, y = df.Etrue.to_numpy() * 1e3, df.Ereco.to_numpy() * 1e3
        t = thr["CC" if k == "CC" else "ES"]
        ymax = max(np.percentile(y, 99.9), 1.5 * t)
        hi = max(x.max(), ymax)
        H, xe, ye = np.histogram2d(x, y, bins=(80, 80), range=((0, hi), (0, hi)))
        ax = axes[0, j]
        filled_map(ax, xe, ye, H.T, "MC events", vmin=1)
        square_one_to_one(ax, 0, hi)
        ax.axhline(t, color="white", ls=":", lw=0.8)       # the threshold
        ax.set_xlabel(ETRUE)
        ax.set_ylabel(EREC)
        ax.set_title(f"{lab}: {len(df):,} MC events")
    for j in range(len(samples), ncol):
        axes[0, j].axis("off")
    ax = axes[1, 0]
    for (ch, path), col in zip(bgs, plt.cm.viridis(np.linspace(0.1, 0.85, max(len(bgs), 1)))):
        df = pd.read_csv(path)
        w = df.weights.to_numpy() if "weights" in df else np.ones(len(df))
        sel = df["mask"].astype(bool).to_numpy() if "mask" in df else np.ones(len(df), bool)
        e = df.Ereco.to_numpy() * 1e3
        bins = np.linspace(0, np.percentile(e, 99.9) * 1.05, 80)
        n_sel, _ = np.histogram(e[sel], bins, weights=w[sel])
        n_all, _ = np.histogram(e, bins, weights=w)
        name = os.path.basename(path).replace("_", r"\_")
        filled_stairs(ax, bins, n_sel, col, col, f"{ch} {name} (selected)", alpha=0.45, lw=1.2)
        ax.stairs(n_all, bins, color=col, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(EREC)
    ax.set_ylabel("MC events (weighted)")
    ax.set_title(r"Background samples (dotted: before selection)")
    axes[1, 2].legend(*ax.get_legend_handles_labels(), loc="center left", fontsize=8, frameon=False)
    axes[1, 2].axis("off")
    ax = axes[1, 1]
    for k, lab, col in samples:
        df = pd.read_csv(block[f"reconstruction_sample_{k}"])
        e = df.Ereco.to_numpy() * 1e3
        bins = np.linspace(0, np.percentile(e, 99.9), 60)
        n_all, _ = np.histogram(e, bins)
        n_sel, _ = np.histogram(e[df["mask"].astype(bool).to_numpy()], bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.stairs(np.where(n_all > 0, n_sel / n_all, np.nan), bins, lw=LW, color=col, label=lab)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(EREC)
    ax.set_ylabel("selected fraction")
    ax.set_title("Signal selection efficiency")
    ax.legend()
    for j in range(3, ncol):
        axes[1, j].axis("off")
    savepage(pdf, fig)


def main():
    args = parse_args()
    det = find_detector(args.prefix, args.detector)
    d = load_jld2(f"{args.prefix}_{det}_angular_stacks.jld2")
    breakdown = {ch: pd.read_csv(p) for ch in ("ES", "CC")
                 if os.path.isfile(p := f"{args.prefix}_{det}_{ch}_breakdown.csv")}
    out = (args.output or os.path.join("images", os.path.basename(args.prefix) + "_inputs")) + ".pdf"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with PdfPages(out) as pdf:
        page_model(pdf, d)
        page_true(pdf, d)
        if breakdown:
            page_rates(pdf, breakdown, det)
            page_backgrounds(pdf, breakdown, det)
        page_response(pdf, d, breakdown)
        page_night(pdf, d)
        record = f"{args.prefix}_config.yaml"
        if not args.no_raw and os.path.isfile(record):
            with open(record) as f:
                cfg = yaml.safe_load(f)["config"]
            block = cfg["detectors"][det] if "detectors" in cfg else cfg
            thr = {"ES": float(block["Ereco_min_ES"]), "CC": float(block["Ereco_min_CC"])}
            page_raw(pdf, block, thr)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
