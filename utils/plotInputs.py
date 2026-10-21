"""Plot what the fitter builds from its inputs, for one detector of a SETUP (or any) run.

Reads the files a run writes next to its ``outFile`` prefix:

- ``<prefix>_<det>_{ES,CC}_breakdown.csv``: signal and backgrounds per E_reco bin at the truth;
- ``<prefix>_<det>_angular_stacks.jld2``: response matrices, true-energy spectra, P_ee;
- ``<prefix>_config.yaml`` (run record): the event-level input files, for the raw-sample page.

Pages: (1) signal and background per E_reco bin, with S/B; (2) response matrices
P(E_reco | E_true) as the fitter bins them, and the fraction above threshold; (3) true-energy
spectra, P_ee by day and night, and the regeneration; (4) the night P_ee map; (5) the raw
event-level samples. Run from the repository root (input paths in the run record are relative
to it)::

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

import plotting  # noqa: F401  applies the shared CMS/LaTeX style

ES_FILL, ES_EDGE = "#ff6b6b", "#b33939"
CC_FILL, CC_EDGE = "#70a1ff", "#1e3799"
BG_FILL, BG_EDGE = "#888888", "#333333"
NUE_COL, NUOTHER_COL = "#0894bc", "#a41007"
DAY_COL, NIGHT_COL = "#E69F00", "#2f4f8f"
LW = 1.5
CMAP = "cividis"

EREC = r"$E_{\rm reco}$ [MeV]"
ETRUE = r"$E_{\rm true}$ [MeV]"


def parse_args():
    p = argparse.ArgumentParser(description="Plot signal, backgrounds, response matrices and P_ee of a run.")
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


def stairs(ax, edges, y, color, label=None, fill=None, lw=LW, **kw):
    if fill is not None:
        ax.stairs(y, edges, fill=True, color=fill, alpha=0.85, label=label, **kw)
        ax.stairs(y, edges, color=color, lw=lw, **kw)
    else:
        ax.stairs(y, edges, color=color, lw=lw, label=label, **kw)


def reco_edges(df):
    """Finite E_reco edges for drawing: the overflow bin is drawn one bin width wide."""
    lo, hi = df.Ereco_lo_MeV.to_numpy(float), df.Ereco_hi_MeV.to_numpy(float)
    w = hi[1] - lo[1]
    lo[0] = hi[0] - w
    hi[-1] = lo[-1] + w
    return np.append(lo, hi[-1])


def page_rates(pdf, breakdown, det):
    chans = [c for c in ("ES", "CC") if c in breakdown]
    fig, axes = plt.subplots(2, len(chans), figsize=(max(7.5 * len(chans), 10), 8.5), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1.2]}, sharex="col")
    for j, ch in enumerate(chans):
        df = breakdown[ch]
        f = df.in_fit.to_numpy(bool)
        e = reco_edges(df)[np.r_[np.flatnonzero(f), np.flatnonzero(f)[-1] + 1]]
        sig, bg = df.signal.to_numpy()[f], df.background.to_numpy()[f]
        comps = [c for c in df.columns if c.startswith("bg_")]
        ax = axes[0, j]
        bottom = np.zeros_like(bg)
        greys = plt.cm.Greys(np.linspace(0.35, 0.7, max(len(comps), 1)))
        for c, col in zip(comps, greys):
            y = df[c].to_numpy()[f]
            ax.stairs(bottom + y, e, baseline=bottom, fill=True, color=col, alpha=0.9,
                      label=c[3:].replace("_", r"\_"))
            bottom = bottom + y
        sig_fill, sig_edge = (ES_FILL, ES_EDGE) if ch == "ES" else (CC_FILL, CC_EDGE)
        ax.stairs(bottom + sig, e, baseline=bottom, fill=True, color=sig_fill, alpha=0.9, label=f"{ch} signal")
        ax.stairs(bottom + sig, e, color=sig_edge, lw=LW)
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
            r.stairs(np.where(bg > 0, sig / bg, np.nan), e, color=sig_edge, lw=LW)
        r.axhline(S / B if B else 0, color=BG_EDGE, ls="--", lw=1, label="window total")
        r.set_ylabel(r"$S/B$ per bin")
        r.set_xlabel(EREC)
        r.legend(fontsize=10)
        if "mc_rel_uncertainty" in df:
            t = r.twinx()
            t.stairs(df.mc_rel_uncertainty.to_numpy()[f], e, color="#009E73", lw=1, ls=":")
            t.set_ylabel("MC rel. unc.", color="#009E73")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_response(pdf, d, breakdown):
    mats = [(k, lab) for k, lab in (("resp/ES_nue", r"ES $\nu_e$"), ("resp/ES_nuother", r"ES $\nu_{\mu,\tau}$"),
                                    ("resp/CC", "CC")) if k in d]
    et = np.linspace(d["meta/Etrue_min"], d["meta/Etrue_max"], int(d["meta/Etrue_n"]) + 1)
    fig, axes = plt.subplots(2, len(mats), figsize=(6.5 * len(mats), 9), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1.3]}, sharex="col")
    for j, (k, lab) in enumerate(mats):
        ch = "CC" if k.endswith("CC") else "ES"
        R = d[k]                                       # (E_true, E_reco), rows sum to 1
        er = reco_edges(breakdown[ch]) if ch in breakdown else \
            np.linspace(d[f"meta/Ereco_{ch}_min"], d[f"meta/Ereco_{ch}_max"], int(d[f"meta/Ereco_{ch}_n"]) + 1)
        ax = axes[0, j]
        Z = np.ma.masked_less_equal(R[:, 1:].T, 0)     # drop the below-threshold bin from the map
        m = ax.pcolormesh(et, er[1:], Z, cmap=CMAP, norm=LogNorm(vmin=max(Z.min(), 1e-4), vmax=Z.max()),
                          rasterized=True)
        ax.plot(et, et, color="white", lw=1, ls="--")
        ax.set_ylim(er[1], er[-1])
        fig.colorbar(m, ax=ax, label=r"$P(E_{\rm reco}\,|\,E_{\rm true})$")
        ax.set_title(f"{lab} response (fit bins; top bin = overflow)")
        ax.set_ylabel(EREC)
        a = axes[1, j]
        a.stairs(R[:, 1:].sum(axis=1), et, color=ES_EDGE if ch == "ES" else CC_EDGE, lw=LW)
        a.set_ylabel("above threshold")
        a.set_xlabel(ETRUE)
        a.set_ylim(0, 1.05)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_true(pdf, d):
    et = np.linspace(d["meta/Etrue_min"], d["meta/Etrue_max"], int(d["meta/Etrue_n"]) + 1)
    fD, fN = d["meta/signal_day_fraction"], d["meta/signal_night_fraction"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax = axes[0]
    for key, lab, col in (("oscillated/ES_nue", r"ES $\nu_e$", NUE_COL), ("oscillated/ES_nuother", r"ES $\nu_{\mu,\tau}$", NUOTHER_COL),
                          ("oscillated/CC", "CC", CC_EDGE)):
        if key + "_day" in d:
            stairs(ax, et, d[key + "_day"] + d[key + "_night"].sum(axis=0), col, label=lab)
    ax.set_yscale("log")
    ax.set_ylim(ax.get_ylim()[1] * 1e-5, None)
    ax.set_xlabel(ETRUE)
    ax.set_ylabel("Events / bin")
    ax.set_title("Oscillated spectra at the truth (day + night)")
    ax.legend()
    ax = axes[1]
    pn = d["osc_probs/nue_8B_night"]                  # (cos z, E_true)
    w = d["oscillated/CC_night"] if "oscillated/CC_night" in d else d["oscillated/ES_nue_night"]
    wz = w.sum(axis=1) / w.sum()                       # night exposure x rate weight per cos z bin
    stairs(ax, et, d["osc_probs/nue_8B_day"], DAY_COL, label=r"$^8$B day")
    stairs(ax, et, wz @ pn, NIGHT_COL, label=r"$^8$B night (average)")
    stairs(ax, et, d["osc_probs/nue_hep_day"], DAY_COL, label="hep day", ls="--")
    ax.set_xlabel(ETRUE)
    ax.set_ylabel(r"$P_{ee}$")
    ax.set_title(r"$\nu_e$ survival probability")
    ax.legend()
    ax = axes[2]
    for key, lab, col, ls in (("oscillated/CC", "CC", CC_EDGE, "-"), ("oscillated/ES_nue", r"ES $\nu_e$", NUE_COL, "--")):
        if key + "_day" in d:
            day, night = d[key + "_day"] / fD, d[key + "_night"].sum(axis=0) / fN
            with np.errstate(divide="ignore", invalid="ignore"):
                stairs(ax, et, np.where(day > 0, night / day - 1, np.nan), col, label=lab, ls=ls)
    ax.axhline(0, color=BG_EDGE, lw=0.8)
    ax.set_xlabel(ETRUE)
    ax.set_ylabel(r"night/day $- 1$")
    ax.set_title("Earth regeneration")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_night(pdf, d):
    et = np.linspace(d["meta/Etrue_min"], d["meta/Etrue_max"], int(d["meta/Etrue_n"]) + 1)
    cz = d["meta/cosz_edges"]
    diff = d["osc_probs/nue_8B_night"] - d["osc_probs/nue_8B_day"][None, :]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    v = np.nanmax(np.abs(diff))
    m = ax.pcolormesh(et, cz, diff, cmap="RdBu_r", vmin=-v, vmax=v, rasterized=True)
    fig.colorbar(m, ax=ax, label=r"$P_{ee}^{\rm night} - P_{ee}^{\rm day}$ ($^8$B)")
    ax.set_xlabel(ETRUE)
    ax.set_ylabel(r"$\cos\theta_z$")
    ax.set_title(r"Night $P_{ee}$ by zenith bin (analysis grid)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_raw(pdf, block, thr):
    samples = [(k, lab) for k, lab in (("reconstruction_sample_ES_nue", r"ES $\nu_e$"),
                                       ("reconstruction_sample_ES_nuother", r"ES $\nu_{\mu,\tau}$"),
                                       ("reconstruction_sample_CC", "CC")) if os.path.isfile(block.get(k, ""))]
    bgs = [(ch, p) for ch in ("ES", "CC") for p in block.get(f"{ch}_background_files", []) if os.path.isfile(p)]
    if not samples and not bgs:
        print("raw-sample page skipped: no input files found from the working directory")
        return
    ncol = max(len(samples), 1)
    fig, axes = plt.subplots(2, ncol, figsize=(6.5 * ncol, 11), squeeze=False)
    for j, (k, lab) in enumerate(samples):
        df = pd.read_csv(block[k])
        x, y = df.Etrue.to_numpy() * 1e3, df.Ereco.to_numpy() * 1e3
        t = thr["CC" if k.endswith("CC") else "ES"]
        ax = axes[0, j]
        h = ax.hist2d(x, y, bins=(120, 120), range=((x.min(), x.max()), (0, max(np.percentile(y, 99.9), 1.5 * t))),
                      cmap=CMAP, norm=LogNorm(), rasterized=True)
        fig.colorbar(h[3], ax=ax, label="MC events")
        ax.axhline(t, color="white", ls="--", lw=1)
        ax.set_xlabel(ETRUE)
        ax.set_ylabel(EREC)
        ax.set_title(f"{lab}: {len(df):,} MC events")
    for j in range(len(samples), ncol):
        axes[0, j].axis("off")
    ax = axes[1, 0]
    for ch, path in bgs:
        df = pd.read_csv(path)
        w = df.weights.to_numpy() if "weights" in df else np.ones(len(df))
        sel = df["mask"].astype(bool).to_numpy() if "mask" in df else np.ones(len(df), bool)
        e = df.Ereco.to_numpy() * 1e3
        bins = np.linspace(0, np.percentile(e, 99.9) * 1.05, 80)
        col = ES_EDGE if ch == "ES" else CC_EDGE
        name = os.path.basename(path).replace("_", r"\_")
        ax.hist(e, bins=bins, weights=w, histtype="step", color=col, lw=1, ls=":", label=f"{ch} {name} (all)")
        ax.hist(e[sel], bins=bins, weights=w[sel], histtype="step", color=col, lw=LW, label=f"{ch} {name} (selected)")
    ax.set_yscale("log")
    ax.set_xlabel(EREC)
    ax.set_ylabel("MC events (weighted)")
    ax.set_title("Background samples")
    if ncol > 2:                                       # legend in the free bottom-right panel
        axes[1, 2].legend(*ax.get_legend_handles_labels(), loc="center left", fontsize=8, frameon=False)
    else:
        ax.legend(fontsize=7, framealpha=0.8)
    ax = axes[1, 1] if ncol > 1 else None
    if ax is not None:
        for (k, lab), col in zip(samples, (NUE_COL, NUOTHER_COL, CC_EDGE)):
            df = pd.read_csv(block[k])
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
    for j in range(2, ncol):
        axes[1, j].axis("off")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    args = parse_args()
    det = find_detector(args.prefix, args.detector)
    d = load_jld2(f"{args.prefix}_{det}_angular_stacks.jld2")
    breakdown = {ch: pd.read_csv(p) for ch in ("ES", "CC")
                 if os.path.isfile(p := f"{args.prefix}_{det}_{ch}_breakdown.csv")}
    out = (args.output or os.path.join("images", os.path.basename(args.prefix) + "_inputs")) + ".pdf"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with PdfPages(out) as pdf:
        if breakdown:
            page_rates(pdf, breakdown, det)
        page_response(pdf, d, breakdown)
        page_true(pdf, d)
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
