"""Corner plot of the 2D likelihood scans of RunMode LLH (src/llhScan.jl).

Off-diagonal: 1/2/3σ regions of each scan (2 parameters: Δχ² = 2.30, 6.18, 11.83), styled like
the posterior corner plots, with the external solar contours on the (sin²θ₁₂, Δm²₂₁) panel.
Diagonal: the likelihood integrated over the other parameter of a scan, with its 1D 1/2/3σ
intervals (Δχ² = 1, 4, 9); each parameter is in two scans, the second is drawn dashed. The scans
are conditional: every other parameter is held at its true value.

    python3 utils/plotLLH.py outputs/tutorial/scan [-o name]   # → images/scan_llh.pdf
"""
import argparse
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotting  # applies the shared CMS/LaTeX style
import style

COLOR, LINECOLOR = "#0894bc", "#08bc8a"        # the default corner-plot colours (plotOutput.py)
DM21_SCALE = 1e4                               # Δm²₂₁ is drawn in 10⁻⁴ eV²
VARS = ["sin2_th12", "sin2_th13", "dm2_21"]
LABELS = {"sin2_th12": r"$\sin^2\theta_{12}$", "sin2_th13": r"$\sin^2\theta_{13}$",
          "dm2_21": r"$\Delta m^2_{21}\ (10^{-4}\,{\rm eV}^2)$"}
SHORT = {"sin2_th12": r"$\sin^2\theta_{12}$", "sin2_th13": r"$\sin^2\theta_{13}$", "dm2_21": r"$\Delta m^2_{21}$"}
SCANS = {("sin2_th12", "sin2_th13"): "_llh_sin2th12_sin2th13.csv",
         ("sin2_th12", "dm2_21"): "_llh_sin2th12_delt2m21.csv",
         ("sin2_th13", "dm2_21"): "_llh_sin2th13_delt2m21.csv"}
LEVELS_2D = [2.30, 6.18, 11.83]
LEVELS_1D = [1.0, 4.0, 9.0]


def read_scan(path):
    """Return (x grid, y grid, truth (x, y) or None, Δχ² (n_x, n_y)) of one scan file."""
    with open(path) as f:
        head = [next(f).strip() for _ in range(3)]
    lim_x = [float(v) for v in head[0].split(": ")[1].split(", ")]
    lim_y = [float(v) for v in head[1].split(": ")[1].split(", ")]
    truth = [float(v) for v in head[2].split(": ")[1].split(", ")] if head[2].startswith("truth:") else None
    m = pd.read_csv(path, skiprows=3 if truth else 2, header=None).to_numpy()   # 2 log L, rows along x
    x = np.linspace(*lim_x, m.shape[0])
    y = np.linspace(*lim_y, m.shape[1])
    return x, y, truth, m.max() - m


def scaled(key, v):
    return np.asarray(v) * DM21_SCALE if key == "dm2_21" else np.asarray(v)


def mix_white(c, f):
    return tuple(np.array(mcolors.to_rgb(c)) * f + (1 - f))


def draw_2d(ax, x, y, chi2, truth, xk, yk):
    X, Y = np.meshgrid(scaled(xk, x), scaled(yk, y))
    shades = [COLOR, mix_white(COLOR, style.WHITE_MIX[1]), mix_white(COLOR, style.WHITE_MIX[0])]
    ax.contourf(X, Y, chi2.T, levels=[0] + LEVELS_2D, colors=shades)
    ax.contour(X, Y, chi2.T, levels=LEVELS_2D, colors=LINECOLOR, linestyles=style.SIGMA_LINESTYLES[::-1],
               linewidths=2)
    if truth:
        ax.plot(scaled(xk, truth[0]), scaled(yk, truth[1]), "+", color="#a41007", markersize=12, mew=2)


def integrated(chi2, axis):
    """Δχ² of the likelihood integrated over one axis of a scan, minimum 0."""
    L = np.exp(-0.5 * (chi2 - chi2.min())).sum(axis=axis)
    return -2 * np.log(np.maximum(L / L.max(), 1e-300))


def draw_1d(ax, grid, q, filled, label):
    L = np.exp(-0.5 * q)
    if filled:
        for lev, alpha in zip(LEVELS_1D, [0.85, 0.5, 0.2]):
            ax.fill_between(grid, L, where=q <= lev, color=COLOR, alpha=alpha, lw=0)
        ax.plot(grid, L, color=COLOR, lw=2, label=label)
    else:
        ax.plot(grid, L, color="0.3", lw=1.5, ls="--", label=label)


def main():
    p = argparse.ArgumentParser(description="Corner plot of the RunMode LLH scans.")
    p.add_argument("input", help="the scan's outFile prefix, e.g. outputs/tutorial/scan")
    p.add_argument("-o", "--output", help="also save as <output>_llh.pdf")
    args = p.parse_args()

    scans = {k: read_scan(args.input + suffix) for k, suffix in SCANS.items()}
    n = len(VARS)
    fig, axes = plt.subplots(n, n, figsize=(5 * n, 5 * n))
    for i, yk in enumerate(VARS):
        for j, xk in enumerate(VARS):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                pairs = [(k, v) for k, v in scans.items() if xk in k]
                for m, (key, (x, y, truth, chi2)) in enumerate(pairs):
                    first = key.index(xk) == 0
                    other = key[1] if first else key[0]
                    grid = scaled(xk, x if first else y)
                    draw_1d(ax, grid, integrated(chi2, 1 if first else 0), m == 0,
                            f"integrated over {SHORT[other]}")
                    if truth:
                        ax.axvline(scaled(xk, truth[0 if first else 1]), color="#a41007", lw=1, ls=":")
                ax.set_ylim(0, 1.05)
                ax.set_yticks([])
                ax.legend(fontsize=11, loc="upper right", frameon=False)
            else:
                x, y, truth, chi2 = scans[(xk, yk)]
                draw_2d(ax, x, y, chi2, truth, xk, yk)
                if (xk, yk) == ("sin2_th12", "dm2_21"):
                    handles = plotting.add_external_solar_data(ax)
                    axes[0, 1].legend(handles=handles, fontsize=16, frameon=False, loc="center right")
                ax.set_xlim(scaled(xk, x)[[0, -1]])
                ax.set_ylim(scaled(yk, y)[[0, -1]])
            if i == n - 1:
                ax.set_xlabel(LABELS[xk])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(LABELS[yk])
            elif i != j:
                ax.set_yticklabels([])
    for j in range(n - 1):                             # diagonal x ranges follow their column
        axes[j, j].set_xlim(axes[n - 1, j].get_xlim())
    axes[n - 1, n - 1].set_xlim(scaled("dm2_21", scans[("sin2_th12", "dm2_21")][1])[[0, -1]])
    axes[0, 2].text(0.95, 0.95, "Conditional scans:\nother parameters at their true values,\nno priors",
                    transform=axes[0, 2].transAxes, ha="right", va="top", fontsize=14, color="0.3")
    fig.tight_layout()
    if args.output:
        plotting.save_figure(fig, args.output + "_llh.pdf", dpi=300, format="pdf")
    plotting.save_figure(fig, os.path.join("images", os.path.basename(args.input) + "_llh.pdf"), dpi=300, format="pdf")
    plt.show()


if __name__ == "__main__":
    main()
