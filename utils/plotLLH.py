"""Corner plot of the 2D likelihood scans of RunMode LLH (src/llhScan.jl).

Off-diagonal: each scan as a colour map of χ²/ndof (χ² = chi2_ref − cell, zero at the truth;
Δχ² from the minimum for older files) with the 2-parameter 1/2/3σ contours (Δχ² = 2.30, 6.18,
11.83). Diagonal: the Δχ² curve of each parameter, the likelihood integrated over the other
parameter of a scan, with the 1D 1/2/3σ levels; each parameter is in two scans, the second is
dashed.

    python3 utils/plotLLH.py outputs/tutorial/scan [--rebin 2] [-o name]   # → images/scan_llh.pdf
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import plotting  # applies the shared CMS/LaTeX style
import style

COLOR = "#0894bc"                              # the default corner-plot colour (plotOutput.py)
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
CMAP = plotting.parula_map
CHI2_MAX = 25.0                                # colour scale: χ² up to this, divided by ndof


def rebin(m, x, y, k):
    """Average the likelihood exp(m/2) over k × k blocks of grid points (m = 2 log L per point)."""
    nx, ny = (m.shape[0] // k) * k, (m.shape[1] // k) * k
    m, x, y = m[:nx, :ny], x[:nx], y[:ny]
    top = m.max()
    L = np.exp((m - top) / 2).reshape(nx // k, k, ny // k, k).mean(axis=(1, 3))
    return top + 2 * np.log(np.maximum(L, 1e-300)), x.reshape(-1, k).mean(1), y.reshape(-1, k).mean(1)


def read_scan(path, k=1):
    """One scan file: grids, header fields, Δχ² from the minimum and χ² from the truth (n_x, n_y),
    with the likelihood averaged over k × k blocks of grid points."""
    head = {}
    with open(path) as f:
        for line in f:
            key, sep, val = line.partition(": ")
            if not sep or not key.replace("_", "").isalnum():
                break
            head[key] = val.strip()
    lims = [[float(v) for v in head[k].split(", ")] for k in list(head)[:2]]
    m = np.loadtxt(path, delimiter=",", skiprows=len(head), ndmin=2)   # 2 log L per point, rows along x
    x, y = np.linspace(*lims[0], m.shape[0]), np.linspace(*lims[1], m.shape[1])
    if k > 1:
        m, x, y = rebin(m, x, y, k)
    s = {"x": x, "y": y,
         "truth": [float(v) for v in head["truth"].split(", ")] if "truth" in head else None,
         "dchi2": m.max() - m,
         "ndof": int(head["ndof"]) if "ndof" in head else None,
         "nuisance": head.get("nuisance", "false")}
    s["chi2"] = np.maximum(float(head["chi2_ref"]) - m, 0) if "chi2_ref" in head else s["dchi2"]
    return s


def scaled(key, v):
    return np.asarray(v) * DM21_SCALE if key == "dm2_21" else np.asarray(v)


def edges(c):
    """Bin edges around evenly spaced grid points."""
    d = c[1] - c[0]
    return np.append(c - d / 2, c[-1] + d / 2)


def draw_2d(ax, s, xk, yk):
    xs, ys = scaled(xk, s["x"]), scaled(yk, s["y"])
    norm = s["ndof"] or 1
    m = ax.pcolormesh(edges(xs), edges(ys), np.minimum(s["chi2"].T, CHI2_MAX) / norm, cmap=CMAP,
                      vmin=0, vmax=CHI2_MAX / norm, shading="flat", rasterized=True)
    X, Y = np.meshgrid(xs, ys)
    ax.contour(X, Y, s["dchi2"].T, levels=LEVELS_2D, colors="white", linestyles=style.SIGMA_LINESTYLES[::-1],
               linewidths=3)
    if s["truth"]:
        ax.plot(scaled(xk, s["truth"][0]), scaled(yk, s["truth"][1]), "+", color="#a41007", markersize=16, mew=3)
    return m


def main():
    p = argparse.ArgumentParser(description="Corner plot of the RunMode LLH scans.")
    p.add_argument("input", help="the scan's outFile prefix, e.g. outputs/tutorial/scan")
    p.add_argument("-o", "--output", help="also save as <output>_llh.pdf")
    p.add_argument("--rebin", type=int, default=1,
                   help="average the likelihood over N x N blocks of grid points (e.g. 2: 20x20 -> 10x10)")
    args = p.parse_args()

    scans = {k: read_scan(args.input + suffix, args.rebin) for k, suffix in SCANS.items()}
    n = len(VARS)
    fig, axes = plt.subplots(n, n, figsize=(5 * n, 5 * n))
    for i, yk in enumerate(VARS):
        for j, xk in enumerate(VARS):
            ax = axes[i, j]
            ax.set_box_aspect(1)
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                pairs = [(k, s) for k, s in scans.items() if xk in k]
                for m, (key, s) in enumerate(pairs):
                    first = key.index(xk) == 0
                    other = key[1] if first else key[0]
                    grid = scaled(xk, s["x"] if first else s["y"])
                    L = np.exp(-0.5 * s["dchi2"]).sum(axis=1 if first else 0)
                    q = -2 * np.log(np.maximum(L / L.max(), 1e-300))
                    ax.plot(grid, q, color=COLOR if m == 0 else "0.3", lw=3 if m == 0 else 2.5,
                            ls="-" if m == 0 else "--", label=f"integrated over {SHORT[other]}")
                    if s["truth"]:
                        ax.axvline(scaled(xk, s["truth"][0 if first else 1]), color="#a41007", lw=2, ls=":")
                for lev, lab in zip(LEVELS_1D, (r"1$\sigma$", r"2$\sigma$", r"3$\sigma$")):
                    ax.axhline(lev, color="0.5", lw=1.5, ls=":")
                    ax.text(0.02, lev, lab, transform=ax.get_yaxis_transform(), va="bottom", fontsize=16, color="0.4")
                ax.set_ylim(0, 12)
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(r"$\Delta\chi^2$")
                ax.legend(fontsize=14, loc="upper right", frameon=False)
            else:
                s = scans[(xk, yk)]
                mesh = draw_2d(ax, s, xk, yk)
                ax.set_xlim(edges(scaled(xk, s["x"]))[[0, -1]])
                ax.set_ylim(edges(scaled(yk, s["y"]))[[0, -1]])
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
    axes[n - 1, n - 1].set_xlim(edges(scaled("dm2_21", scans[("sin2_th12", "dm2_21")]["y"]))[[0, -1]])

    s0 = next(iter(scans.values()))
    prof = s0["nuisance"] != "false"
    axes[0, 2].text(0.95, 0.95, (f"Nuisances marginalised:\naverage over {s0['nuisance']} prior throws"
                                 if prof else "Conditional scans:\nother parameters at their true values"),
                    transform=axes[0, 2].transAxes, ha="right", va="top", fontsize=14, color="0.3")
    cax = axes[0, 2].inset_axes([0.1, 0.55, 0.8, 0.07])
    label = (rf"$\chi^2/{{\rm ndof}}$ (ndof = {s0['ndof']})" if s0["ndof"] else r"$\Delta\chi^2$")
    fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max", label=label)
    axes[0, 2].text(0.5, 0.05, "contours: 1, 2, 3$\\sigma$\n($\\Delta\\chi^2$ = 2.30, 6.18, 11.83)",
                    transform=axes[0, 2].transAxes, ha="center", va="bottom", fontsize=16, color="0.3")
    fig.tight_layout()
    if args.output:
        plotting.save_figure(fig, args.output + "_llh.pdf", dpi=300, format="pdf")
    plotting.save_figure(fig, os.path.join("images", os.path.basename(args.input) + "_llh.pdf"), dpi=300, format="pdf")
    plt.show()


if __name__ == "__main__":
    main()
