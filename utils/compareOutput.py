"""Comparison-mode corner plot: overlay two posteriors on the standard corner.

Draws the standard 3-parameter corner (sin^2 th12, sin^2 th13, dm^2_21) for TWO
chain sets at once, with the global solar contours and the same display spans as
plotOutput, so two runs can be compared directly. Each posterior is drawn as nested
white-graded credible regions (prussian / vermillion) with hard, contour-aligned
edges; where they overlap the fill is their intersection colour (a purple at the
same lightness). Colours and conventions come from utils/style.py.

Run from the repo root (like plotOutput) so inputs/contours/* and outputs/* resolve:

    python3 utils/compareOutput.py -a runA_chain1 runA_chain2 -b runB_chain1 \
        --labels "Run A" "Run B" -o images/corner_comparison.pdf
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
from scipy.ndimage import gaussian_filter, zoom as _ndimage_zoom

import style              # shared palette + plotting conventions (applies CMS/LaTeX rcParams on import)
import plotting           # plotting.add_external_solar_data, two_sig_figs_math, save_figure
import posteriorHelpers

# --- palette & conventions: sourced from style (single source of truth) ---
PRUSSIAN, VERMILLION = style.COMPARE_COLORS                 # posterior A, B
# overlap = true intersection colour: the mean of the two posterior colours, which lands a
# purple at the SAME lightness as the singles (not the darker multiply/dedicated-purple result).
OVERLAP = tuple(0.5 * (np.array(mcolors.to_rgb(PRUSSIAN)) + np.array(mcolors.to_rgb(VERMILLION))))
ZOOM    = 6              # density upsampling factor for crisp, contour-aligned fill edges

SIGMA_FRAC  = style.SIGMA_LEVELS
CONTOUR_LW  = 2.2                 # contour-line and 1D-outline width
SINGLE_MIX   = (0.15, 0.6, 1.0)    # 3-sigma, 2-sigma, 1-sigma fill: colour->white mix (fills fade out)
OVERLAP_MIN  = 0.35                 # minimum overlap mix — must exceed SINGLE_MIX[0] so 3s+3s != lone 3s
LINE_DARKEN  = (0.55, 1.0)         # 2-sigma, 1-sigma line darkness (3-sigma has no contour line)

PARAMETERS = ['sin2_th12', 'sin2_th13', 'dm2_21']
VARIABLES  = [r'$\sin^2\theta_{12}$', r'$\sin^2\theta_{13}$',
              r'$\Delta m^2_{21}$ $(10^{-4} \mathrm{ eV}^2)$']

EXP_PARAMETERS = ['sin2_th12', 'sin2_th13', 'dm2_21',
                  'integrated_8B_flux', 'integrated_HEP_flux']
EXP_VARIABLES  = [r'$\sin^2\theta_{12}$', r'$\sin^2\theta_{13}$',
                  r'$\Delta m^2_{21}$ $(10^{-4} \mathrm{ eV}^2)$',
                  r'$\Phi_\odot^\mathrm{8B}$', r'$\Phi_\odot^\mathrm{hep}$']
# CC asymmetry is appended at runtime if the key is present in both chains
_EXP_ASYM_KEY  = 'derived_CC_asymmetry'
_EXP_ASYM_VAR  = r'$A_{D-N}(\mathrm{CC})$'


def _white_grade(color, mix):
    """`color` mixed toward white by `mix` in [0, 1] (1 = full colour, 0 = white)."""
    c = np.array(mcolors.to_rgb(color))
    return c * mix + (1.0 - mix)


def _darken(color, factor):
    """`color` scaled toward black by `factor` in [0, 1] (1 = full colour, 0 = black)."""
    return tuple(np.array(mcolors.to_rgb(color)) * factor)


def _overlap_mix(s):
    """White-mix for an overlap pixel from the two depths' sum s (2..6).

    Spans OVERLAP_MIN (s=2, 3σ+3σ) to 1.0 (s=6, 1σ+1σ). OVERLAP_MIN > SINGLE_MIX[0]
    so the minimum overlap shade is always visibly distinct from a lone 3σ fill.
    """
    return OVERLAP_MIN + (s - 2) / 4.0 * (SINGLE_MIX[-1] - OVERLAP_MIN)


def _compose_depths(depth_a, depth_b, color_a, color_b, color_o):
    """Build an RGB fill image from the two posteriors' sigma-depth maps.

    depth_* are integers in {0 (outside), 1 (3-sigma), 2 (2-sigma), 3 (1-sigma)} per pixel.
    Single-posterior pixels take that posterior's colour white-graded by its depth (1-sigma full
    -> 3-sigma near-white). Overlap pixels take the overlap colour white-graded by the COMBINED
    depth (depth_a + depth_b), so every sigma combination is a distinct shade. Hard band
    boundaries keep the colour steps exactly on the contour lines.
    """
    out = np.ones((*depth_a.shape, 3))
    both   = (depth_a > 0) & (depth_b > 0)
    only_a = (depth_a > 0) & ~both
    only_b = (depth_b > 0) & ~both
    for d in (1, 2, 3):
        out[only_a & (depth_a == d)] = _white_grade(color_a, SINGLE_MIX[d - 1])
        out[only_b & (depth_b == d)] = _white_grade(color_b, SINGLE_MIX[d - 1])
    ssum = depth_a + depth_b
    for s in range(2, 7):
        out[both & (ssum == s)] = _white_grade(color_o, _overlap_mix(s))
    return out


def _hpd_levels(field):
    """Highest-posterior-density thresholds (ascending: 3-sigma -> 1-sigma) for a density grid."""
    flat = np.sort(field.ravel())[::-1]
    csum = np.cumsum(flat)
    csum = csum / csum[-1]
    lv = sorted(flat[min(np.searchsorted(csum, f), flat.size - 1)] for f in SIGMA_FRAC)
    # enforce strictly increasing so matplotlib contour() accepts the levels
    for k in range(1, len(lv)):
        if lv[k] <= lv[k - 1]:
            lv[k] = lv[k - 1] * (1 + 1e-6) + 1e-12
    return lv


def _depth_field_2d(x, y, w, x_edges, y_edges, smooth=2.0, zoom=ZOOM):
    """Per-pixel sigma depth on an upsampled grid for one posterior.

    Returns (depth, fine_field, levels): the 2D histogram is smoothed, its HPD levels found,
    then upsampled (cubic) so the depth map and the contour lines drawn from `fine_field` are
    smooth and coincide. depth in {0,1,2,3} (3 = 1-sigma core).
    """
    H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=w)
    field = gaussian_filter(H, smooth).T
    levels = _hpd_levels(field)
    fine = np.maximum(_ndimage_zoom(field, zoom, order=3), 0.0)
    depth = ((fine >= levels[0]).astype(int)
             + (fine >= levels[1]).astype(int)
             + (fine >= levels[2]).astype(int))
    return depth, fine, levels


def _fine_centers(edges, n):
    """Pixel-centre coordinates matching an imshow drawn with extent=[edges[0], edges[-1]]."""
    return edges[0] + (np.arange(n) + 0.5) * (edges[-1] - edges[0]) / n


def _sigma_bounds(d):
    """1/2/3-sigma marginal (lo, hi) bounds from the empirical CDF (unweighted, as plot_corner)."""
    sd = np.sort(d)
    cdf = np.arange(1, sd.size + 1) / sd.size
    return [(sd[np.searchsorted(cdf, (1 - s) / 2)],
             sd[np.searchsorted(cdf, 1 - (1 - s) / 2)]) for s in SIGMA_FRAC]


def _values(data_dict, parameters=PARAMETERS):
    """Per-variable sample arrays in display units (dm2_21 scaled to 1e-4 eV^2),
    mirroring plotting.plot_corner."""
    out = []
    for key in parameters:
        v = np.asarray(data_dict[key], dtype=float)
        out.append(v * 1e4 if key == "dm2_21" else v)
    return out


def _read_truth_from_config(chain_path):
    """Read Asimov truth values from <chain>_configSettings.txt.

    Returns a dict with display-unit values keyed by parameter name:
      sin2_th12, sin2_th13, dm2_21 (×1e4), integrated_8B_flux, integrated_HEP_flux,
      derived_CC_asymmetry (always 0 — symmetric Asimov).
    Raises FileNotFoundError / ValueError if the file is missing or unparseable.
    """
    cfg_path = chain_path + "_configSettings.txt"
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"configSettings file not found: {cfg_path}")
    raw = {}
    for line in open(cfg_path):
        line = line.strip()
        if line.startswith("True sin²θ₁₂:"):
            raw['sin2_th12'] = float(line.split(":")[-1])
        elif line.startswith("True sin²θ₁₃:"):
            raw['sin2_th13'] = float(line.split(":")[-1])
        elif line.startswith("True Δm²₂₁:"):
            raw['dm2_21'] = float(line.split(":")[-1])
        elif line.startswith("True ⁸B flux:"):
            raw['integrated_8B_flux'] = float(line.split(":")[-1])
        elif line.startswith("True HEP flux:"):
            raw['integrated_HEP_flux'] = float(line.split(":")[-1])
    missing = [k for k in ('sin2_th12', 'sin2_th13', 'dm2_21') if k not in raw]
    if missing:
        raise ValueError(f"Could not parse truth values {missing} from {cfg_path}")
    print(f"Truth from {cfg_path}: {raw}")
    return {
        'sin2_th12':          raw['sin2_th12'],
        'sin2_th13':          raw['sin2_th13'],
        'dm2_21':             raw['dm2_21'] * 1e4,
        'integrated_8B_flux': raw.get('integrated_8B_flux'),
        'integrated_HEP_flux': raw.get('integrated_HEP_flux'),
        _EXP_ASYM_KEY:        0.0,
    }


def plot_comparison_corner(variables, datasets, colors, labels,
                           bins2D=60, smooth=0.5, externalContours=True, truth=None,
                           parameters=PARAMETERS):
    """Overlay the corner plots of two posteriors with translucent fills.

    datasets   : list of two load_posterior() dicts.
    colors     : list of two fill colours (posterior A, posterior B).
    labels     : list of two legend labels.
    truth      : optional dict mapping parameter name → display-unit value
                 (from _read_truth_from_config); drawn as black triangle with white margin.
    parameters : ordered list of parameter keys matching datasets; defaults to PARAMETERS.
    """
    num_vars = len(variables)
    fig, axes = plt.subplots(num_vars, num_vars, figsize=(5 * num_vars, 5 * num_vars))

    vals    = [_values(d, parameters) for d in datasets]
    weights = [d["weights"] for d in datasets]

    # Shared per-variable range across BOTH posteriors so bins/marginals align.
    gmin = [min(vals[s][k].min() for s in range(len(datasets))) for k in range(num_vars)]
    gmax = [max(vals[s][k].max() for s in range(len(datasets))) for k in range(num_vars)]

    # Global solar (nuFit) contour only, in a contrasting colour, on the dm2_21 vs sin2_th12 panel.
    if externalContours:
        ext_handles = plotting.add_external_solar_data(axes[2, 0],
                                                       experiments=('solar',), color=style.COMPARE_SOLAR)
    else:
        ext_handles = []

    for i in range(num_vars):
        for j in range(num_vars):
            ax = axes[i, j]

            if i == j:
                # --- diagonal: 1D marginals; fills darken outward, overlap in OVERLAP colour ---
                x_edges = np.linspace(gmin[i], gmax[i], bins2D + 1)
                xc = 0.5 * (x_edges[:-1] + x_edges[1:])
                heights, depths_x = [], []
                for s in (0, 1):
                    h, _ = np.histogram(vals[s][i], bins=x_edges, weights=weights[s], density=True)
                    dx = np.zeros(bins2D, dtype=int)             # sigma depth per x-bin (outer->inner)
                    for lvl, (lo, hi) in enumerate(_sigma_bounds(vals[s][i])[::-1], start=1):
                        dx[(xc >= lo) & (xc <= hi)] = lvl
                    heights.append(h); depths_x.append(dx)
                ymax = max(h.max() for h in heights) * 1.10
                yc = np.linspace(0.0, ymax, 256)
                depth_a = np.where(yc[:, None] < heights[0][None, :], depths_x[0][None, :], 0)
                depth_b = np.where(yc[:, None] < heights[1][None, :], depths_x[1][None, :], 0)
                img = _compose_depths(depth_a, depth_b, colors[0], colors[1], OVERLAP)
                ax.imshow(img, extent=[x_edges[0], x_edges[-1], 0.0, ymax],
                          origin='lower', aspect='auto', interpolation='nearest', zorder=0)
                for s in (0, 1):                                  # crisp marginal outline
                    ax.step(xc, heights[s], where='mid', color=colors[s], lw=CONTOUR_LW, alpha=0.95, zorder=2)
                ax.set_xlim(gmin[i], gmax[i]); ax.set_ylim(0.0, ymax)
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(plotting.two_sig_figs_math))
                ax.yaxis.set_ticks_position('right')
                ax.set_yticklabels([])
                if i == 0:
                    ax.set_ylabel(r'$dP$')
                    ax.yaxis.set_label_coords(-0.2, 0.6)

            elif i > j:
                # --- lower triangle: white-graded credible regions; hard edges on the contour line ---
                x_edges = np.linspace(gmin[j], gmax[j], bins2D + 1)
                y_edges = np.linspace(gmin[i], gmax[i], bins2D + 1)
                depth_a, fine_a, lev_a = _depth_field_2d(vals[0][j], vals[0][i], weights[0], x_edges, y_edges, smooth=smooth)
                depth_b, fine_b, lev_b = _depth_field_2d(vals[1][j], vals[1][i], weights[1], x_edges, y_edges, smooth=smooth)
                img = _compose_depths(depth_a, depth_b, colors[0], colors[1], OVERLAP)
                ax.imshow(img, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                          origin='lower', aspect='auto', interpolation='nearest', zorder=0)
                xcf = _fine_centers(x_edges, fine_a.shape[1])
                ycf = _fine_centers(y_edges, fine_a.shape[0])
                # 1-sigma (solid) and 2-sigma (dash-dot) lines only; 3-sigma is fill only
                for fine, lev, col in ((fine_a, lev_a, colors[0]), (fine_b, lev_b, colors[1])):
                    shades = [_darken(col, f) for f in LINE_DARKEN]
                    ax.contour(xcf, ycf, fine, levels=[lev[2]],
                               colors=[shades[1]], linestyles='-',   linewidths=CONTOUR_LW, zorder=2.5)
                    ax.contour(xcf, ycf, fine, levels=[lev[1]],
                               colors=[shades[0]], linestyles='-.', linewidths=CONTOUR_LW, zorder=2.5)
                # Best-fit points: white ring (outer) + posterior colour (inner)
                for fine, col in ((fine_a, colors[0]), (fine_b, colors[1])):
                    r, c = np.unravel_index(fine.argmax(), fine.shape)
                    x_hpd, y_hpd = xcf[c], ycf[r]
                    ax.plot(x_hpd, y_hpd, marker='o', color='white',   markersize=7, zorder=5, linestyle='')
                    ax.plot(x_hpd, y_hpd, marker='o', color=col,       markersize=4, zorder=6, linestyle='')
                # Asimov truth point: white triangle (margin) + black triangle (fill)
                if truth is not None:
                    x_tr = truth.get(parameters[j])
                    y_tr = truth.get(parameters[i])
                    if x_tr is not None and y_tr is not None:
                        ax.plot(x_tr, y_tr, marker='^', color='black', markersize=7,
                                markeredgecolor='white', markeredgewidth=2.0, zorder=8, linestyle='')
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(plotting.two_sig_figs_math))
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(plotting.two_sig_figs_math))

            else:
                ax.axis('off')

            ax.set_box_aspect(1)

    # --- axis labels on the outer edges (with same label_coords as plot_corner) ---
    for i in range(num_vars):
        for j in range(num_vars):
            if i <= j:
                continue
            if i == num_vars - 1:
                axes[i, j].set_xlabel(variables[j])
                axes[i, j].xaxis.set_label_coords(0.6, -0.2)
            else:
                axes[i, j].set_xticklabels([])
            if j == 0:
                axes[i, j].set_ylabel(variables[i])
                ycoord = 0.9 if i >= num_vars - 1 else 0.6
                axes[i, j].yaxis.set_label_coords(-0.3, ycoord)
            else:
                axes[i, j].set_yticklabels([])
    # diagonal bottom-right: x-label for the last parameter (skipped by i<=j guard above)
    axes[num_vars - 1, num_vars - 1].set_xlabel(variables[num_vars - 1])
    axes[num_vars - 1, num_vars - 1].xaxis.set_label_coords(0.9, -0.2)

    # --- fixed display spans ---
    idx_th12 = parameters.index('sin2_th12') if 'sin2_th12' in parameters else None
    idx_dm2  = parameters.index('dm2_21')    if 'dm2_21'    in parameters else None
    idx_hep  = parameters.index('integrated_HEP_flux') if 'integrated_HEP_flux' in parameters else None
    idx_asym = parameters.index(_EXP_ASYM_KEY) if _EXP_ASYM_KEY in parameters else None
    for k in range(num_vars):
        if idx_th12 is not None:
            axes[k, idx_th12].set_xlim(0.23, 0.43)
        if idx_dm2 is not None:
            axes[k, idx_dm2].set_xlim(0.2, 1.4)
            if k != idx_dm2:
                axes[idx_dm2, k].set_ylim(0.2, 1.4)
        # zero-lines for HEP flux and CC asymmetry columns
        if idx_hep is not None and k != idx_hep:
            axes[idx_hep, k].axhline(0, color='red',  lw=1.5, ls='--', zorder=3)
            axes[k, idx_hep].axvline(0, color='red',  lw=1.5, ls='--', zorder=3)
        if idx_asym is not None and k != idx_asym:
            axes[idx_asym, k].axhline(0, color='gray', lw=1.5, ls='--', zorder=3)
            axes[k, idx_asym].axvline(0, color='gray', lw=1.5, ls='--', zorder=3)
    # sync each diagonal's xlim to its column's bottom-row panel
    for k in range(num_vars):
        axes[k, k].set_xlim(axes[num_vars - 1, k].get_xlim())

    # --- legend (posteriors + overlap + external experiments) in the empty (0,1) cell ---
    from matplotlib.lines import Line2D
    post_handles = [Patch(facecolor=colors[0], edgecolor=colors[0], label=labels[0]),
                    Patch(facecolor=colors[1], edgecolor=colors[1], label=labels[1]),
                    Patch(facecolor=OVERLAP,   edgecolor=OVERLAP,   label='overlap')]
    if truth is not None:
        post_handles.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='black',
                                   markeredgecolor='white', markeredgewidth=2.0,
                                   markersize=9, label='truth', linestyle=''))
    axes[0, 1].legend(handles=post_handles + ext_handles, fontsize=24,
                      framealpha=0.0, loc='center', handlelength=1.4)

    plt.subplots_adjust(hspace=0.0, wspace=-0.015)
    return fig, axes


def main():
    import sys

    # Extract --labels values before argparse sees them, because argparse rejects
    # label values that start with '-' (e.g. '-1$\sigma$') as unknown flags.
    argv = sys.argv[1:]
    labels_override = None
    if '--labels' in argv:
        idx = argv.index('--labels')
        if idx + 2 < len(argv):
            labels_override = [argv[idx + 1], argv[idx + 2]]
            argv = argv[:idx] + argv[idx + 3:]

    parser = argparse.ArgumentParser(
        description="Overlay the standard corner plot for two posteriors (comparison mode).")
    parser.add_argument('-a', '--chains-a', nargs='+', required=True,
                        help="First chain set (one or more chain files).")
    parser.add_argument('-b', '--chains-b', nargs='+', required=True,
                        help="Second chain set (one or more chain files).")
    parser.add_argument('--labels', nargs=2, default=['A', 'B'],
                        help="Legend labels for the two posteriors (default: A B).")
    parser.add_argument('--burnin', type=int, default=100,
                        help="Burnin steps to discard (default: 100).")
    parser.add_argument('--bins', type=int, default=60,
                        help="Number of 2D histogram bins per axis (default: 60).")
    parser.add_argument('--smooth', type=float, default=0.5,
                        help="Gaussian smoothing sigma (in bins) applied to the 2D histograms "
                             "before drawing contours; affects the displayed credible regions "
                             "(default: 0.5).")
    parser.add_argument('--exclude', nargs='*', type=int, default=[20],
                        help="Chain IDs to exclude (default: 20). Use --exclude with no args to keep all.")
    parser.add_argument('-o', '--output', type=str, default='images/corner_comparison.pdf',
                        help="Output file (default: images/corner_comparison.pdf).")
    parser.add_argument('--truth', action='store_true', default=False,
                        help="Overlay the Asimov truth point read from the first chain's "
                             "_configSettings.txt. Drawn as a black triangle with white margin.")
    parser.add_argument('-e', '--expanded', action='store_true', default=False,
                        help="Expanded corner: adds 8B flux, HEP flux, and CC day-night "
                             "asymmetry panels (if available in the chains).")
    parser.add_argument('--detector', type=str, default=None, help="Detector whose derived quantities to use, for chains derived with several detectors (default: the first).")
    args = parser.parse_args(argv)
    if labels_override is not None:
        args.labels = labels_override

    exclude = args.exclude if args.exclude is not None else []
    print(f"[info] 2D histogram smoothing: sigma={args.smooth} bins (change with --smooth)")

    truth_display = None
    if args.truth:
        truth_display = _read_truth_from_config(args.chains_a[0])

    if args.expanded:
        active_params = list(EXP_PARAMETERS)
        active_vars   = list(EXP_VARIABLES)
        test_key      = [(_EXP_ASYM_KEY, _EXP_ASYM_VAR)]
    else:
        active_params = list(PARAMETERS)
        active_vars   = list(VARIABLES)
        test_key      = None

    data_a = posteriorHelpers.load_posterior(args.chains_a, active_params,
                                             burnin=args.burnin, exclude_chains=exclude,
                                             test=test_key, detector=args.detector)
    data_b = posteriorHelpers.load_posterior(args.chains_b, active_params,
                                             burnin=args.burnin, exclude_chains=exclude,
                                             test=test_key, detector=args.detector)

    if args.expanded and _EXP_ASYM_KEY in data_a and data_a[_EXP_ASYM_KEY].size > 0:
        if _EXP_ASYM_KEY in data_b and data_b[_EXP_ASYM_KEY].size > 0:
            active_params.append(_EXP_ASYM_KEY)
            active_vars.append(_EXP_ASYM_VAR)

    out_path = args.output
    if args.expanded and not out_path.endswith('_expanded.pdf'):
        stem = out_path[:-4] if out_path.endswith('.pdf') else out_path
        out_path = stem + '_expanded.pdf'

    fig, _ = plot_comparison_corner(
        active_vars, [data_a, data_b],
        colors=[PRUSSIAN, VERMILLION], labels=args.labels, bins2D=args.bins,
        smooth=args.smooth, truth=truth_display, parameters=active_params,
    )

    print(f"saving output as {out_path}")
    plotting.save_figure(fig, out_path, dpi=300, format='pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
