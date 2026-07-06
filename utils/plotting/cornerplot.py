"""Corner-plot rendering (main posterior figure)."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
import cmasher
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom as _ndimage_zoom
import matplotlib.patches as patches
from matplotlib import colors
import matplotlib.ticker as ticker
import corner

import style  # shared plotting conventions (sigma levels, line styles, white-mix ratios)
from .colormaps import create_sequential_colormap, scaled_cmap
from .contours import add_external_solar_data, get_contours
from .formatting import sigma_to_prob, two_sig_figs_math


def plot_corner(variables, data_dict, externalContours=False, colorlist=['b', 'purple', 'red'], linecolors=['red', 'white', 'blue'], fill=True, bins2D=60, cbar=None, diagnostics=None, smooth=1.0):
    """
    Create a corner plot of 2D histograms for the given variables.

    :param variables: List of variable names.
    :param data: List of numpy arrays corresponding to each variable.
    """

    data = []
    scaling_info = {}

    for i, key in enumerate(data_dict.keys()):
        if key in {"chains", "weights"}:
            continue  # Skip these keys
        
        value = data_dict[key]
        
        if key in {'integrated_8B_flux', 'integrated_HEP_flux'}:
            # Assuming the central value is the mean (modify if needed)
            central_value = np.mean(value)  
            order_of_magnitude = int(np.floor(np.log10(np.abs(central_value))))
            scaled_value = value / (10 ** order_of_magnitude)
            variables[i] += rf'$(\, 10^{{{order_of_magnitude}}} \, \mathrm{{cm}}^{{-2}}\mathrm{{s}}^{{-1}})$'
            data.append(scaled_value)

        elif key == "dm2_21":
            data.append(value * 1e4)  # Apply the scaling for dm2_21
        else:
            data.append(value)  # Default case (no scaling)

    weights = data_dict["weights"]

    # Resolve colormap name (handle case aliases; others pass through unchanged)
    _cmap_aliases = {
        'blues': 'Blues', 'blues_r': 'Blues_r',
        'cosmic': 'cmr.cosmic', 'cosmic_r': 'cmr.cosmic_r',
        'torch': 'cmr.torch', 'torch_r': 'cmr.torch_r',
        'freeze': 'cmr.freeze', 'freeze_r': 'cmr.freeze_r',
        'ember': 'cmr.ember', 'ember_r': 'cmr.ember_r',
    }
    _cmap_name = _cmap_aliases.get(cbar, cbar)
    use_cmap = cbar is not None

    saved_levels = None
    num_vars = len(variables)
    fig, axes = plt.subplots(num_vars, num_vars, figsize=(5 * num_vars, 5 * num_vars))

    # Calculate global min and max for each variable
    global_min = [np.min(d) for d in data]
    global_max = [np.max(d) for d in data]

    # N_eff-per-bin stability metric.  For each histogram bin we compute the effective
    # number of independent posterior samples it contains:
    #   N_eff_bin = ESS_total × (fraction of posterior in bin) / tau
    # where ESS_total accounts for importance-weight imbalance and tau for autocorrelation.
    # Thresholds from MCMC literature (Gelman/Vehtari):
    #   N_eff < 5  → bin too sparse to trust (red)
    #   5 ≤ N_eff < 25 → marginal (amber)
    #   N_eff ≥ 25 → reliable for contour estimation (blue)
    _stab_t1, _stab_t2 = 5, 25   # N_eff thresholds
    _stab_colors = ['#d6604d', '#fdcc8a', '#4393c3']   # red / amber / blue
    _stab_cmap = mpl.colors.ListedColormap(_stab_colors, name='stab_diag')
    # Use zone-index [0,1,2] as the image data so the colorbar scale is uniform
    _stab_norm = mpl.colors.BoundaryNorm([0, 1, 2, 3], _stab_cmap.N)
    # N_unique: total accepted positions — divided by tau below to get ESS
    _n_samples = float(len(weights))

    # Plot contours from other experiments if desired (assumes the order of parameters is sin2th12, sin2th13, dm21)
    if externalContours:
        legend_handles = add_external_solar_data(axes[2, 0])
        # Place legend in the (0,1) white space cell
        axes[0, 1].legend(handles=legend_handles, fontsize=24,
                          framealpha=0.0, loc='center right',
                          handlelength=1.2)

    for i in range(num_vars):
        for j in range(num_vars):
            if i == j:
                # Set colors
                if i <= 2: color=colorlist[0] 
                elif i <= 4: color=colorlist[2]
                else: color='gray'

                # Plot histogram with constant range (invisible) to get bin counts
                n, bins, patches = axes[i, j].hist(
                    data[i], bins=bins2D, range=(global_min[i], global_max[i]),
                    color=color, alpha=0.0, density=False, weights=weights)

                # Pre-compute marginal sigma boundaries from the data (these are used
                # for both the vertical lines and to assign bins to sigma regions so
                # the colours match the vlines exactly).
                sorted_data = np.sort(data[i])
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                sigma_levels = style.SIGMA_LEVELS  # 1σ, 2σ, 3σ

                sigma_bounds = []
                for sigma in sigma_levels:
                    lower_bound = sorted_data[np.searchsorted(cdf, (1 - sigma) / 2)]
                    upper_bound = sorted_data[np.searchsorted(cdf, 1 - (1 - sigma) / 2)]
                    # Use histogram-aligned bin thresholds for consistency with plotting
                    lower_bound_hist = bins[np.argmin(np.abs(bins - lower_bound))]
                    upper_bound_hist = bins[np.argmin(np.abs(bins - upper_bound))]
                    sigma_bounds.append((lower_bound_hist, upper_bound_hist))

                # Now build region colours from the Wilks-like transform evaluated at
                # the bin counts at the sigma boundaries. This way the colour assigned
                # to each region corresponds to the colour at the boundary above it.
                if use_cmap:
                    _cmap_fn = plt.get_cmap(_cmap_name)
                    counts = np.array(n, dtype=float)
                    counts_max = counts.max() if counts.max() > 0 else 1.0
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])

                    # For each sigma boundary take the count of the bin nearest the
                    # outer boundary (upper_bound_hist) as the boundary count.
                    boundary_counts = []
                    for (low_b, up_b) in sigma_bounds:
                        # find nearest bin center to the outer edge (up_b)
                        idx = np.argmin(np.abs(bin_centers - up_b))
                        boundary_counts.append(counts[idx])

                    _sigma_cut = 3.0
                    cutoff_d_wilks = []
                    for c in boundary_counts:
                        d_rel = (c / (counts_max + 1e-300))
                        d_rel = np.clip(d_rel, np.exp(-_sigma_cut**2 / 2.0), 1.0)
                        d_w = 1.0 - np.sqrt(-2.0 * np.log(d_rel)) / _sigma_cut
                        cutoff_d_wilks.append(np.clip(d_w, 0.0, 1.0))

                    # region_colors: innermost (1σ) uses cmap(1.0), next uses cmap at
                    # the 1σ boundary, next uses cmap at the 2σ boundary, outside uses cmap(0.0)
                    region_colors = [
                        _cmap_fn(1.0),
                        _cmap_fn(cutoff_d_wilks[0] if len(cutoff_d_wilks) > 0 else 0.0),
                        _cmap_fn(cutoff_d_wilks[1] if len(cutoff_d_wilks) > 1 else 0.0),
                        _cmap_fn(0.0),
                    ]

                    # Assign each bin to a sigma-region based on its center falling
                    # inside the marginal sigma bounds computed above.
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    for k, p in enumerate(patches):
                        bc = bin_centers[k]
                        # check innermost first
                        if (sigma_bounds[0][0] <= bc) and (bc <= sigma_bounds[0][1]):
                            col = region_colors[0]
                        elif (sigma_bounds[1][0] <= bc) and (bc <= sigma_bounds[1][1]):
                            col = region_colors[1]
                        elif (sigma_bounds[2][0] <= bc) and (bc <= sigma_bounds[2][1]):
                            col = region_colors[2]
                        else:
                            col = region_colors[3]
                        p.set_facecolor(col)
                        p.set_edgecolor(col)
                        p.set_alpha(1.0)
                else:
                    # leave patches invisible; overlays below will draw the coloured regions
                    pass

                # Calculate empirical percentiles for 1, 2, and 3 sigma levels
                sorted_data = np.sort(data[i])
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

                sigma_levels = style.SIGMA_LEVELS  # Corresponding to 1, 2, and 3 sigma
                alphas = [0.85, 0.5, 0.2]

                if use_cmap:
                    _cmap_fn = plt.get_cmap(_cmap_name)
                    # Use the region colours we computed above (innermost->outer). If for
                    # some reason region_colors is not present, fall back to a simple mapping.
                    try:
                        _hist_colors = [region_colors[0], region_colors[1], region_colors[2]]
                    except Exception:
                        # 1σ→1.0, 2σ→0.75, 3σ→0.5, background→0.0 (fallback)
                        _hist_colors = [_cmap_fn(1.0 - (s - 1) / 4.0) for s in [1, 2, 3]]
                else:
                    _hist_colors = [color, color, color]

                # Plot histogram for each sigma region with different alpha
                if diagnostics != 'stab':
                    for sigma, alpha, sigma_color in zip(sigma_levels, alphas, _hist_colors):
                        lower_bound = sorted_data[np.searchsorted(cdf, (1 - sigma) / 2)]
                        upper_bound = sorted_data[np.searchsorted(cdf, 1 - (1 - sigma) / 2)]

                        lower_bound_hist = bins[np.argmin(np.abs(bins - lower_bound))]
                        upper_bound_hist = bins[np.argmin(np.abs(bins - upper_bound))]

                        mask = (data[i] >= lower_bound_hist) & (data[i] <= upper_bound_hist)
                        if weights is not None:
                            axes[i, j].hist(data[i][mask], bins=bins, range=(global_min[i], global_max[i]), color=sigma_color, alpha=alpha, density=False, weights=weights[mask])
                        else:
                            axes[i, j].hist(data[i][mask], bins=bins, range=(global_min[i], global_max[i]), color=sigma_color, alpha=alpha, density=False)

                        # Lines to separate the confidence intervals in the 1D histograms
                        _line_color = plt.get_cmap(_cmap_name)(0.0) if use_cmap else 'w'
                        axes[i, j].axvline(lower_bound_hist, color=_line_color, lw=1)
                        axes[i, j].axvline(upper_bound_hist, color=_line_color, lw=1)

                # ── 1D stability map ────────────────────────────────────────
                if diagnostics == 'stab':
                    _x_edges_1d = np.linspace(global_min[i], global_max[i], bins2D + 1)
                    _cids_1d = np.unique(data_dict['chains'])

                    def _autocorr_tau_stab(chains_x):
                        """Sokal IACT from ACF pooled across chains.
                        Cap at n//50 lags: the Sokal window condition requires
                        chain_len >> 50*tau; beyond this cap the estimate is
                        unreliable and inflates catastrophically.
                        Returns (tau, pooled_acf_array)."""
                        acfs = []
                        max_lag = 1
                        for x in chains_x:
                            n = len(x)
                            if n < 4: continue
                            max_lag = max(max_lag, n // 50)
                            xc = x - x.mean()
                            if np.var(xc) < 1e-300: continue
                            f = np.fft.rfft(xc, n=2 * n)
                            acf = np.fft.irfft(f * np.conj(f))[:n].real
                            acf /= (acf[0] + 1e-300)
                            acfs.append(acf[:max_lag])
                        if not acfs:
                            return 1, np.array([1.0])
                        min_len = min(len(a) for a in acfs)
                        pooled = np.mean([a[:min_len] for a in acfs], axis=0)
                        tau = 1.0
                        for k in range(1, min_len):
                            tau += 2.0 * pooled[k]
                            if k >= 5 * tau: break
                        return max(1, int(np.ceil(tau))), pooled

                    _tau_1d, _pooled_acf = _autocorr_tau_stab([data[i][data_dict['chains'] == _c] for _c in _cids_1d])
                    _acf_lags = [1, 5, 10, 25, 50, 100, 200, 500]
                    _acf_str = "  ".join(
                        f"ρ({lag})={_pooled_acf[lag]:.3f}" for lag in _acf_lags if lag < len(_pooled_acf)
                    )
                    print(f"[stab 1D] var={variables[i]!r:35s}  tau={_tau_1d:4d}  ESS≈{_n_samples / _tau_1d:.0f}  {_acf_str}")

                    # N_eff-per-bin: effective independent samples per bin
                    _H_full_1d, _ = np.histogram(data[i], bins=_x_edges_1d, weights=weights)
                    _H_norm_1d = _H_full_1d.astype(float) / (_H_full_1d.sum() + 1e-300)
                    _N_eff_1d = _n_samples * _H_norm_1d / _tau_1d
                    _y_max_1d = _H_full_1d.max() * 1.15
                    axes[i, j].set_ylim(0, _y_max_1d)
                    axes[i, j].set_facecolor('white')
                    _bin_zone = np.digitize(_N_eff_1d, bins=[_stab_t1, _stab_t2])
                    _bin_colors_1d = [_stab_cmap.colors[min(z, 2)] for z in _bin_zone]
                    axes[i, j].bar(
                        _x_edges_1d[:-1], _H_full_1d,
                        width=np.diff(_x_edges_1d), align='edge',
                        color=_bin_colors_1d, edgecolor='k', linewidth=0.3, zorder=1,
                    )
                    # Nominal sigma vlines: solid=1σ (innermost), dash-dot=2σ, dotted=3σ (outermost)
                    for _si, (_lo, _hi) in enumerate(sigma_bounds):
                        _ls1d = ['-', '-.', ':'][_si]
                        axes[i, j].axvline(_lo, color='black', lw=1.5, linestyle=_ls1d, zorder=3)
                        axes[i, j].axvline(_hi, color='black', lw=1.5, linestyle=_ls1d, zorder=3)

                # ── 1D bootstrap: faint + min/max vlines ────────────────────
                elif diagnostics == 'bootstrap':
                    if use_cmap:
                        _sigma_cut_1d = 3.0
                        _cmap_fn_1d = plt.get_cmap(_cmap_name)
                        _boot_colors_1d = [
                            tuple(1.0 - c for c in _cmap_fn_1d(1.0 - _s / _sigma_cut_1d)[:3])
                            for _s in [1, 2, 3]
                        ]
                    else:
                        _boot_colors_1d = [color] * 3
                    # Estimate autocorrelation time per chain for this parameter
                    def _autocorr_tau_1d(x):
                        n = len(x)
                        if n < 4: return 1
                        xc = x - x.mean()
                        if np.var(xc) < 1e-300: return 1
                        f = np.fft.rfft(xc, n=2 * n)
                        acf = np.fft.irfft(f * np.conj(f))[:n].real
                        acf /= (acf[0] + 1e-300)
                        tau = 1.0
                        for k in range(1, n // 2):
                            tau += 2.0 * acf[k]
                            if k >= 5 * tau: break
                        return max(1, int(np.ceil(tau)))

                    _cids_1d = np.unique(data_dict['chains'])
                    _nc_1d = len(_cids_1d)
                    _tau_1d = max(
                        _autocorr_tau_1d(data[i][data_dict['chains'] == _c])
                        for _c in _cids_1d
                    )
                    # Build blocks per chain so they never span chain boundaries
                    _blocks_1d = []
                    for _c in _cids_1d:
                        _m_idx = np.where(data_dict['chains'] == _c)[0]
                        _nblocks_c = max(1, len(_m_idx) // _tau_1d)
                        for _k in range(_nblocks_c):
                            _blocks_1d.append(_m_idx[_k * _tau_1d : (_k + 1) * _tau_1d])
                    _nblocks_1d = len(_blocks_1d)
                    _nkeep_1d = max(2, _nblocks_1d // 2)

                    def _get_1d_bounds(bx):
                        _bs = np.sort(bx); _bc = np.arange(1, len(_bs) + 1) / len(_bs)
                        return [
                            (bins[np.argmin(np.abs(bins - _bs[np.searchsorted(_bc, (1 - _s) / 2)]))],
                             bins[np.argmin(np.abs(bins - _bs[np.searchsorted(_bc, 1 - (1 - _s) / 2)]))])
                            for _s in sigma_levels
                        ]

                    _boot_bounds_1d = []
                    # Chain-level resamples (60 for a smooth envelope)
                    for _ in range(60):
                        _rs = np.random.choice(_cids_1d, size=_nc_1d, replace=True)
                        _bx = np.concatenate([data[i][data_dict['chains'] == _c] for _c in _rs])
                        _boot_bounds_1d.append(_get_1d_bounds(_bx))
                    # Block subsampling resamples using per-chain blocks (50 resamples)
                    for _ in range(50):
                        _kept = np.random.choice(_nblocks_1d, size=_nkeep_1d, replace=False)
                        _idx = np.concatenate([_blocks_1d[_k] for _k in _kept])
                        _boot_bounds_1d.append(_get_1d_bounds(data[i][_idx]))
                    # Median ± 1σ band (avoids single-outlier resample inflating the span)
                    for _si in range(3):
                        _lo_vals = np.array([_b[_si][0] for _b in _boot_bounds_1d])
                        _hi_vals = np.array([_b[_si][1] for _b in _boot_bounds_1d])
                        for _vals in (_lo_vals, _hi_vals):
                            _med = np.median(_vals)
                            _sig = np.std(_vals)
                            axes[i, j].axvspan(_med - _sig, _med + _sig, color=_boot_colors_1d[_si], alpha=0.5, zorder=0.5)

                axes[i, j].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axes[i, j].xaxis.get_major_formatter().set_scientific(True)
                axes[i, j].xaxis.get_major_formatter().set_powerlimits((-2, 2))
                axes[i, j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axes[i, j].yaxis.get_major_formatter().set_scientific(True)
                axes[i, j].yaxis.get_major_formatter().set_powerlimits((-2, 2))

                axes[i, j].yaxis.set_ticks_position('right')

                if i == num_vars - 1:
                    axes[i, j].set_xlabel(variables[i])
                    axes[i, j].xaxis.set_label_coords(0.6, -0.2)  # Center the x-label
                else:
                    axes[i, j].set_xticklabels([])
                if i == 0:
                    axes[i, j].set_ylabel(r'$dP$')
                    axes[i, j].yaxis.set_label_coords(-0.2, 0.6)  # Center the y-label
                axes[i, j].set_yticklabels([])

            elif i > j:
                # Set colors
                if i <= 2: color, linecolor = colorlist[0], linecolors[0]
                elif j <= 2 and i <= 4: color,linecolor = colorlist[1], linecolors[1]
                elif i <= 4: color, linecolor = colorlist[2], linecolors[2]
                else: color, linecolor = 'gray', 'black'

                # ── shared bootstrap helper ──────────────────────────────
                def _autocorr_tau(x):
                    """Integrated autocorrelation time via Sokal's windowed FFT estimator.
                    Capped at n//50 lags: reliable estimation requires chain_len >> 50*tau."""
                    n = len(x)
                    if n < 4:
                        return 1
                    xc = x - x.mean()
                    if np.var(xc) < 1e-300:
                        return 1
                    max_lag = max(10, n // 50)
                    f = np.fft.rfft(xc, n=2 * n)
                    acf = np.fft.irfft(f * np.conj(f))[:max_lag].real
                    acf /= (acf[0] + 1e-300)
                    tau = 1.0
                    for k in range(1, max_lag):
                        tau += 2.0 * acf[k]
                        if k >= 5 * tau:
                            break
                    return max(1, int(np.ceil(tau)))

                def _run_bootstrap_chains(n=15):
                    """Chain-level bootstrap: resample whole chains with replacement."""
                    _cids = np.unique(data_dict['chains'])
                    _nc = len(_cids)
                    _results = []
                    for _ in range(n):
                        _rs = np.random.choice(_cids, size=_nc, replace=True)
                        _bx, _by, _bw = [], [], []
                        for _c in _rs:
                            _m = (data_dict['chains'] == _c)
                            _bx.append(data[j][_m]); _by.append(data[i][_m]); _bw.append(weights[_m])
                        _bx = np.concatenate(_bx); _by = np.concatenate(_by); _bw = np.concatenate(_bw)
                        _results.append(get_contours(_bx, _by, _bw, bins2D, smooth=smooth))
                    return _results

                def _run_bootstrap_blocks(n=15):
                    """Block subsampling: keep ~50% of tau-sized blocks, respecting chain boundaries."""
                    _cids = np.unique(data_dict['chains'])
                    _tau = 1
                    for _c in _cids:
                        _m = (data_dict['chains'] == _c)
                        _tau = max(_tau, _autocorr_tau(data[i][_m]), _autocorr_tau(data[j][_m]))
                    # Build index blocks per chain so blocks never span chain boundaries
                    _all_blocks = []
                    for _c in _cids:
                        _m_idx = np.where(data_dict['chains'] == _c)[0]
                        _nblocks_c = max(1, len(_m_idx) // _tau)
                        for _k in range(_nblocks_c):
                            _all_blocks.append(_m_idx[_k * _tau : (_k + 1) * _tau])
                    _nblocks = len(_all_blocks)
                    _nkeep = max(2, _nblocks // 2)
                    _results = []
                    for _ in range(n):
                        _kept = np.random.choice(_nblocks, size=_nkeep, replace=False)
                        _idx = np.concatenate([_all_blocks[_k] for _k in _kept])
                        _bx = data[j][_idx]; _by = data[i][_idx]; _bw = weights[_idx]
                        _results.append(get_contours(_bx, _by, _bw, bins2D, smooth=smooth))
                    return _results

                # ── Lower triangle: 2D visualisation ─────────────────────
                if diagnostics == 'stab':
                    # N_eff-per-bin 2D stability map
                    _x_edges = np.linspace(data[j].min(), data[j].max(), bins2D + 1)
                    _y_edges = np.linspace(data[i].min(), data[i].max(), bins2D + 1)
                    _cids = np.unique(data_dict['chains'])
                    # Pooled multi-chain tau for each parameter; take worst-case
                    _tau_j_2d, _ = _autocorr_tau_stab([data[j][data_dict['chains'] == _c] for _c in _cids])
                    _tau_i_2d, _ = _autocorr_tau_stab([data[i][data_dict['chains'] == _c] for _c in _cids])
                    # Geometric mean: for independent updates τ_2D ≈ harmonic mean of τ_i and τ_j;
                    # geometric mean is a reasonable middle ground between harmonic and max.
                    _tau_2d = max(1, int(np.ceil(np.sqrt(_tau_j_2d * _tau_i_2d))))
                    _H_full_2d, _, _ = np.histogram2d(data[j], data[i],
                                                       bins=[_x_edges, _y_edges], weights=weights)
                    _H_norm_2d = _H_full_2d.astype(float) / (_H_full_2d.sum() + 1e-300)
                    _N_eff_2d = (_n_samples * _H_norm_2d / _tau_2d).T  # (y, x)
                    # Upsample N_eff with bicubic interpolation — no pre-smoothing so peak values
                    # are not deflated (gaussian_filter would spread the peak and demote blue→amber)
                    _N_eff_2d_up = np.maximum(0, _ndimage_zoom(_N_eff_2d, 8, order=3))
                    _zone_2d = np.digitize(_N_eff_2d_up, bins=[_stab_t1, _stab_t2]).astype(float)
                    axes[i, j].imshow(
                        _zone_2d, origin='lower', aspect='auto',
                        extent=[_x_edges[0], _x_edges[-1], _y_edges[0], _y_edges[-1]],
                        cmap=_stab_cmap, norm=_stab_norm, interpolation='nearest',
                    )
                    axes[i, j].set_facecolor(_stab_cmap.colors[0])  # red = no samples
                    # Nominal contours as reference
                    x_grid, y_grid, density, levels, x_HPD, y_HPD = get_contours(data[j], data[i], weights, bins2D, smooth=1.)
                    axes[i, j].contour(x_grid, y_grid, density, levels=levels[:3],
                                       colors='black', linestyles=style.SIGMA_LINESTYLES, linewidths=1.5)

                elif use_cmap:
                    # Smooth 2D density rendered with Wilks-stretched colormap
                    x_grid, y_grid, density, levels, x_HPD, y_HPD = get_contours(data[j], data[i], weights, bins2D, smooth=smooth * 2.)
                    _sigma_cut = 3.0
                    _d = density / (density.max() + 1e-300)
                    _d = np.clip(_d, np.exp(-_sigma_cut**2 / 2), 1.0)
                    _d_wilks = 1.0 - np.sqrt(-2.0 * np.log(_d)) / _sigma_cut
                    axes[i, j].imshow(
                        _d_wilks, origin='lower', aspect='auto',
                        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
                        cmap=_cmap_name, vmin=0, vmax=1, interpolation='bilinear',
                    )
                    axes[i, j].set_facecolor(plt.get_cmap(_cmap_name)(0.0))
                    _cmap_fn_2d = plt.get_cmap(_cmap_name)
                    axes[i, j].contourf(x_grid, y_grid, density, levels=levels,
                                        colors=[_cmap_fn_2d(0.2), _cmap_fn_2d(0.5), _cmap_fn_2d(0.85)],
                                        alpha=0.4)
                    _bg_2d = _cmap_fn_2d(0.0)
                    _lum_2d = 0.299 * _bg_2d[0] + 0.587 * _bg_2d[1] + 0.114 * _bg_2d[2]
                    _cline_2d = 'white' if _lum_2d < 0.5 else 'black'
                    axes[i, j].contour(x_grid, y_grid, density, levels=levels[:3],
                                       colors=[_cline_2d]*3, linestyles=style.SIGMA_LINESTYLES, linewidths=1.5)

                elif fill:
                    sns.histplot(x=data[j], y=data[i], ax=axes[i, j], bins=bins2D, cmap=create_sequential_colormap('white', color), weights=weights, alpha=0.5, fill=False)
                    x_grid, y_grid, density, levels, x_HPD, y_HPD = get_contours(data[j], data[i], weights, bins2D, smooth=smooth)
                else:
                    hist = sns.histplot(x=data[j], y=data[i], ax=axes[i, j], bins=bins2D, cmap=cmasher.guppy_r, edgecolor='face', weights=weights, alpha=1, fill=True)
                    hist.collections[0].set_norm(colors.LogNorm(vmin=1))
                    x_grid, y_grid, density, levels, x_HPD, y_HPD = get_contours(data[j], data[i], weights, bins2D, smooth=smooth)

                # Add custom colour transitions for levels
                original_color = colors.to_rgb(color)
                mix_ratios = style.WHITE_MIX
                strong_white = tuple(np.array(original_color) * mix_ratios[0] + np.array((1, 1, 1)) * (1 - mix_ratios[0]))
                weaker_white = tuple(np.array(original_color) * mix_ratios[1] + np.array((1, 1, 1)) * (1 - mix_ratios[1]))

                # Plot contours (non-cmap, non-stab modes only)
                if not use_cmap and diagnostics != 'stab':
                    if fill:
                        axes[i, j].contourf(x_grid, y_grid, density, levels=levels, colors=[strong_white, weaker_white, color], alpha=1)
                        axes[i, j].contour(x_grid, y_grid, density, levels=levels, colors=linecolor, linestyles=style.SIGMA_LINESTYLES, linewidths=2)
                    else:
                        axes[i, j].contour(x_grid, y_grid, density, levels=levels, colors=linecolor, linestyles=style.SIGMA_LINESTYLES, linewidths=3)

                # Plot highest posterior density
                axes[i, j].plot(x_HPD, y_HPD, marker='o', color='white', markersize=4)

                # Chain-level bootstrap: overlay contours to show statistical stability
                if diagnostics == 'bootstrap':
                    if use_cmap:
                        # Per-sigma-level brightness complement: white lines on dark backgrounds,
                        # black lines on bright backgrounds, matched to each contour's local region.
                        _sigma_cut = 3.0
                        _cmap_fn = plt.get_cmap(_cmap_name)
                        _boot_colors = [
                            tuple(1.0 - c for c in _cmap_fn(1.0 - _s / _sigma_cut)[:3])
                            for _s in [3, 2, 1]
                        ]
                    else:
                        _boot_colors = [linecolor] * 3
                    _boot_results = _run_bootstrap_chains(60) + _run_bootstrap_blocks(50)

                    # Draw all bootstrap contours per sigma level
                    for _si in range(3):
                        for _xg, _yg, _bd, _bl, _, _ in _boot_results:
                            axes[i, j].contour(
                                _xg, _yg, _bd, levels=[_bl[_si]],
                                colors=[_boot_colors[_si]],
                                linestyles=['-'],
                                linewidths=0.3,
                                alpha=0.3,
                            )

                # Set scientific notation for tick labels
                axes[i, j].xaxis.set_major_formatter(ticker.FuncFormatter(two_sig_figs_math))
                #axes[i, j].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                #axes[i, j].xaxis.get_major_formatter().set_scientific(True)
                #axes[i, j].xaxis.get_major_formatter().set_powerlimits((-3, 3))

                axes[i, j].yaxis.set_major_formatter(ticker.FuncFormatter(two_sig_figs_math))
                #axes[i, j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                #axes[i, j].yaxis.get_major_formatter().set_scientific(True)
                #axes[i, j].yaxis.get_major_formatter().set_powerlimits((-3, 3))



            else:
                # Upper triangle: leave empty
                axes[i, j].axis('off')

            # Set aspect ratio to be equal
            axes[i, j].set_box_aspect(1)


    for i in range(num_vars):
        for j in range(num_vars):
            if i > j:
                if i == num_vars - 1:
                    axes[i, j].set_xlabel(variables[j])
                    if j < 2 or j > 4:
                        axes[i, j].xaxis.set_label_coords(0.6, -0.2)  # Center the x-label
                    else:
                        axes[i, j].xaxis.set_label_coords(0.9, -0.2)  # Center the x-label                        
                else:
                    axes[i, j].set_xticklabels([])
                if j == 0:
                    axes[i, j].set_ylabel(variables[i])
                    if i < 2 or i > 4:
                        axes[i, j].yaxis.set_label_coords(-0.3, 0.6)  # Center the y-label
                    else:
                        axes[i, j].yaxis.set_label_coords(-0.3, 0.9)  # Center the x-label
                else:
                    axes[i, j].set_yticklabels([])

            elif i == j:
                axes[i, j].set_xlim(axes[num_vars - 1, j].get_xlim())
                if i == num_vars-1:
                    axes[i, j].set_xlim(axes[num_vars - 1, 0].get_ylim())
                    axes[i, j].xaxis.set_label_coords(0.9, -0.2)

    # Adjust spacing between plots
    plt.subplots_adjust(hspace=0.0, wspace=-0.015)

    if use_cmap:
        _bg = plt.get_cmap(_cmap_name)(0.0)
        # Choose axis/tick colour based on background luminance
        _lum = 0.299 * _bg[0] + 0.587 * _bg[1] + 0.114 * _bg[2]
        _ax_color = 'white' if _lum < 0.5 else 'black'
        for i in range(num_vars):
            axes[i, i].set_facecolor(_bg)
        for ax in axes.flat:
            for spine in ax.spines.values():
                spine.set_edgecolor(_ax_color)
            ax.tick_params(which='both', color=_ax_color)
    elif diagnostics == 'stab':
        # Discrete map has clear colour blocks; black spines are legible against all three
        for _r in range(num_vars):
            for _c in range(num_vars):
                for spine in axes[_r, _c].spines.values():
                    spine.set_edgecolor('black')
                axes[_r, _c].tick_params(which='both', color='black')

        # Colorbar in the top-right empty corner – double height, top edge kept fixed
        _stab_sm = plt.cm.ScalarMappable(cmap=_stab_cmap, norm=_stab_norm)
        _stab_sm.set_array([])
        _cb_ax = axes[0, num_vars - 1]
        _cb_ax.set_visible(False)
        _cb_pos = _cb_ax.get_position()
        _cbar_ax = fig.add_axes([
            _cb_pos.x0 + _cb_pos.width * 0.15,
            _cb_pos.y0 - _cb_pos.height * 0.80,
            _cb_pos.width * 0.25,
            _cb_pos.height * 1.70,
        ])
        _cbar = fig.colorbar(_stab_sm, cax=_cbar_ax)
        _cbar.set_label(r'Indep.\ posterior samples per bin ($N_\mathrm{eff}$)',
                        color='black', fontsize=22, labelpad=30)
        # Ticks at zone boundaries on the outside, labelled with actual N_eff cutoffs
        _cbar.set_ticks([1, 2])
        _cbar.set_ticklabels([r'$N_\mathrm{eff}=5$', r'$N_\mathrm{eff}=25$'])
        _cbar.ax.yaxis.set_tick_params(color='black', length=6, width=1.5)
        _cbar.outline.set_edgecolor('black')
        plt.setp(_cbar.ax.yaxis.get_ticklabels(), color='black', fontsize=20)
        # Zone labels inside each color band (rotated to fit narrow bar)
        for _yt, _lab, _tc in [
            (1.0/6.0, r'\textbf{unreliable}', 'black'),
            (1.0/2.0, r'\textbf{marginal}',   'black'),
            (5.0/6.0, r'\textbf{reliable}',   'white'),
        ]:
            _cbar_ax.text(0.5, _yt, _lab,
                          transform=_cbar_ax.transAxes, ha='center', va='center',
                          color=_tc, fontsize=18, rotation=90)
        # Thin separator lines at zone boundaries
        _cbar.ax.axhline(1, color='#333333', linewidth=1.0)
        _cbar.ax.axhline(2, color='#333333', linewidth=1.0)

    return fig, axes, saved_levels


def plot_default_corner(data, diagnostics=False):
    """
    Generate a corner plot with standardized styling.
    
    Parameters:
    -----------
    data : dict
        Dictionary of arrays with parameter samples, including 'weights' key
    diagnostics : bool
        If True, uses cmasher.tropical_r colormap and data ranges
        If False, uses solid blue colors and fixed ranges
    """
    with mpl.rc_context(rc=mpl.rcParamsDefault): # Reset to default settings
        plt.rcParams['text.usetex'] = True 
        hep.style.use("ATLAS")

        # Extract metadata to avoid plotting it
        weights = data.pop('weights', None)
        chains = data.pop('chains', None)
        steps = data.pop('stepno', None)
        
        # Convert remaining data dict to numpy array format
        variables = [var for var in data.keys() if data[var] is not None and len(data[var]) > 0 and np.min(data[var]) != np.max(data[var])]  # get rid of empty vars and vars with no dynamic range
        samples = np.column_stack([data[var] for var in variables])

        ndim = len(variables)
        label_size = max(18 * np.exp(-0.02 * ndim), 6)
        
        plot_kwargs = {
            'plot_datapoints': False,
            'fill_contours': True,
            'alpha': 1,
            'levels': style.SIGMA_LEVELS,  # 1σ, 2σ, 3σ
            'smooth': 0.8,
            'show_titles': True,
            'title_kwargs': {'fontsize': 12},
            
            # Custom contour appearance (line color and fill)
            'contour_kwargs': {
                'colors': 'red',   # Set contour lines to white
                'linewidths': 1,   # Set the width of contour lines
                'linestyles': style.SIGMA_LINESTYLES
            },

            'hist_kwargs': { "color": "#0894bc", "alpha": 0.6, "linewidth": 1.5},
            'label_kwargs': {"fontsize": label_size},
            
            # Custom contour fill appearance with alpha transparency
            'contourf_kwargs': {
                'alpha': [0, 0.2, 0.55, 1],  # Transparency for each contour level
                'colors': ['#0894bc']  # Blue shades for contour fills
            },
        }
        
        # Style configuration
        if diagnostics:
            # Diagnostics mode - tropical_r colormap, data ranges
            plot_kwargs.update({
                'range': None,  # Auto-range
                'levels': sigma_to_prob(np.linspace(0, 3, 12)[1:]),
                'fill_contours': True,
                'no_fill_contours': True,
                'plot_density': True,
                'smooth': 0.,
                'color': None,
                'hist_kwargs': { "color": "black", "alpha": 0.5, "histtype":'stepfilled'},
                'contour_kwargs': {'colors': 'white', "linewidths": 1.5},
                'contourf_kwargs': {'colors': None,'cmap': scaled_cmap(cmasher.tropical_r)}
            })
        
        # Create the plot
        fig = corner.corner(
            samples,
            bins=40,
            labels=variables,
            weights=data.get('weights', None),
            **plot_kwargs
        )

        return fig
