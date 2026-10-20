"""Per-bin maps, posterior-predictive spectra and bin-series panels."""
import numpy as np
from typing import Optional


def plot_binmap(ax, Z, x_edges=None, y_edges=None, title=None,
                symmetric=False, cmap=None, vmin=None, vmax=None,
                mask=None, add_colorbar=True, cbar_kw=None,
                topk=None, topk_style=None):
    """
    Plot a 2D bin map (e.g. CCnight_var, corr, score) onto an existing Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    Z : 2D array
        Map to plot.
    x_edges, y_edges : 1D arrays or None
        Bin edges for pcolormesh. If None, uses pixel coordinates.
    symmetric : bool
        If True, set vmin/vmax symmetric about 0 (good for correlations).
    cmap : str or Colormap or None
        If None, matplotlib default.
    vmin, vmax : float or None
        Color limits. If symmetric=True and both None, auto symmetric.
    mask : 2D bool array or None
        If provided, mask out bins where mask==False (set to NaN).
    add_colorbar : bool
    cbar_kw : dict or None
        Passed to fig.colorbar.
    topk : tuple(rows, cols) or tuple(rows, cols, vals) or None
        Overlay markers at top bins (indices in Z).
    topk_style : dict or None
        Scatter kwargs (e.g. dict(s=30, facecolors='none', edgecolors='k')).

    Returns
    -------
    mappable : QuadMesh or AxesImage
    """
    Zp = np.array(Z, dtype=float, copy=True)
    if mask is not None:
        Zp[~mask] = np.nan

    if symmetric and (vmin is None) and (vmax is None):
        m = np.nanmax(np.abs(Zp))
        vmin, vmax = -m, m

    if x_edges is not None and y_edges is not None:
        # pcolormesh, not imshow: the cos z grid is non-uniform. Accepts centres or edges.
        def _as_edges(e, n):
            e = np.asarray(e, dtype=float)
            if e.size == n + 1:
                return e
            if e.size == n and n >= 2:                      # centres → edges
                mid = 0.5 * (e[1:] + e[:-1])
                return np.concatenate(([2 * e[0] - mid[0]], mid, [2 * e[-1] - mid[-1]]))
            return np.linspace(e[0], e[-1], n + 1)
        xe = _as_edges(x_edges, Zp.shape[1])
        ye = _as_edges(y_edges, Zp.shape[0])
        mappable = ax.pcolormesh(xe, ye, Zp, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
        ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(ye[0], ye[-1])
    else:
        mappable = ax.imshow(Zp, origin="lower", aspect="auto",
                             cmap=cmap, vmin=vmin, vmax=vmax)

    if x_edges is not None:
        ax.set_xlabel(r"$E_{reco}(MeV)$", fontsize=20)
    if title:
        ax.set_title(title)

    if topk is not None:
        rows, cols = topk[0], topk[1]
        style = dict(s=40, facecolors="none", edgecolors="g", linewidths=1.5, marker='s')
        if topk_style:
            style.update(topk_style)

        if x_edges is not None and y_edges is not None:
            ax.scatter(x_edges[cols], y_edges[rows], **style)
        else:
            ax.scatter(cols, rows, **style)

    if add_colorbar:
        fig = ax.figure
        if cbar_kw is None:
            cbar_kw = {}
        fig.colorbar(mappable, ax=ax, **cbar_kw)

    ax.tick_params(axis="both", which="major", labelsize=20)
    return mappable


def plot_predictive_spectrum(ax, pp_mean, bands=None, x_centers=None,
                             pp_lo=None, pp_hi=None,
                             data=None, data_err=None,
                             total_rate=None,
                             bg_var=None,
                             inflate=False,
                             title=None, ylabel="Events"):
    """
    T2K-style posterior predictive plot for a 1D energy spectrum.

    - Blue filled step histogram  : posterior predictive mean
    - Nested red bands            : 1σ/2σ/3σ PPD credible intervals (Poisson-inflated)
    - Black dots with error bars  : Asimov / observed data (background subtracted)

    Parameters
    ----------
    bands : list of (lo, hi) array pairs, outermost first (3σ → 1σ)
    pp_lo, pp_hi : legacy single-band fallback
    total_rate : total expected rate μ_total per bin (signal + bg); adds Poisson shot noise
                 to the PPD: half-width² = rate_hw² + n²·(μ_total + bg_var)
    bg_var : posterior variance of the background mean per bin (enters the PPD; data error bars
             stay Poisson-only)
    """
    # Legacy single-band compat
    if bands is None and pp_lo is not None and pp_hi is not None:
        bands = [(pp_lo, pp_hi)]

    pp_mean = np.asarray(pp_mean, dtype=float)
    above   = pp_mean > 0

    n = len(pp_mean)
    if n < 2:
        return

    step  = x_centers[1] - x_centers[0]
    edges = np.empty(n + 1)
    edges[:-1] = x_centers - step / 2.0
    edges[-1]  = x_centers[-1] + step / 2.0

    # Optional: rate intervals → count intervals, half-width² = hw_rate² + n²·(μ_total + bg_var).
    # Bands are outermost first.
    if inflate and bands and total_rate is not None:
        μ = np.maximum(np.asarray(total_rate, dtype=float), 0.0)
        bv = np.maximum(np.asarray(bg_var, dtype=float), 0.0) if bg_var is not None else 0.0
        nb_bands = len(bands)
        inflated = []
        for k, (lo, hi) in enumerate(bands):
            nsig = nb_bands - k        # 3, 2, 1 for outermost→innermost
            lo  = np.asarray(lo,  dtype=float)
            hi  = np.asarray(hi,  dtype=float)
            hlo = np.maximum(pp_mean - lo, 0.0)
            hhi = np.maximum(hi - pp_mean, 0.0)
            extra_var = (nsig ** 2) * (μ + bv)
            inflated.append((
                pp_mean - np.sqrt(hlo**2 + extra_var),
                pp_mean + np.sqrt(hhi**2 + extra_var),
            ))
        bands = inflated

    # Nested credible bands — drawn outermost first so inner bands paint over them
    # ALPHAS indexed from outermost: 3σ=0.15, 2σ=0.25, 1σ=0.40
    _ALPHAS = [0.15, 0.25, 0.40]
    _LABELS = [r"$3\sigma$", r"$2\sigma$", r"$1\sigma$"]
    if bands:
        nb = len(bands)
        x_rep = np.repeat(edges, 2)[1:-1]
        for i, (lo, hi) in enumerate(bands):
            alpha = _ALPHAS[-(nb - i)]
            label = _LABELS[-(nb - i)]
            # Clip at zero — negative signal rates are unphysical. Band edges (and the
            # mean / data below) are floored at 0; Poisson error bars on the data are kept.
            lo = np.maximum(np.asarray(lo, dtype=float), 0.0)
            hi = np.maximum(np.asarray(hi, dtype=float), 0.0)
            lo_step = np.where(above, lo, np.nan)
            hi_step = np.where(above, hi, np.nan)
            ax.fill_between(x_rep, np.repeat(lo_step, 2), np.repeat(hi_step, 2),
                            color="tab:red", alpha=alpha, linewidth=0, label=label)

    mean_plot = np.where(above, np.maximum(pp_mean, 0.0), np.nan)
    ax.step(edges, np.r_[mean_plot, mean_plot[-1]], where="post",
            color="royalblue", lw=1.8, label="P-pred. mean")

    if data is not None:
        data = np.asarray(data, dtype=float)
        err  = data_err if data_err is not None else np.sqrt(np.maximum(data, 0.0))
        ax.errorbar(x_centers[above], np.maximum(data[above], 0.0), yerr=err[above],
                    fmt="k.", ms=5, capsize=2, lw=1.0,
                    label="Data (Asimov)", zorder=5)

    if title:
        ax.set_title(title)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel(r"$E_{reco}$ (MeV)", fontsize=20)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_xlim(edges[0], edges[-1])


def plot_binseries(ax, y, x_edges=None, title=None, ylabel=None,
                   kind="step", marker=None, mask=None,
                   topk_idx=None, topk_style=None, ylim=None, bar_kwargs=None):
    """
    Plot a 1D binned series (e.g. CCday_var, CCday_score, corr vs energy).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    y : 1D array
    x_edges : 1D array or None
        If given, plot vs bin centers (line) or as a step curve.
    kind : {"step","line","bar"}
    marker : str or None
    mask : 1D bool array or None
        If provided, masked points are set to NaN.
    topk_idx : 1D int array or None
        Indices of top bins to mark.
    topk_style : dict or None
        kwargs for ax.scatter markers

    Returns
    -------
    artist
    """
    if ylim is not None:
        ax.set_ylim(*ylim)

    y = np.array(y, dtype=float, copy=True)
    if mask is not None:
        y[~mask] = np.nan

    bk = bar_kwargs or {}
    if x_edges is None:
        x = np.arange(len(y))
        if kind == "bar":
            artist = ax.bar(x, y, **bk)
        else:
            artist = ax.plot(x, y, marker=marker)[0]
    else:
        if kind == "step":
            # step wants edges; provide y per bin
            artist = ax.step(x_edges, np.r_[y, y[-1]], where="post")[0]
        elif kind == "bar":
            width = x_edges[1]-x_edges[0]
            artist = ax.bar(x_edges, y, width=width, align="center", **bk)
        else:  # "line"
            artist = ax.plot(x_edges, y, marker=marker)[0]

    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if x_edges is not None:
        ax.set_xlabel(r"$E_{reco}(MeV)$", fontsize=20)

    if topk_idx is not None and x_edges is not None:
        style = dict(s=35, facecolors="none", edgecolors="k", linewidths=1.0)
        if topk_style:
            style.update(topk_style)
        x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
        ax.scatter(x_cent[topk_idx], y[topk_idx], **style)

    ax.tick_params(axis="both", which="major", labelsize=20)
    return artist
