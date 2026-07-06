"""2D posterior contour extraction and external-experiment overlays."""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline


def get_contours(x, y, weights, bins, smooth=0.8):
    data_range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Parse the bin specifications.
    try:
        bins = [int(bins) for _ in data_range]
    except TypeError:
        if len(bins) != len(data_range):
            raise ValueError("Dimension mismatch between bins and range")

    bins_2d = []
    bins_2d.append(np.linspace(min(data_range[0]), max(data_range[0]), bins[0] + 1))
    bins_2d.append(np.linspace(min(data_range[1]), max(data_range[1]), bins[0] + 1))
    levels = [0, 0.6827, 0.9545, 0.9973]

    try:
        H, X, Y = np.histogram2d(
            x.flatten(),
            y.flatten(),
            bins=bins_2d,
            weights=weights,
        )
    except ValueError:
        raise ValueError(
            "It looks like at least one of your sample columns "
            "have no dynamic range. You could try using the "
            "'range' argument."
        )

    if H.sum() == 0:
        raise ValueError(
            "It looks like the provided 'range' is not valid "
            "or the sample is empty."
        )

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    # Get HPD point
    max_index = np.unravel_index(np.argmax(H), H.shape)

    # The coordinates of the highest density point are in 'max_index'
    x_HPD = X[max_index[0]]
    y_HPD = Y[max_index[1]]

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except IndexError:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        warnings.warn("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate(
        [
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ]
    )
    Y2 = np.concatenate(
        [
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ]
    )

    return X2, Y2, H2.T, V, x_HPD, y_HPD


def overlay_contours(data, ax, fill=False, smooth=True, **plotKwargs):
    hull = ConvexHull(data)
    ordered_points = data[hull.vertices]
    # Extract x and y coordinates from ordered points
    x, y = ordered_points[:, 0], ordered_points[:, 1]

    # Close the contour by adding the first point at the end
    sorted_x = np.append(x, x[0])
    sorted_y = np.append(y, y[0])

    if smooth:
        t = np.arange(0, len(sorted_x))
        cs_x = CubicSpline(t, sorted_x)
        cs_y = CubicSpline(t, sorted_y)
        num_points = int(len(x) * 2.7)
        smoothed_t = np.linspace(0, len(sorted_x) - 1, num_points)
        plot_x = cs_x(smoothed_t)
        plot_y = cs_y(smoothed_t)
    else:
        plot_x = sorted_x
        plot_y = sorted_y

    if not fill:
        ax.plot(plot_x, plot_y, **plotKwargs)
    else:
        polygon = np.asarray([plot_x, plot_y]).T
        pol = plt.Polygon(polygon, closed=False, fill=True, **plotKwargs)
        ax.add_patch(pol)


def add_external_solar_data(ax, experiments=('kamland', 'solar', 'juno'), color=None):
    """Overlay external-experiment contours on the sin2_th12 vs dm2_21 panel.

    experiments : which of 'kamland', 'solar' (global/nuFit), 'juno' to draw.
    color       : if given, draw all selected experiments in this single colour
                  (e.g. 'gray'); otherwise each uses its default colour.
    Returns the legend handles for the experiments actually drawn.
    """
    # Add external data
    solar1 = np.genfromtxt('inputs/contours/contour1.csv', delimiter=',')
    solar1[:, 0] = np.sin(np.arctan(np.sqrt(solar1[:, 0])))**2
    solar1[:, 1] = solar1[:, 1]*1e0

    solar2 = np.genfromtxt('inputs/contours/contour2.csv', delimiter=',')
    solar2[:, 0] = np.sin(np.arctan(np.sqrt(solar2[:, 0])))**2
    solar2[:, 1] = solar2[:, 1]*1e0

    solar3 = np.genfromtxt('inputs/contours/contour3.csv', delimiter=',')
    solar3[:, 0] = np.sin(np.arctan(np.sqrt(solar3[:, 0])))**2
    solar3[:, 1] = solar3[:, 1]*1e0

    bestFitSolar = np.genfromtxt('inputs/contours/bestFitSolar.csv', delimiter=',')
    bestFitSolar[0] = np.sin(np.arctan(np.sqrt(bestFitSolar[0])))**2
    bestFitSolar[1] = bestFitSolar[1]*1e0

    kamLAND1 = np.genfromtxt('inputs/contours/kamLAND1.csv', delimiter=',')
    kamLAND1[:, 0] = np.sin(np.arctan(np.sqrt(kamLAND1[:, 0])))**2
    kamLAND1[:, 1] = kamLAND1[:, 1]*1e0

    kamLAND2 = np.genfromtxt('inputs/contours/kam2.csv', delimiter=',')
    kamLAND2[:, 0] = np.sin(np.arctan(np.sqrt(kamLAND2[:, 0])))**2
    kamLAND2[:, 1] = kamLAND2[:, 1]*1e0

    kamLAND3 = np.genfromtxt('inputs/contours/kamland3.csv', delimiter=',')
    kamLAND3[:, 0] = np.sin(np.arctan(np.sqrt(kamLAND3[:, 0])))**2
    kamLAND3[:, 1] = kamLAND3[:, 1]*1e0

    bestFitkamLAND = np.genfromtxt('inputs/contours/bestFitkamLAND.csv', delimiter=',')
    bestFitkamLAND[0] = np.sin(np.arctan(np.sqrt(bestFitkamLAND[0])))**2
    bestFitkamLAND[1] = bestFitkamLAND[1]*1e0

    juno1 = np.genfromtxt('inputs/contours/juno_1sigma.csv', delimiter=',')
    juno1[:, 1] *= 1e-1

    juno2 = np.genfromtxt('inputs/contours/juno_2sigma.csv', delimiter=',')
    juno2[:, 1] *= 1e-1

    juno3 = np.genfromtxt('inputs/contours/juno_3sigma.csv', delimiter=',')
    juno3[:, 1] *= 1e-1

    bestFitJuno = np.genfromtxt('inputs/contours/bestFitJuno.csv', delimiter=',')
    bestFitJuno[1] *= 1e-1

    global1 = np.genfromtxt('inputs/contours/nuFit_1sigma.csv', delimiter=',')
    global1[:, 1] = global1[:, 1]*1e-1

    global2 = np.genfromtxt('inputs/contours/nuFit_2sigma.csv', delimiter=',')
    global2[:, 1] = global2[:, 1]*1e-1

    global3 = np.genfromtxt('inputs/contours/nuFit_3sigma.csv', delimiter=',')
    global3[:, 1] = global3[:, 1]*1e-1

    bestFitGlobal = np.genfromtxt('inputs/contours/bestFitGlobal.csv', delimiter=',')
    bestFitGlobal[1] = bestFitGlobal[1]*1e-1

    from matplotlib.lines import Line2D

    kamcol  = color or 'purple'
    globcol = color or 'orange'
    junocol = color or 'crimson'

    # contours first, then best-fit markers, then legend handles (order preserved)
    if 'kamland' in experiments:
        overlay_contours(kamLAND1, ax, color=kamcol, lw=2, ls='-')
        overlay_contours(kamLAND2, ax, color=kamcol, lw=2, label='KamLAND', ls='-.')
        overlay_contours(kamLAND3, ax, color=kamcol, lw=2, ls=':')
    if 'solar' in experiments:
        overlay_contours(global1, ax, color=globcol, lw=3, ls='-')
        overlay_contours(global2, ax, color=globcol, lw=3, label='Solar', ls='-.')
        overlay_contours(global3, ax, color=globcol, lw=3, ls=':')
    if 'juno' in experiments:
        overlay_contours(juno1, ax, color=junocol, lw=2, ls='-')
        overlay_contours(juno2, ax, color=junocol, lw=2, label='JUNO', ls='-')
        overlay_contours(juno3, ax, color=junocol, lw=2, ls='-')

    handles = []
    if 'kamland' in experiments:
        ax.plot(bestFitkamLAND[0], bestFitkamLAND[1], color=kamcol, linestyle='', marker='s', markersize=5)
        handles.append(Line2D([0], [0], color=kamcol, lw=2, label='KamLAND'))
    if 'solar' in experiments:
        ax.plot(bestFitGlobal[0], bestFitGlobal[1], color=globcol, linestyle='', marker='P', markersize=6)
        handles.append(Line2D([0], [0], color=globcol, lw=2, label='Solar (current)'))
    if 'juno' in experiments:
        ax.plot(bestFitJuno[0], bestFitJuno[1], color=junocol, linestyle='', marker='d', markersize=5)
        handles.append(Line2D([0], [0], color=junocol, lw=2, label='JUNO'))

    return handles    
