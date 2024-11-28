import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
from matplotlib.ticker import ScalarFormatter
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib import colors

plt.rcParams['text.usetex'] = True
hep.style.use("CMS")


cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952,
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286],
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238,
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571],
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571,
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429],
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667,
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286],
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571,
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429],
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524,
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048,
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381,
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381],
 [0.0589714286, 0.6837571429, 0.7253857143],
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429,
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048],
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619,
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667],
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524,
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905],
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476,
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143],
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
 [0.7184095238, 0.7411333333, 0.3904761905],
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667,
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762],
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857,
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619],
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857,
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381],
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333,
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333],
 [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
parula_map_r = LinearSegmentedColormap.from_list('parula_r', np.flip(cm_data, axis=0))


def gaussian(x, mu, sigma):
    """
    Calculate the value of a Gaussian function at x.

    Parameters:
    x (float): The point at which to evaluate the Gaussian.
    mu (float): The mean (center) of the Gaussian.
    sigma (float): The standard deviation of the Gaussian.

    Returns:
    float: The value of the Gaussian function at x.
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def create_sequential_colormap(start_color, end_color, num_steps=256):
    # Convert color strings to RGBA tuples
    start_color_rgba = colors.to_rgba(start_color)
    end_color_rgba = colors.to_rgba(end_color)

    # Linearly interpolate RGB values between the two colors
    r = np.linspace(start_color_rgba[0], end_color_rgba[0], num_steps)
    g = np.linspace(start_color_rgba[1], end_color_rgba[1], num_steps)
    b = np.linspace(start_color_rgba[2], end_color_rgba[2], num_steps)

    # Create a colormap dictionary
    colormap_dict = {
        'red': [(i / (num_steps - 1), r[i], r[i]) for i in range(num_steps)],
        'green': [(i / (num_steps - 1), g[i], g[i]) for i in range(num_steps)],
        'blue': [(i / (num_steps - 1), b[i], b[i]) for i in range(num_steps)]
    }

    # Create and return the colormap
    colormap = LinearSegmentedColormap('sequential', colormap_dict)
    return colormap


def fade_color_to_white(color, alpha):
    """
    Fades a given color towards white based on the alpha-like parameter.

    Args:
        color (str or tuple): The input color in any valid matplotlib format (e.g., 'red', '#FF5733', (0.2, 0.4, 0.6)).
        alpha (float): The fade parameter, where 0.0 means no fading (original color) and 1.0 means completely white.

    Returns:
        tuple: A tuple representing the faded color in RGB format.
    """
    # Convert the input color to RGB tuple
    rgb_color = colors.to_rgba(color)[:3]

    # Calculate the faded color by interpolating towards white
    faded_color = tuple(np.array(rgb_color) * (1 - alpha) + alpha)

    return faded_color


def plot_corner(variables, data, externalContours=False, color='b', weights=None):
    """
    Create a corner plot of 2D histograms for the given variables.

    :param variables: List of variable names.
    :param data: List of numpy arrays corresponding to each variable.
    """

    num_vars = len(variables)
    fig, axes = plt.subplots(num_vars, num_vars, figsize=(5 * num_vars, 5 * num_vars))

    # Calculate global min and max for each variable
    global_min = [np.min(d) for d in data]
    global_max = [np.max(d) for d in data]

    # Plot contours from other experiments if desired (assumes the order of parameters is sin2th12, sin2th13, dm21)
    if externalContours:
        add_external_solar_data(axes[2, 0])

    for i in range(num_vars):
        for j in range(num_vars):
            if i == j:
                # Plot histogram with constant range
                n, bins, patches = axes[i, j].hist(data[i], bins=30, range=(global_min[i], global_max[i]), color=color, alpha=0.0, density=False, weights=weights)

                # Calculate empirical percentiles for 1, 2, and 3 sigma levels
                sorted_data = np.sort(data[i])
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

                sigma_levels = [0.6827, 0.9545, 0.9973]  # Corresponding to 1, 2, and 3 sigma
                alphas = [0.85, 0.5, 0.2]

                # Plot histogram for each sigma region with different alpha
                for sigma, alpha in zip(sigma_levels, alphas):
                    lower_bound = sorted_data[np.searchsorted(cdf, (1 - sigma) / 2)]
                    upper_bound = sorted_data[np.searchsorted(cdf, 1 - (1 - sigma) / 2)]
                    mask = (data[i] >= lower_bound) & (data[i] <= upper_bound)
                    if weights != None:
                        axes[i, j].hist(data[i][mask], bins=30, range=(global_min[i], global_max[i]), color=color, alpha=alpha, density=False, weights=weights[mask])
                    else:
                        axes[i, j].hist(data[i][mask], bins=30, range=(global_min[i], global_max[i]), color=color, alpha=alpha, density=False)

                axes[i, j].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axes[i, j].xaxis.get_major_formatter().set_scientific(True)
                axes[i, j].xaxis.get_major_formatter().set_powerlimits((-2, 2))
                axes[i, j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axes[i, j].yaxis.get_major_formatter().set_scientific(True)
                axes[i, j].yaxis.get_major_formatter().set_powerlimits((-2, 2))

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
                # Lower triangle: 2D histograms
                sns.histplot(x=data[j], y=data[i], ax=axes[i, j], bins=30, cmap=create_sequential_colormap('white', color), weights=weights)

                # Estimate the density
                xy = np.vstack([data[j], data[i]])
                kde = gaussian_kde(xy)
                if externalContours:
                    x_min, x_max = axes[num_vars - 1, j].get_xlim()
                    y_min, y_max = axes[i, 0].get_ylim()
                else:
                    x_min, x_max = axes[i, j].get_xlim()
                    y_min, y_max = axes[i, j].get_ylim()

                x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
                density = kde(positions).reshape(x_grid.shape)

                # Calculate the cumulative distribution function (CDF)
                sorted_density = np.sort(density.ravel())
                cdf = np.cumsum(sorted_density)
                cdf /= cdf[-1]

                # Find density levels for the desired confidence intervals
                levels = [sorted_density[np.searchsorted(cdf, level)] for level in [0.003, 0.05, 0.32]]

                # Plot contours
                axes[i, j].contour(x_grid, y_grid, density, levels=levels, colors='red', linestyles=[ ':','-.', '-'], linewidths=2.5, weights=weights)

                # Set scientific notation for tick labels
                axes[i, j].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axes[i, j].xaxis.get_major_formatter().set_scientific(True)
                axes[i, j].xaxis.get_major_formatter().set_powerlimits((-2, 2))
                axes[i, j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axes[i, j].yaxis.get_major_formatter().set_scientific(True)
                axes[i, j].yaxis.get_major_formatter().set_powerlimits((-2, 2))

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
                    axes[i, j].xaxis.set_label_coords(0.6, -0.2)  # Center the x-label
                else:
                    axes[i, j].set_xticklabels([])
                if j == 0:
                    axes[i, j].set_ylabel(variables[i])
                    axes[i, j].yaxis.set_label_coords(-0.2, 0.6)  # Center the y-label
                else:
                    axes[i, j].set_yticklabels([])

            elif i == j:
                axes[i, j].set_xlim(axes[num_vars - 1, j].get_xlim())
                if i == num_vars-1:
                    axes[i, j].set_xlim(axes[num_vars - 1, 0].get_ylim())

    # Adjust spacing between plots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    return fig, axes


def add_external_solar_data(ax):
    def overlay_contours(data, ax, fill=False, **plotKwargs):
        hull = ConvexHull(data)
        ordered_points = data[hull.vertices]
        # Extract x and y coordinates from ordered points
        x, y = ordered_points[:, 0], ordered_points[:, 1]

        # Close the contour by adding the first point at the end
        sorted_x = np.append(x, x[0])
        sorted_y = np.append(y, y[0])

        t = np.arange(0, len(sorted_x))
        cs_x = CubicSpline(t, sorted_x)
        cs_y = CubicSpline(t, sorted_y)

        # Generate the smoothed perimeter points
        num_points = int(len(x) * 2.7 )  # Adjust as needed
        smoothed_t = np.linspace(0, len(sorted_x) - 1, num_points)
        smoothed_x = cs_x(smoothed_t)
        smoothed_y = cs_y(smoothed_t)

        # Plot the smoothed contour

        if not fill:
            ax.plot(smoothed_x, smoothed_y, **plotKwargs)
        else:
            polygon = np.asarray([smoothed_x, smoothed_y]).T
            pol = plt.Polygon(polygon, closed=False, fill=True, **plotKwargs)
            ax.add_patch(pol)
        
    # Add external data
    solar1 = np.genfromtxt('inputs/contours/contour1.csv', delimiter=',')
    solar1[:, 0] = np.sin(np.arctan(np.sqrt(solar1[:, 0])))**2
    solar1[:, 1] = solar1[:, 1]*10**-4

    solar2 = np.genfromtxt('inputs/contours/contour2.csv', delimiter=',')
    solar2[:, 0] = np.sin(np.arctan(np.sqrt(solar2[:, 0])))**2
    solar2[:, 1] = solar2[:, 1]*10**-4

    solar3 = np.genfromtxt('inputs/contours/contour3.csv', delimiter=',')
    solar3[:, 0] = np.sin(np.arctan(np.sqrt(solar3[:, 0])))**2
    solar3[:, 1] = solar3[:, 1]*10**-4

    bestFitSolar = np.genfromtxt('inputs/contours/bestFitSolar.csv', delimiter=',')
    bestFitSolar[0] = np.sin(np.arctan(np.sqrt(bestFitSolar[0])))**2
    bestFitSolar[1] = bestFitSolar[1]*10**-4

    kamLAND1 = np.genfromtxt('inputs/contours/kamLAND1.csv', delimiter=',')
    kamLAND1[:, 0] = np.sin(np.arctan(np.sqrt(kamLAND1[:, 0])))**2
    kamLAND1[:, 1] = kamLAND1[:, 1]*10**-4

    kamLAND2 = np.genfromtxt('inputs/contours/kam2.csv', delimiter=',')
    kamLAND2[:, 0] = np.sin(np.arctan(np.sqrt(kamLAND2[:, 0])))**2
    kamLAND2[:, 1] = kamLAND2[:, 1]*10**-4

    kamLAND3 = np.genfromtxt('inputs/contours/kamland3.csv', delimiter=',')
    kamLAND3[:, 0] = np.sin(np.arctan(np.sqrt(kamLAND3[:, 0])))**2
    kamLAND3[:, 1] = kamLAND3[:, 1]*10**-4

    bestFitkamLAND = np.genfromtxt('inputs/contours/bestFitkamLAND.csv', delimiter=',')
    bestFitkamLAND[0] = np.sin(np.arctan(np.sqrt(bestFitkamLAND[0])))**2
    bestFitkamLAND[1] = bestFitkamLAND[1]*10**-4

    contcol = 'orange'
    kamcol = 'purple'
    globcol = 'orange'
    snocol = 'green'

    overlay_contours(solar1, ax, color=contcol, lw=2, ls='-')  # linewidth=10)
    overlay_contours(solar2, ax, color=contcol, lw=2, label='Solar', ls='-.')
    overlay_contours(solar3, ax, color=contcol, lw=2, ls=':')
    ax.plot(bestFitSolar[0], bestFitSolar[1], color=contcol, linestyle='', marker='d', markersize=5)

    overlay_contours(kamLAND1, ax, color=kamcol, lw=2, ls='-')  # linewidth=10)
    overlay_contours(kamLAND2, ax, color=kamcol, lw=2, label='KamLAND', ls='-.')
    overlay_contours(kamLAND3, ax, color=kamcol, lw=2, ls=':')
    ax.plot(bestFitkamLAND[0], bestFitkamLAND[1], color=kamcol, linestyle='', marker='s', markersize=5)