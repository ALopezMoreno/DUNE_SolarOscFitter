import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
import cmasher
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib import colors
import matplotlib.ticker as ticker
from scipy.special import erf
import corner

plt.rcParams['text.usetex'] = True
hep.style.use("CMS")
plt.rcParams.update({
    'font.size': 14,           # Default text size (affects all text)
    'axes.titlesize': 16,      # Title size
    'axes.labelsize': 32,  
    'axes.linewidth': 2.5,

    'xtick.labelsize': 30,
    'ytick.labelsize': 30,     # Y-axis tick label size
    
    "xtick.major.width": 2.5,
    "ytick.major.width": 2.5,
    "xtick.minor.width": 1.5,
    "ytick.minor.width": 1.5,
})

#fmt = ticker.ScalarFormatter(useMathText=True)
#fmt.set_scientific(True)
#fmt.set_powerlimits((-3, 3))
#fmt.set_useOffset(False)

# Override the formatting function to enforce 2 sig figs
def format_func(x):
    return f"{x:.1g}"

#fmt.format_data = format_func  # Affects tick labels

def get_contours(x, y, weights, bins, smooth=0.2):
    range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Parse the bin specifications.
    try:
        bins = [int(bins) for _ in range]
    except TypeError:
        if len(bins) != len(range):
            raise ValueError("Dimension mismatch between bins and range")

    bins_2d = []
    bins_2d.append(np.linspace(min(range[0]), max(range[0]), bins[0] + 1))
    bins_2d.append(np.linspace(min(range[1]), max(range[1]), bins[0] + 1))
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
        print(fr'smoothing with a gaussian filter of std = {smooth}')
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
    if np.any(m) and not quiet:
        logging.warning("Too few points to create valid contours")
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


def two_sig_figs_math(x, pos):
    # Format to 2 significant figures, then wrap in MathText
    return "${}$".format(f"{x:.2g}")  # LaTeX-style formatting

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

def sigma_to_prob(sigma):
    return 0.5 * (1 + erf(sigma / np.sqrt(2)))


def scaled_cmap(cmap, ncolors=256, power=0.3, white_bottom=True):
    
    # Generate quadratic spacings (t^power)
    t = np.linspace(0, 1, ncolors)
    quad_spacings = t ** power
    
    # Sample colors from the original colormap
    colors_array = cmap(quad_spacings)
    
    # Replace the first color with white if desired
    if white_bottom:
        colors_array[0] = [1., 1., 1., 1.]  # white (RGBA)
        colors_array[0] = [1., 1., 1., 1.]
    
    # Create a new colormap
    new_cmap = colors.ListedColormap(colors_array)
    
    return new_cmap


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
    faded_color = np.array(rgb_color, dtype='float') * (1 - alpha) + alpha

    # Convert the numpy array to a list of Python floats
    return faded_color.tolist()


def plot_corner(variables, data_dict, externalContours=False, colorlist=['b', 'purple', 'red'], linecolors=['red', 'white', 'blue'], fill=True, bins2D=60):
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

    saved_levels = None
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
                # Set colors
                if i <= 2: color=colorlist[0] 
                elif i <= 4: color=colorlist[2]
                else: color='gray'

                # Plot histogram with constant range
                n, bins, patches = axes[i, j].hist(data[i], bins=100, range=(global_min[i], global_max[i]), color=color, alpha=0.0, density=False, weights=weights)

                # Calculate empirical percentiles for 1, 2, and 3 sigma levels
                sorted_data = np.sort(data[i])
                cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

                sigma_levels = [0.6827, 0.9545, 0.9973]  # Corresponding to 1, 2, and 3 sigma
                alphas = [0.85, 0.5, 0.2]

                # Plot histogram for each sigma region with different alpha
                for sigma, alpha in zip(sigma_levels, alphas):
                    lower_bound = sorted_data[np.searchsorted(cdf, (1 - sigma) / 2)]
                    upper_bound = sorted_data[np.searchsorted(cdf, 1 - (1 - sigma) / 2)]

                    lower_bound_hist = bins[np.argmin(np.abs(bins - lower_bound))]
                    upper_bound_hist = bins[np.argmin(np.abs(bins - upper_bound))]

                    mask = (data[i] >= lower_bound_hist) & (data[i] <= upper_bound_hist)
                    if weights is not None:
                        # print('plotting according to weights')
                        axes[i, j].hist(data[i][mask], bins=bins, range=(global_min[i], global_max[i]), color=color, alpha=alpha, density=False, weights=weights[mask])
                    else:
                        axes[i, j].hist(data[i][mask], bins=bins, range=(global_min[i], global_max[i]), color=color, alpha=alpha, density=False)

                    # White lines to separate the confidence intervals in the 1D histograms
                    axes[i, j].axvline(lower_bound_hist, color='w', lw=1)
                    axes[i, j].axvline(upper_bound_hist, color='w', lw=1)
                        

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

                # Lower triangle: 2D histograms
                if fill:
                    sns.histplot(x=data[j], y=data[i], ax=axes[i, j], bins=bins2D, cmap=create_sequential_colormap('white', color), weights=weights, alpha=0.5, fill=False)
                else:
                    hist = sns.histplot(x=data[j], y=data[i], ax=axes[i, j], bins=bins2D, cmap=cmasher.guppy_r, edgecolor='face', weights=weights, alpha=1, fill=True)
                    hist.collections[0].set_norm(colors.LogNorm(vmin=1))


                x_grid, y_grid, density, levels, x_HPD, y_HPD = get_contours(data[j], data[i], weights, bins2D, smooth=0.5)
                # Add custom colour transitions for levels
                original_color = colors.to_rgb(color)  # (0.2549, 0.4118, 0.8824)

                # Mix with white (30%, 60%)
                mix_ratios = [0.15, 0.6]
                strong_white = tuple(np.array(original_color) * mix_ratios[0] + np.array((1, 1, 1)) * (1 - mix_ratios[0]))
                weaker_white = tuple(np.array(original_color) * mix_ratios[1] + np.array((1, 1, 1)) * (1 - mix_ratios[1]))

                # Plot contours
                # axes[i, j].contour(x_grid, y_grid, density, levels=levels, colors='red', linestyles=[ ':','-.', '-'], linewidths=2.5, weights=weights)
                if fill:
                    axes[i, j].contourf(x_grid, y_grid, density, levels=levels, colors=[strong_white, weaker_white, color], alpha=1)
                    axes[i, j].contour(x_grid, y_grid, density, levels=levels, colors=linecolor, linestyles=[ ':','-.', '-'], linewidths=2)
                else:
                    axes[i, j].contour(x_grid, y_grid, density, levels=levels, colors=linecolor, linestyles=[ ':','-.', '-'], linewidths=3)
                # axes[i, j].contour(x_grid, y_grid, density, levels=levels, colors='white', linestyles=[ '-','-', '-'], linewidths=1.5, weights=weights)

                # Plot highest posterior density
                axes[i, j].plot(x_HPD, y_HPD, marker='o', color='white', markersize=4)

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
    # plt.subplots_adjust(hspace=0.04, wspace=0.025)
    plt.subplots_adjust(hspace=0.0, wspace=-0.015)  # No spacing

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
            'levels': [0.6827, 0.9545, 0.9973],  # 1σ, 2σ, 3σ
            'smooth': 0.3,
            'show_titles': True,
            'title_kwargs': {'fontsize': 12},
            
            # Custom contour appearance (line color and fill)
            'contour_kwargs': {
                'colors': 'red',   # Set contour lines to white
                'linewidths': 1,   # Set the width of contour lines
                'linestyles': ['dotted', 'dashdot', 'solid']
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

def add_external_solar_data(ax):
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

    global1 = np.genfromtxt('inputs/contours/nuFit_1sigma.csv', delimiter=',')
    global1[:, 1] = global1[:, 1]*1e-1

    global2 = np.genfromtxt('inputs/contours/nuFit_2sigma.csv', delimiter=',')
    global2[:, 1] = global2[:, 1]*1e-1

    global3 = np.genfromtxt('inputs/contours/nuFit_3sigma.csv', delimiter=',')
    global3[:, 1] = global3[:, 1]*1e-1

    bestFitGlobal = np.genfromtxt('inputs/contours/bestFitGlobal.csv', delimiter=',')
    bestFitGlobal[1] = bestFitGlobal[1]*1e-1

    contcol = 'darksalmon'
    kamcol = 'purple'
    globcol = 'orange'
    snocol = 'green'

    #kamcol = '#009E73'
    #globcol = '#E69F00'

    #overlay_contours(solar1, ax, color=contcol, lw=2, ls='-', alpha=0.45)  # linewidth=10)
    #overlay_contours(solar2, ax, color=contcol, lw=2, label='Solar', ls='-.', alpha=0.45)
    #overlay_contours(solar3, ax, color=contcol, lw=2, ls=':', alpha=0.45)
    #ax.plot(bestFitSolar[0], bestFitSolar[1], color=contcol, linestyle='', marker='d', markersize=5, alpha=0.45)

    overlay_contours(kamLAND1, ax, color=kamcol, lw=2, ls='-')  # linewidth=10)
    overlay_contours(kamLAND2, ax, color=kamcol, lw=2, label='KamLAND', ls='-.')
    overlay_contours(kamLAND3, ax, color=kamcol, lw=2, ls=':')
    ax.plot(bestFitkamLAND[0], bestFitkamLAND[1], color=kamcol, linestyle='', marker='s', markersize=5)

    overlay_contours(global1, ax, color=globcol, lw=3, ls='-')  # linewidth=10)
    overlay_contours(global2, ax, color=globcol, lw=3, label='Global', ls='-.')
    overlay_contours(global3, ax, color=globcol, lw=3.5, ls=':')
    ax.plot(bestFitGlobal[0], bestFitGlobal[1], color=globcol, linestyle='', marker='P', markersize=6)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_covariance(raw_matrix, colormap='viridis', labels=None, 
                           title='cov matrix', colorbar=True, 
                           figsize=(12, 8), fontsize=16, 
                           vmin=None, vmax=None):

    fig, ax = plt.subplots(figsize=figsize)

    std_devs = np.sqrt(np.diag(raw_matrix))  # Standard deviations (σ_i)
    matrix = raw_matrix / np.outer(std_devs, std_devs)

    row_indices, col_indices = np.diag_indices_from(matrix)
    matrix[row_indices, col_indices] = np.nan

    n = matrix.shape[0]  # Number of bins
    vlim = np.max([np.abs(np.nanmin(matrix)), np.abs(np.nanmax(matrix))])  # cbar limits

    cax = ax.matshow(matrix, cmap=colormap, origin='lower', 
                    extent=[0, n, 0, n], aspect='auto', vmin=-vlim, vmax=vlim)

    ax.grid(True, linestyle='-', linewidth=1, color='white', alpha=0.5)  # Enable grid
    
    # Set ticks at edges (N+1 ticks)
    edge_ticks = np.arange(n + 1)
    ax.set_xticks(edge_ticks)
    ax.set_yticks(edge_ticks)
    
    # Set labels at centers (N labels)
    if labels is not None:
        center_positions = np.arange(n) + 0.5
        ax.set_xticks(center_positions + 0.3, minor=True)
        ax.set_yticks(center_positions, minor=True)
        
        ax.set_xticklabels(labels, minor=True, 
                         fontsize=fontsize, rotation=90, ha='right')
        ax.set_yticklabels(labels, minor=True,
                         fontsize=fontsize)
        
        # Hide major tick labels (keep ticks)
        ax.set_xticklabels([], minor=False)
        ax.set_yticklabels([], minor=False)
    
    # Style adjustments
    ax.tick_params(axis='both', which='major', length=5)  # Show edge ticks
    ax.tick_params(axis='both', which='minor', length=0)  # Hide center ticks

    ax.xaxis.set_ticks_position('bottom')

    ax.set_box_aspect(1)
    
    if colorbar:
        fig.colorbar(cax, ax=ax)
    if title:
        ax.set_title(title, pad=20)
    
    plt.tight_layout()
    return fig, ax

