import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import plotting  # applies shared CMS/LaTeX style on import
import os
import argparse

parser = argparse.ArgumentParser(description="Display LLH scan and optionally specify an output file.")

parser.add_argument('input', help="Input LLH scan file name.")

parser.add_argument('-o', '--output', type=str, help="Output file (optional).")

args = parser.parse_args()

inFile = args.input
outFile = args.output


def read_llh(file_path):
    with open(file_path, 'r') as file:
        # Read the first two lines for headers
        header_lines = [next(file).strip() for _ in range(2)]

    # Extract the limits from the headers
    lim_dim1 = [float(x) for x in header_lines[0].split(': ')[1].split(', ')]
    lim_dim2 = [float(x) for x in header_lines[1].split(': ')[1].split(', ')]

    # Read the matrix data using pandas, skipping the first two header lines
    matrix_data = pd.read_csv(file_path, skiprows=2, header=None)

    # Convert the DataFrame to a NumPy array if needed
    matrix = matrix_data.to_numpy()
    return lim_dim1, lim_dim2, matrix


# Injected (Asimov true) parameter values, marked with the red cross/lines.
# dm2_21 is given in the plotted (x10^-4 eV^2) units.
DM21_SCALE = 1e4
TRUTH = {
    'sin2_th12': 0.307,
    'sin2_th13': 0.022,
    'dm2_21':    0.75,
}
LABELS = {
    'sin2_th12': r'$\sin^2\theta_{12}$',
    'sin2_th13': r'$\sin^2\theta_{13}$',
    'dm2_21':    r'$\Delta m^2_{21} (\times 10^{-4}eV^2)$',
}

# (csv suffix, x-axis key, y-axis key, contour linewidth)
PANELS = [
    ('_llh_sin2th12_sin2th13.csv', 'sin2_th12', 'sin2_th13', 2.0),
    ('_llh_sin2th12_delt2m21.csv', 'sin2_th12', 'dm2_21',    2.5),
    ('_llh_sin2th13_delt2m21.csv', 'sin2_th13', 'dm2_21',    2.0),
]


def plot_llh_panel(ax, fig, path, xkey, ykey, linewidth):
    """Render one 2D -logL scan panel onto ``ax``."""
    lim_x, lim_y, matrix = read_llh(path)

    # The dm2_21 axis is plotted in (x10^-4 eV^2) units.
    scale_y = ykey == 'dm2_21'
    if scale_y:
        lim_y = [lim_y[0] * DM21_SCALE, lim_y[1] * DM21_SCALE]

    im = ax.imshow(matrix.T, cmap=plotting.parula_map, aspect='auto', origin='lower',
                   extent=[lim_x[0], lim_x[1], lim_y[0], lim_y[1]])

    X = np.linspace(lim_x[0], lim_x[1], matrix.shape[1])
    Y = np.linspace(lim_y[0], lim_y[1], matrix.shape[0])
    XX, YY = np.meshgrid(X, Y)
    ax.contour(XX, YY, matrix.T - np.max(matrix),
               levels=[-9, -4, -1], colors='white',
               linewidths=linewidth, linestyles='-')

    truth_x, truth_y = TRUTH[xkey], TRUTH[ykey]
    ax.plot(truth_x, truth_y, 'r+', markersize=10, markeredgewidth=2)
    ax.axhline(y=truth_y, color='r', linestyle='dashed', linewidth=1)
    ax.axvline(x=truth_x, color='r', linestyle='dashed', linewidth=1)
    ax.set_xlabel(LABELS[xkey])
    ax.set_ylabel(LABELS[ykey])
    ax.set_box_aspect(1)

    # Panels against the dm2_21 axis use a plain (non-offset) x-tick format.
    if scale_y:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='x', style='plain')
        ax.xaxis.offsetText.set_visible(False)

    fig.colorbar(im, ax=ax, label=r'$-\log\mathcal{L}$', fraction=0.05, pad=0.1)
    return im


fig, axs = plt.subplots(1, 3, figsize=(20, 5))
for ax, (suffix, xkey, ykey, linewidth) in zip(axs, PANELS):
    plot_llh_panel(ax, fig, inFile + suffix, xkey, ykey, linewidth)

plt.tight_layout()

# Adjust layout and save the figure
if outFile:
    plotting.save_figure(fig, outFile + "_llh.pdf", dpi=300, format='pdf')

plotting.save_figure(fig, 'images/' + os.path.basename(inFile) + "_llh.pdf", dpi=300, format='pdf')

plt.show()
