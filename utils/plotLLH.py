import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
from cmap import Colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
import plotting
import os

plt.rcParams['text.usetex'] = True
hep.style.use("CMS")

inFile = "outputs/both_leptonReco_3MeV_woBG_PDGlike_fast"

th12th13_path = inFile + "_llh_sin2th12_sin2th13.csv"
th12dm21_path = inFile + "_llh_sin2th12_delt2m21.csv"
th13dm21_path = inFile + "_llh_sin2th13_delt2m21.csv"

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

lim_th12, lim_th13, matrix1 = read_llh(th12th13_path)

fig, axs = plt.subplots(1, 3, figsize=(21, 7))


# First plot
im1 = axs[0].imshow(matrix1, cmap=plotting.parula_map, aspect='auto', origin='lower',
                     extent=[lim_th12[0], lim_th12[1], lim_th13[0], lim_th13[1]], vmin=np.max([-1e2, np.min(matrix1)]), vmax=0)

axs[0].plot(0.022, 0.307, 'r+', markersize=10, markeredgewidth=2)
axs[0].set_ylabel(r'$\sin^2\theta_{12}$')
axs[0].set_xlabel(r'$\sin^2\theta_{13}$')
axs[0].set_box_aspect(1)

# Create a colorbar for the first subplot
divider1 = make_axes_locatable(axs[0])
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im1, cax=cax1, label=r'$-\log\mathcal{L}$')
cax1.set_aspect(0.2)


lim_th12, lim_dm21, matrix2 = read_llh(th12dm21_path)



# Second plot
im2 = axs[1].imshow(matrix2.T, cmap=plotting.parula_map, aspect='auto', origin='lower',
                     extent=[lim_th12[0], lim_th12[1], lim_dm21[0]*10000, lim_dm21[1]*10000], vmin=np.max([-1e2, np.min(matrix2)]), vmax=0)

axs[1].plot(0.307, 0.753, 'r+', markersize=10, markeredgewidth=2)
axs[1].set_xlabel(r'$\sin^2\theta_{12}$')
axs[1].set_ylabel(r'$\Delta m^2_{21} (\times 10^{-5}eV^2)$')
axs[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axs[1].ticklabel_format(axis='x', style='plain')
axs[1].xaxis.offsetText.set_visible(False)
axs[1].set_box_aspect(1)

# Create a colorbar for the second subplot
divider2 = make_axes_locatable(axs[1])
cax2 = divider2.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im2, cax=cax2, label=r'$-\log\mathcal{L}$')
cax2.set_aspect(0.2)

# Read the likelihood data for sin2_th13 vs dm2_21
lim_th13, lim_dm21, matrix3 = read_llh(th13dm21_path)

# Third plot
im3 = axs[2].imshow(matrix3.T, cmap=plotting.parula_map, aspect='auto', origin='lower',
                    extent=[lim_th13[0], lim_th13[1], lim_dm21[0]*10000, lim_dm21[1]*10000], 
                    vmin=np.max([-1e2, np.min(matrix3)]), vmax=0)

axs[2].plot(0.022, 0.753, 'r+', markersize=10, markeredgewidth=2)
axs[2].set_xlabel(r'$\sin^2\theta_{13}$')
axs[2].set_ylabel(r'$\Delta m^2_{21} (\times 10^{-5}eV^2)$')
axs[2].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axs[2].ticklabel_format(axis='x', style='plain')
axs[2].xaxis.offsetText.set_visible(False)
axs[2].set_box_aspect(1)

# Create a colorbar for the third subplot
divider3 = make_axes_locatable(axs[2])
cax3 = divider3.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im3, cax=cax3, label=r'$-\log\mathcal{L}$')
cax3.set_aspect(0.2)

# Adjust layout and save the figure
plt.tight_layout()
fig.savefig('images/' + os.path.basename(inFile) + "_llh.png")
plt.show()

