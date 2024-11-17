import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
from cmap import Colormap
from matplotlib.ticker import ScalarFormatter


plt.rcParams['text.usetex'] = True
hep.style.use("CMS")

th12th13_path = "outputs/llh_sin2th12_sin2th13.csv"
th12dm21_path = "outputs/llh_sin2th12_delt2m21.csv"

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

lim_th12, lim_th13, matrix = read_llh(th12th13_path)

fig,ax = plt.subplots(figsize = (8, 5))


# Plot the matrix
cax = ax.imshow(matrix, cmap='viridis', aspect='auto', origin='lower', 
                extent=[lim_th13[0], lim_th13[1], lim_th12[0], lim_th12[1]], vmax=0, vmin=-100)

fig.colorbar(cax, ax=ax, label='Log likelihood')  # Add a colorbar to show the scale
ax.plot(0.022, 0.307, 'r+', markersize=10, markeredgewidth=2)

ax.set_ylabel(r'$\sin^2\theta_{12}$')
ax.set_xlabel(r'$\sin^2\theta_{13}$')
ax.set_title("llh_scan")
ax.set_box_aspect(1)
plt.tight_layout()
plt.show()


lim_th12, lim_dm21, matrix = read_llh(th12dm21_path)

fig,ax = plt.subplots(figsize = (8, 5))

# Plot the matrix
cax = ax.imshow(matrix, cmap='viridis', aspect='auto', origin='lower', 
                extent=[lim_dm21[0]*10000, lim_dm21[1]*10000, lim_th12[0], lim_th12[1]], vmax=0, vmin=-100)

fig.colorbar(cax, ax=ax, label='Log likelihood')

# Add a red cross at the specified point
ax.plot(0.753, 0.307, 'r+', markersize=10, markeredgewidth=2)

# Set axis labels
ax.set_ylabel(r'$\sin^2\theta_{12}$')
ax.set_xlabel(r'$\Delta m^2_{21} (\times 10^{-5}eV^2)$')

# Format the x-axis to show order 1 values
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='x', style='plain')
ax.xaxis.offsetText.set_visible(False)

# Add a label for the scale
#ax.annotate(r'$\times 10^{-5}$', xy=(1.5, -0.3), xycoords='axes fraction',
#            xytext=(-10, 10), textcoords='offset points', ha='right', va='bottom')

ax.set_title("llh_scan")
ax.set_box_aspect(1)
plt.tight_layout()
plt.show()