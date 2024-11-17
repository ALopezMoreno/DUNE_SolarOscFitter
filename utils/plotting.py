import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
from matplotlib.ticker import ScalarFormatter


plt.rcParams['text.usetex'] = True
hep.style.use("CMS")

def plot_corner(variables, data):
    """
    Create a corner plot of 2D histograms for the given variables.

    :param variables: List of variable names.
    :param data: List of numpy arrays corresponding to each variable.
    """
    num_vars = len(variables)
    fig, axes = plt.subplots(num_vars, num_vars, figsize=(5 * num_vars, 5 * num_vars))

    for i in range(num_vars):
        for j in range(num_vars):
            if i == j:
                # Diagonal plots can be used for 1D histograms or left empty
                axes[i, j].hist(data[i], bins=30, color='gray', alpha=0.7)
                axes[i, j].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axes[i, j].xaxis.get_major_formatter().set_scientific(True)
                axes[i, j].xaxis.get_major_formatter().set_powerlimits((-2, 2))
                axes[i, j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axes[i, j].yaxis.get_major_formatter().set_scientific(True)
                axes[i, j].yaxis.get_major_formatter().set_powerlimits((-2, 2))
                
                if i == num_vars - 1:
                    axes[i, j].set_xlabel(variables[i])
                    axes[i, j].xaxis.set_label_coords(0.65, -0.15)  # Center the x-label
                else:
                    axes[i, j].set_xticklabels([])
                if i == 0:
                    axes[i, j].set_ylabel(r'$dP$')
                    axes[i, j].yaxis.set_label_coords(-0.15, 0.65)  # Center the y-label
                axes[i, j].set_yticklabels([])

            elif i > j:
                # Lower triangle: 2D histograms
                sns.histplot(x=data[j], y=data[i], ax=axes[i, j], bins=30, cmap='Blues')
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
                    axes[i, j].xaxis.set_label_coords(0.65, -0.15)  # Center the x-label
                else:
                    axes[i, j].set_xticklabels([])
                if j == 0:
                    axes[i, j].set_ylabel(variables[i])
                    axes[i, j].yaxis.set_label_coords(-0.15, 0.65)  # Center the y-label
                else:
                    axes[i, j].set_yticklabels([])

                

    # Minimize space between plots
    #plt.subplots_adjust(hspace=0.05, wspace=0.05)
    return fig, axes