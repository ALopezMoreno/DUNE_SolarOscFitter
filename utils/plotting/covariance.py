"""Posterior covariance-matrix heatmap."""
import numpy as np
import matplotlib.pyplot as plt


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
