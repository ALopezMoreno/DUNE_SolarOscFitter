import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
import networkx as nx
import mplhep as hep
import time
import sys
import plotting
import posteriorHelpers
from termcolor import colored


parser = argparse.ArgumentParser(description="Process MCMC chains and optionally specify an output file.")

parser.add_argument('chains', nargs='+', help="Input MCMC chain files.")

parser.add_argument('-o', '--output', type=str, help="Output file (optional).")
parser.add_argument('-d', '--diagnostics', action='store_true',
                    help="Diagnostics mode: produces both a bootstrap contour plot (freeze cmap by default, override with --cbar) and a stability heatmap (RdYlGn: green=stable, red=needs more stats).")
parser.add_argument('-e', '--expanded', action="store_true", help="Make additional corner plots including other parameters of interest: HEP flux (discovery), 8B flux, DN_asymmetry")
parser.add_argument('-f', '--full_output', action="store_true", help="Make additional corner plot over **all** parameters (default corner.py plot).")
parser.add_argument('-p', '--parameters', nargs='+', help="Choose parameters to be plotted (default corner.py plot).")
parser.add_argument('-b', '--burnin', type=int, default=5_000, help="Burnin steps to discard (default: 20000).")
parser.add_argument('--bins', type=int, default=60, help="Number of 2D histogram bins per axis (default: 60).")
parser.add_argument('--exclude', nargs='*', type=int, default=[20],
                    help="Chain IDs to exclude (default: 20). Use --exclude with no args to keep all chains.")
parser.add_argument('--cbar', type=str, default=None,
                    choices=['mako', 'mako_r', 'viridis', 'viridis_r', 'cividis', 'cividis_r',
                             'blues', 'blues_r', 'parula', 'parula_r',
                             'inferno', 'inferno_r', 'plasma', 'plasma_r',
                             'magma', 'magma_r', 'rocket', 'rocket_r',
                             'cosmic', 'cosmic_r', 'torch', 'torch_r',
                             'freeze', 'freeze_r', 'ember', 'ember_r'],
                    help="Replace contours with a solid colormap (bright = higher posterior density).")

args = parser.parse_args()

mcmc_chains = args.chains
output_name = args.output
diagnostics = args.diagnostics
expanded = args.expanded
full = args.full_output
customParameters = args.parameters
burnin = args.burnin
cbar = args.cbar
bins = args.bins
exclude = args.exclude if args.exclude is not None else []

# Check if at least one file is provided
if not mcmc_chains:
    print("Error: No MCMC output files provided.")
    sys.exit(1)


#####################################################################
################ ----- Begin default execution ----- ################
#####################################################################

start_time = time.time()

parameters = ['sin2_th12', 'sin2_th13', 'dm2_21']
variables = [r'$\sin^2\theta_{12}$', r'$\sin^2\theta_{13}$', r'$\Delta m^2_{21}$ $(10^{-4} \mathrm{ eV}^2)$']

data = posteriorHelpers.load_posterior(mcmc_chains, parameters, burnin=burnin, exclude_chains=exclude)

##### Some useful colours #####
#----------------------------------------------#
#    -- Muted dark blue/turquoise: '#006c94'   #
#    -- T2K-NOvA blue: '#0894bc'               #
#    -- ChatGPT bluish green: '#009E73'        #
#    -- ChatGPT blue: '#332288'                #
#----------------------------------------------#


# Define parameter sets based on diagnostics flag
plot_params = {
    'diagnostics': {
        'color': ['dimgray']*3,
        'linecolor': ['w']*3,
        'fill': False,
    },
    'default': {
        'color': ['#0894bc', '#9a1b9a' , '#a41007'],
        'linecolor': ['#08bc8a', '#5b1b9a', '#a45f07'],
        'fill': True,
    }
}

def _fix_axes_diag(axes, data, variables):
    """Set axes limits to data range (diagnostics mode)."""
    lim0 = [np.min(data['sin2_th12']), np.max(data['sin2_th12'])]
    lim2 = [np.min(data['dm2_21'])*1e4, np.max(data['dm2_21']*1e4)]
    for k in range(len(variables)):
        axes[k, 0].set_xlim(lim0[0], lim0[1])
        axes[k, 2].set_xlim(lim2[0], lim2[1])
        if k != 2:
            axes[2, k].set_ylim(lim2[0], lim2[1])

def _fix_axes_default(axes, variables):
    """Set axes limits for display of global constraints."""
    for k in range(len(variables)):
        axes[k, 0].set_xlim(0.23, 0.43)
        axes[k, 2].set_xlim(0.2, 1.4)
        if k != 2:
            axes[2, k].set_ylim(0.2, 1.4)

def _save(fig, tag, output_name):
    path = f'images/corner_output{tag}.pdf' if output_name is None else f'{output_name}{tag}.pdf'
    print(f"saving output as {path}")
    fig.savefig(path, dpi=300, format='pdf', bbox_inches='tight')
    plt.close(fig)

_diag_params   = plot_params['diagnostics']
_normal_params = plot_params['default']

if diagnostics:
    # -d mode: produce both a bootstrap and a stability plot
    _boot_cmap = cbar if cbar else 'freeze_r'   # --cbar overrides bootstrap default

    # ── Bootstrap plot ──────────────────────────────────────────────
    fig, axes, l = plotting.plot_corner(
        variables, data, externalContours=True,
        colorlist=_diag_params['color'], linecolors=_diag_params['linecolor'],
        fill=_diag_params['fill'], cbar=_boot_cmap, diagnostics='bootstrap',
        bins2D=bins,
    )
    _fix_axes_diag(axes, data, variables)
    _save(fig, '_bootstrap', output_name)

    # ── Stability plot ───────────────────────────────────────────────
    fig, axes, l = plotting.plot_corner(
        variables, data, externalContours=True,
        colorlist=_diag_params['color'], linecolors=_diag_params['linecolor'],
        fill=_diag_params['fill'], cbar=None, diagnostics='stab',
        bins2D=bins,
    )
    _fix_axes_diag(axes, data, variables)
    _save(fig, '_stab', output_name)

else:
    # Normal mode
    fig, axes, l = plotting.plot_corner(
        variables, data, externalContours=True,
        colorlist=_normal_params['color'], linecolors=_normal_params['linecolor'],
        fill=_normal_params['fill'], cbar=cbar, diagnostics=None,
        bins2D=bins,
    )
    _fix_axes_default(axes, variables)
    _save(fig, '', output_name)

end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")

###################################################################
################ ----- Begin execution cases ----- ################
###################################################################

if customParameters is not None:
    print(f"Generating corner plot with custom parameters: {customParameters}")
    param_str = "_".join(customParameters)

    start_time = time.time()

    data = posteriorHelpers.load_posterior(mcmc_chains, customParameters, burnin=burnin, exclude_chains=exclude)

    # Create the corner plot
    fig = plotting.plot_default_corner(data, diagnostics=diagnostics)


    # Save figs
    if output_name is None:
        print(f"saving output as images/corner_output_{customParameters}.pdf")
        fig.savefig(f"images/corner_output_{param_str}.pdf", dpi=300, format='pdf', bbox_inches='tight')
    else:
        print("saving output as " + output_name + f"_{param_str}.pdf")
        fig.savefig(output_name + f"_{param_str}.pdf", dpi=300, format='pdf', bbox_inches='tight')

    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")


if full:
    print(f"Generating corner plot with all parameters")
    start_time = time.time()

    data = posteriorHelpers.load_posterior(mcmc_chains, 'all', burnin=burnin, exclude_chains=exclude)

    chain_indexes = np.unique(data["chains"])
    print("Plotting traces")
    for idx in chain_indexes:
        mask = (data["chains"] == idx)
        print(f"Chain {idx}: {len(data['stepno'][mask])} steps")

    _trace_labels = {
        'sin2_th12':           r'$\sin^2\theta_{12}$',
        'sin2_th13':           r'$\sin^2\theta_{13}$',
        'dm2_21':              r'$\Delta m^2_{21}$',
        'integrated_8B_flux':  r'$\Phi_\odot^\mathrm{8B}$',
        'integrated_HEP_flux': r'$\Phi_\odot^\mathrm{hep}$',
    }
    _colors_tr = plt.get_cmap('tab20').colors

    # Build legend handles once, shared across both pages
    _leg_handles = [
        plt.Line2D([0], [0], color=_colors_tr[_ci % len(_colors_tr)], lw=1.5, label=str(idx))
        for _ci, idx in enumerate(chain_indexes)
    ]

    def _draw_traces(ax, key):
        for _ci, idx in enumerate(chain_indexes):
            mask = (data["chains"] == idx)
            ax.plot(data["stepno"][mask] / 1e3, data[key][mask],
                    color=_colors_tr[_ci % len(_colors_tr)], alpha=0.7, lw=0.8)
        ax.set_box_aspect(1)
        ax.set_xlabel(r"Step $(\times 10^3)$", fontsize=14)
        ax.set_ylabel(_trace_labels.get(key, key), fontsize=14)
        ax.set_title(_trace_labels.get(key, key), fontsize=15)
        ax.tick_params(labelsize=13)

    _trace_path = (output_name or 'images/corner_output') + '_traces.pdf'
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(_trace_path) as _pdf:

        # Page 1: osc parameters in 2×2, legend in bottom-right cell
        _osc_keys = [k for k in ['sin2_th12', 'sin2_th13', 'dm2_21']
                     if k in data and data[k].size > 0]
        if _osc_keys:
            _fig_tr, _axes_tr = plt.subplots(2, 2, figsize=(12, 12))
            for _axi, key in enumerate(_osc_keys):
                _draw_traces(_axes_tr[_axi // 2, _axi % 2], key)
            _axes_tr[1, 1].axis('off')
            _axes_tr[1, 1].legend(handles=_leg_handles, title="Chain ID",
                                  fontsize=7, title_fontsize=8, loc='center', ncol=2)
            _fig_tr.suptitle("Oscillation parameter traces", fontsize=10)
            _fig_tr.tight_layout()
            _pdf.savefig(_fig_tr, bbox_inches='tight')
            plt.close(_fig_tr)

        # Page 2: flux parameters in 1×3, legend in rightmost cell
        _flux_keys = [k for k in ['integrated_8B_flux', 'integrated_HEP_flux']
                      if k in data and data[k].size > 0]
        if _flux_keys:
            _fig_tr, _axes_tr = plt.subplots(1, 3, figsize=(18, 6))
            for _axi, key in enumerate(_flux_keys):
                _draw_traces(_axes_tr[_axi], key)
            _axes_tr[2].axis('off')
            _axes_tr[2].legend(handles=_leg_handles, title="Chain ID",
                               fontsize=7, title_fontsize=8, loc='center', ncol=2)
            _fig_tr.suptitle("Solar flux traces", fontsize=10)
            _fig_tr.tight_layout()
            _pdf.savefig(_fig_tr, bbox_inches='tight')
            plt.close(_fig_tr)

    print(f"saving traces as {_trace_path}")

    # Create the corner plot
    fig = plotting.plot_default_corner(data, diagnostics=diagnostics)

    # Save figs
    if output_name is None:
        print("saving output as images/corner_output_all.pdf")
        fig.savefig('images/corner_output_all.pdf', dpi=300, format='pdf', bbox_inches='tight')
    else:
        print("saving output as " + output_name + "_all.pdf")
        fig.savefig(output_name + '_all.pdf', dpi=300, format='pdf', bbox_inches='tight')

    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")


if expanded:
    print("Generating expanded corner plot.")
    start_time = time.time()

    parameters = ['sin2_th12', 'sin2_th13', 'dm2_21', 'integrated_8B_flux', 'integrated_HEP_flux']
    variables = [r'$\sin^2\theta_{12}$', r'$\sin^2\theta_{13}$', r'$\Delta m^2_{21}$ $(10^{-4} \, \mathrm{ eV}^2)$', r'$\Phi_\odot^\mathrm{8B}$', r'$\Phi_\odot^\mathrm{hep}$']

    test = [("derived_CC_asymmetry", r'$A_{D-N}(CC)$')]

    data = posteriorHelpers.load_posterior(mcmc_chains, parameters, burnin=burnin, test=test, exclude_chains=exclude)

    for v in test:
        if v[0] in data:
            variables.append(v[1])

    fig, axes, l = plotting.plot_corner(
        variables,
        data,
        externalContours=True,
        colorlist=_normal_params['color'],
        linecolors=_normal_params['linecolor'],
        fill=_normal_params['fill'],
        cbar=cbar,
        diagnostics=None,
        bins2D=bins,
    )

    # get axis limits
    i = parameters.index('sin2_th12')
    j = parameters.index('dm2_21')
    k = parameters.index('integrated_HEP_flux')

    lim0 = axes[i, i].get_xlim()
    lim2 = axes[j, j].get_xlim()
    lim4 = axes[k, k].get_xlim()

    if diagnostics:
        # Fix axes limits to data
        for k, par in enumerate(variables):
            axes[k, 0].set_xlim(lim0[0], lim0[1])
            axes[k, 2].set_xlim(lim2[0], lim2[1])
            if k != 2:
                axes[2, k].set_ylim(lim2[0], lim2[1])


    else:
        # Fix axes limits to display global constraints nicely
        for k, par in enumerate(axes):
            axes[k, 0].set_xlim(0.23, 0.38)
            axes[k, 0].set_xlim(0.15, 0.45)
            axes[k, 2].set_xlim(0.26, 1)
            if len(axes) > 5   :
                axes[k, 5].set_xlim(-0.12, 0.02)

            if k < 4:
                axes[4, k].axhline(0, color='red', linewidth=2, linestyle='dashed')
            else:
                axes[k, 4].axvline(0, color='red', linewidth=2, linestyle='dashed')
            if k < 5 and len(axes) > 5:
                axes[5, k].axhline(0, color='gray', linewidth=2, linestyle='dashed')   
            elif len(axes) > 5:
                axes[k, 5].axvline(0, color='gray', linewidth=2, linestyle='dashed')


            if lim4[0] >= 0:
                axes[k, 4].set_xlim(-lim4[0]/2, lim4[1])
            if k != 2:
                axes[2, k].set_ylim(0.26, 1)
            if k != 4:
                axes[4, k].set_ylim(-lim4[0]/2, lim4[1])
            if k !=5 and len(axes) > 5:
                axes[5, k].set_ylim(-0.12, 0.02)

        


        # Save figs
    if output_name is None:
        print(f"saving output as images/corner_output_expanded.pdf")
        fig.savefig(f"images/corner_output_expanded.pdf", dpi=300, format='pdf', bbox_inches='tight')
    else:
        print("saving output as " + output_name + "_expanded.pdf")
        fig.savefig(output_name + "_expanded.pdf", dpi=300, format='pdf', bbox_inches='tight')

    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")


exit()
import h5py
#
# posterior = np.vstack((sin2_th12_all, sin2_th13_all, dm2_21_all, stepno_all, B8flux_all))
# 
# 
covariance = np.cov(data)
print(data.keys())
exit()
# 
# Save the covariance matrix to a JLD2 file
with h5py.File('outputs/posterior_covariance.jld2', 'w') as f:
    f.create_dataset('posterior_cov', data=covariance)
# 
# 
# data = [sin2_th12_all, sin2_th13_all, dm2_21_all * 1e4]



if diagnostics:
    print(colored("Running in diagnostics mode. 2D hists will use a logarithmic colormap", "yellow"))

    # If running in diagnostics, show full histogram so that bin population is visible
    fig, axes, l = plotting.plot_corner(variables, data, weights=weights_all, externalContours=True,
                                        color='dimgray', linecolor='w', fill=False)

    # Fix axes limits to data
    lim0 = [np.min(data[0]), np.max(data[0])]
    lim1 = [np.min(data[1]), np.max(data[1])]
    lim2 = [np.min(data[2]), np.max(data[2])]

    axes[0, 0].set_xlim(lim0[0], lim0[1])
    axes[1, 0].set_xlim(lim0[0], lim0[1])
    axes[2, 0].set_xlim(lim0[0], lim0[1])

    axes[2, 0].set_ylim(lim2[0], lim2[1])
    axes[2, 1].set_ylim(lim2[0], lim2[1])
    axes[2, 2].set_xlim(lim2[0], lim2[1])




if diagnostics:
    poisson_noise = [np.sqrt(lev)/lev for lev in l[::-1]]

    for i, level in enumerate(poisson_noise):
        frac_uncertainty = level * 100  # Convert to percentage

        # Create the base message
        message = f"Fractional poissonian error at {i+1}σ is {frac_uncertainty:.1f}%"

        # Add conditional message
        if frac_uncertainty < (i+1)*5:
            full_message = message + ". Statistics are sufficient to draw this contour."
            print(colored(full_message, 'green'))
        else:
            full_message = message + ". Not enough statistics to draw this contour. Try running larger chains."
            print(colored(full_message, 'red'))

exit()



print("Plotting traces")
mask = (chains_all == chain_indexes[0])
for field in data:
    plt.figure()
    # Loop over every unique chain index
    for idx in chain_indexes:
         mask = (chains_all == idx)
         print(f"Chain {idx}: {len(stepno_all[mask])} steps")
         plt.plot(stepno_all[mask], field[mask], label=f'Chain {idx}')
    plt.xlabel("Step Number")
    plt.ylabel("Field Value")
    plt.title("Chain Plot for Field")
    plt.legend()
    plt.show()

for field in data:
    # Calculate the difference between contiguous elements
    differences = np.diff(field[mask])
    
    # Count how many times the difference is zero
    zero_diff_count = np.sum(differences == 0)
    print(f"Number of zero differences: {zero_diff_count}")
    
    # Plot the differences
    plt.plot(stepno_all[mask][1:], differences)  # stepno_all[mask][1:] to align with differences length
    plt.title("Differences between contiguous elements")
    plt.xlabel("Step Number")
    plt.ylabel("Difference")
    plt.show()