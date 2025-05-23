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
parser.add_argument('-d', '--diagnostics', action="store_true", help="Run in plot diagnostics mode.")
parser.add_argument('-e', '--expanded', action="store_true", help="Make additional corner plots including other parameters of interest: HEP flux (discovery), 8B flux, DN_asymmetry")
parser.add_argument('-f', '--full_output', action="store_true", help="Make additional corner plot over **all** parameters (default corner.py plot).")
parser.add_argument('-p', '--parameters', nargs='+', help="Choose parameters to be plotted (default corner.py plot).")

args = parser.parse_args()

mcmc_chains = args.chains 
output_name = args.output   
diagnostics = args.diagnostics
expanded = args.expanded
full = args.full_output
customParameters = args.parameters

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

data = posteriorHelpers.load_posterior(mcmc_chains, parameters)

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

# Select which parameters to use
params = plot_params['diagnostics'] if diagnostics else plot_params['default']


fig, axes, l = plotting.plot_corner(
    variables, 
    data, 
    externalContours=True,
    colorlist=params['color'],
    linecolors=params['linecolor'],
    fill=params['fill']
)

if diagnostics:
    # Fix axes limits to data
    lim0 = [np.min(data['sin2_th12']), np.max(data['sin2_th12'])]
    lim2 = [np.min(data['dm2_21'])*1e4, np.max(data['dm2_21']*1e4)]

    for k, par in enumerate(variables):
        axes[k, 0].set_xlim(lim0[0], lim0[1])
        axes[k, 2].set_xlim(lim2[0], lim2[1])
        if k != 2:
            axes[2, k].set_ylim(lim2[0], lim2[1])

else:
    # Fix axes limits to display global constraints nicely
    for k, par in enumerate(variables):
        axes[k, 0].set_xlim(0.23, 0.38)
        axes[k, 2].set_xlim(0.26, 1)
        if k != 2:
            axes[2, k].set_ylim(0.26, 1)


# Save figs
if output_name is None:
    print("saving output as images/corner_output.pdf")
    fig.savefig('images/corner_output.pdf', dpi=300, format='pdf')
else:
    print("saving output as " + output_name + ".pdf")
    fig.savefig(output_name + '.pdf', dpi=300, format='pdf')

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

    data = posteriorHelpers.load_posterior(mcmc_chains, customParameters)

    # Create the corner plot
    fig = plotting.plot_default_corner(data, diagnostics=diagnostics)


    # Save figs
    if output_name is None:
        print(f"saving output as images/corner_output_{customParameters}.pdf")
        fig.savefig(f"images/corner_output_{param_str}.pdf", dpi=300, format='pdf')
    else:
        print("saving output as " + output_name + f"_{param_str}.pdf")
        fig.savefig(output_name + f"_{param_str}.pdf", dpi=300, format='pdf')

    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")


if full:
    print(f"Generating corner plot with all parameters")
    start_time = time.time()

    data = posteriorHelpers.load_posterior(mcmc_chains, 'all')

    chain_indexes = np.unique(data["chains"])
    print("Plotting traces")
    mask = (data["chains"] == 0)
    for key in ["dm2_21"]:
        field = data[key]
        plt.figure()
        # Loop over every unique chain index
        for idx in chain_indexes:
            mask = (data["chains"] == idx)
            print(f"Chain {idx}: {len(data['stepno'][mask])} steps")
            plt.plot(data["stepno"][mask], field[mask], label=f'Chain {idx}', alpha=0.5)
        plt.xlabel("Step Number")
        plt.ylabel("Field Value")
        plt.title("Chain Plot for Field")
        plt.legend()
        plt.show()

    # Create the corner plot
    fig = plotting.plot_default_corner(data, diagnostics=diagnostics)

    # Save figs
    if output_name is None:
        print("saving output as images/corner_output_all.pdf")
        fig.savefig('images/corner_output.pdf', dpi=300, format='pdf')
    else:
        print("saving output as " + output_name + "_all.pdf")
        fig.savefig(output_name + '_all.pdf', dpi=300, format='pdf')

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

    data = posteriorHelpers.load_posterior(mcmc_chains, parameters, test=test)

    for v in test:
        if v[0] in data:
            variables.append(v[1])

    fig, axes, l = plotting.plot_corner(
        variables, 
        data, 
        externalContours=True,
        colorlist=params['color'],
        linecolors=params['linecolor'],
        fill=params['fill']
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
        fig.savefig(f"images/corner_output_expanded.pdf", dpi=300, format='pdf')
    else:
        print("saving output as " + output_name + "_expanded.pdf")
        fig.savefig(output_name + "_expanded.pdf", dpi=300, format='pdf')

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