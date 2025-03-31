import numpy as np
import matplotlib.pyplot as plt
import h5py
import networkx as nx
import mplhep as hep
import time
import sys
import plotting
import os
import subprocess

def convert_mcmc_to_jld2(bin_file, info_file, out_jld2):
    result = subprocess.run(
    ["julia", "./utils/convert_mcmc_to_jld2.jl", bin_file, info_file, out_jld2],
    check=False,
    capture_output=False,
    text=False)

    print(result.stdout)
    # If there's an error message sent to stderr, print that as well
    if result.stderr:
        print("Julia stderr:", result.stderr)
        return -1
    else:
        return 0



mcmc_chain =  sys.argv[1] #"ES_testReco_3MeV_woBG_PDGlike_fast_smallTheta13_mcmc.jld2"

mcmc_chains = sys.argv[1:]

# Check if at least one file is provided
if not mcmc_chains:
    print("Error: No MCMC output files provided.")
    sys.exit(1)

# Initialize lists to store concatenated data
sin2_th12_all = []
sin2_th13_all = []
dm2_21_all = []
stepno_all = []
B8flux_all = []
weights_all = []
chains_all = []

start_time = time.time()

# Iterate over each MCMC chain file
for mcmc_chain in mcmc_chains:
    if not os.path.exists(mcmc_chain+".jld2"):
        print(f"The file '{mcmc_chain}.jld2' does not exist. Looking for binaries")
        
        if not os.path.exists(mcmc_chain+"_mcmc.bin"):
            print(f"Error: The mcmc file '{mcmc_chain}_mcmc.bin' does not exist.")

        if not os.path.exists(mcmc_chain+"_info.txt"):
            print(f"Error: The info file '{mcmc_chain}_info.txt' does not exist.")  
        
        convert_mcmc_to_jld2(mcmc_chain+"_mcmc.bin", mcmc_chain+"_info.txt", mcmc_chain+".jld2")

    with h5py.File(mcmc_chain+".jld2", 'r') as f:
        sin2_th12 = np.array(f['sin2_th12'][()])
        sin2_th13 = np.array(f['sin2_th13'][()])
        dm2_21 = np.array(f['dm2_21'][()])
        stepno = np.array(f['stepno'][()])
        B8flux = np.array(f['integrated_8B_flux'])
        weights = np.array(f['weights'])
        chains = np.array(f['chainid'])
        chain_indexes = np.unique(chains)
        print(f"Unique chain IDs: {chain_indexes}")
        
        print(f'number of steps in raw posterior: {len(dm2_21)}')
        
        # Create a boolean mask for burnin
        mask = (stepno > 4000)
        
        # Apply the mask to filter the data
        sin2_th12_burnin = sin2_th12[mask]
        sin2_th13_burnin = sin2_th13[mask]
        dm2_21_burnin = dm2_21[mask]
        stepno_burnin = stepno[mask]
        B8flux_burnin = B8flux[mask]
        weights_burnin = weights[mask]
        chains_burnin = chains[mask]
        
        """
        # Apply the mask to filter the data
        sin2_th12_burnin = sin2_th12[mask]
        sin2_th13_burnin = sin2_th13[mask]
        dm2_21_burnin = dm2_21[mask]
        stepno_burnin = stepno[mask]
        B8flux_burnin = B8flux[mask]
        chains_burnin = chains[mask]
        """

        # Append the filtered data to the lists
        sin2_th12_all.append(sin2_th12_burnin)
        sin2_th13_all.append(sin2_th13_burnin)
        dm2_21_all.append(dm2_21_burnin)
        stepno_all.append(stepno_burnin)
        weights_all.append(weights_burnin)
        B8flux_all.append(B8flux_burnin)
        chains_all.append(chains_burnin)


# Concatenate all data
sin2_th12_all = np.concatenate(sin2_th12_all)
sin2_th13_all = np.concatenate(sin2_th13_all)
dm2_21_all = np.concatenate(dm2_21_all)
stepno_all = np.concatenate(stepno_all)
weights_all = np.concatenate(weights_all)
B8flux_all = np.concatenate(B8flux_all)
chains_all = np.concatenate(chains_all)

print(len(dm2_21_all), len(weights_all))

print("number of steps in posterior: " + str(np.sum(weights_all)))
print('generating covariance matrix')

import h5py

posterior = np.vstack((sin2_th12_all, sin2_th13_all, dm2_21_all, stepno_all, B8flux_all))
covariance = np.cov(posterior)

# Save the covariance matrix to a JLD2 file
with h5py.File('outputs/posterior_covariance.jld2', 'w') as f:
    f.create_dataset('posterior_cov', data=covariance)


variables = [r'$\sin^2\theta_{12}$', r'$\sin^2\theta_{13}$', r'$\Delta m^2_{21}$']
data = [sin2_th12_all, sin2_th13_all, dm2_21_all]

# Assuming plotting.plot_corner is a function from an internal library
fig, axes = plotting.plot_corner(variables, data, weights=weights_all, externalContours=True, color='#006c94')

# This changes the limits to mimic Sergio's plot
axes[0, 0].set_xlim(0.15, 0.45)
axes[1, 0].set_xlim(0.15, 0.45)
axes[2, 0].set_xlim(0.15, 0.45)

axes[2, 0].set_ylim(0.3e-4, 1e-4)
axes[2, 1].set_ylim(0.3e-4, 1e-4)
axes[2, 2].set_xlim(0.3e-4, 1e-4)

axes[2, 0].plot(0.307, 6e-5, markersize=10, marker='+', color='red')

fig.savefig('images/corner_combined.png', dpi=300, format='png')

end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")
plt.show()


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