import numpy as np
import subprocess
import h5py
import time
import os


def convert_mcmc_to_jld2(bin_file, info_file, out_jld2):
    result = subprocess.run(
    ["julia", "./utils/convert_mcmc_to_jld2.jl", bin_file, info_file, out_jld2],
    check=False,
    capture_output=False,
    text=False)

    # If there's an error message sent to stderr, print that as well
    if result.stderr:
        print("Julia stderr:", result.stderr)
        return -1
    else:
        return 0


def load_posterior(mcmc_chains, parameters, burnin=30_000, test=None):

    for mcmc_chain in mcmc_chains:
        if not os.path.exists(mcmc_chain+".jld2"):
            print(f"The file '{mcmc_chain}.jld2' does not exist. Looking for binaries")
            
            if not os.path.exists(mcmc_chain+"_mcmc.bin"):
                print(f"Error: The mcmc file '{mcmc_chain}_mcmc.bin' does not exist.")
                continue

            if not os.path.exists(mcmc_chain+"_info.txt"):
                print(f"Error: The info file '{mcmc_chain}_info.txt' does not exist.")  
                continue
            
            convert_mcmc_to_jld2(mcmc_chain+"_mcmc.bin", mcmc_chain+"_info.txt", mcmc_chain+".jld2")

    if test is not None:
        with h5py.File(mcmc_chains[0]+".jld2", 'r') as f:
            for v in test:
                if v[0] in f:
                    parameters.append(v[0])

    # Initialize dictionary to store concatenated data
    results = {param: [] for param in parameters}
    results['chains'] = []  # Always track chain IDs
    results['weights'] = [] # Always get weights 
    results['stepno'] = []  # Always track step numbers
    
    # Iterate over each MCMC chain file
    for mcmc_chain in mcmc_chains:
        with h5py.File(mcmc_chain+".jld2", 'r') as f:
            # Detect all parameter names if requested
            if parameters == "all":
                skip_keys = {'stepno', 'chainid'}
                parameters = [key for key in f.keys() if key not in skip_keys]
                # parameters.append('weights')  # Ensure weights are always included
                for param in parameters:
                    results[param] = []

            
            else:
                parameters.append('weights')

            # Get step numbers and chain IDs first to create burnin mask
            stepno = np.array(f['stepno'][()])
            chains = np.array(f['chainid'][()])
            
            # Create burnin mask ## and bad chains mask
            mask = (stepno > burnin)   & (~np.isin(chains, [17]))
            
            # Store chain IDs (after burnin)
            results['chains'].append(chains[mask])
            results['stepno'].append(stepno[mask])
            
            # Load each requested parameter
            for param in parameters:
                if param in f:
                    data = np.array(f[param][()])
                    results[param].append(data[mask])
                else:
                    raise ValueError(f"Parameter '{param}' not found in MCMC chain file")    


    # Concatenate all data for each parameter
    for key in results:
        if results[key]:  # Only concatenate if there's data
            results[key] = np.concatenate(results[key])

    chain_indexes = np.unique(results['chains'])
    # print(f"Parameters loaded: {results}")
    print(f"Unique chain IDs: {chain_indexes}")
    print("Number of effective steps in posterior:", np.sum(results['weights']))
    return results


