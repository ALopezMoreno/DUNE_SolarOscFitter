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


def load_posterior(mcmc_chains, parameters, burnin=5_000, test=None):
    valid_chains = []
    for mcmc_chain in mcmc_chains:
        if not os.path.exists(mcmc_chain + ".jld2"):
            print(f"The file '{mcmc_chain}.jld2' does not exist. Looking for binaries")  
            
            if not os.path.exists(mcmc_chain + "_mcmc.bin"):
                print(f"Error: The mcmc file '{mcmc_chain}_mcmc.bin' does not exist.")
                continue

            if not os.path.exists(mcmc_chain + "_info.txt"):
                print(f"Error: The info file '{mcmc_chain}_info.txt' does not exist.")  
                continue
            
            convert_mcmc_to_jld2(mcmc_chain + "_mcmc.bin", mcmc_chain + "_info.txt", mcmc_chain + ".jld2")
        
        valid_chains.append(mcmc_chain)
    
    if not valid_chains:
        raise ValueError("No valid MCMC chains found")

    # Handle test parameters if specified
    if test is not None:
        with h5py.File(valid_chains[0] + ".jld2", 'r') as f:
            for v in test:
                if v[0] in f and v[0] not in parameters:
                    parameters.append(v[0])

    # Initialize dictionary to store concatenated data
    results = {param: [] for param in parameters}
    results['chains'] = []  # Always track chain IDs
    results['weights'] = [] # Always get weights 
    results['stepno'] = []  # Always track step numbers
    
    # Iterate over each valid MCMC chain file
    for mcmc_chain in valid_chains:
        with h5py.File(mcmc_chain + ".jld2", 'r') as f:
            # Detect all parameter names if requested
            if parameters == "all":
                skip_keys = {'stepno', 'chainid', 'weights'}
                parameters = [key for key in f.keys() if key not in skip_keys]
                results.update({param: [] for param in parameters})
            
            # Get step numbers and chain IDs first to create burnin mask
            stepno = np.array(f['stepno'][()])
            chains = np.array(f['chainid'][()])
            
            # Create burnin mask (excluding chain 17 as in your original code)
            mask = (stepno > burnin) # & (~np.isin(chains, [17]))
            
            # Store chain IDs and step numbers (after burnin)
            results['chains'].append(chains[mask])
            results['stepno'].append(stepno[mask])
            
            # Load each requested parameter
            for param in parameters:
                if param in f:
                    data = np.array(f[param][()])
                    # Ensure we're working with 1D arrays
                    if data.ndim > 1:
                        data = data.squeeze()
                        print("we had to flatten" + param)
                    results[param].append(data[mask])
                else:
                    raise ValueError(f"Parameter '{param}' not found in MCMC chain file")
            
            # Handle weights separately if they exist
            if 'weights' in f:
                weights = np.array(f['weights'][()])
                results['weights'].append(weights[mask])
    
    # Concatenate all data for each parameter
    for key in results:
        if results[key]:  # Only concatenate if there's data
            results[key] = np.concatenate(results[key])
        else:
            results[key] = np.array([])  # Ensure empty arrays for missing data

    chain_indexes = np.unique(results['chains'])
    print(f"Unique chain IDs: {chain_indexes}")
    print("Number of effective steps in posterior:", np.sum(results['weights']))
    
    return results


