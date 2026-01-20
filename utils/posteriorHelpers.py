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

def infer_thinning(length, max_steps, tol=0.05):
    ratio = max_steps / length
    thin = max(1, int(round(ratio)))
    if abs(ratio - thin) > tol:
        print(f"Warning: inferred thinning {thin} from ratio {ratio:.3f}")
    return thin

def downsample_mask(mask_dense, length_param, max_steps):
    """
    Convert a boolean mask defined on the densest step grid (len=max_steps)
    into a boolean mask aligned to a parameter array of length length_param.
    """
    thin = infer_thinning(length_param, max_steps)
    idx_dense = np.arange(length_param) * thin          # map param index -> dense index
    idx_dense = np.minimum(idx_dense, max_steps - 1)    # safety clamp
    return mask_dense[idx_dense]


def load_posterior(mcmc_chains, parameters, burnin=10_000, test=None):
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
            mask = (stepno > burnin) & (~np.isin(chains, [16]))
            max_steps = len(mask)
            
            # Store chain IDs and step numbers (after burnin)
            results['chains'].append(chains[mask])
            results['stepno'].append(stepno[mask])
            
            # Load each requested parameter
            for param in parameters:
                if "_llh" in param:
                    continue  # skip by-bin loglik diagnostics parameters

                if param in f:
                    data = np.array(f[param][()])
                    if data.ndim > 1:
                        data = data.squeeze()
                        print("we had to flatten " + param)

                    # align mask to this parameter's length
                    mask_param = downsample_mask(mask, len(data), max_steps)

                    results[param].append(data[mask_param])
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



def load_bin_diagnostics(mcmc_chains, require_all=True, verbose=True):
    keys = [
        # --- CC night (per-bin) ---
        "derived_CCnight_mean_llh",
        "derived_CCnight_var_llh",
        "derived_CCnight_corr_llh_sin2_th12",
        "derived_CCnight_corr_llh_dm2_21",

        # --- CC day (per-bin) ---
        "derived_CCday_mean_llh",
        "derived_CCday_var_llh",
        "derived_CCday_corr_llh_sin2_th12",
        "derived_CCday_corr_llh_dm2_21",

        # --- ES night (per-bin) ---
        "derived_ESnight_mean_llh",
        "derived_ESnight_var_llh",
        "derived_ESnight_corr_llh_sin2_th12",
        "derived_ESnight_corr_llh_dm2_21",

        # --- ES day (per-bin) ---
        "derived_ESday_mean_llh",
        "derived_ESday_var_llh",
        "derived_ESday_corr_llh_sin2_th12",
        "derived_ESday_corr_llh_dm2_21",

        # --- Sample totals (scalars) ---
        "derived_CCnight_tot_mean_llh",
        "derived_CCnight_tot_var_llh",
        "derived_CCnight_tot_corr_llh_sin2_th12",
        "derived_CCnight_tot_corr_llh_dm2_21",

        "derived_CCday_tot_mean_llh",
        "derived_CCday_tot_var_llh",
        "derived_CCday_tot_corr_llh_sin2_th12",
        "derived_CCday_tot_corr_llh_dm2_21",

        "derived_ESnight_tot_mean_llh",
        "derived_ESnight_tot_var_llh",
        "derived_ESnight_tot_corr_llh_sin2_th12",
        "derived_ESnight_tot_corr_llh_dm2_21",

        "derived_ESday_tot_mean_llh",
        "derived_ESday_tot_var_llh",
        "derived_ESday_tot_corr_llh_sin2_th12",
        "derived_ESday_tot_corr_llh_dm2_21",

        # --- Posterior predictive moments (rates) ---
        "derived_CCday_pp_mean",
        "derived_CCday_pp_var",
        "derived_CCnight_pp_mean",
        "derived_CCnight_pp_var",

        "derived_ESday_pp_mean",
        "derived_ESday_pp_var",
        "derived_ESnight_pp_mean",
        "derived_ESnight_pp_var",

        # --- Shapes (for de-serialisation / sanity checks) ---
        "derived_CCnight_shape",
        "derived_ESnight_shape",
        "derived_CCday_shape",
        "derived_ESday_shape",

        "derived_CCday_pp_shape",
        "derived_CCnight_pp_shape",
        "derived_ESday_pp_shape",
        "derived_ESnight_pp_shape",
    ]

    chains = []
    for c in mcmc_chains:
        j = c + ".jld2"
        if not os.path.exists(j):
            if verbose:
                print(f"Missing '{j}'. Looking for binaries.")
            b, info = c + "_mcmc.bin", c + "_info.txt"
            if not (os.path.exists(b) and os.path.exists(info)):
                if verbose:
                    print(f"Skip '{c}': missing {b} or {info}")
                continue
            convert_mcmc_to_jld2(b, info, j)
        chains.append(c)
    if not chains:
        raise ValueError("No valid MCMC chains found")

    per_chain = {k: [] for k in keys}
    missing = {k: [] for k in keys}

    for c in chains:
        with h5py.File(c + ".jld2", "r") as f:
            have = set(f.keys())
            for k in keys:
                if k not in have:
                    missing[k].append(c)
                    continue
                a = np.array(f[k][()])
                if a.ndim > 1 and 1 in a.shape:  # squeeze degenerate axes only
                    a = np.squeeze(a)
                per_chain[k].append(a)

    if require_all:
        bad = {k: v for k, v in missing.items() if v}
        if bad:
            raise ValueError(
                "Missing keys:\n" + "\n".join(f"  {k}: {', '.join(v)}" for k, v in bad.items())
            )

    out = {}
    for k, arrs in per_chain.items():
        if not arrs:
            continue
        base = arrs[0]
        same = all((a.shape == base.shape) and np.array_equal(a, base) for a in arrs[1:])
        out[k] = base if same else arrs
        if verbose and len(arrs) > 1:
            print(f"{k}: {'consistent' if same else 'DIFF'} across {len(arrs)} chains")

    if verbose:
        print("Loaded diagnostics from:", ", ".join(chains))
    return out



def reshape_julia_matrix(x, shape, name="array"):
    """
    Ensure x is a 2D numpy array with given shape.
    If x arrives flattened (as from Julia), reshape using column-major order.
    """
    x = np.asarray(x)

    # Already 2D
    if x.ndim == 2:
        if x.shape == shape:
            return x
        if x.shape == shape[::-1]:
            # handle accidental transpose
            return x.T
        raise ValueError(
            f"{name}: expected shape {shape} (or {shape[::-1]}), got {x.shape}"
        )

    # Flattened Julia matrix
    if x.ndim == 1:
        ny, nx = shape
        if x.size != ny * nx:
            raise ValueError(
                f"{name}: cannot reshape length {x.size} into {shape}"
            )
        # Julia is column-major → order='F'
        return x.reshape((ny, nx), order="F")

    raise ValueError(
        f"{name}: expected 1D or 2D array, got ndim={x.ndim}, shape={x.shape}"
    )