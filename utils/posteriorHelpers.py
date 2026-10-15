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


def _is_batched_jld2(h5file):
    """Return True if the JLD2 file uses the batch_N group format."""
    return any(k.startswith("batch_") for k in h5file.keys())


_UNSET = object()


def derived_detectors(h5file):
    """Detectors written by derived mode (src/derive_variables_from_chain.jl), in config
    order; [] for files derived before multi-detector support (unprefixed keys)."""
    if "derived_detectors" not in h5file:
        return []
    v = h5file["derived_detectors"][()]
    v = v.decode() if isinstance(v, bytes) else str(v)
    return [d for d in v.split(",") if d]


def pick_derived_detector(h5file, detector=None, verbose=True):
    """Choose the detector whose derived_* keys to read (default: the first one)."""
    dets = derived_detectors(h5file)
    if not dets:
        if detector is not None:
            raise ValueError(f"{h5file.filename} has no per-detector derived quantities (derived before "
                             f"multi-detector support): cannot select detector '{detector}'. Re-run derived mode.")
        return None
    if detector is None:
        detector = dets[0]
        if verbose and len(dets) > 1:
            print(f"Derived quantities for detectors {dets}; using '{detector}' (choose with --detector).")
    elif detector not in dets:
        raise ValueError(f"Detector '{detector}' not in derived detectors {dets}")
    return detector


def resolve_key(h5file, key, detector=None):
    """Map a 'derived_X' key to the stored key: 'derived_<detector>_X' for per-detector
    quantities, else 'derived_X' (shared quantities and older files). Other keys unchanged."""
    if detector is not None and key.startswith("derived_"):
        per_det = "derived_" + detector + "_" + key[len("derived_"):]
        if per_det in h5file:
            return per_det
    return key


def _read_jld2_batched(h5file, keys, detector=None):
    """
    Read a batched JLD2 file (batch_0, batch_1, ...) and return concatenated
    arrays for each key in `keys`.  Batches are processed in numeric order.
    Keys not stored per batch (the per-step derived_* arrays, written at the top level by
    derived mode) are read from the top level.
    """
    batch_names = sorted(
        [k for k in h5file.keys() if k.startswith("batch_")],
        key=lambda x: int(x.split("_")[1])
    )
    accum = {k: [] for k in keys}
    for batch in batch_names:
        g = h5file[batch]
        for k in keys:
            if k in g:
                accum[k].append(np.array(g[k][()]))
    out = {k: np.concatenate(v) if v else np.array([]) for k, v in accum.items()}
    n_steps = sum(len(h5file[b]["stepno"]) for b in batch_names)
    for k in keys:
        stored = resolve_key(h5file, k, detector)
        if out[k].size == 0 and stored in h5file:
            arr = np.array(h5file[stored][()])
            # One entry per chain step; a chain resumed after deriving has more steps than the array.
            if arr.ndim == 1 and arr.shape[0] != n_steps:
                raise ValueError(f"{h5file.filename}: '{stored}' has {arr.shape[0]} entries but the chain has "
                                 f"{n_steps} steps (resumed after deriving?). Re-run derived mode.")
            out[k] = arr
    return out


def _read_jld2_flat(h5file, keys, detector=None):
    """Read a flat (legacy) JLD2 file written by convert_mcmc_to_jld2.jl."""
    keys_stored = {k: resolve_key(h5file, k, detector) for k in keys}
    return {k: np.array(h5file[ks][()]) if ks in h5file else np.array([]) for k, ks in keys_stored.items()}


def load_posterior(mcmc_chains, parameters, burnin=100, test=None, exclude_chains=None, detector=None):
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
            detector = pick_derived_detector(f, detector)
            if _is_batched_jld2(f):
                first_batch = sorted([k for k in f.keys() if k.startswith("batch_")],
                                     key=lambda x: int(x.split("_")[1]))[0]
                sample_keys = set(f[first_batch].keys())
            else:
                sample_keys = set()
            # per-step derived arrays (e.g. derived_CC_asymmetry) live at the top level
            for v in test:
                if (v[0] in sample_keys or resolve_key(f, v[0], detector) in f) and v[0] not in parameters:
                    parameters.append(v[0])

    # Initialize dictionary to store concatenated data
    results = {param: [] for param in parameters}
    results['chains'] = []
    results['weights'] = []
    results['stepno'] = []

    chosen_detector = _UNSET          # every chain must resolve to the same derived detector
    for mcmc_chain in valid_chains:
        with h5py.File(mcmc_chain + ".jld2", 'r') as f:
            batched = _is_batched_jld2(f)

            # Detect all parameter names if requested
            if parameters == "all":
                skip_keys = {'stepno', 'chainid', 'weights'}
                if batched:
                    first_batch = sorted([k for k in f.keys() if k.startswith("batch_")], key=lambda x: int(x.split("_")[1]))[0]
                    sample_keys = f[first_batch].keys()
                else:
                    sample_keys = f.keys()
                parameters = [k for k in sample_keys if k not in skip_keys]
                results.update({param: [] for param in parameters})

            meta_keys = ['stepno', 'chainid', 'weights']
            all_keys  = list(parameters) + meta_keys

            det = pick_derived_detector(f, detector, verbose=False)
            if chosen_detector is not _UNSET and det != chosen_detector:
                raise ValueError(f"{mcmc_chain}: derived detector '{det}' differs from '{chosen_detector}' "
                                 "used for the previous chain (mixed old/new derived files or detector lists).")
            chosen_detector = det
            raw = _read_jld2_batched(f, all_keys, det) if batched else _read_jld2_flat(f, all_keys, det)

        stepno  = raw['stepno']
        chains  = raw['chainid']
        weights = raw.get('weights', np.array([]))

        _excl = list(exclude_chains) if exclude_chains is not None else [20]
        mask = (stepno > burnin) & (~np.isin(chains, _excl))
        max_steps = len(mask)

        results['chains'].append(chains[mask])
        results['stepno'].append(stepno[mask])

        for param in parameters:
            if "_llh" in param:
                continue

            data = raw.get(param, None)
            if data is None or data.size == 0:
                raise ValueError(f"Parameter '{param}' not found in MCMC chain file")

            if data.ndim > 1:
                data = data.squeeze()
                print("we had to flatten " + param)

            mask_param = downsample_mask(mask, len(data), max_steps)
            results[param].append(data[mask_param])

        if weights.size > 0:
            results['weights'].append(weights[mask])

    for key in results:
        if results[key]:
            results[key] = np.concatenate(results[key])
        else:
            results[key] = np.array([])

    chain_indexes = np.unique(results['chains'])
    print(f"Unique chain IDs: {chain_indexes}")
    print(f"Total MCMC steps in posterior (sum of weights): {np.sum(results['weights']):.0f}")

    return results



def load_bin_diagnostics(mcmc_chains, require_all=True, verbose=True, detector=None):
    es_keys = [
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

        # --- ES sample totals (scalars) ---
        "derived_ESnight_tot_mean_llh",
        "derived_ESnight_tot_var_llh",
        "derived_ESnight_tot_corr_llh_sin2_th12",
        "derived_ESnight_tot_corr_llh_dm2_21",

        "derived_ESday_tot_mean_llh",
        "derived_ESday_tot_var_llh",
        "derived_ESday_tot_corr_llh_sin2_th12",
        "derived_ESday_tot_corr_llh_dm2_21",

        # --- ES posterior predictive moments and 1σ/2σ/3σ quantile bands (rates) ---
        "derived_ESday_pp_mean",
        "derived_ESday_pp_var",
        "derived_ESday_pp_lo",
        "derived_ESday_pp_hi",
        "derived_ESday_pp_lo2",
        "derived_ESday_pp_hi2",
        "derived_ESday_pp_lo3",
        "derived_ESday_pp_hi3",
        "derived_ESnight_pp_mean",
        "derived_ESnight_pp_var",

        # --- ES background moments for signal-only display and data error bars ---
        "derived_BGESday_pp_mean",
        "derived_BGESday_pp_var",

        # --- ES shapes (for de-serialisation / sanity checks) ---
        "derived_ESnight_shape",
        "derived_ESday_shape",
        "derived_ESday_pp_shape",
        "derived_ESnight_pp_shape",

        # --- Asimov data and threshold indices (optional: present after re-derive) ---
        "derived_data_ESday",
        "derived_data_ESnight",
        "derived_index_ES",
        "derived_index_CC",
    ]

    cc_keys = [
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

        # --- CC sample totals (scalars) ---
        "derived_CCnight_tot_mean_llh",
        "derived_CCnight_tot_var_llh",
        "derived_CCnight_tot_corr_llh_sin2_th12",
        "derived_CCnight_tot_corr_llh_dm2_21",

        "derived_CCday_tot_mean_llh",
        "derived_CCday_tot_var_llh",
        "derived_CCday_tot_corr_llh_sin2_th12",
        "derived_CCday_tot_corr_llh_dm2_21",

        # --- CC posterior predictive moments and 1σ/2σ/3σ quantile bands (rates) ---
        "derived_CCday_pp_mean",
        "derived_CCday_pp_var",
        "derived_CCday_pp_lo",
        "derived_CCday_pp_hi",
        "derived_CCday_pp_lo2",
        "derived_CCday_pp_hi2",
        "derived_CCday_pp_lo3",
        "derived_CCday_pp_hi3",

        # --- CC background moments for signal-only display and data error bars ---
        "derived_BGCCday_pp_mean",
        "derived_BGCCday_pp_var",
        "derived_CCnight_pp_mean",
        "derived_CCnight_pp_var",

        # --- CC shapes (for de-serialisation / sanity checks) ---
        "derived_CCnight_shape",
        "derived_CCday_shape",
        "derived_CCday_pp_shape",
        "derived_CCnight_pp_shape",
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

    # Detect inclusive mode from the first chain.
    # Fall back to inferring from key presence for pre-fix JLD2 files.
    inclusive_mode = False
    with h5py.File(chains[0] + ".jld2", "r") as f:
        detector = pick_derived_detector(f, detector, verbose)
        k_incl = resolve_key(f, "derived_inclusive_mode", detector)
        if k_incl in f:
            inclusive_mode = bool(f[k_incl][()])
        elif resolve_key(f, "derived_CCnight_mean_llh", detector) not in f:
            inclusive_mode = True  # backward compat: CC keys absent => inclusive

    if verbose and inclusive_mode:
        print("Inclusive mode detected: CC diagnostics not present in chain.")

    keys = es_keys + ([] if inclusive_mode else cc_keys)

    # Optional metadata keys (plotting-axis bin edges); present only after re-deriving
    # with the edge-saving fix. Loaded if available but never required.
    optional_keys = [
        "derived_cosz_edges",
        "derived_Ereco_edges_ES",
        "derived_Ereco_edges_CC",
        "derived_BGESnight_pp_mean",
        "derived_BGCCnight_pp_mean",
        "derived_data_CCday",
        "derived_data_CCnight",
        "derived_signal_CCday",
        "derived_signal_CCnight",
        "derived_signal_ESday",
        "derived_signal_ESnight",
        "derived_CCbg_comp_day",
        "derived_CCbg_comp_names",
        # best-fit oscillation map + day-night survival-ratio panel
        "derived_oscmap_8B_day",
        "derived_oscmap_8B_night",
        "derived_oscmap_Etrue_edges",
        "derived_exposure_weights",
        "derived_BGCCday_pp_mean",
        "derived_BGESday_pp_mean",
    ]
    load_keys = keys + optional_keys

    per_chain = {k: [] for k in load_keys}
    missing = {k: [] for k in load_keys}

    for c in chains:
        with h5py.File(c + ".jld2", "r") as f:
            if pick_derived_detector(f, detector, verbose=False) != detector:
                raise ValueError(f"{c}: derived detectors differ from the first chain's; load them separately.")
            have = set(f.keys())
            for k in load_keys:
                stored = resolve_key(f, k, detector)   # per-detector key, or shared/legacy
                if stored not in have:
                    missing[k].append(c)
                    continue
                a = np.array(f[stored][()])
                if a.ndim > 1 and 1 in a.shape:  # squeeze degenerate axes only
                    a = np.squeeze(a)
                per_chain[k].append(a)

    if require_all:
        bad = {k: v for k, v in missing.items() if v and k in keys}
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

    out["inclusive_mode"] = inclusive_mode  # propagate flag to callers
    out["detector"] = detector              # None for files derived before multi-detector support

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