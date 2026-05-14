# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the fitter

```bash
# Run with default config.yaml
julia -t auto src/readConfig.jl

# Run with a specific config
julia -t auto src/readConfig.jl path/to/config.yaml

# Run tests
julia tests/runtests.jl
```

> `-t auto` uses all available threads. **Do not use multithreading with nuFast** (`nuFast: true`).

## Run modes

`RunMode` in `config.yaml` selects the analysis:

| Mode | Script | Purpose |
|------|--------|---------|
| `MCMC` | `src/mcmc.jl` | Bayesian posterior sampling via BAT.jl (HMC) |
| `LLH` | `src/llhScan.jl` | 2D likelihood scans over parameter pairs |
| `derived` | `src/derive_variables_from_chain.jl` | Post-process chains: appends day-night asymmetry and per-bin likelihoods; requires `prevFile` set in config |
| `PROFILE` | `src/profiling.jl` | Wall-time benchmarking; use `configs/profilingConfig.yaml` |

## Python utils

All utilities live in `utils/` and are invoked directly with Python. Run any with `--help` for full options.

```bash
# MCMC posterior plots (corner, traces, diagnostics)
python3 utils/plotOutput.py outFileName [outFileName2 ...] [-o output] [-d] [-e] [-f] [-p param1 param2] [--burnin N] [--bins N] [--cbar colormap]

# Likelihood scan contours
python3 utils/plotLLH.py outFileName [-o output]

# Per-bin diagnostic plots (use after RunMode: derived)
python3 utils/plotBinDiagnostics.py outFileName [-o output]

# Build a proposal covariance matrix from a chain (feed back via config proposal_matrix key)
python3 utils/makeOutputCovariance.py outFileName [-o output] [-p]

# Overlay oscillation probability samples from a chain
python3 utils/plotSamples.py outFileName
```

Key `plotOutput.py` flags:
- `-d` diagnostics: bootstrap contour + stability heatmap
- `-e` expanded: includes HEP flux and day-night asymmetry panels
- `-f` full: corner plot over all sampled parameters
- `--burnin N`: steps to discard (default 5000)
- `--exclude id [id ...]`: chain IDs to drop (default: 20); `--exclude` with no args keeps all

## Architecture

`src/readConfig.jl` is the single entry point. It parses the YAML config into Julia globals, then `include`s `src/setup.jl` which runs the full initialization pipeline before dispatching to the mode script.

### Initialization pipeline (`setup.jl`)

1. Load solar model, Earth density profile, and neutrino flux tables
2. Load unoscillated MC samples (ES and CC channels) and detector response matrices
3. Load and normalize backgrounds
4. Generate Asimov (fake) data by calling `propagateSamples` with true parameters
5. Build the likelihood function (`likelihood_all_samples`) over the Asimov dataset

### Per-event propagation (`propagation/propagation_main.jl`)

Called once per likelihood evaluation with a new parameter point:

```
params → get_mixing_parameters → oscPars{T}
       → setup_earth_propagation   (matter effect lookup + Earth paths)
       → compute_oscillation_probs (day/night × 8B/HEP × νe/νother)
       → apply_response_matrices   (true E → reco E smearing)
       → normalize_backgrounds
       → event rates per channel/period
```

### Oscillation calculation (`oscillations/osc.jl`)

Three parallel implementations exist for the Earth matter integral:
- `NumOsc.Slow` — full 3×3 matrix exponentiation per energy/path segment, used when `fastFit: false`
- `NumOsc.Fast` — pre-allocated loop with eigendecomposition lookup (`osc_prob_earth`); the hot path when `fastFit: true`
- `nuFastOsc` (optional) — wraps a compiled C++ library (`libnufast_earth.so`); incompatible with Julia multithreading

`oscPars{T<:Real}` is parametric so ForwardDiff Dual numbers propagate through the full pipeline for HMC gradient computation. For plain Float64 (LLH mode), Julia specialises to identical machine code.

### MCMC implementation (`mcmc.jl` + `mcmcHelpers.jl`)

Sampling is done via BAT.jl `TransformedMCMC`. The proposal algorithm and context are defined once as module-level globals in `mcmc.jl`:

```julia
proposal_algorithm = HamiltonianMC(termination=GeneralisedNoUTurn(max_depth=4))
hmc_context        = BATContext(ad = ADSelector(:ForwardDiff))
```

`runMCMCbatch` in `mcmcHelpers.jl` consumes these globals. To tune HMC (e.g. `max_depth`) edit only `mcmc.jl`. Chains are written in batches of `batchSteps` steps to avoid memory pressure; each batch is appended as `batch_N` inside a single JLD2 file.

### Key globals (set by `readConfig.jl`, consumed everywhere)

The codebase is script-based (no module encapsulation), so many parameters live as `global` variables after config load. The most commonly referenced:

- `run_mode`, `fast`, `nuFast`, `Asimov`, `singleChannel`, `CC_mode`, `ES_mode`
- `inclusive_analysis` — when `true`, CC signal is projected onto the ES reconstructed energy/angle axes and summed into the ES likelihood; CC backgrounds are absorbed into ES backgrounds; the separate CC likelihood branch is disabled (`use_CC = CC_mode && !inclusive_analysis`). Set via `inclusiveMode` in config.
- `outFile`, `prevFile`
- `mcmcSteps`, `mcmcChains`, `tuningSteps`, `maxTuningAttempts`
- `earthUncertainty`, `earth_lookup`, `earth_paths`, `earth_normalisation_prior`
- `propMatrix` — covariance matrix for the HMC proposal, loaded from `proposal_matrix` config key if present

## nuFast build

Required only when `nuFast: true`. Needs the `NuFast-Earth` C++ library installed separately:

```bash
NUFAST_DIR=/path/to/NuFast-Earth make
```

This produces `src/oscillations/libnufast_earth.so`.

## Important operational notes

- **Always delete existing output files before a new run.** JLD2 append semantics can silently produce corrupted or mixed-run output files.
- `prevFile` in config resumes chains from a previous run's `_chainState.bin` serialised state.
- `proposal_matrix` in config feeds a JLD2 covariance matrix (from `makeOutputCovariance.py`) into the MH proposal distribution.

## Known limitations / open TODOs

- **No background floor in conditional likelihood** (`likelihood_statistical.jl`): when `row_sum_expected == 0` the conditional likelihood returns 0 regardless of observed counts. This is intentional pending a proper background floor implementation.
- **Earth systematics diagonalised** (`mcmc.jl`): BAT.jl does not yet support multivariate Gaussian priors, so the Earth density covariance matrix is approximated as independent 1-D normals (diagonal only). Update when BAT adds `MvNormal` prior support.
- **Barlow-Beeston half-count substitution** (`likelihood_statistical.jl`): when `e == 0` or `m == 0` the BB formula substitutes 0.5 (a Wald lower bound). Intentional regularisation — revisit once a background floor is in place.
- **CC background ×10 factor** (`backgrounds.jl`): CC background MC is normalised to 1 kt-year. The signal MC uses `detector_nAr40 = 10 kT × detection_time`, so the signal scales as `10 kT × CC_normalisation`. The `×10` in the background scaling matches this — it is not the exposure itself but the 10 kT factor embedded in `detector_nAr40`. If `detector_nAr40` ever changes, update this factor to match.
- **`src/responseSys.jl` and `src/xsecSys.jl`** are empty stubs. Energy-scale and cross-section systematics are not yet implemented.
- **Runs are not reproducible**: the random seed in `mcmc.jl` is commented out. Set `seed` in the BAT context if reproducibility is needed.
