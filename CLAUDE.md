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
| `derived` | `src/derive_variables_from_chain.jl` | Post-process an existing chain: appends `derived_*` arrays to the same JLD2. Computes per-sample day-night asymmetries `2(D−N)/(D+N)` (background-subtracted) for ES and CC; posterior predictive distributions (mean ± σ bands on signal-only rates); weighted first-/second-moment statistics of per-bin log-likelihoods vs sin²θ₁₂ and Δm²₂₁. Requires `prevFile` set in config. |
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
params → compute_shared_osc_probs   (oscillation probs on fine grid, block-averaged)
       → compute_oscillated_samples  (unosc × P × flux, per channel/period)
       → normalize_backgrounds       (scale by nuisance parameters)
       → compute_ES/CC_event_rates   (apply response matrices, add BG)
       → return day/night rates per channel
```

**Unoscillated baseline** (`unoscillatedSample.jl`, built once at init): for each process (8B, HEP) and flavour channel (ES\_νe, ES\_νother, CC), `flux(E) × σ(E)` is numerically integrated over each `Etrue` bin and scaled by `detector_ne` (ES) or `detector_nAr40 × detection_time × normalisation` (CC), giving vectors of length `n_Etrue`.

**Response matrices** (`response.jl`, built once at init): a 2D MC histogram `C[i_true, j_reco]` is row-normalised to `R[i_true, j_reco] = P(E_reco | E_true)` so each row sums to 1. Separate matrices exist for ES\_νe, ES\_νother, CC, and — in inclusive/semi-inclusive modes — CC\_inclusive remapped to ES reco bins. Per-reco-bin selection efficiency `eff[j_reco] = N_selected / N_total` is derived from MC.

**Oscillation probabilities** (`propagation_osc.jl`): computed on a finer grid than the analysis binning — 2× in energy for day, 3× in zenith and 2× in energy for night — then block-averaged (`block_average` in `propagation_core.jl`) to analysis resolution. Night array convention: shape `(n_cosz, n_Etrue)`, rows index zenith bins. ν_other by unitarity: `P_other = 1 − P_νe`.

**Unoscillated baseline** (`unoscillatedSample.jl`) also computes a per-bin log-energy vector `log_E_norm = log.(E_bins / E_pivot)` where `E_pivot` is the flux-weighted mean CC event energy. This is stored in the `unoscillatedSample` named tuple and used at runtime for spectral shape distortions. By construction, `sum(unosc_CC_8B .* log_E_norm) ≈ 0`, making `cc_xsec_tilt` orthogonal to `cc_xsec_norm`.

**Oscillated samples** (`compute_oscillated_samples`):
- Day: element-wise product `unosc .* P_νe .* flux` → `(n_Etrue,)`.
- Night: outer product with per-zenith `exposure_weights` folded in → `(n_cosz, n_Etrue)`.
- CC signal is multiplied by `xsec_shape = exp(cc_xsec_tilt × logE + cc_xsec_curv × logE²) × cc_xsec_norm`, a per-Etrue-bin weight vector. CC backgrounds are not affected (they are cosmogenic in origin).

**Response application** (`propagation_core.jl`):
- `apply_day_response(osc, R, eff)` = `0.5 × (R' × osc) .* eff` → `(n_Ereco,)`. The 0.5 accounts for 50% day-time exposure.
- `apply_night_response(osc_rows, R, eff)` loops over the `n_cosz` rows of `osc_rows(n_cosz, n_Etrue)`, computes `0.5 × (row' × R) .* eff` per zenith slice, and vcat-s to `(n_cosz, n_Ereco)`.

**Background normalisation** (`propagation_bg.jl`): applied at every likelihood call via nuisance parameters.
- ES: `ES_conversions(side, norm)` = detector face area in cm² × `norm` (long faces: 2 × 12 m × 64 m; end caps: 2 × 12 m × 12 m).
- CC: `CC_conversions(norm)` = `norm / 2.2×10⁻⁶` (MC normalised to 2.2×10⁻⁶ neutrons cm⁻² s⁻¹).
- Backgrounds enter the final rate as `0.5 × BG` (day) and `0.5 × BG .* exposure_weights` (night).

**Inclusive / semi-inclusive modes** (`propagation_main.jl`, `propagation_reco.jl`): in `inclusiveMode` CC signal is folded into ES reco bins via the `CC_inclusive` response matrix and CC backgrounds are disabled. In `semiInclusiveMode` the forward hemisphere uses the CC-inclusive projection; the backward hemisphere retains a separate CC channel plus ES events mis-identified as CC.

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
- `semi_inclusive_analysis` — per-detector flag (`semiInclusiveMode` in config): forward hemisphere uses CC-inclusive in ES bins; backward hemisphere uses CC exclusive + ES mis-ID
- `CC_xsec_scale` — **deprecated**. Fixed multiplier baked into the unoscillated CC sample at setup time. Superseded by `true_cc_xsec_norm` + `prior_cc_xsec_norm`. Keep at 1.0; stacking with `cc_xsec_norm` multiplies both effects.
- `true_cc_xsec_norm` — Asimov true value for `cc_xsec_norm`; default 1.0 (`true_cc_xsec_norm` in config). Decoupled from the prior centre, enabling bias/wrong-prior studies.
- `prior_cc_xsec_norm` — prior on the CC cross-section overall normalisation nuisance parameter; default `truncated(Normal(1.0, 0.1), 0.1, 2.0)`. Centre independently of `true_cc_xsec_norm`.
- `prior_cc_xsec_tilt` — prior on the CC xsec spectral tilt `α₁`; applied as `exp(α₁ × log(E/E_pivot))`; default `Normal(0.0, 0.1)`. Orthogonal to `cc_xsec_norm` by choice of pivot.
- `prior_cc_xsec_curv` — prior on the CC xsec spectral curvature `α₂`; applied as `exp(α₂ × log²(E/E_pivot))`; default `Normal(0.0, 0.05)`. Orthogonal to both norm and tilt by symmetry.
- `active_detectors` — list of detector names to include in the fit (`active_detectors` in config)

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
- **`src/responseSys.jl` and `src/xsecSys.jl`** are empty stubs. Energy-scale systematics and response systematics are not yet implemented. CC cross-section shape uncertainty is handled via `cc_xsec_norm`, `cc_xsec_tilt`, and `cc_xsec_curv` fit parameters (see globals section).
- **Runs are not reproducible**: the random seed in `mcmc.jl` is commented out. Set `seed` in the BAT context if reproducibility is needed.
- **Temporary efficiency boost in inclusive/semi-inclusive mode** (`response.jl`, `backgrounds.jl`): a Hermite smooth-step ramp artificially raises ES (and CC\_inclusive) masking efficiency to a 90% plateau between 10 and 11.5 MeV, with the same boost applied to ES background efficiency for self-consistency. This is a temporary measure to study high-energy sensitivity without retuning the selection. Remove both blocks (tagged `TEMPORARY` in both files) once the selection is properly tuned.
- **ES background normalisations fully correlated** (`propagation_bg.jl`): all ES background components currently share a single nuisance parameter (`{det}_ES_bg_norm_1`) because `norm_index` is not incremented in the ES loop (line 57 is commented out). This is intentional to match a frozen reference run. CC backgrounds correctly have independent parameters per component. Un-comment line 57 when running with independent ES background shapes.
