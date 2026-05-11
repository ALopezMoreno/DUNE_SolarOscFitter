# Configuration Reference

All parameters are read from `config.yaml` by `src/readConfig.jl`. Optional keys have defaults shown; required keys will raise an error if absent.

---

## Output and run type

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `outFile` | String | required | Output file prefix. Results written as `{outFile}_mcmc.jld2`, `{outFile}_configSettings.txt`, etc. |
| `prevFile` | String | `nothing` | Previous run output prefix. Required when `RunMode: derived`. If set in MCMC mode, resumes chains from `{prevFile}_chainState.bin`. |
| `RunMode` | String | required | `"MCMC"`, `"LLH"`, `"derived"`, or `"PROFILE"`. Selects the analysis script. |
| `Asimov` | Bool | required | `true` = use theoretical prediction as data. `false` = read real data (not yet implemented). |
| `fastFit` | Bool | required | `true` = NumOsc.Fast + average solar density. `false` = NumOsc.Slow + production-region integration. See [oscillation_physics.md §5](oscillation_physics.md). |
| `nuFast` | Bool | required | `true` = use external C++ nuFast library (overrides `fastFit`). Incompatible with multithreading. |
| `singleChannel` | Bool or String | `false` | `false` = both channels. `"ES"` = ES only. `"CC"` = CC only. |
| `inclusiveMode` | Bool | `false` | `true` = fold CC signal into ES Ereco bins; disable separate CC likelihood. |

---

## Asimov true values

Used only when `Asimov: true` to generate the fake dataset.

| Key | Type | Description |
|-----|------|-------------|
| `true_sin2_th12` | Float | True sin²θ₁₂ (dimensionless) |
| `true_sin2_th13` | Float | True sin²θ₁₃ (dimensionless) |
| `true_dm2_21` | Float | True Δm²₂₁ (eV²) |
| `true_integrated_HEP_flux` | Float | True HEP flux (ν/cm²/s) |

The true ⁸B flux is taken as the mean of `prior_8B_flux` (not a separate config key).

---

## Prior distributions

Strings are evaluated as Julia `Distributions.jl` expressions.

| Key | Description |
|-----|-------------|
| `prior_sin2_th12` | Prior on sin²θ₁₂. Default: `Uniform(0.15, 0.5)`. |
| `prior_sin2_th13` | Prior on sin²θ₁₃. Default: `truncated(Normal(0.022, 0.0007), 0.005, 0.035)` (reactor constraint). |
| `prior_dm2_21` | Prior on Δm²₂₁ (eV²). Default: `Uniform(2e-5, 3e-4)`. |
| `prior_8B_flux` | Prior on integrated ⁸B flux (ν/cm²/s). Default: `truncated(Normal(5.25e6, 0.28e6), 1.5e6, 1e7)`. |
| `prior_HEP_flux` | Prior on integrated HEP flux (ν/cm²/s). Default: `Uniform(0, 2e4)`. |

---

## MCMC parameters

| Key | Type | Description |
|-----|------|-------------|
| `nChains` | Int | Number of parallel HMC chains. Recommended: equal to available cores. |
| `nSteps` | Int | Post-tuning steps per chain. Total samples = nChains × nSteps. |
| `maxTuningAttempts` | Int | Maximum adaptive tuning rounds. `0` = skip tuning. |
| `nTuning` | Int | Steps per tuning round. |
| `earth_potential_uncertainties` | Bool | Include Earth density nuisance parameters. Requires `earth_normalisation_prior_file`. |
| `proposal_matrix` | String | (optional) Path to JLD2 file with `posterior_cov` matrix for mass-matrix warm-start. |

---

## LLH scan parameters

| Key | Type | Description |
|-----|------|-------------|
| `llh_bins` | Int | Bins per parameter dimension for 2D likelihood scans (`RunMode: LLH`). |

---

## Derived parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `thinning` | Int | `1` | Thinning factor for per-bin diagnostic calculations in `RunMode: derived`. `1` = no thinning. |

---

## Detector configuration

| Key | Type | Description |
|-----|------|-------------|
| `CC_exposure` | Float | CC exposure in module-years (1 module-year ≈ 10 kt-year). Sets `CC_normalisation`. |
| `ES_exposure` | Float | ES exposure in module-years. Sets `ES_normalisation`. |
| `ES_background_normalisations` | Float[] | Per-background flux normalisation (cm⁻²s⁻¹). Length must match `ES_background_files`. |
| `CC_background_normalisations` | Float[] | Per-background flux normalisation (cm⁻²s⁻¹). Length must match `CC_background_files`. |
| `ES_background_systematics` | Float[] | Fractional uncertainty on each ES background. `0` = fixed (no nuisance parameter). |
| `CC_background_systematics` | Float[] | Fractional uncertainty on each CC background. |

---

## Energy and angle binning

| Key | Type | Description |
|-----|------|-------------|
| `nBins_Etrue` | Int | Number of true-energy bins for oscillation calculation. |
| `range_Etrue` | [Float, Float] | True energy range [MeV]. Converted to GeV internally. |
| `nBins_Ereco_ES` | Int | Reconstructed energy bins for ES channel. |
| `nBins_Ereco_CC` | Int | Reconstructed energy bins for CC channel. |
| `range_Ereco_ES` | [Float, Float] | ES reco energy range [MeV]. |
| `range_Ereco_CC` | [Float, Float] | CC reco energy range [MeV]. |
| `Ereco_min_ES` | Float | ES energy threshold [MeV]. Events below this bin are excluded from the likelihood. |
| `Ereco_min_CC` | Float | CC energy threshold [MeV]. |
| `nBins_cosz` | Int | Number of cos(zenith) bins for day/night split (range −1 to 0). |
| `nBins_cos_scatter` | Int | Number of cos(scatter angle) bins for ES angular reconstruction. |
| `use_scattering_info` | Bool | `true` = enable angular reconstruction likelihood. Changes ES rate shapes from 1D/2D to 2D/3D. |
| `ES_cos_cut` | Float | Optional (default −1.0). Forward-hemisphere angular cut: events with cos(θ_scatter) < this value are masked. −1.0 = no cut; 0.0 = forward hemisphere only. |

---

## Input files

| Key | Type | Description |
|-----|------|-------------|
| `solar_model_file` | String | JLD2 file with solar density profile and ⁸B/HEP production fractions. |
| `flux_model_file` | String | JLD2 file with neutrino energy spectra. |
| `earth_model_file` | String | PREM Earth density profile (plain text). |
| `earth_uncertainty_file` | String | JLD2 file with Earth layer density uncertainties. |
| `earth_normalisation_prior_file` | String | JLD2 file with `covariance` matrix and `true` vector for Earth density nuisance priors. Required if `earth_potential_uncertainties: true`. |
| `solar_exposure_file` | String | CSV file with detector exposure vs. solar zenith angle. Sets `exposure_weights`. |
| `reconstruction_sample_ES_nue` | String | CSV file (Etrue, Ereco, mask) for νₑ ES response matrix. |
| `reconstruction_sample_ES_nuother` | String | CSV file for νμ,τ ES response matrix. |
| `reconstruction_sample_ES_angle` | String | CSV file (Ereco, cos_scatter) for ES angular response matrix. |
| `reconstruction_sample_CC` | String | CSV file (Etrue, Ereco, mask) for CC response matrix. |
| `ES_background_files` | String[] | CSV files for each ES background component. |
| `CC_background_files` | String[] | CSV files for each CC background component. |
| `data_file` | String | (unused) Placeholder for real data when `Asimov: false`. |
