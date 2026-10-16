# Configuration Reference

All parameters are read from `config.yaml` by `src/readConfig.jl`. Optional keys have defaults shown; required keys will raise an error if absent.

---

## Output and run type

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `outFile` | String | required | Output file prefix. Results written as `{outFile}_mcmc.jld2`, `{outFile}_configSettings.txt`, etc. |
| `prevFile` | String | `nothing` | Previous run output prefix. Required when `RunMode: derived`. If set in MCMC mode, resumes chains from `{prevFile}_chainState.bin`. |
| `RunMode` | String | required | `"MCMC"`, `"LLH"`, `"derived"`, `"PROFILE"`, or `"SETUP"` (build the likelihood and stop). Selects the analysis script. |
| `Asimov` | Bool | required | `true` = use theoretical prediction as data. `false` = read real data (not yet implemented). |
| `fastFit` | Bool | required | `true` = mean solar production density (and NumOsc.Fast when `nightOsc: false`). `false` = production-region integration (and NumOsc.Slow). See [oscillation_physics.md §5](oscillation_physics.md). |
| `nightOsc` | Bool | `true` | `true` = NightOsc bin-averaged night Earth crossing. `false` = legacy 3-flavour per-path NumOsc. Was `binnedNight`, which is still read with a warning. See [oscillation_physics.md §7](oscillation_physics.md). |
| `singleChannel` | Bool or String | `false` | `false` = both channels. `"ES"` = ES only. `"CC"` = CC only. |
| `inclusiveMode` | Bool | `false` | `true` = fold CC signal into ES Ereco bins; disable separate CC likelihood. |
| `semiInclusiveMode` | Bool | `false` | `true` = forward hemisphere treated inclusively (CC folded into ES bins) while the backward hemisphere keeps a separate CC likelihood. Requires `singleChannel: false`; mutually exclusive with `inclusiveMode`. |

---

## Multi-detector configuration

Detector-level keys can be grouped under a `detectors:` section, one named block per detector, instead of being placed flat at the top level:

```yaml
detectors:
  HD:
    ES_exposure: 40
    CC_exposure: 40
    # ... all per-detector keys ...
active_detectors: ["HD"]   # optional; defaults to all blocks in detectors:
```

Per-detector keys are: `singleChannel`, `inclusiveMode`, `semiInclusiveMode`, `use_scattering_info`, `ES_cos_cut`, the exposures, the reconstruction-sample and background file lists with their normalisations/systematics, and the Ereco binning/threshold keys (`nBins_Ereco_*`, `range_Ereco_*`, `Ereco_min_*`, `nBins_cos_scatter`). A flat config without a `detectors:` section is parsed as a single detector named `default`.

---

## Asimov true values

Used only when `Asimov: true` to generate the fake dataset.

| Key | Type | Description |
|-----|------|-------------|
| `true_sin2_th12` | Float | True sin²θ₁₂ (dimensionless) |
| `true_sin2_th13` | Float | True sin²θ₁₃ (dimensionless) |
| `true_dm2_21` | Float | True Δm²₂₁ (eV²) |
| `true_integrated_HEP_flux` | Float | True HEP flux (ν/cm²/s) |
| `true_8B_flux` | Float | (optional) True ⁸B flux (ν/cm²/s). Defaults to `mean(prior_8B_flux)`. Set it away from the prior centre to study bias from flux mis-expectation. |
| `true_cc_xsec_norm` | Float | (optional, default `1.0`) True CC cross-section normalisation. |

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
| `prior_cc_xsec_norm` | Prior on the CC cross-section normalisation (relative to MARLEY). Default: `truncated(Normal(1.0, 0.1), 0.1, 2.0)`. |
| `prior_cc_xsec_tilt` | Prior on the CC cross-section power-law tilt α₁, applied as `exp(α₁·log(E/E_pivot))`. Default: `Normal(0.0, 0.1)`. |
| `prior_cc_xsec_curv` | Prior on the CC cross-section quadratic curvature α₂, applied as `exp(α₂·log²(E/E_pivot))`. Default: `Normal(0.0, 0.05)`. |
| `fix_cc_xsec` | Bool (default `false`). `true` = fix the three CC cross-section shape parameters at their true values instead of fitting them. |

---

## MCMC parameters

| Key | Type | Description |
|-----|------|-------------|
| `nChains` | Int | Number of parallel HMC chains. Recommended: equal to available cores. |
| `nSteps` | Int | Post-tuning steps per chain. Total samples = nChains × nSteps. |
| `maxTuningAttempts` | Int | Maximum adaptive tuning rounds. `0` = skip tuning. |
| `nTuning` | Int | Steps per tuning round. |
| `earth_potential_uncertainties` | Bool | Include Earth density nuisance parameters. Requires `earth_normalisation_prior_file`. |
| `hmc_max_depth` | Int | (default `4`) NUTS tree-depth cap: each MCMC step uses at most 2^`hmc_max_depth` gradient evaluations. Each batch logs the gradients actually used per step (with the Enzyme backend) and warns when steps run into the cap, which cuts trajectories short of their natural end and lowers the effective sample size. |
| `proposal_matrix` | String | (optional, currently unused) Path to a JLD2 file with a `posterior_cov` matrix. The HMC sampler tunes its own metric, so setting it only logs a warning. |

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
| `ES_background_systematics` | Float[] | Fractional uncertainty on each ES background. `0` = fixed (no nuisance parameter). If the normalisation/systematic arrays are shorter than the file list, only the first N files are used (with a warning). A nonzero systematic needs a normalisation > 0. |
| `CC_background_systematics` | Float[] | Fractional uncertainty on each CC background. |
| `module_mass_kt` | Float | (default `10`) Fiducial mass of one module [kt]. Sets the signal target counts (from the LAr molar mass) and scales the CC background MC, which is normalised per kt-year. |
| `ES_bg_generated_events` | Float | (default `50e6`) Events generated for each ES background MC file; each file's weighted event count over this is the fraction that reaches the detector. |
| `ES_bg_face_areas_m2` | [Float, Float] | (default `[1536, 288]`, i.e. 2·12·64 and 2·12·12 m²) Exposed area of the long faces (`side` = 0) and the end caps (`side` = 1); files without a `side` use the sum. Multiplies the ES background flux normalisation. |
| `mc_stat_likelihood` | String | (default `Poisson`) MC statistical uncertainty of the predictions in the likelihood: `Poisson` (none), `DembinskiAbdelmotteleb` (recommended) or `BarlowBeeston` (Conway's approximation). Can also be set at the top level as the default for all detectors. See [mcstat/mcstat.pdf](mcstat/mcstat.pdf). |
| `mc_stat_day_night` | String | (default `shared`) `shared`: one MC scale factor per E_reco bin for its day and all night bins, which come from the same MC events; `independent`: one per bin, as MaCh3 does. Top level or per detector. |
| `CC_bg_reference_flux` | Float | (default `2.2e-6` cm⁻²s⁻¹) Flux at which the CC background MC is normalised; each CC background is scaled by its normalisation divided by this. |
| `CC_xsec_scale` | Float | (default `1.0`) Fixed scale factor on the CC cross section relative to the MARLEY prediction. Applied to both Asimov data and prediction (unlike `cc_xsec_norm`, which is a fitted nuisance). |

---

## Energy and angle binning

| Key | Type | Description |
|-----|------|-------------|
| `nBins_Etrue` | Int | Number of true-energy bins for oscillation calculation. |
| `range_Etrue` | [Float, Float] | True energy range [MeV]. Converted to GeV internally. |
| `nBins_Ereco_ES` | Int | Number of fitted ES reco-energy bins, uniform over [`Ereco_min_ES`, `range_Ereco_ES[2]`]. One extra bin below the threshold collects everything reconstructed below it (never fitted); the last bin also collects everything above the range. |
| `nBins_Ereco_CC` | Int | Number of fitted CC reco-energy bins (as for ES). |
| `range_Ereco_ES` | [Float, Float] | ES reco energy range [MeV]. Only the upper end is used: the binning starts at `Ereco_min_ES`. |
| `range_Ereco_CC` | [Float, Float] | CC reco energy range [MeV]. Only the upper end is used. |
| `Ereco_min_ES` | Float | ES analysis threshold [MeV]. The fitted bins start exactly here (see `nBins_Ereco_ES`). |
| `Ereco_min_CC` | Float | CC analysis threshold [MeV]; the fitted CC bins start exactly here. |
| `TPC_threshold` | Float | (default `0.2`) True electron-recoil detection threshold [MeV]; lower integration limit of the ES cross-section. Must match the MC generation threshold. Distinct from the `Ereco_min_ES` analysis cut. |
| `nBins_cosz` | Int | Number of cos(zenith) bins for day/night split (range −1 to 0). |
| `nBins_cosz_fine` | Int | (default `0`) Horizon-refined cos(z) binning: number of graded bins in `[cosz_fine_edge, 0]`, concentrated where the day-night regeneration first peak lives; the rest of the grid is PREM-aligned. `0` = legacy uniform-piecewise grid. |
| `cosz_fine_edge` | Float | (default `−0.12`) Lower edge of the fine cos(z) zone. Should sit at ≈ the first-peak cos(z) for the slowest oscillation frequency (lower bound of `prior_dm2_21`). |
| `nBins_cos_scatter` | Int | Number of cos(scatter angle) bins for ES angular reconstruction. |
| `use_scattering_info` | Bool | `true` = enable angular reconstruction likelihood. Changes ES rate shapes from 1D/2D to 2D/3D. Required by `semiInclusiveMode`. |
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
