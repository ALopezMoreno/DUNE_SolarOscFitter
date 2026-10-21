# Configuration Reference

All parameters are read from `config.yaml` by `src/readConfig.jl`. Optional keys have defaults shown; required keys will raise an error if absent. Unknown keys (e.g. a misspelling) are an error that names the closest known key; the accepted keys are listed in `DETECTOR_CONFIG_KEYS` and `TOP_LEVEL_CONFIG_KEYS` in `src/readConfig.jl`, and a new key must be added there. Obsolete keys (`proposal_matrix`, `binnedNight`, `nuFast`) are accepted with a warning.

---

## Output and run type

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `outFile` | String | required | Output file prefix. Results written as `{outFile}.jld2` (MCMC), `{outFile}_configSettings.txt`, etc. Every run also writes the run record `{outFile}_config.yaml` (see "Run record" below). |
| `prevFile` | String | `nothing` | Previous run output prefix. Required when `RunMode: derived`. If set in MCMC mode, resumes chains from `{prevFile}_chainState.bin`. |
| `RunMode` | String | required | `"MCMC"`, `"LLH"`, `"derived"`, `"PROFILE"`, or `"SETUP"` (build the likelihood and stop). Selects the analysis script. |
| `Asimov` | Bool | `true` | `true` = use theoretical prediction as data. `false` = read real data (not yet implemented: rejected with an error). `data_file` is reserved for it and not read. |
| `data_file` | String | — | Reserved for real data (`Asimov: false`); accepted but not read yet. |
| `fastFit` | Bool | required | `true` = mean solar production density (and NumOsc.Fast when `nightOsc: false`). `false` = production-region integration (and NumOsc.Slow). See [oscillation_physics.md §5](oscillation_physics.md). |
| `nightOsc` | Bool | `true` | `true` = NightOsc bin-averaged night Earth crossing. `false` = legacy 3-flavour per-path NumOsc. Was `binnedNight`, which is still read with a warning. See [oscillation_physics.md §7](oscillation_physics.md). |
| `ad_backend` | String | `"Enzyme"` if `nightOsc`, else `"ForwardDiff"` | Gradient backend for HMC: `"Enzyme"` (reverse mode) or `"ForwardDiff"`. |
| `mcmc_seed` | Int | random | Seed for the MCMC random number generator, for reproducible chains. |
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

## Config inheritance (`base:`)

A config can build on other config files instead of repeating them:

```yaml
base: [configs/physics.yaml, configs/detectors/VD.yaml]   # or a single path
outFile: "outputs/es_only"
detectors:
  VD:
    singleChannel: "ES"      # only this key of VD changes
```

Paths in `base:` are relative to the file that names them; the bases may have bases of their own.
The bases are merged in order and the file itself goes on top: mappings (such as `detectors:` or a
detector block) are merged key by key at every depth, the later file wins, and any other value,
a list included, is replaced whole; so is a key set to null (`key:` with no value) rather than
inherited. An override cannot delete a key. A file reached twice, as when two bases share a base,
is applied once, where it is first reached, so it cannot undo an override made after it. A config
without `base:` is read as before. The run record's `config:` section holds the merged result, and its provenance
lists every base file with its SHA-256 (`config_base_files`). `src/configFiles.jl` implements it;
[tutorial_example/](../tutorial_example/README.md) is a worked example.

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
| `prior_cc_xsec_tilt` | Prior on the CC cross-section tilt, applied as `exp(tilt·p₁)`, `p₁ = L − ⟨L⟩`, `L = ln(E/E_pivot)` (orthogonal basis, see [mcmc_and_priors.md](mcmc_and_priors.md)). Default: `Normal(0.0, 0.1)`. |
| `prior_cc_xsec_curv` | Prior on the CC cross-section curvature, applied as `exp(curv·p₂)`, `p₂ = L² − a·L − b` (orthogonal to 1 and p₁). Default: `Normal(0.0, 0.05)`. |
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
| `warm_start_chain` | String | (optional) `outFile` of an earlier MCMC run with the same fitted parameters and priors. Its samples set the mass matrix (their covariance in BAT's prior-normal space), which is then kept fixed, and the new chains start from its draws. The step size is still tuned. Ignored when resuming with `prevFile`. See [mcmc_and_priors.md](mcmc_and_priors.md#4-tuning). |
| `proposal_matrix` | String | (unused) Setting it only logs a warning; use `warm_start_chain`. |

---

## LLH scan parameters

| Key | Type | Description |
|-----|------|-------------|
| `llh_bins` | Int | Bins per parameter dimension for 2D likelihood scans (`RunMode: LLH`). |
| `llh_range_sin2_th12` | [Float, Float] | (default `[0.2, 0.4]`) Scanned sin²θ₁₂ range. |
| `llh_range_sin2_th13` | [Float, Float] | (default `[1e-4, 0.25]`) Scanned sin²θ₁₃ range. |
| `llh_range_dm2_21` | [Float, Float] | (default `[4e-5, 1.5e-4]`) Scanned Δm²₂₁ range [eV²]. |
| `nuisance` | `false`, Int or `{top: k, method: laplace\|grid, points: 2\|3}` | (default `false`) The parameters not scanned. `false` holds them at their true values (a conditional scan). `N` averages the likelihood of each grid point over `N` throws of the nuisances from their priors, the same throws for every point: a Monte Carlo marginalisation, noisy unless `N` is large. Thrown: sin²θ₁₃ when not scanned, the fluxes, CC cross section, background norms and Earth layers (jointly, with their covariance). `{top: k}` ranks the nuisances by their share of the sin²θ₁₂ and Δm²₂₁ variance in the Fisher matrix at the truth (printed in the log) and marginalises the top `k`: `method: laplace` (default) maximises log L + log π over them at each point (damped Newton, warm-started from the neighbouring point) and adds −½ log det of the Hessian, which reproduces the posterior widths (tutorial, `top: 3`: 20 × 20 in 8.5 min on 10 threads); `method: grid` averages them on a fixed grid (prior mean and ±√3σ, or ±1σ with `points: 2`), which is too narrow when the data pin a nuisance more tightly than its prior. The rest stay at their true values, and so does the other solar oscillation parameter; grid rows run on all threads. |

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
| `mc_stat_likelihood` | String | (default `Poisson`) MC statistical uncertainty of the predictions in the likelihood: `Poisson` (none), `DembinskiAbdelmotteleb` (recommended) or `BarlowBeeston` (Conway's approximation). Can also be set at the top level as the default for all detectors. See [likelihood.md](likelihood.md). |
| `mc_stat_day_night` | String | (default `shared`) `shared`: one MC scale factor per E_reco bin for its day and all night bins, which come from the same MC events; `independent`: one per bin, as MaCh3 does. Top level or per detector. |
| `CC_bg_reference_flux` | Float | (default `2.2e-6` cm⁻²s⁻¹) Flux at which the CC background MC is normalised; each CC background is scaled by its normalisation divided by this. |
| `CC_xsec_scale` | Float | (default `1.0`) Fixed scale factor on the CC cross section relative to the table in `cc_xsec_file` (default MARLEY 2.0.0). Applied to both Asimov data and prediction (unlike `cc_xsec_norm`, which is a fitted nuisance). Note for stress tests: the DEAP-3600 measurement (arXiv:2605.12769), 2.4 (+1.3/−1.0), is quoted relative to BGG2009 / MARLEY 1.2 (the old table); relative to the MARLEY 2.0.0 default it corresponds to ≈ 2.6–2.7 (2.4 / 0.902 = 2.66, with 0.902 the ⁸B-folded MARLEY 2.0.0 / 1.2 ratio over the DEAP 12–14.5 MeV window; errors scale alike, +1.4/−1.1). |

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
| `solar_model_file` | String | JLD2 file with solar density profile and ⁸B/HEP production fractions (keys `radii` [R☉], `n_e` [N_A/cm³], `prodFractionBoron`, `prodFractionHep`; the weights are normalised by the fitter). Shipped: `inputs/SF3_GS98.jld2` (SF-III GS98, high-Z) and `inputs/SF3_AGSS09met.jld2` (SF-III AGSS09met, low-Z), built by `utils/make_solar_model_inputs.jl` from Zenodo 10822316 and carrying the model's ⁸B/hep fluxes and the source hashes. The older `AGSS09_high_z.jld2` and `GS98_low_z.jld2` are B16-AGSS09met (low-Z) and B16-GS98 (high-Z): their names have the metallicity swapped. |
| `flux_model_file` | String | JLD2 file with neutrino energy spectra (`energies` in GeV, `flux8B`, `fluxHep`). Validated (finite, strictly increasing energies, flux ≥ 0) and never extrapolated (SPEC 8.1). |
| `earth_model_file` | String | PREM Earth density profile (plain text, no header: r/R⊕, density, electron fraction). Validated (finite, strictly increasing radii, values ≥ 0) and never extrapolated. |
| `earth_uncertainty_file` | String | JLD2 file with Earth layer density uncertainties. |
| `earth_normalisation_prior_file` | String | JLD2 file with `covariance` matrix and `true` vector for Earth density nuisance priors. Required if `earth_potential_uncertainties: true`. |
| `solar_exposure_file` | String | CSV with `#` comments and the header `cosz,time_weight,flux_weight`, made by `utils/make_solar_exposure.py` (shipped: `inputs/solarExposure_v2.csv`). `time_weight` (fraction of time per unit cos z) sets the backgrounds' day/night split and night cos z weights; `flux_weight` (× (1 AU/r)²) the signal's (SPEC 4.6). f_night = ∫_{cos z<0} / ∫_all, f_day = 1 − f_night, so it must reach both sides of cos z = 0. A legacy two-column file is an error. Validated (finite, strictly increasing cos z, weights ≥ 0) and never extrapolated. |
| `cc_xsec_file` | String | (default `inputs/CC_nue_40Ar_total_marley_v2.0.0.csv`) Total CC ν_e–⁴⁰Ar cross-section table: `#` comment lines, one header row, then `E_nu [MeV], sigma [1e-42 cm² per ⁴⁰Ar]`, strictly increasing energies, finite, ≥ 0 (otherwise setup stops with an error). Interpolated linearly and never extrapolated: `range_Etrue` must lie inside the table (an error otherwise). A relative path is taken from the working directory if it exists there, else from the repository root. The default is MARLEY 2.0.0 (HF-CRPA + Bhattacharya 2009, 0–30 MeV; provenance in the file header). The previous table, MARLEY 1.2 with Bhattacharya 2009 (`inputs/CC_nue_40Ar_total_marley.csv`, to 15.5 MeV, no header), is kept for reference but cannot be used as is: it has no header and does not cover `range_Etrue` up to 21 MeV. |
| `reconstruction_sample_ES_nue` | String | CSV file (Etrue, Ereco, mask) for νₑ ES response matrix. |
| `reconstruction_sample_ES_nuother` | String | CSV file for νμ,τ ES response matrix. |
| `reconstruction_sample_ES_angle` | String | CSV file (Ereco, cos_scatter) for ES angular response matrix. |
| `reconstruction_sample_CC` | String | CSV file (Etrue, Ereco, mask) for CC response matrix. |
| `ES_background_files` | String[] | CSV files for each ES background component. |
| `CC_background_files` | String[] | CSV files for each CC background component. |

## Run record (`{outFile}_config.yaml`)

Every run writes `{outFile}_config.yaml` before the RunMode script starts (src/runRecord.jl), so a
chain can be traced back to what produced it:

- `provenance`: config file (absolute path, SHA-256), git commit and branch, `git_dirty` (tracked
  files differ from the commit; untracked files are ignored) with the list of changed files,
  Julia version, host, threads, RunMode, working directory, local timestamp, and `input_sha256`:
  the SHA-256 of every input file the config names (most of `inputs/` is not in git).
- `defaults_applied`: the keys absent from the config file and the values used for them.
- `config`: the configuration actually used (the file plus `defaults_applied`).
  `YAML.load_file("{outFile}_config.yaml")["config"]` is a complete config for a re-run. With
  `mcmc_seed` unset the seed actually drawn is not recorded, so such a re-run is statistically,
  not bitwise, equivalent.

An existing record is moved to `{outFile}_config.old<k>.yaml`, never overwritten. MCMC runs also
store the same text in the chain file, as the string `run_config` (`run_config_2`, ... when a
resumed run appends to a file that already has one). A new MCMC run whose `{outFile}.jld2` exists
stops before writing anything. Derived mode writes into `{prevFile}.jld2` (`outFile` is unused), so
its record is `{prevFile}_derived_config.yaml`; it is not added to the chain file.
