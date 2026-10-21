# SolarOscFitter

A Julia-based oscillation calculator and fitter for solar neutrino oscillations at the Deep Underground Neutrino Experiment (DUNE).

## Overview

SolarOscFitter is a comprehensive analysis framework for studying solar neutrino oscillations using the DUNE detector. It provides tools for:

- **Bayesian parameter estimation** using Markov Chain Monte Carlo (MCMC)
- **Likelihood scanning** for quick parameter space exploration
- **Detector response modeling** for both Elastic Scattering (ES) and Charged Current (CC) channels
- **Earth matter effect calculations** with systematic uncertainties
- **Comprehensive plotting**: Python utilities for result visualization

## Installation

### Prerequisites

- Julia 1.12.3 (pinned in `Manifest.toml`)
- Python 3.10 or higher (the plotting package uses `X | None` annotations)

### Julia Dependencies

The repo includes a committed `Manifest.toml` with the exact package versions the results were produced with, and `[compat]` bounds in `Project.toml`. Install Julia and instantiate:

```bash
# Recommended: install Julia via juliaup
curl -fsSL https://install.julialang.org | sh
juliaup add 1.12.3 && juliaup default 1.12.3

# Instantiate exact pinned dependencies from Manifest.toml
julia --project=./ -e "using Pkg; Pkg.instantiate()"
```

If you're on a different Julia version and hit compatibility errors, run `Pkg.resolve()` before `Pkg.instantiate()` to re-solve the dependency graph. Commit `Manifest.toml` whenever you add or update a package, so that runs (local, cluster, git worktrees) use the same versions.

### Python Dependencies

Install the required Python packages:

```bash
pip install matplotlib numpy pandas scipy h5py seaborn mplhep cmasher cmap \
            networkx corner Pillow termcolor peanuts pyyaml
```

`plotting/` (a package: shared style, corner plots, contours, layouts), `style.py`, `posteriorHelpers.py`, and `binImportanceHelpers.py` are local modules inside `utils/`. Run plotting scripts from within that directory, or set `PYTHONPATH`:

```bash
cd utils && python plotOutput.py ...
# or from anywhere:
PYTHONPATH=/path/to/SolarOscFitter/utils python utils/plotOutput.py ...
```

## Quick Start

New here? [TUTORIAL.md](TUTORIAL.md) runs a complete fit on idealised inputs that a fresh clone
can build itself. Most of `inputs/` is not in git, so the production `config.yaml` needs the
detector samples from the analysis group.

### 1. Basic Usage

Run the analysis with the default configuration (`config.yaml` in the repository root):

```bash
julia -t auto --project src/readConfig.jl
```

Or specify a configuration file:

```bash
julia -t 8 --project src/readConfig.jl path/to/your/config.yaml
```

`--project` makes Julia use this repository's pinned environment (without it, Julia uses your
default environment, which may not have the packages). `-t` sets the number of threads.
Add `--plot` to plot the configuration's inputs instead of running it (see "Inputs as the fitter
sees them" below).

### 2. Configuration

The analysis is controlled through YAML configuration files. The main `config.yaml` file contains:

- **Analysis mode**: MCMC, LLH scanning, derived quantities, profiling, or setup only
- **Data type**: Asimov (simulated) data; fitting real data is not implemented yet
- **Detector parameters**: Exposure, thresholds, binning
- **Prior distributions**: Parameter constraints from external measurements
- **File paths**: Input data and output locations

Key configuration options:

```yaml
# Analysis mode
RunMode: "MCMC"          # "MCMC", "LLH", "derived", "PROFILE", or "SETUP" (build the likelihood and stop)
Asimov: true             # the only data mode for now; Asimov: false stops with "not implemented"
singleChannel: false     # false (both), "ES", or "CC"

# MCMC settings
nChains: 10              # Number of parallel chains
nSteps: 15000            # Steps per chain

# Detector configuration
CC_exposure: 40          # Module-years
ES_exposure: 40          # Module-years

# Shared binning
TPC_threshold: 0.2       # True electron-recoil detection threshold (MeV); lower
                         # integration limit of the ES cross-section. Must match
                         # the MC generation threshold (distinct from Ereco_min_ES).

# Asimov truth overrides
true_8B_flux: 5.53e6     # Asimov 8B flux; defaults to mean(prior_8B_flux) if omitted.
                         # Set != prior centre to study bias from flux mis-expectation.
```

The example above is a flat (single-detector) config. Detector-level keys (`singleChannel`,
`CC_exposure`, binning, input samples, backgrounds, ...) can instead be grouped into named blocks
under a `detectors:` section (with an optional `active_detectors` list) to fit several detector
configurations simultaneously; the root `config.yaml` does this. With a `detectors:` section,
detector-level keys must go inside each block: unknown or misplaced keys stop the run with a
"did you mean" hint, so a typo cannot silently fall back to a default. A config can also build on
other config files with a `base:` key, changing only what differs; [tutorial_example/](tutorial_example/README.md)
is a complete example, with idealised inputs, split into subconfigs this way. See [docs/configuration_reference.md](docs/configuration_reference.md) for every key, and [docs/index.md](docs/index.md) for the developer documentation.

### 3. Output Visualization

#### MCMC Results

```bash
python3 utils/plotOutput.py --help
python3 utils/plotOutput.py outFileName
```

Both `plotOutput.py` and `compareOutput.py` apply a Gaussian smoothing of
`--smooth` sigma (in bins, default 0.5) to the 2D histograms before drawing
contours. This affects the displayed credible regions; the value used is
printed at startup. Set `--smooth 0` for raw histograms.

#### Compare two posteriors (overlaid corner)

```bash
python3 utils/compareOutput.py --help
python3 utils/compareOutput.py -a chainsA -b chainsB --labels "Run A" "Run B"
```
#### Inputs as the fitter sees them

To look at a configuration's inputs before fitting, call the fitter with `--plot`. It plots
instead of running: it builds the likelihood (as `RunMode: SETUP`) under the prefix
`<outFile>_inputs`, writes the plots to `images/<name>_inputs_<detector>.pdf` (`<name>` is the last
part of `outFile`) and stops.

```bash
julia -t auto --project src/readConfig.jl path/to/config.yaml --plot
```

The plots show one detector's inputs at the Asimov truth:
- the solar fluxes (⁸B, hep) and the ES and CC cross sections the fitter evaluates;
- the oscillated true-energy spectra (all interactions and the part above threshold), P_ee by day
  and night, and the regeneration;
- signal and each background per E_reco bin in the fit window, with S/B per bin;
- each background spectrum on its own;
- the response matrices P(E_reco | E_true) as binned by the fitter, and the fraction above threshold;
- the day-night asymmetry of P_ee, 2(N − D)/(N + D) (positive for regeneration), by zenith and
  E_true bin;
- the raw event-level samples, the background samples and the selection efficiency.

1D quantities are drawn as filled histograms and 2D ones as filled contours, all in viridis. The flux
and cross-section page needs a run made with the current fitter, which stores them.

`utils/plotInputs.py` makes them, and can also be run on the outputs of any earlier run (run it
from the repository root, so that the input paths in the run record resolve):

```bash
python3 utils/plotInputs.py outFileName [--detector HD] -o images/my_inputs   # → images/my_inputs.pdf
```

#### Bin diagnostic results (derived mode)

```bash
python3 utils/plotBinDiagnostics.py --help
python3 utils/plotBinDiagnostics.py outFileName
```

#### Likelihood Scans

```bash
python3 utils/plotLLH.py --help
python3 utils/plotLLH.py outFileName
```

## Project Structure

```
SolarOscFitter/
├── src/                    # Main Julia source code
│   ├── readConfig.jl      # Main entry point
│   ├── mcmc.jl           # MCMC implementation
│   ├── oscillations/     # Neutrino oscillation calculations
│   └── ...
├── utils/                 # Python plotting utilities
│   ├── plotOutput.py     # MCMC result visualization
│   ├── compareOutput.py  # Compare two posteriors
│   └── plotLLH.py        # Likelihood scan plots
├── tests/                # Unit, end-to-end and regression tests
├── docs/                 # Physics and code documentation
├── inputs/               # Input data files
├── outputs/              # Analysis results
├── configs/              # Configuration file examples
├── tutorial_example/     # Tutorial: idealised VD inputs and a config split into subconfigs
├── config.yaml           # Main configuration file
└── SPEC.md               # Validation criteria (what the tests check)
```

## Analysis Modes

### MCMC (Markov Chain Monte Carlo)

Performs Bayesian parameter estimation:

```yaml
RunMode: "MCMC"
nChains: 10
nSteps: 15000
```

Outputs:
- `{outFile}.jld2`: the chains, appended in batches (`batch_1`, `batch_2`, ...), plus the run record
  (`run_config`)
- `{outFile}_chainState.bin`: sampler state, for resuming with `prevFile`
- `{outFile}_config.yaml`: the run record (resolved config, input-file hashes, git commit); an
  existing one is moved to `_config.old<k>.yaml`, never overwritten
- `{outFile}_configSettings.txt`: summary of the settings
- `{outFile}_<DET>_<ES|CC>_breakdown.csv`: per-bin data, signal and backgrounds at the truth

A new run stops if `{outFile}.jld2` already exists. To start from an earlier run's posterior
(mass-matrix warm start, much faster mixing), set `warm_start_chain` to that run's `outFile`.

### Likelihood Scanning

Quick 2D parameter space exploration:

```yaml
RunMode: "LLH"
llh_bins: 15
```

### Derived Quantities

Post-process existing MCMC chains to append the Day-Night asymmetry at each step, as well as by-bin likelihood contributions (not done by default due to its heavy computational cost). Every active detector is processed in one pass; results are stored per detector as `derived_<DET>_*` keys (shared ones, such as the best-fit oscillation map, as `derived_*`). The plotting scripts use the first detector unless given `--detector <DET>`:

```yaml
RunMode: "derived"
prevFile: "path/to/previous/analysis"   # chain prefix, without .jld2
```

Results are written into `{prevFile}.jld2` (`outFile` is not used), and the run record to
`{prevFile}_derived_config.yaml`.

## Input Data Requirements

The repository ships the physics model: solar model (`inputs/SF3_GS98.jld2`), solar fluxes
(`inputs/fluxes.jld2`), Earth density table (`inputs/EARTH_MODEL_PREM_DENSE.dat`), solar exposure
(`inputs/solarExposure_v2.csv`) and the CC ν_e–⁴⁰Ar cross section (MARLEY 2.0.0). The fitter
computes the ES cross sections itself. **An analyser provides the detector**: for each detector
in `detectors:`, event-level Monte Carlo samples for the signal response and for each background.

### What to provide per detector

| Config key | Content | Columns |
|---|---|---|
| `reconstruction_sample_ES_nue` | ν_e–e⁻ scattering events | `Etrue`, `Ereco`, `weights`, `mask` |
| `reconstruction_sample_ES_nuother` | ν_μ,τ–e⁻ scattering events (a different recoil spectrum, so a separate sample) | same |
| `reconstruction_sample_CC` | ν_e ⁴⁰Ar CC events | same |
| `reconstruction_sample_ES_angle` | ES electron direction (always read; used with `use_scattering_info: true`) | `Ereco`, `cos_scatter` |
| `ES_background_files`, `CC_background_files` | one file per background component | `Ereco`, `weights`, `mask` (ES: optional `side`) |

Rules for every file:
- **One row per simulated event, never a histogram.** The fitter bins the events itself. A
  histogram fed in as if it were events is re-binned onto the analysis grid and aliases. Setup
  warns about samples that look pre-binned.
- **Energies in GeV.** `Etrue` is the neutrino energy. `Ereco` is your reconstructed energy
  observable.
- **Keep rejected events, with `mask = false`.** For the signal, the fitter takes the response
  P(E_reco | E_true) from all events of each E_true bin, and the selection efficiency from
  selected / all events of each E_reco bin. Dropping rejected events would set the efficiency to 1.
- **Signal samples need no flux or cross-section weighting.** The fitter applies flux × σ ×
  oscillation itself. The E_ν distribution of the sample only needs to cover `range_Etrue` with
  enough events in every E_true bin; a flat distribution is fine. Non-unit signal `weights` are
  ignored, with a warning.
- **ES samples start at the TPC threshold.** Generate the electron recoils down to
  `TPC_threshold` (the lower limit of the fitter's ES cross section), not down to the analysis
  cut `Ereco_min_ES`. The analysis cut is applied by the binning.

### How backgrounds are normalised

- **ES backgrounds** (external, e.g. neutrons entering the detector): each file holds the events
  that reached the detector out of `ES_bg_generated_events` thrown on the detector faces. Expected
  events per E_reco bin = `ES_background_normalisations` [cm⁻² s⁻¹] × face area (`side` 0 = long
  faces, 1 = end caps, absent = all, `ES_bg_face_areas_m2`) × 1 year × `ES_exposure` × Σ weights of
  the selected events in the bin / `ES_bg_generated_events`.
- **CC backgrounds**: the weighted selected events are events per kt-year at the flux
  `CC_bg_reference_flux`. Expected events = Σ weights × `module_mass_kt` × `CC_exposure` ×
  `CC_background_normalisations` / `CC_bg_reference_flux`.
- `*_background_systematics` sets a Gaussian prior of that relative width on each normalisation
  (0 fixes it).

### Example: an ES ν_e sample

`tutorial_example/inputs/ES_nue_ideal.csv`, made by `tutorial_example/make_inputs.jl`. E_ν is
drawn flat in 1–21 MeV, the recoil T from dσ/dT above `TPC_threshold`, and `Ereco` = T smeared by
10%:

```
Etrue,Ereco,weights,mask
0.013494352138785793,0.007846489500269371,1.0,true
0.0017821080575948726,0.0010736813495904106,1.0,true
0.0025061557219528577,0.00046075219640512516,1.0,true
```

All events pass the selection here (an ideal detector). A real sample keeps the rejected
events, with `mask = false`. In the config:

```yaml
detectors:
  VD:
    ES_exposure: 10                  # module-years (× module_mass_kt = 100 kt-yr)
    module_mass_kt: 10
    Ereco_min_ES: 5                  # analysis cut [MeV]: the fit starts here
    range_Ereco_ES: [2, 20]          # upper edge [MeV]; events above go to the last bin
    nBins_Ereco_ES: 30
    reconstruction_sample_ES_nue:     "tutorial_example/inputs/ES_nue_ideal.csv"
    reconstruction_sample_ES_nuother: "tutorial_example/inputs/ES_numutau_ideal.csv"
    reconstruction_sample_ES_angle:   "tutorial_example/inputs/ES_angular_ideal.csv"
    ES_background_files:          ["tutorial_example/inputs/ES_background_flat.csv"]
    ES_background_normalisations: [1.86511e-11]   # cm⁻² s⁻¹
    ES_background_systematics:    [0.02]
```

### Checking the inputs

`julia --project src/readConfig.jl config.yaml --plot` builds the likelihood and plots what the
fitter made of the inputs (see "Inputs as the fitter sees them"). Every run also prints a table
per input file (rows, selected rows, weight range, events in the fit window) and writes
`<outFile>_<DET>_<ES|CC>_breakdown.csv`, with data, signal and each background per E_reco bin
at the truth. All model tables are validated at load and never extrapolated
([SPEC.md](SPEC.md) 8.1). Every key is listed in
[docs/configuration_reference.md](docs/configuration_reference.md).

## Physics Parameters

The analysis fits the following parameters:

### Oscillation Parameters
- `sin²θ₁₂`: Solar mixing angle
- `sin²θ₁₃`: Reactor mixing angle  
- `Δm²₂₁`: Solar mass-squared difference

### Flux Parameters
- `⁸B flux`: Boron-8 solar neutrino flux
- `HEP flux`: He3+p (HEP) neutrino flux

### Systematic Parameters
- Earth matter density uncertainties (per-layer normalisations)
- Background normalizations
- MC statistical uncertainty of the predictions (optional, see below)
- (Detector energy-scale and resolution uncertainties are not implemented yet.)
- CC cross-section shape uncertainties (relative to the MARLEY 2.0.0 nominal table, `cc_xsec_file`):
  - `cc_xsec_norm` — overall scale; default `truncated(Normal(1.0, 0.1), 0.1, 2.0)`
  - `cc_xsec_tilt` — spectral slope, applied as `exp(tilt × p₁)` with `p₁ = L − ⟨L⟩`, `L = ln(E/E_pivot)`; default `Normal(0, 0.1)`
  - `cc_xsec_curv` — spectral curvature, applied as `exp(curv × p₂)` with `p₂ = L² − a·L − b`; default `Normal(0, 0.05)`

  `p₁`, `p₂` are orthogonal to 1 and to each other under the unoscillated CC ⁸B rate per E_true bin, so at the nominal point tilt and curv leave the CC ⁸B rate unchanged to first order and do not correlate with `cc_xsec_norm` or each other (see [docs/mcmc_and_priors.md](docs/mcmc_and_priors.md), SPEC 3.3).

> ⚠️ **Warning: Delete Existing Output Files Before Rewriting**
>
> When regenerating results, make sure to **delete any existing output files** first.
> Some serialization and output routines may **not cleanly overwrite** previous files,
> which can lead to corrupted, partially overwritten, or unexpectedly appended data.
>
> To avoid inconsistent results, always remove old output files manually before running
> a job that rewrites them. (A new MCMC run already refuses to start over an existing
> `{outFile}.jld2`, and run records are never overwritten; other outputs, such as LLH scans and
> breakdown CSVs, are.)


## Testing

The tests come in tiers:

```bash
julia --project tests/runtests.jl                              # fast: unit tests, ~2 min
JULIA_TESTS=slow julia --project tests/runtests.jl             # slow: + Enzyme, warm start, derived mode, ~5 min
JULIA_REGRESSION=1 julia --project tests/runtests.jl           # + full-likelihood regression, ~5 min more
julia --project tests/regression_likelihood.jl night --grad    # Enzyme vs ForwardDiff on the full likelihood
```

Run the fast tier while working, the slow tier once before merging, and the regression and the
gradient check only when a change can move the likelihood value or touches the Enzyme path.

The regression compares the likelihood at fixed points with frozen values in
`tests/regression_reference.toml`; any intended change to the likelihood value needs a deliberate
`--update` of it. What each test checks, and against which tolerance, is listed in
[SPEC.md](SPEC.md).

Benchmarks and profiling runs are opt-in via `JULIA_LIKELIHOOD_BENCH=1` and `JULIA_PROFILE=1`.


## MC statistical uncertainty

The predictions come from finite MC samples. Set `mc_stat_likelihood: DembinskiAbdelmotteleb`
(or `BarlowBeeston`) to include their statistical uncertainty in the likelihood; the setup log
reports the MC uncertainty per channel either way. Method: Dembinski & Abdelmotteleb,
[arXiv:2206.12346](https://arxiv.org/abs/2206.12346); implementation in [docs/likelihood.md](docs/likelihood.md).

## Performance Tips

- Use multiple threads: `julia -t auto` or `julia -t N`
- Fast mode: `fastFit: true` uses the mean solar production density. It is accurate to ≤ 0.3% for
  ⁸B, but the hep day survival probability can be off by up to 4.5%; use `fastFit: false` for hep
  studies
- Start from an earlier run's posterior with `warm_start_chain` (mass-matrix warm start; much higher
  effective sample size per gradient than a cold start)
- Resume an interrupted run with `prevFile` instead of re-tuning
- Adjust MCMC parameters based on available compute time


## Acknowledgments

- Thanks to Philipp Eller for providing the Newthrino source code, upon which the julia oscillation calculations are based
## Upcoming plans

- **Earth model.** Replace the shipped density table with a full PREM-based model, including the
  crust and the detector depth. Add Earth parametrisations beyond per-layer scaling: boundary
  shifts, and local and large-scale density perturbations.
- **Solar spectra.** Use the ⁸B spectrum of Winter et al. (2006) with its full high-energy tail, and
  the hep spectrum of Bahcall (1997).
- **CC cross section.** Use physically motivated uncertainties instead of the ad hoc
  normalisation, tilt and curvature: a Fermi/Gamow-Teller split, per-transition MARLEY modes, and
  data-set variations. Study how well a joint ES+CC fit constrains them.
- **Detector systematics.** Add an energy scale and resolution shared by ES and CC.
- **Statistics.** Use the MC-statistics likelihood by default once all inputs are event-level.
  Add R-hat and ESS convergence diagnostics, and check the coverage of the credible intervals
  with toy data.
- **Oscillation accuracy and speed.** Move the oscillation settings into their own config file,
  with a check that tunes the calculation grids to a target accuracy.
- **Sampling.** Estimate the mass matrix from the run's own tuning (self warm start).
- **Diagnostics.** Base pulls, impacts and per-bin information in derived mode on exact gradients,
  and add oscillograms to the bin diagnostics.
- **Configs.** Convert the example configs in `configs/` to overrides of a shared base
  (`base:`).
- **Real data.** Fit measured data (`Asimov: false`) as well as Asimov data sets.
