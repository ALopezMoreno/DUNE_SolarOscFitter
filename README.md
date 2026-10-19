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
- Python 3.7 or higher

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
            networkx corner Pillow termcolor peanuts
```

`plotting/` (a package: shared style, corner plots, contours, layouts), `style.py`, `posteriorHelpers.py`, and `binImportanceHelpers.py` are local modules inside `utils/`. Run plotting scripts from within that directory, or set `PYTHONPATH`:

```bash
cd utils && python plotOutput.py ...
# or from anywhere:
PYTHONPATH=/path/to/SolarOscFitter/utils python utils/plotOutput.py ...
```

## Quick Start

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
"did you mean" hint, so a typo cannot silently fall back to a default. See [docs/configuration_reference.md](docs/configuration_reference.md) for every key, and [docs/index.md](docs/index.md) for the developer documentation.

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

At setup, every detector's input samples are checked: a per-file table (rows, selected rows,
weight range, events in the fit window) is printed, with warnings for pre-binned samples,
signal and ES background samples with every event selected, and identical files. A per-bin breakdown of data,
signal and each background component at the true parameters is written to
`<outFile>_<DET>_<ES|CC>_breakdown.csv`.


The analysis requires several input files:

- **Solar model**: `inputs/SF3_GS98.jld2` (SF-III GS98, high-Z; `inputs/SF3_AGSS09met.jld2` is the low-Z alternative)
- **Neutrino fluxes**: `inputs/fluxes.jld2`
- **Earth model**: `inputs/EARTH_MODEL_PREM_DENSE.dat` (a PREM-like table without crust; the Earth
  modelling is being reworked)
- **CC ν_e–⁴⁰Ar cross section**: `inputs/CC_nue_40Ar_total_marley_v2.0.0.csv` (MARLEY 2.0.0, HF-CRPA + Bhattacharya 2009; config key `cc_xsec_file`)
- **Detector response**: event-level Monte Carlo samples for the CC and ES channels (csv, one row
  per event: `Etrue`, `Ereco`, `weights`, `mask`)
- **Background samples**: event-level Monte Carlo samples for the neutron and gamma backgrounds
  (csv, one row per event: `Ereco`, `weights`, `mask`)

> **Inputs must be event-level, never pre-binned histograms.** The fitter bins the events itself;
> a histogram fed in as if it were events is re-binned onto the analysis grid and aliases (for a
> 200 keV histogram on 0.67 MeV bins, a ±20% sawtooth). Setup warns about samples that look
> pre-binned. All model tables (fluxes, cross section, exposure, Earth) are validated at load and
> never extrapolated silently ([SPEC.md](SPEC.md) 8.1).

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
reports the MC uncertainty per channel either way. Method and implementation:
[docs/mcstat/mcstat.pdf](docs/mcstat/mcstat.pdf).

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