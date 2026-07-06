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

The repo includes a committed `Manifest.toml` with pinned versions. Install Julia and instantiate:

```bash
# Recommended: install Julia via juliaup
curl -fsSL https://install.julialang.org | sh
juliaup add 1.12.3 && juliaup default 1.12.3

# Instantiate exact pinned dependencies from Manifest.toml
julia --project=./ -e "using Pkg; Pkg.instantiate()"
```

If you're on a different Julia version and hit compatibility errors, run `Pkg.resolve()` before `Pkg.instantiate()` to re-solve the dependency graph.

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

### C++ Dependencies

Install **nuFast-Earth**:  
<https://github.com/PeterDenton/NuFast-Earth>

Edit the following line in the Makefile to point at your nuFast-Earth directory:
`NUFAST_DIR ?= ../NuFast-Earth`

## Quick Start

### 1. Basic Usage

Run the analysis with default configuration:

```bash
julia -t auto src/readConfig.jl
```

Or specify a custom configuration file:

```bash
julia -t 8 src/readConfig.jl path/to/your/config.yaml
```

The `-t` flag sets the number of threads for parallel processing.

### 2. Configuration

The analysis is controlled through YAML configuration files. The main `config.yaml` file contains:

- **Analysis mode**: MCMC, LLH scanning, or derived quantities
- **Data type**: Asimov (simulated) or real data
- **Detector parameters**: Exposure, thresholds, binning
- **Prior distributions**: Parameter constraints from external measurements
- **File paths**: Input data and output locations

Key configuration options:

```yaml
# Analysis mode
RunMode: "MCMC"          # "MCMC", "LLH", "derived", or "PROFILE"
Asimov: true             # true for simulated data, false for real data
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

Detector-level keys can also be grouped into named blocks under a `detectors:` section (with an optional `active_detectors` list) to fit several detector configurations simultaneously. See [docs/configuration_reference.md](docs/configuration_reference.md) for every key, and [docs/index.md](docs/index.md) for the developer documentation.

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
├── inputs/               # Input data files
├── outputs/              # Analysis results
├── configs/              # Configuration file examples
└── config.yaml           # Main configuration file
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
- `outFile_mcmc.bin`: MCMC chains
- `outFile_info.txt`: Analysis summary
- `outFile_posterior.jld2`: Posterior samples

### Likelihood Scanning

Quick 2D parameter space exploration:

```yaml
RunMode: "LLH"
llh_bins: 15
```

### Derived Quantities

Post-process existing MCMC chains to append the Day-Night asymmetry at each step, as well as by-bin likelihood contributions (not done by default due to its heavy computational cost):

```yaml
RunMode: "derived"
prevFile: "path/to/previous/analysis"
```

## Input Data Requirements

The analysis requires several input files:

- **Solar model**: `inputs/AGSS09_high_z.jld2`
- **Neutrino fluxes**: `inputs/fluxes.jld2`
- **Earth model**: `inputs/EARTH_MODEL_PREM_DENSE.dat`
- **Detector response**: Monte Carlo samples with E_truth and E_reco for CC and ES channels (csv)
- **Background samples**: Monte Carlo samples with E_reco for Neutron and gamma backgrounds (csv)

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
- Earth matter density uncertainties
- Background normalizations
- Detector response uncertainties
- CC cross-section shape uncertainties (relative to MARLEY nominal):
  - `cc_xsec_norm` — overall scale; default `truncated(Normal(1.0, 0.1), 0.1, 2.0)`
  - `cc_xsec_tilt` — power-law spectral slope, applied as `exp(α₁ × log(E/E_pivot))`; default `Normal(0, 0.1)`
  - `cc_xsec_curv` — quadratic spectral curvature, applied as `exp(α₂ × log²(E/E_pivot))`; default `Normal(0, 0.05)`

  `E_pivot` is the flux-weighted mean CC event energy; by construction `cc_xsec_tilt` is orthogonal to `cc_xsec_norm`, and `cc_xsec_curv` is orthogonal to both.

> ⚠️ **Warning: Delete Existing Output Files Before Rewriting**
>
> When regenerating results, make sure to **delete any existing output files** first.
> Some serialization and output routines may **not cleanly overwrite** previous files,
> which can lead to corrupted, partially overwritten, or unexpectedly appended data.
>
> To avoid inconsistent results, always remove old output files manually before running
> a job that rewrites them.


## Testing

Run the unit tests (likelihood correctness and wiring) with the project environment:

```bash
julia --project tests/runtests.jl
```

Benchmarks and profiling runs are opt-in via `JULIA_LIKELIHOOD_BENCH=1` and `JULIA_PROFILE=1`.


## Performance Tips

- Use multiple threads: `julia -t auto` or `julia -t N` . **NOT COMPATIBLE WITH NUFAST**
- Enable fast mode: `fastFit: true` in config
- Adjust MCMC parameters based on available compute time
- Use proposal matrices from previous runs for faster convergence


## Acknowledgments

- Thanks to Philipp Eller for providing the Newthrino source code, upon which the julia oscillation calculations are based