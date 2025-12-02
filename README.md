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

- Julia 1.6 or higher
- Python 3.7 or higher

### Julia Dependencies

Install the required Julia packages:

```julia
using Pkg
Pkg.add([
    "YAML", "Distributions", "JLD2", "HDF5", "PDMats", "Plots", "StatsBase", "BAT", "DensityInterface",
    "IntervalSets", "CSV", "DataFrames", "StaticArrays", "Interpolations", "QuadGK",
    "DataStructures", "ArraysOfArrays", "StructArrays", "ElasticArrays"
])
```

### Python Dependencies

Install the required Python packages:

```bash
pip install numpy matplotlib seaborn mplhep cmasher scipy corner h5py pandas
```

### C++ Dependencies

Install **nuFast-Earth**:  
<https://github.com/PeterDenton/NuFast-Earth>

After building the project, copy the compiled library into:

`src/oscillations/`

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
RunMode: "MCMC"          # "MCMC", "LLH", or "derived"
Asimov: true             # true for simulated data, false for real data
singleChannel: false     # false (both), "ES", or "CC"

# MCMC settings
nChains: 10              # Number of parallel chains
nSteps: 15000            # Steps per chain

# Detector configuration
CC_exposure: 40          # Module-years
ES_exposure: 40          # Module-years
```

### 3. Output Visualization

#### MCMC Results

```bash
python3 utils/plotOutput.py --help
python3 utils/plotOutput.py outFileName
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

Post-process existing MCMC chains to append the Day-Night asymmetry at each step (not done by default due to its heavy computational cost):

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

## Performance Tips

- Use multiple threads: `julia -t auto` or `julia -t N` . **NOT COMPATIBLE WITH NUFAST**
- Enable fast mode: `fastFit: true` in config
- Adjust MCMC parameters based on available compute time
- Use proposal matrices from previous runs for faster convergence


## Acknowledgments

- Thanks to Philipp Eller for providing the Newthrino source code, upon which the julia oscillation calculations are based