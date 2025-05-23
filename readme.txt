Oscillation calculator and fitter for solar neutrino oscillations at DUNE

Julia dependencies:

-YAML
-Distributions
-JLD2
-PDMats
-Plots
-StatsBase
-BAT
-DensityInterface
-IntervalSets
-CSV
-DataFrames
-StaticArrays
-Interpolations
-QuadGK
-DataStructures
-ArraysOfArrays
-StructArrays

Install via:

using Pkg
Pkg.add([
    "YAML", "Distributions", "JLD2", "PDMats", "Plots", "StatsBase", "BAT", "DensityInterface",
    "IntervalSets", "CSV", "DataFrames", "StaticArrays", "Interpolations", "QuadGK",
    "DataStructures", "ArraysOfArrays", "StructArrays"
])

Python dependencies:

-numpy
-matplotlib
-seaborn
-mplhep
-cmasher
-scipy
-corner
-h5py
-pandas

Install via:

pip install numpy matplotlib seaborn mplhep cmasher scipy corner h5py pandas

################################################################################################
################################################################################################


Usage: julia -t nProcesses src/readConfig.jl path_to_config.yaml (will default to ./config.yaml)

Display outputs:

   MCMC: python3 utils/plotOutput.py --options (see --help) outFileName(same string as used in config.yaml; no suffixes)
   LLH:  python3 utils/plotLLH.py --options (see --help) outFileName(same string as used in config.yaml; no suffixes)
