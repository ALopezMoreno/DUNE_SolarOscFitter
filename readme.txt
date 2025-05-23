Oscillation calculator and fitter for solar neutrino oscillations at DUNE

Usage: julia -t nProcesses src/readConfig.jl path_to_config.yaml

Display outputs:

   MCMC: python3 utils/plotOutput.py --options (see --help) outFileName(same string as used in config.yaml; no suffixes)
   LLH:  python3 utils/plotLLH.py --options (see --help) outFileName(same string as used in config.yaml; no suffixes)
