Oscillation calculator and fitter for solar neutrino oscillations at DUNE

Usage: julia -t nProcesses src/readConfig.jl path_to_config.yaml

Display outputs:

   MCMC: python3 utils/plotOutput.py --options outFileName(same string as used in config.yaml; no suffixes)
   LLH:  python3 utils/plotLLH.py --options outFileNmae(same string as used in config.yaml; no suffixes)
