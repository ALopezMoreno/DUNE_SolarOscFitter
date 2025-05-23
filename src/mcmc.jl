"""
This script performs Bayesian parameter estimation for neutrino oscillation parameters using Markov Chain 
Monte Carlo (MCMC) sampling. It leverages the BAT.jl library to sample from the posterior distribution 
defined by a likelihood function and prior distributions over the parameters of interest.

Modules and Libraries:
- Utilizes Julia packages such as `LinearAlgebra`, `Statistics`, `Distributions`, `StatsBase`, `BAT`, 
  `DensityInterface`, `IntervalSets`, `Plots`, and `JLD2` for mathematical operations, statistical 
  distributions, Bayesian analysis, plotting, and data storage.
- Includes a setup script from `../src/setup.jl` to configure the fitting process.

Parameters:
- `prior`: A product of distributions defining uniform priors for the squared sine of mixing angles 
  (`sin²θ₁₂`, `sin²θ₁₃`) and the squared mass difference (`Δm²₂₁`).
- `fast`: A boolean flag that determines which likelihood function to use (`likelihood_all_samples_ctr` 
  or `likelihood_all_samples_avg`).

Process:
1. Defines a Bayesian model using the `PosteriorMeasure` with the specified likelihood and prior.
2. Configures MCMC chain parameters, including initialization, burn-in, and convergence settings.
3. Executes MCMC sampling using the Metropolis-Hastings algorithm, logging the number of chains and steps.
4. Extracts parameter samples and metadata (step number and chain ID) from the MCMC results.
5. Saves the extracted data and additional sample information to a JLD2 file for further analysis.

Output:
- A JLD2 file containing arrays of sampled parameter values (`sin²θ₁₂`, `sin²θ₁₃`, `Δm²₂₁`), step numbers, 
  chain IDs, and sample data (`ES_nue`, `ES_nuother`, `CC`).

Note:
- The script assumes the existence of certain global variables such as `mcmcChains`, `mcmcSteps`, 
  `tuningSteps`, `maxTuningAttempts`, `outFile`, and `ereco_data`.
- Logging is set up but commented out; adjust the logging level as needed for debugging.
"""

using Logging

# DEBUGGING AND TESTING: Set the logging level to Warn to suppress Info messages
# global_logger(ConsoleLogger(stderr, Logging.Warn))

using LinearAlgebra, Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using Plots, JLD2

using Random
# Random.seed!(1234)  # Set global seed once at startup if you want RNG reproducibility

include("../src/setup.jl")
include("../src/mcmcHelpers.jl")


# Set prior distributions from config
# Construct a dictionary of priors
priors = Dict{Symbol,Any}(
  # miximg parameters
  :sin2_th12 => prior_sin2_th12,
  :sin2_th13 => prior_sin2_th13,
  :dm2_21 => prior_dm2_21,

  # HEP discovery
  :integrated_HEP_flux => prior_HEP_flux,

  # systematic parameters
  :integrated_8B_flux => prior_8B_flux
)


# Conditionally add nuisance parameters
if earthUncertainty
  priors[:earth_norm] = earth_normalisation_prior
end

if !isempty(ES_bg_norms_pars)
  for (i, norm) in enumerate(ES_bg_norms_pars)
      priors[Symbol("ES_bg_norm_$i")] = norm
  end
end

if !isempty(CC_bg_norms_pars)
  for (i, norm) in enumerate(CC_bg_norms_pars)
      priors[Symbol("CC_bg_norm_$i")] = norm
  end
end

# Use splatting to pass the priors to distprod
prior = distprod(; priors...)

# Define Bayesian model
if fast
  posterior = PosteriorMeasure(likelihood_all_samples_ctr, prior)
else # THIS HAS TO DISAPPEAR. WE NO DOING NO AVG!
  posterior = PosteriorMeasure(likelihood_all_samples_avg, prior)
end

# Set chain parameters
init = MCMCChainPoolInit(
  init_tries_per_chain=IntervalSets.ClosedInterval(1, 180),  # Example interval
  nsteps_init=500,
  initval_alg=InitFromTarget()
)

if maxTuningAttempts != 0
  burnin = MCMCMultiCycleBurnin(
    nsteps_per_cycle=tuningSteps,
    max_ncycles=maxTuningAttempts,
    nsteps_final=tuningSteps / 10
  )
else
  burnin = MCMCNoBurnin()
end

convergence = AssumeConvergence()

# Check if there is a covariance matrix for the step proposal function
if propMatrix !== nothing
  # Use BAT's GenericMvTDist for a multivariate t-distribution proposal.
  d = size(propMatrix, 1)
  df = 1.5
  proposal_distribution = TDist(df)
  @logmsg MCMC "found proposal covariance matrix: using MvTDist with df = $df."
else
  # Default to a univariate T distribution if no covariance matrix is provided
  proposal_distribution = TDist(1.0)
  @logmsg MCMC "no proposal covariance matrix given. Running with default parameters"
end


@logmsg MCMC "running $mcmcChains chains with $mcmcSteps steps each."


# Skip tuning if desired
if maxTuningAttempts == 0
  proposal_algorithm = MetropolisHastings(proposal=proposal_distribution, tuning=AdaptiveMHTuning(α=ClosedInterval(0, 1)))
  @logmsg MCMC "skipping the tuning stage."
else
  proposal_algorithm = MetropolisHastings(proposal=proposal_distribution)
  @logmsg MCMC "tuning will be performed with $tuningSteps steps up to a maximum of $maxTuningAttempts times."
end

println(" ")


# Run MCMC in batches to not overwhelm the RAM
# Setting a batch size of 10K steps, count tuning as the zeroth batch
batchSteps = 1_000
nBatches = ceil(Int, mcmcSteps / batchSteps)

currentBatch = 0
chainState = nothing

@logmsg MCMC "Running chains in $nBatches post-tuning batches of $batchSteps steps each."
println(" ")

# Main MCMC loop
global currentBatch = 0
global chain_state

# Run the mcmc for the tuning stage
chain_state = runMCMCbatch(currentBatch)

currentBatch += 1

# Run MCMC in batches to prevent RAM overload
while currentBatch <= nBatches
  # Declare global scope for trans-loop variables
  global currentBatch, chain_state

  chain_state = runMCMCbatch(currentBatch, chain_state, priors)

  GC.gc()

  currentBatch += 1
end
