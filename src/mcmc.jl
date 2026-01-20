#=
mcmc.jl

Bayesian parameter estimation for neutrino oscillation parameters using Markov Chain 
Monte Carlo (MCMC) sampling with the BAT.jl library.

This script performs posterior sampling from the likelihood function defined by the
solar neutrino data and prior distributions over oscillation parameters.

Key Features:
- Metropolis-Hastings MCMC sampling with adaptive tuning
- Support for systematic uncertainties (Earth matter, backgrounds)
- Batch processing to manage memory usage
- Configurable proposal distributions and covariance matrices
- Comprehensive logging and progress tracking

Workflow:
1. Define prior distributions for all parameters
2. Configure MCMC settings (chains, steps, tuning)
3. Run MCMC sampling in batches
4. Save results for posterior analysis

Dependencies:
- BAT.jl for Bayesian analysis toolkit
- Distributions.jl for probability distributions
- JLD2.jl for data storage

Author: [Author name]
=#

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
if earthUncertainty  if !isempty(CC_bg_norms_pars)
    for (i, norm) in enumerate(CC_bg_norms_pars)
        priors[Symbol("CC_bg_norm_$i")] = norm
    end
  end
  # FOR NOW, SET THE INPUTS AS A SERIRES OF INDEPENDENT VARIABLES: TRANSFORM THE MVNORMAL INTO AN ARRAY OF 1DNORMALS
  means = mean(earth_normalisation_prior)
  covmat = cov(earth_normalisation_prior)
  n = length(means)
  for i in 1:n
    priors[Symbol("earth_norm_$i")] = Truncated(Normal(means[i], sqrt(covmat[i, i])), 0.0, 2.0)
  end
  # param_bounds = Dict(:earth_norm => (0.0, 2.0)) # We want these to be fixed between 0 and 2
end

if ES_mode
  if !isempty(ES_bg_norms_pars)
    for (i, norm) in enumerate(ES_bg_norms_pars)
        priors[Symbol("ES_bg_norm_$i")] = norm
    end
  end
end

if CC_mode
  if !isempty(CC_bg_norms_pars)
    for (i, norm) in enumerate(CC_bg_norms_pars)
        priors[Symbol("CC_bg_norm_$i")] = norm
    end
  end
end

# Use splatting to pass the priors to distprod
prior = distprod(; priors...)

# Check if explicit param bounds have been given:
# param_names = collect(keys(prior))
# Create aligned bounds vector
# bounds = [haskey(param_bounds, name) ? param_bounds[name] : (-Inf, Inf) for name in param_names]

# Define Bayesian model
posterior = PosteriorMeasure(likelihood_all_samples, prior)

# Apply parameter bounds
#bounded_domain = BoundedDomain(bounds)
#transformed_posterior = TransformedDensity(bounded_domain, posterior)

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
# Setting a batch size of 1K steps, count tuning as the zeroth batch
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
