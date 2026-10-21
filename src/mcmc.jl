using Logging

# DEBUGGING AND TESTING: Set the logging level to Warn to suppress Info messages
# global_logger(ConsoleLogger(stderr, Logging.Warn))

using LinearAlgebra, Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using Plots, JLD2
using ForwardDiff, AdvancedHMC
import AutoDiffOperators: ADSelector

using Random
# Random.seed!(1234)  # Set global seed once at startup if you want RNG reproducibility

include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "mcmcHelpers.jl"))
include(joinpath(@__DIR__, "warmStart.jl"))


include(joinpath(@__DIR__, "priors.jl"))
priors = build_priors()
if earthUncertainty
  means = mean(earth_normalisation_prior)
  covmat = cov(earth_normalisation_prior)
end

# Use splatting to pass the priors to distprod
prior = distprod(; priors...)

# Check if explicit param bounds have been given:
# param_names = collect(keys(prior))
# Create aligned bounds vector
# bounds = [haskey(param_bounds, name) ? param_bounds[name] : (-Inf, Inf) for name in param_names]

# The Earth-layer priors above are independent normals. If the covariance is correlated, add
# the correlation term to the log-density so the product becomes the full MvNormal.
if earthUncertainty && !isdiag(covmat)
  @logmsg MCMC "Earth prior covariance is correlated: adding the correlation term to the log-density"
  global likelihood_all_samples = let f = total_llh,
                                      M = Matrix(inv(covmat) - Diagonal(1 ./ diag(covmat))),
                                      μ = Vector{Float64}(means)
    logfuncdensity(p -> f(p) + earth_prior_correlation(p, M, μ))
  end
end

# Define Bayesian model
posterior = PosteriorMeasure(likelihood_all_samples, prior)

# Apply parameter bounds
#bounded_domain = BoundedDomain(bounds)
#transformed_posterior = TransformedDensity(bounded_domain, posterior)

# Set chain parameters
init = MCMCChainPoolInit(
  init_tries_per_chain=IntervalSets.ClosedInterval(4, 180),  # min>1 avoids BAT v4 bug in else-branch of chain selection
  nsteps_init=15,
  initval_alg=InitFromTarget()
)

if maxTuningAttempts != 0
  burnin = MCMCMultiCycleBurnin(
    nsteps_per_cycle=tuningSteps,
    max_ncycles=maxTuningAttempts,
    nsteps_final=tuningSteps ÷ 10
  )
else
  burnin = MCMCNoBurnin()
end

convergence = AssumeConvergence()

# Warm start from an earlier run's posterior: fixed mass matrix + chains started from its draws.
# Only for a fresh run; a resumed run carries its own transform in the chain state.
warm_start = nothing
if warm_start_chain !== nothing
  if prevFile !== nothing
    @logmsg MCMC "warm_start_chain is ignored when resuming from prevFile"
  else
    global warm_start = build_warm_start(warm_start_chain, posterior, prior, PriorToNormal(); init = init)
    global init = warm_start.init
    @logmsg MCMC "Warm start from $warm_start_chain: mass matrix from $(warm_start.nsamples) samples, " *
                 "kept fixed (step size still tuned); chains start from its draws"
    log_transform_scales("Warm-start", warm_start.scales, warm_start.names)
  end
end

@logmsg MCMC "running $mcmcChains chains with $mcmcSteps steps each (HMC)."

# NUTS tree-depth cap: at most 2^max_depth leapfrog steps (gradients) per MCMC step. Every
# batch logs how close the steps come to it (log_sampler_cost in mcmcHelpers.jl).
proposal_algorithm = HamiltonianMC(termination=GeneralisedNoUTurn(max_depth=hmc_max_depth))
if ad_backend == "Enzyme" && !uses_night_osc(OSC)
  error("ad_backend: Enzyme requires nightOsc: true " *
        "(the legacy NumOsc night calculation is not Enzyme-differentiable)")
end
hmc_context = if mcmc_seed === nothing
  BATContext(ad = ADSelector(Symbol(ad_backend)))
else
  BATContext(ad = ADSelector(Symbol(ad_backend)),
             rng = BAT.Random123.Philox4x((UInt64(mcmc_seed), UInt64(0))))
end
@logmsg MCMC "HMC gradients: $ad_backend" * (ad_backend_defaulted ? " (default; set ad_backend to override)" : "") *
             (mcmc_seed === nothing ? "" : ", RNG seed $mcmc_seed")

batchSteps = 200   # checkpoint/output cadence only — tuning persists across batches
nBatches   = ceil(Int, mcmcSteps / batchSteps)

@logmsg MCMC "Running chains in $nBatches post-tuning batches of $batchSteps steps each."
println(" ")

# Batch 0: init + tune (or load chain state from prevFile if resuming)
chain_state  = runMCMCbatch(0, priors)
currentBatch = 1

# Sampling loop: all batches continue from the tuned chain state
while currentBatch <= nBatches
    global currentBatch, chain_state
    chain_state = runMCMCbatch(currentBatch, priors, chain_state)
    GC.gc()
    currentBatch += 1
end
