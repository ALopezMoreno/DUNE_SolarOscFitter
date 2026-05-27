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
  :integrated_8B_flux => prior_8B_flux,
  :cc_xsec_norm => prior_cc_xsec_norm,
  :cc_xsec_tilt => prior_cc_xsec_tilt,
  :cc_xsec_curv => prior_cc_xsec_curv
)


# Conditionally add nuisance parameters
if earthUncertainty
  # FOR NOW, SET THE INPUTS AS A SERIRES OF INDEPENDENT VARIABLES: TRANSFORM THE MVNORMAL INTO AN ARRAY OF 1DNORMALS
  means = mean(earth_normalisation_prior)
  covmat = cov(earth_normalisation_prior)
  n = length(means)
  for i in 1:n
    priors[Symbol("earth_norm_$i")] = truncated(Normal(means[i], sqrt(covmat[i, i])), 0.0, 2.0)
  end
  # param_bounds = Dict(:earth_norm => (0.0, 2.0)) # We want these to be fixed between 0 and 2
end

# Add per-detector background nuisance parameters (names prefixed by detector name)
for (dname, out) in detector_outputs
    det = detector_configs[dname]
    if det.ES_mode
        for (i, dist) in enumerate(out.ES_bg_norms_pars)
            priors[Symbol("$(dname)_ES_bg_norm_$i")] = dist
        end
    end
    if det.CC_mode && !det.inclusive_analysis
        for (i, dist) in enumerate(out.CC_bg_norms_pars)
            priors[Symbol("$(dname)_CC_bg_norm_$i")] = dist
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

@logmsg MCMC "running $mcmcChains chains with $mcmcSteps steps each (HMC)."

proposal_algorithm = HamiltonianMC(termination=GeneralisedNoUTurn(max_depth=4))
hmc_context        = BATContext(ad = ADSelector(:ForwardDiff))

batchSteps = 50
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
