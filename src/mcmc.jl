using Logging


using LinearAlgebra, Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using Plots, JLD2
using ForwardDiff, AdvancedHMC
import AutoDiffOperators: ADSelector

using Random

include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "mcmcHelpers.jl"))
include(joinpath(@__DIR__, "warmStart.jl"))


priors = Dict{Symbol,Any}(
  :sin2_th12 => prior_sin2_th12,
  :sin2_th13 => prior_sin2_th13,
  :dm2_21 => prior_dm2_21,

  :integrated_HEP_flux => prior_HEP_flux,

  :integrated_8B_flux => prior_8B_flux,
  # not sampled with fix_cc_xsec (the Asimov values are used)
  (fix_cc_xsec ? () : (:cc_xsec_norm => prior_cc_xsec_norm,
                        :cc_xsec_tilt => prior_cc_xsec_tilt,
                        :cc_xsec_curv => prior_cc_xsec_curv))...
)


if earthUncertainty
  # independent per layer; correlations are added to the log-density below
  means = mean(earth_normalisation_prior)
  covmat = cov(earth_normalisation_prior)
  n = length(means)
  for i in 1:n
    priors[Symbol("earth_norm_$i")] = truncated(Normal(means[i], sqrt(covmat[i, i])), 0.0, 2.0)
  end
end

# per-detector background norms (names prefixed by the detector)
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

prior = distprod(; priors...)

# A correlated Earth covariance: add the correlation term, making the product the full MvNormal.
if earthUncertainty && !isdiag(covmat)
  @logmsg MCMC "Earth prior covariance is correlated: adding the correlation term to the log-density"
  global likelihood_all_samples = let f = total_llh,
                                      M = Matrix(inv(covmat) - Diagonal(1 ./ diag(covmat))),
                                      μ = Vector{Float64}(means)
    logfuncdensity(p -> f(p) + earth_prior_correlation(p, M, μ))
  end
end

posterior = PosteriorMeasure(likelihood_all_samples, prior)

init = MCMCChainPoolInit(
  init_tries_per_chain=IntervalSets.ClosedInterval(4, 180),  # min > 1 avoids a BAT v4 bug
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

# Warm start (fresh runs only): fixed mass matrix and starting points from an earlier posterior
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

# NUTS: at most 2^hmc_max_depth gradients per MCMC step
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

batchSteps = 200   # checkpoint cadence; tuning persists across batches
nBatches   = ceil(Int, mcmcSteps / batchSteps)

@logmsg MCMC "Running chains in $nBatches post-tuning batches of $batchSteps steps each."
println(" ")

# Batch 0: init and tuning, or the chain state of a resumed run
chain_state  = runMCMCbatch(0, priors)
currentBatch = 1

while currentBatch <= nBatches
    global currentBatch, chain_state
    chain_state = runMCMCbatch(currentBatch, priors, chain_state)
    GC.gc()
    currentBatch += 1
end
