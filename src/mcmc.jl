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
using HDF5
# DEBUGGING AND TESTING: Set the logging level to Warn to suppress Info messages
# global_logger(ConsoleLogger(stderr, Logging.Warn))

using LinearAlgebra, Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using BAT: MCMCIterator, MCMCInitAlgorithm, MCMCAlgorithm, BATMeasure, BATContext, AbstractMCMCTunerInstance, ConvergenceTest, DensitySampleVector, mcmc_info
using Plots, JLD2

using Random
Random.seed!(1234)  # Set global seed once at startup if you want RNG reproducibility

include("../src/setup.jl")

# Override mv_proposaldist to inject a custom covariance matrix
function mv_proposaldist(T::Type{<:AbstractFloat}, d::TDist, varndof::Integer)
  df = only(Distributions.params(d))
  μ = fill(zero(T), varndof)
  # Use cov matrix if available, otherwise fall back to the identity
  Σ = (propMatrix !== nothing) ? propMatrix : PDMat(Matrix(I(varndof) * one(T)))
  
  # Construct a multivariate t-distribution with your custom covariance.
  # Use MvTDist from Distributions, which takes for arguments:
  #   - degrees of freedom (as a scalar),
  #   - mean vector, and
  #   - scale matrix (a regular matrix)
  return MvTDist(convert(T, df), μ, Matrix(Σ))
end

# Custom init algorithm and necessary structures for continuing a chain from where we left off:
struct ChainState
  chains::Vector{<:MCMCIterator}
  tuners::Vector{AdaptiveMHTuning}
  step_numbers::Vector{Int}
  chain_ids::Vector{Int32}
end


struct ContinueChains <: MCMCInitAlgorithm
  state::ChainState
end


function BAT.mcmc_init!(
  algorithm::MCMCAlgorithm,
  target::BATMeasure,
  nchains::Integer,
  init::ContinueChains,
  tuning,
  nonzero_weights::Bool,
  callback::Function,
  context::BATContext
)
  length(init.state.chains) == nchains || throw(ArgumentError("Chain count mismatch"))
  return (
      chains = init.state.chains,
      tuners = init.state.tuners,
      chain_outputs = DensitySampleVector[
          DensitySampleVector(c) for c in init.state.chains
      ]
  )
end


function save_chain_state(samples)
  generator = samples.generator
  ChainState(
      generator.chains,
      [c.algorithm.tuning for c in generator.chains],
      [BAT.nsteps(c) for c in generator.chains],
      [mcmc_info(c).id for c in generator.chains]
  )
end


# New burnin algorithm type definition
struct MCMCNoBurnin <: MCMCBurninAlgorithm end

# Required BAT.jl interface implementation
function BAT.mcmc_burnin!(
  outputs::Union{Nothing, Vector{<:DensitySampleVector}},
  tuners::Vector{AdaptiveMHTuning},
  chains::Vector{<:MCMCIterator},
  burnin::MCMCNoBurnin,
  convergence::ConvergenceTest,
  strict::Bool,
  nonzero_weights::Bool,
  callback::Function
)
  # No-op implementation that matches BAT's return type expectations
  return (chains = chains, tuners = MCMCNoOpTuning())
end


# Function for saving chain ouput to file
function saveBatch(samples)
  # Initialize arrays to store the extracted data
  sin2_th12 = Float64[]
  sin2_th13 = Float64[]
  dm2_21 = Float64[]
  integrated_8B_flux = Float64[]
  stepno = Int64[]
  chainid = Int32[]

  # Iterate over each sample and extract the desired fields
  for sample in samples
      # Extract values from the NamedTuple `v`
      push!(sin2_th12, sample.v.sin2_th12)
      push!(sin2_th13, sample.v.sin2_th13)
      push!(dm2_21, sample.v.dm2_21)
      
      # Extract nuisance parameters
      push!(integrated_8B_flux, sample.v.integrated_8B_flux)
  
      # Extract values from the `info` StructVector
      push!(stepno, sample.info.stepno)
      push!(chainid, sample.info.chainid)
  end

  fileName = outFile * "_mcmc.h5"  # using HDF5 file extension
  batch_length = length(sin2_th12)

  try
    if !isfile(fileName)
        # Create new file and datasets with unlimited dimension on the first axis.
        # Choosing a chunk size is required for extendable datasets; here we use the minimum of batch_length and 1024.
        h5open(fileName, "w") do f
            d_create(f, "sin2_th12", sin2_th12; dims=(batch_length,), maxdims=(Inf,), chunk=(min(batch_length, 1024),))
            d_create(f, "sin2_th13", sin2_th13; dims=(batch_length,), maxdims=(Inf,), chunk=(min(batch_length, 1024),))
            d_create(f, "dm2_21", dm2_21; dims=(batch_length,), maxdims=(Inf,), chunk=(min(batch_length, 1024),))
            d_create(f, "integrated_8B_flux", integrated_8B_flux; dims=(batch_length,), maxdims=(Inf,), chunk=(min(batch_length, 1024),))
            d_create(f, "stepno", stepno; dims=(batch_length,), maxdims=(Inf,), chunk=(min(batch_length, 1024),))
            d_create(f, "chainid", chainid; dims=(batch_length,), maxdims=(Inf,), chunk=(min(batch_length, 1024),))
        end
    else
        # Append to the existing datasets without loading them entirely into memory.
        h5open(fileName, "r+") do f
            # For each dataset, extend the first dimension by the number of new samples.
            for dset_name in ["sin2_th12", "sin2_th13", "dm2_21", "integrated_8B_flux", "stepno", "chainid"]
                dset = f[dset_name]
                old_size = size(dset, 1)
                new_size = old_size + batch_length
                dset.resize!(new_size)
            end
            # Write the new batch into the extended portions of each dataset.
            f["sin2_th12"][end-batch_length+1:end] = sin2_th12
            f["sin2_th13"][end-batch_length+1:end] = sin2_th13
            f["dm2_21"][end-batch_length+1:end] = dm2_21
            f["integrated_8B_flux"][end-batch_length+1:end] = integrated_8B_flux
            f["stepno"][end-batch_length+1:end] = stepno
            f["chainid"][end-batch_length+1:end] = chainid
        end
    end
  catch err
    error("Failed to save batch to $(fileName): $(err)")
  end
end


# Set prior distributions from config
# Construct a dictionary of priors
priors = Dict{Symbol, Any}(
    # miximg parameters
    :sin2_th12 => prior_sin2_th12,
    :sin2_th13 => prior_sin2_th13,
    :dm2_21   => prior_dm2_21,

    # systematic parameters
    :integrated_8B_flux => prior_8B_flux
)

# Conditionally add nuisance parameters
if earthUncertainty
    priors[:earth_norm] = earth_normalisation_prior
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

burninAttempts = max(1, maxTuningAttempts)

burnin = MCMCMultiCycleBurnin(
    nsteps_per_cycle=tuningSteps,
    max_ncycles=burninAttempts,
    nsteps_final=tuningSteps/10
)


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
  proposal_algorithm = MetropolisHastings(proposal=proposal_distribution, tuning = AdaptiveMHTuning(α=ClosedInterval(0, 1)))
  @logmsg MCMC "skipping the tuning stage."
else
  proposal_algorithm = MetropolisHastings(proposal=proposal_distribution)
  @logmsg MCMC "tuning will be performed with $tuningSteps steps up to a maximum of $maxTuningAttempts times."
end

println(" ")  


# Run MCMC in batches to not overwhelm the RAM
# Setting a batch size of 15K steps, count tuning as the zeroth batch
batchSteps = 2_000
nBatches = ceil(Int, mcmcSteps / batchSteps)

currentBatch = 0
chainState = nothing

@logmsg MCMC "Running chains in $nBatches post-tuning batches of $batchSteps steps each."
println(" ")  

# Main MCMC loop
while true
  global currentBatch
  global chain_state

  if currentBatch == 0
      # First batch: Run the tuning phase (~500 steps) for stability
      samples = bat_sample(posterior, MCMCSampling(mcalg=proposal_algorithm,
                              nsteps=500,
                              nchains=mcmcChains,
                              init=init,
                              burnin=burnin,
                              convergence=convergence
                              ))

      # For debugging: print available keys in the samples named tuple (if needed)
      # println("Sample keys: ", collect(keys(samples)))
      
      # Save chain state from the last sample of the tuning run
      chain_state = save_chain_state(samples)
      
      # Update batch counter and log progress
      currentBatch += 1
      @logmsg MCMC "Tuning finished, running batches"
      println("")
  
  elseif currentBatch < nBatches
      # Regular batch: Run with batchSteps steps and skip burnin
      elapsed_time = @elapsed samples = bat_sample(posterior, MCMCSampling(mcalg=proposal_algorithm,
                                          nsteps=batchSteps,
                                          nchains=mcmcChains,
                                          init=ContinueChains(chain_state),
                                          burnin=MCMCNoBurnin(),
                                          convergence=convergence
                                          ))
      
      # Estimate remaining time
      remaining = elapsed_time * (nBatches - currentBatch)
      @logmsg MCMC "Ran batch $(currentBatch) of $(nBatches) in $(elapsed_time) seconds. Expect ~$(remaining) seconds remaining."

      # Save the batch and update the chain state
      saveBatch(samples.result)
      chain_state = save_chain_state(samples)
      
      # Increment batch counter for next iteration
      currentBatch += 1
  
  else
      # Final batch: If batchSteps does not equally divide mcmcSteps, use the remainder
      remainder = mcmcSteps % batchSteps
      lastBatchSteps = (remainder == 0) ? batchSteps : remainder
      
      samples = bat_sample(posterior, MCMCSampling(mcalg=proposal_algorithm,
                          nsteps=lastBatchSteps,
                          nchains=mcmcChains,
                          init=ContinueChains(chain_state),
                          burnin=MCMCNoBurnin(),
                          convergence=convergence
                          ))
      
      # Save the final batch and exit the loop
      saveBatch(samples.result)
      break
  end
end

println(" ")  

@logmsg Setup ("Truth: $true_params")

println(" ")

@logmsg Output "Mode: $(mode(samples))"
@logmsg Output "Mean: $(mean(samples))"
@logmsg Output "Stddev: $(std(samples))"

println(" ")

@logmsg Output "$(mcmcChains) output chain saved to : $(outFile)"

# Initialize arrays to store the extracted data
# sin2_th12 = Float64[]
# sin2_th13 = Float64[]
# dm2_21 = Float64[]
# integrated_8B_flux = Float64[]
# stepno = Int64[]
# chainid = Int32[]  # Assuming chainid is Int32 based on your structure

# Iterate over each sample and extract the desired fields
# for sample in samples
      # Extract values from the NamedTuple `v`
#     push!(sin2_th12, sample.v.sin2_th12)
#     push!(sin2_th13, sample.v.sin2_th13)
#     push!(dm2_21, sample.v.dm2_21)

#     # Extract nuisance parameters
#     push!(integrated_8B_flux, sample.v.integrated_8B_flux)
#     
#     # Extract values from the `info` StructVector
#     push!(stepno, sample.info.stepno)
#     push!(chainid, sample.info.chainid)
# end

# sample_ES_nue = ereco_data.ES_nue_night
# sample_ES_nuother = ereco_data.ES_nuother_night
# sample_CC = ereco_data.CC_night

# @save outFile*"_mcmc.jld2" sin2_th12 sin2_th13 dm2_21 integrated_8B_flux stepno chainid
# sample_ES_nue_night sample_ES_nuother_night sample_CC_night

# println(" ")

# Unshape sample for statistical treatment
# unshaped_samples, f_flatten = bat_transform(Vector, samples)
# this can be used to generate a covariance matrix