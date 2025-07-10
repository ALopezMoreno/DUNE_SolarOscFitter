#=
mcmcHelpers.jl

Helper functions and custom algorithms for MCMC sampling in the Solar Oscillation Fitter.
This module provides utilities for managing MCMC chains, including batch processing,
chain continuation, custom proposal distributions, and memory management.

Key Features:
- Custom proposal distribution with user-defined covariance matrices
- Chain state management for batch processing and continuation
- Memory-efficient MCMC execution with garbage collection
- Serialization utilities for chain persistence
- Custom burnin algorithms including no-burnin option

These utilities enable efficient MCMC sampling for large parameter spaces
while managing memory usage and allowing for interrupted/resumed analyses.

Author: Andres Lopez Moreno, based on Philipp Eller's Newthrino
=#

using BAT: MCMCIterator, MCMCInitAlgorithm, MCMCAlgorithm, BATMeasure, BATContext, AbstractMCMCTunerInstance, ConvergenceTest, DensitySampleVector, mcmc_info
using Serialization  # For chain state persistence

# Custom proposal distribution with user-defined covariance matrix
function mv_proposaldist(T::Type{<:AbstractFloat}, d::TDist, varndof::Integer)
    """
    Override BAT's default proposal distribution to inject custom covariance matrix.
    
    This allows using pre-computed covariance matrices from previous runs
    to improve MCMC efficiency and convergence.
    """
    df = only(Distributions.params(d))
    μ = fill(zero(T), varndof)
    
    # Use custom covariance matrix if available, otherwise use identity
    Σ = (propMatrix !== nothing) ? propMatrix : PDMat(Matrix(I(varndof) * one(T)))

    # Construct multivariate t-distribution with custom covariance
    # Arguments: degrees of freedom, mean vector, scale matrix
    return MvTDist(convert(T, df), μ, Matrix(Σ))
end

# Chain state management for batch processing and continuation

struct ChainState
    """
    Container for MCMC chain state information.
    
    Stores all necessary information to resume MCMC chains from a previous state,
    enabling batch processing and interrupted analysis recovery.
    """
    chains::Vector{<:MCMCIterator}      # Active MCMC chain iterators
    tuners::Vector{AdaptiveMHTuning}    # Adaptive tuning state for each chain
    step_numbers::Vector{Int}           # Current step number for each chain
    chain_ids::Vector{Int32}            # Unique identifiers for each chain
    end_step::Int                       # Total number of completed steps
end

struct ContinueChains <: MCMCInitAlgorithm
    """
    Custom MCMC initialization algorithm for continuing from saved state.
    
    This allows resuming MCMC sampling from a previously saved ChainState,
    maintaining all tuning information and chain positions.
    """
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
    """
    Initialize MCMC chains from a saved state.
    
    Validates chain count consistency and returns the saved chain state
    for continuation of sampling.
    """
    length(init.state.chains) == nchains || throw(ArgumentError("Chain count mismatch"))
    return (
        chains=init.state.chains,
        tuners=init.state.tuners,
        chain_outputs=DensitySampleVector[
            DensitySampleVector(c) for c in init.state.chains
        ]
    )
end



function keep_last_step(chain::BAT.MCMCIterator)
  # Preserve original storage types
  last_samples = chain.samples[end:end]
  
  empty!(chain.samples)
  GC.gc()

  chain.samples = last_samples

  chain.nsamples = 1
  chain.stepno = 1

  chain.samples.info[1] = BAT.MCMCSampleID(mcmc_info(chain).id, mcmc_info(chain).cycle, chain.stepno, BAT.CURRENT_SAMPLE)
  
  # Set the weight to zero as to not count it twice
  chain.samples.weight[1] = 0
  
  return chain

end

function save_chain_state(samples, starting_step)
  # Generator holds the full chain history
  generator = samples.generator
  # tempGen = map(keep_last_step, generator.chains)

  # Build the ChainState using these truncated chains
  return ChainState(
      generator.chains,
      [c.algorithm.tuning for c in generator.chains],
      # Store only 1 as the step count for each chain,
      fill(1, length(generator.chains)),
      [mcmc_info(c).id for c in generator.chains],
      # Keep the final step number from the run, plus the offset:
      [sample.info.stepno for sample in samples.result][end] + starting_step
  )
end

function save_chainstate_serialized(cs::ChainState, filename::String)
  open(filename, "w") do io
      serialize(io, cs)
  end
end

function load_chainstate_serialized(filename::String)
  open(filename, "r") do io
      return deserialize(io)
  end
end


# New burnin algorithm type definition
struct MCMCNoBurnin <: MCMCBurninAlgorithm end

# Required BAT.jl interface implementation
function BAT.mcmc_burnin!(
  outputs::Union{Nothing,Vector{<:DensitySampleVector}},
  tuners::Vector{AdaptiveMHTuning},
  chains::Vector{<:MCMCIterator},
  burnin::MCMCNoBurnin,
  convergence::ConvergenceTest,
  strict::Bool,
  nonzero_weights::Bool,
  callback::Function
)
  # No-op implementation that matches BAT's return type expectations
  return (chains=chains, tuners=MCMCNoOpTuning())
end

function BAT.mcmc_burnin!(
  outputs::Union{Nothing,Vector{<:DensitySampleVector}},
  tuners::Vector{<:AbstractMCMCTunerInstance},
  chains::Vector{<:MCMCIterator},
  burnin::MCMCNoBurnin,
  convergence::ConvergenceTest,
  strict::Bool,
  nonzero_weights::Bool,
  callback::Function
)
  # No-op implementation that matches BAT's return type expectations
  return (chains=chains, tuners=MCMCNoOpTuning())
end


# Function to save output chunks into the same file

function saveBatch(samples, start_step_number, priors)
  println("Saving with starting step number of $start_step_number")

  # 1) pick out only the priors you care about
  param_names = filter(p -> p in keys(priors),
                       fieldnames(typeof(samples[1].v)))

  # 2) build a Dict that can hold anything
  param_data = Dict{Symbol, Any}()

  for pname in param_names
      vals = [getfield(s.v, pname) for s in samples]
      param_data[pname] = vals
  end

  # 3) metadata
  stepno  = [s.info.stepno  for s in samples] .+ start_step_number
  chainid = [s.info.chainid for s in samples]
  weights = [s.weight         for s in samples]

  println("Number of samples to save: ", length(samples))
  # 4) write in one go
  fileName = outFile * "_mcmc.bin"
  try
    open(fileName, isfile(fileName) ? "a" : "w") do io
      serialize(io, (param_data, weights, stepno, chainid))
    end
  
  catch err
      @error "Failed to write MCMC data to $fileName.\nError: $err"
      return
  end

  # 5) simple info file
  infoFileName = outFile * "_info.txt"
  open(infoFileName, "w") do io
    println(io, "# Saved fields in binary (in order):")
    println(io, "1: param_data (Dict{Symbol,Any})")
    println(io, "   Parameter names: ", join(string.(keys(param_data)), ", "))
    println(io, "2: weights (Vector)")
    println(io, "3: stepno  (Vector)")
    println(io, "4: chainid (Vector)")
  end
end



# Function for running the MCMC in batches
function runMCMCbatch(currentBatch, args...)
  if currentBatch == 0
      if isnothing(prevFile)
          @logmsg MCMC "No previous MCMC file indicated. Starting chain from zero"
          # First batch: Run the tuning phase (~500 steps) for stability
          samples = bat_sample(posterior, MCMCSampling(mcalg=proposal_algorithm,
                              nsteps=2_500,
                              nchains=mcmcChains,
                              init=init,
                              burnin=burnin,
                              convergence=convergence
                              ))

          # For debugging: print available keys in the samples named tuple (if needed)
          # println("Sample keys: ", collect(keys(samples)))
      
          # Save chain state from the last sample of the tuning run
          # saveBatch(samples.result, 0)
          chain_state = save_chain_state(samples, 0)
          # Clear heavy data immediately
          samples = nothing
          GC.gc()  # force garbage collection
          
          @logmsg MCMC "Tuning finished, running batches"

      else
          chain_state = load_chainstate_serialized(prevFile * "_chainState.bin")
          @logmsg MCMC ("Continuing chains from " * prevFile)
      end
      # @logmsg MCMC ("Chain at $(chain_state.end_step) steps")


  elseif currentBatch < nBatches
      chain_state = args[1]
      starting_step = chain_state.end_step
      # Regular batch: Run with batchSteps steps and skip burnin
      elapsed_time = @elapsed begin
          samples = bat_sample(posterior, MCMCSampling(mcalg=proposal_algorithm,
                                                      nsteps=batchSteps,
                                                      nchains=mcmcChains,
                                                      init=ContinueChains(chain_state),
                                                      burnin=MCMCNoBurnin(),
                                                      convergence=convergence))
          
          # Save batch
          saveBatch(samples.result, starting_step, priors)
          chain_state = save_chain_state(samples, starting_step)
          save_chainstate_serialized(chain_state, outFile*"_chainState.bin")
          # Clear heavy data immediately
          samples = nothing
      end 
      
      # Force garbage collection after batch completes
      GC.gc()

      remaining = elapsed_time * (nBatches - currentBatch)
      @logmsg MCMC "Ran batch $(currentBatch) of $(nBatches) in $(elapsed_time) seconds. Expectation: ~$(ceil(Int32, remaining/60)) minutes until completion."
      # @logmsg MCMC ("Chain at $(chain_state.end_step) steps")

  else
      chain_state = args[1]
      starting_step = chain_state.end_step
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


          
      # Save batch
      saveBatch(samples.result, starting_step, priors)
      
      # Store statistics before clearing samples
      mode_result = mode(samples.result)
      mean_result = mean(samples.result)
      std_result = std(samples.result)
      
      chain_state = save_chain_state(samples, starting_step)
      save_chainstate_serialized(chain_state, outFile*"_chainState.bin")
      
      # Clear heavy data
      samples = nothing
      GC.gc()

      println(" ")  

      @logmsg Setup ("Truth: $true_params")

      println(" ")

      @logmsg Output "Mode: $(mode_result)"
      @logmsg Output "Mean: $(mean_result)"
      @logmsg Output "Stddev: $(std_result)"

      println(" ")

      @logmsg Output "$(mcmcChains) output chain(s) saved to : $(outFile) (+_mcmc.bin/_info.txt)"
  end

  # Final cleanup to ensure no lingering references
  samples = nothing
  GC.gc()

  return(chain_state)
end