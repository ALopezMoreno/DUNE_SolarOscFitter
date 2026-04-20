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

using BAT: MCMCIterator, MCMCInitAlgorithm, MCMCAlgorithm, BATMeasure, BATContext,
           ConvergenceTest, DensitySampleVector, mcmc_info, MCMCState, MCMCChainState
using Serialization  # For chain state persistence

# Chain state management for batch processing and continuation.
# v4: MCMCSampleGenerator stores MCMCChainState (MCMCIterator) objects, not MCMCState.
# We store chain_states (MCMCChainState) and reconstruct full MCMCState on continuation.

struct ChainState
    chain_states::Vector{<:MCMCIterator}  # MCMCChainState objects from generator
    chain_ids::Vector{Int32}
    end_step::Int
end

struct ContinueChains <: MCMCInitAlgorithm
    state::ChainState
end

function BAT.mcmc_init!(
    samplingalg::TransformedMCMC,
    target::BATMeasure,
    init_alg::ContinueChains,
    callback::Function,
    context::BATContext
)
    length(init_alg.state.chain_states) == samplingalg.nchains ||
        throw(ArgumentError("Chain count mismatch"))
    # Reconstruct full MCMCState (chain + tuner) from stored MCMCChainState.
    # Fresh tuner states are created from samplingalg; the adapted f_transform
    # inside each chain_state carries over the learned proposal geometry.
    mcmc_states = map(init_alg.state.chain_states) do cs
        trafo_tuner   = BAT.create_trafo_tuner_state(samplingalg.transform_tuning, cs, 0)
        proposal_tuner = BAT.create_proposal_tuner_state(samplingalg.proposal_tuning, cs, 0)
        temperer      = BAT.create_temperering_state(samplingalg.tempering, cs)
        MCMCState(cs, proposal_tuner, trafo_tuner, temperer)
    end
    outputs = [BAT._empty_chain_outputs(cs) for cs in init_alg.state.chain_states]
    return (mcmc_states=mcmc_states, outputs=outputs)
end

function save_chain_state(samples, starting_step)
    generator = samples.generator
    return ChainState(
        generator.chain_states,                          # Vector{MCMCChainState}
        [mcmc_info(s).id for s in generator.chain_states],
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

# Required BAT.jl interface implementation - v4 signature
function BAT.mcmc_burnin!(
    outputs::Union{AbstractVector{<:AbstractVector{<:DensitySampleVector}}, Nothing},
    mcmc_states::AbstractVector{<:MCMCState},
    samplingalg::TransformedMCMC,
    callback::Function
)
    return mcmc_states
end


# Function to save output chunks into the same JLD2 file, one group per batch.
# Each batch is written as "batch_N" so that groups can be appended without
# extending existing datasets (which JLD2 does not support).

function saveBatch(samples, start_step_number, priors, batchNumber)
  println("Saving batch $batchNumber (starting step $start_step_number)")

  # 1) pick out only the sampled parameters
  param_names = filter(p -> p in keys(priors),
                       fieldnames(typeof(samples[1].v)))

  # 2) extract per-sample values
  param_data = Dict{Symbol, Any}()
  for pname in param_names
      param_data[pname] = [getfield(s.v, pname) for s in samples]
  end

  # 3) metadata
  stepno  = [s.info.stepno  for s in samples] .+ start_step_number
  chainid = [s.info.chainid for s in samples]
  weights = [s.weight       for s in samples]

  println("Number of samples to save: ", length(samples))

  # 4) write as a new group inside the shared JLD2 file.
  #    "a+" opens for appending (creates file if absent); writing a new group
  #    never requires extending an existing dataset.
  fileName = outFile * ".jld2"
  try
    jldopen(fileName, "a+") do f
      g = JLD2.Group(f, "batch_$batchNumber")
      for (k, v) in param_data
          g[string(k)] = v
      end
      g["weights"] = weights
      g["stepno"]  = stepno
      g["chainid"] = chainid
    end
  catch err
    @error "Failed to write MCMC data to $fileName.\nError: $err"
  end
end



# Function for running the MCMC in batches
function runMCMCbatch(currentBatch, priors, args...)
  if currentBatch == 0
      if isnothing(prevFile)
          @logmsg MCMC "No previous MCMC file indicated. Starting chain from zero"
          # First batch: Run the tuning phase (~500 steps) for stability
          samples = bat_sample(posterior, TransformedMCMC(proposal=proposal_algorithm,
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
          samples = bat_sample(posterior, TransformedMCMC(proposal=proposal_algorithm,
                                                      nsteps=batchSteps,
                                                      nchains=mcmcChains,
                                                      init=ContinueChains(chain_state),
                                                      burnin=MCMCNoBurnin(),
                                                      convergence=convergence))
          
          # Save batch
          saveBatch(samples.result, starting_step, priors, currentBatch)
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
    
      samples = bat_sample(posterior, TransformedMCMC(proposal=proposal_algorithm,
                          nsteps=lastBatchSteps,
                          nchains=mcmcChains,
                          init=ContinueChains(chain_state),
                          burnin=MCMCNoBurnin(),
                          convergence=convergence
                          ))


          
      # Save batch
      saveBatch(samples.result, starting_step, priors, currentBatch)

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

      @logmsg Output "$(mcmcChains) output chain(s) saved to : $(outFile).jld2"
  end

  # Final cleanup to ensure no lingering references
  samples = nothing
  GC.gc()

  return(chain_state)
end