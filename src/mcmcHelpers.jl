#=
mcmcHelpers.jl — MCMC batch processing with persistent native tuning.

Based on Newthrino (Philipp Eller), extended by Andres Lopez Moreno.

Batching strategy
-----------------
The run is written out in fixed-size batches (for checkpointing / bounded
memory), but the sampler state must be CONTINUOUS across batches and resumes —
otherwise BAT's tuners (RAM affine transform + dual-averaging step size) get
reset every batch and never converge, the step size drifts, and tree depth (and
wall-time) blow up on continuation.

`bat_sample` cannot give us this: it returns a generator that keeps only the
`chain_state` and *discards* the tuner states
(`MCMCSampleGenerator(mcmc_states) = getfield.(mcmc_states, :chain_state)`).

So instead of one `bat_sample` per batch, we drive BAT's MCMC loop directly
(exactly what `bat_sample_impl` does), holding the full `Vector{MCMCState}`
(chain_state + proposal_tuner_state + trafo_tuner_state + temperer) across
batches and serialising it whole. Native tuning therefore persists and keeps
converging with its diminishing adaptation rate (which remains ergodic-valid),
and a resume continues tuning mid-stream rather than restarting cold.
=#

using BAT: DensitySampleVector, MCMCState, MCMCBurninAlgorithm
using Serialization  # For run-state persistence


# ─────────────────────────────────────────────────────────────────────────────
# Persistent run state carried across batches and serialised for resume.
# `mcmc_states` holds the full BAT MCMCState per chain (incl. tuner state);
# `f_pretransform` is the (prior→normal) map needed to push samples back to
# parameter space.
# ─────────────────────────────────────────────────────────────────────────────
struct RunState
    mcmc_states::Vector       # Vector{MCMCState}  (chain + tuners + temperer)
    f_pretransform            # PriorToNormal unshaping transform
    end_step::Int
end

save_runstate(rs::RunState, filename::String) =
    open(io -> serialize(io, rs), filename, "w")
load_runstate(filename::String)::RunState =
    open(deserialize, filename, "r")


# No-op burn-in (used when maxTuningAttempts == 0 → no native tuning phase).
struct MCMCNoBurnin <: MCMCBurninAlgorithm end

function BAT.mcmc_burnin!(
    outputs::Union{AbstractVector{<:AbstractVector{<:DensitySampleVector}}, Nothing},
    mcmc_states::AbstractVector{<:MCMCState},
    samplingalg::TransformedMCMC,
    callback::Function
)
    return mcmc_states
end


# Build the sampling algorithm. Tuning fields (transform_tuning=RAMTuning,
# proposal_tuning=StepSizeAdaptor, pretransform=PriorToNormal, …) take BAT's
# native HMC defaults — that's the tuning we now persist rather than reset.
_build_samplingalg() = TransformedMCMC(
    proposal    = proposal_algorithm,
    nchains     = mcmcChains,
    init        = init,
    burnin      = burnin,
    convergence = convergence,
    nsteps      = batchSteps,   # placeholder; per-batch length is passed explicitly
)


# On resume, rebuild each chain's likelihood functions from the LIVE current
# session, keeping the numeric state (positions, tuner accumulators, affine
# transform, step size, metric) from the checkpoint. The serialize round-trip
# type-erases the target/likelihood CLOSURES into Serialization stand-ins; running
# the identical arithmetic through those boxes/allocates ~7-8x more per batch
# (~600 GB vs ~80 GB measured) → heavy GC → a resume runs ~2x slower than one
# continuous run, for no mathematical reason (it's a Markov chain). Swapping in the
# live `posterior` + `f_pretransform` and rebuilding the proposal's ℓπ/∂ℓπ∂θ
# restores fresh-run allocation (verified: rebuilt batch == fresh batch). The
# ongoing per-step RAM tuning also rebuilds via chain_state.target, so the swap
# keeps that path live too.
# Warm the process the way a fresh run's init does, so the FIRST resumed batch does not
# hit a cold-heap GC collapse. A fresh run gradually grows the GC heap (and JIT-compiles
# the threaded sampling path) during mcmc_init!/burnin; a resume otherwise jumps straight
# from a cold heap into ~60 GB/batch of churn across N threads, where Julia's GC falls into
# a sustained over-collection equilibrium (~14x slower at 10 chains, threads==chains).
# This runs a throwaway mcmc_init! purely for that warming; the loaded numeric chain state
# is untouched (these throwaway states are discarded). Verified: brings a resumed batch back
# to fresh-run wall-time. Its RNG draws are isolated (fresh contexts), so the real chains'
# randomness is unaffected.
function warmup_process()
    salg = _build_samplingalg()
    tmw, fptw = BAT.transform_and_unshape(salg.pretransform, posterior, hmc_context)
    warm_states, _ = BAT.mcmc_init!(
        salg, tmw, BAT.apply_trafo_to_init(fptw, salg.init), BAT.nop_func, hmc_context)
    warm_states = nothing
    GC.gc()
    return nothing
end


function rebuild_live_state(run_state::RunState)
    salg = _build_samplingalg()
    transformed_m, f_pretransform =
        BAT.transform_and_unshape(salg.pretransform, posterior, hmc_context)
    new_states = map(run_state.mcmc_states) do ms
        cs = ms.chain_state
        mk(target, proposal) = BAT.MCMCChainState(
            target, proposal, cs.f_transform, cs.weighting,
            cs.current, cs.proposed, cs.output, cs.accepted,
            cs.info, cs.rngpart_cycle, cs.nsamples, cs.stepno, cs.context)
        cs_live   = mk(transformed_m, cs.proposal)
        prop_live = BAT.set_proposal_transform!!(cs_live.proposal, cs_live)  # live ℓπ/∂ℓπ∂θ
        cs_final  = mk(transformed_m, prop_live)
        MCMCState(cs_final, ms.proposal_tuner_state, ms.trafo_tuner_state, ms.temperer_state)
    end
    return RunState(new_states, f_pretransform, run_state.end_step)
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


# ─────────────────────────────────────────────────────────────────────────────
# Drive one batch of sampling on a live `Vector{MCMCState}`. Tuning state is
# preserved (we never recreate the tuners). Returns (new_states, batch_samples)
# with `batch_samples` already mapped back to parameter space.
# ─────────────────────────────────────────────────────────────────────────────
function _sample_one_batch(mcmc_states, f_pretransform, nsteps)
    BAT.next_cycle!.(mcmc_states)
    outputs     = BAT._empty_chain_outputs.(mcmc_states)
    new_states  = BAT.mcmc_iterate!!(outputs, mcmc_states; max_nsteps = nsteps, nonzero_weights = true)
    merged      = BAT._merge_chain_outputs(first(new_states), outputs)
    batch_samples = BAT.transform_samples(BAT.inverse(f_pretransform), merged)
    return new_states, batch_samples
end

# Mean last-step NUTS tree depth across chains (sampling-health diagnostic).
function _log_tree_depth(mcmc_states)
    try
        ds = [Int(AdvancedHMC.stat(s.chain_state.proposal.transition).tree_depth) for s in mcmc_states]
        @logmsg MCMC "Mean NUTS tree depth: $(round(sum(ds)/length(ds), digits=2)) (per-chain: $ds)"
    catch
    end
end


# Function for running the MCMC in batches.
function runMCMCbatch(currentBatch, priors, args...)
  if currentBatch == 0
      # ── Batch 0: native init + tuning (fresh), or load a resumed run state ──
      if isnothing(prevFile)
          @logmsg MCMC "No previous MCMC file indicated. Starting chain from zero"
          outputFile = outFile * ".jld2"
          isfile(outputFile) && error("Output file $outputFile already exists. Delete it or choose a different outFile before starting a new run.")

          salg = _build_samplingalg()
          # Replicate bat_sample_impl's setup, but keep the full mcmc_states.
          transformed_m, f_pretransform =
              BAT.transform_and_unshape(salg.pretransform, posterior, hmc_context)
          mcmc_states, _ = BAT.mcmc_init!(
              salg, transformed_m,
              BAT.apply_trafo_to_init(f_pretransform, salg.init),
              BAT.nop_func, hmc_context,
          )
          mcmc_states = BAT.mcmc_burnin!(nothing, mcmc_states, salg, BAT.nop_func)
          @logmsg MCMC "Tuning finished, running batches"

          run_state = RunState(mcmc_states, f_pretransform, 0)
      else
          run_state = load_runstate(prevFile * "_chainState.bin")
          run_state = rebuild_live_state(run_state)   # live closures; numeric state from disk
          if get(ENV, "RESUME_WARMUP", "1") == "1"
              @logmsg MCMC "Warming process (heap + JIT) before continuing..."
              warmup_process()                        # avoids cold-heap GC collapse on resume
          end
          @logmsg MCMC ("Continuing chains from " * prevFile)
      end
      return run_state

  else
      # ── Sampling batch (tuning persists in the carried mcmc_states) ──
      run_state     = args[1]
      starting_step = run_state.end_step
      is_last       = currentBatch >= nBatches
      remainder     = mcmcSteps % batchSteps
      nsteps        = is_last ? (remainder == 0 ? batchSteps : remainder) : batchSteps

      local batch_samples
      elapsed_time = @elapsed begin
          new_states, batch_samples =
              _sample_one_batch(run_state.mcmc_states, run_state.f_pretransform, nsteps)
          saveBatch(batch_samples, starting_step, priors, currentBatch)
          run_state = RunState(new_states, run_state.f_pretransform, starting_step + nsteps)
          save_runstate(run_state, outFile * "_chainState.bin")
      end

      _log_tree_depth(run_state.mcmc_states)

      if is_last
          mode_result = mode(batch_samples)
          mean_result = mean(batch_samples)
          std_result  = std(batch_samples)
          batch_samples = nothing
          GC.gc()
          println(" ")
          @logmsg Setup  ("Truth: $true_params")
          println(" ")
          @logmsg Output "Mode: $(mode_result)"
          @logmsg Output "Mean: $(mean_result)"
          @logmsg Output "Stddev: $(std_result)"
          println(" ")
          @logmsg Output "$(mcmcChains) output chain(s) saved to : $(outFile).jld2"
      else
          batch_samples = nothing
          GC.gc()
          remaining = elapsed_time * (nBatches - currentBatch)
          @logmsg MCMC "Ran batch $(currentBatch) of $(nBatches) in $(elapsed_time) seconds. Expectation: ~$(ceil(Int32, remaining/60)) minutes until completion."
      end

      return run_state
  end
end
