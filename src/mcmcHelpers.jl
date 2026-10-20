#=
mcmcHelpers.jl — MCMC in batches with tuning carried across batches and resumes (after Newthrino,
P. Eller). `bat_sample` returns only the chain states and drops the tuner states, so tuning would
restart every batch; instead BAT's MCMC loop is driven directly and the full `Vector{MCMCState}`
(chain, proposal and transform tuners, temperer) is kept and serialised.
=#

using BAT: DensitySampleVector, MCMCState, MCMCBurninAlgorithm
using Serialization  # For run-state persistence


# Run state carried across batches and serialised for resume. `f_pretransform` maps samples back
# to parameter space.
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


# BAT's native HMC tuning. With a warm start (warmStart.jl) the affine transform comes from the
# previous posterior and stays fixed: RAM tuning would pull it back towards identity.
_build_samplingalg() = warm_start === nothing ?
    TransformedMCMC(
        proposal    = proposal_algorithm,
        nchains     = mcmcChains,
        init        = init,
        burnin      = burnin,
        convergence = convergence,
        nsteps      = batchSteps,   # placeholder; per-batch length is passed explicitly
    ) :
    TransformedMCMC(
        proposal           = proposal_algorithm,
        adaptive_transform = warm_start.transform,
        transform_tuning   = BAT.NoMCMCTransformTuning(),
        nchains            = mcmcChains,
        init               = init,
        burnin             = burnin,
        convergence        = convergence,
        nsteps             = batchSteps,
    )

# Scales of the affine transform (the square root of the mass-matrix diagonal), in units of
# each parameter's prior width (BAT's PriorToNormal space). Identity = 1 everywhere.
@noinline function log_transform_scales(label, scales, names)
    order = sortperm(scales)
    show_idx = unique(vcat(order[1:min(3, end)], order[max(1, end - 2):end]))
    @logmsg MCMC "$label transform scales (posterior/prior width): " *
                 join(["$(names[i]) $(round(scales[i], sigdigits = 2))" for i in show_idx], ", ") *
                 " (smallest and largest of $(length(scales)))"
end


# Load-bearing: a throwaway mcmc_init! warms the GC heap and the JIT as a fresh run's init does.
# Without it the first resumed batch falls into a GC collapse (~14x slower at 10 chains). The
# loaded chain state and the chains' RNG are untouched.
function warmup_process()
    salg = _build_samplingalg()
    tmw, fptw = BAT.transform_and_unshape(salg.pretransform, posterior, hmc_context)
    warm_states, _ = BAT.mcmc_init!(
        salg, tmw, BAT.apply_trafo_to_init(fptw, salg.init), BAT.nop_func, hmc_context)
    warm_states = nothing
    GC.gc()
    return nothing
end


# On resume, rebuild each chain's target and proposal gradient from the live session and keep only
# the numeric state (positions, tuners, transform, step size, metric) from the checkpoint:
# deserialised closures run the same arithmetic with ~7x the allocation, so a resume ran ~2x slower.
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


# Append one batch to the JLD2 output as a new group (JLD2 cannot extend datasets).
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

  # 4) write as the next free batch group: a resumed run restarts its counter at 1
  fileName = outFile * ".jld2"
  try
    jldopen(fileName, "a+") do f
      name = next_batch_name(f)
      name == "batch_$batchNumber" || @logmsg MCMC "Batch $batchNumber written as $name ($fileName already has earlier batches)"
      g = JLD2.Group(f, name)
      for (k, v) in param_data
          g[string(k)] = v
      end
      g["weights"] = weights
      g["stepno"]  = stepno
      g["chainid"] = chainid
      @isdefined(store_run_record!) && store_run_record!(f)   # run_config (runRecord.jl), once
    end
  catch err
    @error "Failed to write MCMC data to $fileName; stopping so the samples are not silently lost." exception = (err, catch_backtrace())
    rethrow()
  end
end



# One batch on the live states (tuners kept); samples are returned in parameter space.
function _sample_one_batch(mcmc_states, f_pretransform, nsteps)
    BAT.next_cycle!.(mcmc_states)
    outputs     = BAT._empty_chain_outputs.(mcmc_states)
    new_states  = BAT.mcmc_iterate!!(outputs, mcmc_states; max_nsteps = nsteps, nonzero_weights = true)
    merged      = BAT._merge_chain_outputs(first(new_states), outputs)
    batch_samples = BAT.transform_samples(BAT.inverse(f_pretransform), merged)
    return new_states, batch_samples
end

# Gradients per MCMC step in the last batch against the 2^hmc_max_depth cap (counted by the
# NightOsc Enzyme rule, so Enzyme + nightOsc only). Near the cap, NUTS is cut short of a U-turn.
const _GRADIENTS_AT_BATCH_START = Ref(0)
function log_sampler_cost(nsteps, nchains)
    g = NightOsc.GRADIENT_COUNT[]
    used = g - _GRADIENTS_AT_BATCH_START[]
    _GRADIENTS_AT_BATCH_START[] = g
    (used > 0 && nsteps > 0) || return nothing
    per_step, cap = used / (nsteps * nchains), 2^hmc_max_depth
    @logmsg MCMC @sprintf("Gradients per MCMC step: %.1f of at most %d (hmc_max_depth = %d)", per_step, cap, hmc_max_depth)
    per_step > 0.9 * cap &&
        @warn "NUTS trajectories hit the tree-depth cap ($(round(per_step, digits=1)) of $cap gradients per step): " *
              "they are cut short of a U-turn, which lowers the effective sample size. Consider raising " *
              "hmc_max_depth or tuning longer (nTuning × maxTuningAttempts)."
    return nothing
end

# Last-step NUTS tree depth per chain (the depth of each chain's final transition only).
function _log_tree_depth(mcmc_states)
    try
        ds = [Int(AdvancedHMC.stat(s.chain_state.proposal.transition).tree_depth) for s in mcmc_states]
        @logmsg MCMC "NUTS tree depth of the last transition per chain: $ds"
    catch
    end
end


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
          try
              A = mcmc_states[1].chain_state.f_transform.A
              log_transform_scales("Chain 1 tuned", sqrt.(diag(A * A')), collect(keys(BAT.varshape(prior))))
          catch err
              @logmsg MCMC "Could not log the tuned transform scales: $(sprint(showerror, err))"
          end
          _GRADIENTS_AT_BATCH_START[] = NightOsc.GRADIENT_COUNT[]   # count sampling batches only

          run_state = RunState(mcmc_states, f_pretransform, 0)
      else
          run_state = load_runstate(prevFile * "_chainState.bin")
          run_state = rebuild_live_state(run_state)   # live closures; numeric state from disk
          if get(ENV, "RESUME_WARMUP", "1") == "1"
              @logmsg MCMC "Warming process (heap + JIT) before continuing..."
              warmup_process()                        # avoids cold-heap GC collapse on resume
          end
          @logmsg MCMC ("Continuing chains from " * prevFile)
          _GRADIENTS_AT_BATCH_START[] = NightOsc.GRADIENT_COUNT[]   # not the warm-up's gradients
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
      log_sampler_cost(nsteps, length(run_state.mcmc_states))

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
