# Build one likelihood closure per detector, then sum them.
# Oscillation parameters are shared; background nuisance params are namespaced by detector.
detector_llh_fns    = OrderedDict{String, Function}()   # same order as detector_configs
detector_perbin_fns = OrderedDict{String, Function}()

for (dname, out) in detector_outputs
    det = detector_configs[dname]
    li  = out.likelihood_inputs

    use_CC = det.CC_mode && (!det.inclusive_analysis || det.semi_inclusive_analysis)

    # Channel likelihoods: plain Poisson, or with the MC statistical uncertainty of the
    # predictions (mc_stat_likelihood; src/likelihoods/mcstat.jl).
    if det.mc_stat_likelihood == "Poisson"
        ES_llh,  CC_llh  = det.angular_reco ? llh_ES_angle : llh_ES_poisson, llh_CC_poisson
        ES_pbin, CC_pbin = det.angular_reco ? llh_ES_angle_perbin : llh_ES_poisson_perbin, llh_CC_poisson_perbin
    else
        method = mcstat_method(det.mc_stat_likelihood)
        shared = det.mc_stat_day_night == "shared"
        a_ES = haskey(out.mc_counts, :ES) ? out.mc_counts[:ES].a : Float64[]
        a_CC = haskey(out.mc_counts, :CC) ? out.mc_counts[:CC].a : Float64[]
        ES_llh,  CC_llh  = llh_ES_mcstat(method, shared, a_ES), llh_CC_mcstat(method, shared, a_CC)
        ES_pbin, CC_pbin = llh_ES_mcstat_perbin(method, shared, a_ES), llh_CC_mcstat_perbin(method, shared, a_CC)
    end

    det_llh = make_likelihood(li; use_ES = det.ES_mode, use_CC = use_CC, ES_llh = ES_llh, CC_llh = CC_llh)
    det_perbin = make_perbin_likelihood(li; use_ES = det.ES_mode, use_CC = use_CC, ES_llh = ES_pbin, CC_llh = CC_pbin)
    detector_llh_fns[dname]    = det_llh
    detector_perbin_fns[dname] = det_perbin
end

_det_llh_tuple    = Tuple(values(detector_llh_fns))
_det_perbin_tuple = Tuple(values(detector_perbin_fns))
_shared_ssm       = solarModel

# Sum over the detector likelihoods by tuple recursion rather than a generator: a generator
# stores the (constant) closures next to the (differentiable) oscillation probabilities,
# which Enzyme's activity analysis rejects.
_sum_det_llh(::Tuple{}, parameters, osc) = false
@inline _sum_det_llh(fns::Tuple, parameters, osc) =
    first(fns)(parameters; precomputed_osc=osc) + _sum_det_llh(Base.tail(fns), parameters, osc)

total_llh = let fns = _det_llh_tuple, ssm = _shared_ssm
    parameters -> begin
        shared_osc = compute_shared_osc_probs(parameters, ssm)
        _sum_det_llh(fns, parameters, shared_osc)
    end
end

# Per-bin log L, one entry per detector: the detectors' bin grids differ, so they cannot be summed.
per_bin_llh = parameters -> OrderedDict(dname => f(parameters) for (dname, f) in detector_perbin_fns)

global likelihood_all_samples = logfuncdensity(total_llh)
