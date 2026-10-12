# Build one likelihood closure per detector, then sum them.
# Oscillation parameters are shared; background nuisance params are namespaced by detector.
detector_llh_fns    = Dict{String, Function}()
detector_perbin_fns = Dict{String, Function}()

for (dname, out) in detector_outputs
    det = detector_configs[dname]
    li  = out.likelihood_inputs

    det_llh = make_likelihood(li;
        use_ES = det.ES_mode,
        use_CC = det.CC_mode && (!det.inclusive_analysis || det.semi_inclusive_analysis),
        ES_llh = det.angular_reco ? llh_ES_angle   : llh_ES_poisson,
        CC_llh = llh_CC_poisson,
    )
    det_perbin = make_perbin_likelihood(li;
        use_ES = det.ES_mode,
        use_CC = det.CC_mode && (!det.inclusive_analysis || det.semi_inclusive_analysis),
        ES_llh = det.angular_reco ? llh_ES_angle_perbin : llh_ES_poisson_perbin,
        CC_llh = llh_CC_poisson_perbin,
    )
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

per_bin_llh = parameters -> sum(f(parameters) for f in _det_perbin_tuple)

global likelihood_all_samples = logfuncdensity(total_llh)
