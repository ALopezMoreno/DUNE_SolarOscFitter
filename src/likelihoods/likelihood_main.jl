include(joinpath(@__DIR__, "likelihood_core.jl"))
include(joinpath(@__DIR__, "likelihood_debug.jl"))
include(joinpath(@__DIR__, "likelihood_statistical.jl"))
include(joinpath(@__DIR__, "likelihood_builder.jl"))

likelihood_inputs = LikelihoodInputs(
    ereco_data,
    bin_edges,
    responseMatrices,
    solarModel,
    unoscillatedSample,
    backgrounds,
    propagateSamples,
    ES_mode,
    CC_mode,
    index_ES,
    index_CC,
)

total_llh = make_likelihood(likelihood_inputs;
    use_ES = ES_mode,
    use_CC = CC_mode,
    ES_llh = llh_ES_poisson,
    CC_llh = llh_CC_poisson,
)

per_bin_llh = make_perbin_likelihood(likelihood_inputs;
    use_ES = ES_mode,
    use_CC = CC_mode,
    ES_llh = llh_ES_poisson_perbin,
    CC_llh = llh_CC_poisson_perbin,
)

if run_mode == "MCMC"
    likelihood_all_samples = logfuncdensity(total_llh)
end