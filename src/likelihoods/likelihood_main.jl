include("../likelihoods/likelihood_core.jl")
include("../likelihoods/likelihood_debug.jl")
include("../likelihoods/likelihood_statistical.jl")
include("../likelihoods/likelihood_builder.jl")

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

likelihood_all_samples = logfuncdensity(total_llh)