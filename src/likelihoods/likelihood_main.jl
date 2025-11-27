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

function expected_rates(d::LikelihoodInputs, parameters)
    expectedRate_ES_day,
    expectedRate_CC_day,
    expectedRate_ES_night,
    expectedRate_CC_night,
    BG_ES_tot,
    BG_CC_tot = d.f(d.MC_no_osc, d.Mreco, parameters, d.SSM, d.energies, d.BG)

    return (
        ES_day    = expectedRate_ES_day,
        CC_day    = expectedRate_CC_day,
        ES_night  = expectedRate_ES_night,
        CC_night  = expectedRate_CC_night,
        BG_ES_tot = BG_ES_tot,
        BG_CC_tot = BG_CC_tot,
    )
end

total_llh = make_likelihood(likelihood_inputs;
    use_ES = ES_mode,
    use_CC = CC_mode,
    ES_llh = llh_ES_poisson,
    CC_llh = llh_CC_poisson,
)


likelihood_all_samples = logfuncdensity(total_llh)