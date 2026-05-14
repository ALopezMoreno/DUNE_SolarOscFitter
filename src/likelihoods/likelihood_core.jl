# Structure of inputs for the likelihood function called in BAT
struct LikelihoodInputs{F<:Function}
    nObserved      # should have ES_day, ES_night, CC_day, CC_night
    energies
    Mreco
    SSM
    MC_no_osc
    BG
    f::F           # propagation closure: (unosc, resp, params, ssm, edges, bg) -> 7-tuple
    ES_mode::Bool
    CC_mode::Bool
    index_ES::Int
    index_CC::Int
    det_name::String
end

function expected_rates(d::LikelihoodInputs, parameters; precomputed_osc=nothing)
    expectedRate_ES_day,
    expectedRate_CC_day,
    expectedRate_ES_night,
    expectedRate_CC_night,
    BG_ES_tot,
    BG_CC_tot,
    _CC_incl_spectrum = d.f(d.MC_no_osc, d.Mreco, parameters, d.SSM, d.energies, d.BG;
                            precomputed_osc=precomputed_osc)

    return (
        ES_day    = expectedRate_ES_day,
        CC_day    = expectedRate_CC_day,
        ES_night  = expectedRate_ES_night,
        CC_night  = expectedRate_CC_night,
        BG_ES_tot = BG_ES_tot,
        BG_CC_tot = BG_CC_tot,
    )
end
