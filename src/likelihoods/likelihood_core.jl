# Structure of inputs for the likelihood function called in BAT
struct LikelihoodInputs
    nObserved      # should have ES_day, ES_night, CC_day, CC_night
    energies
    Mreco
    SSM
    MC_no_osc
    BG
    f              # propagation function
    ES_mode::Bool
    CC_mode::Bool
    index_ES::Int
    index_CC::Int
end

function expected_rates(d::LikelihoodInputs, parameters)
    expectedRate_ES_day,
    expectedRate_CC_day,
    expectedRate_ES_night,
    expectedRate_CC_night,
    # eventRate_ES_angular,
    BG_ES_tot,
    BG_CC_tot = d.f(d.MC_no_osc, d.Mreco, parameters, d.SSM, d.energies, d.BG)

    return (
        ES_day    = expectedRate_ES_day,
        CC_day    = expectedRate_CC_day,
        ES_night  = expectedRate_ES_night,
        CC_night  = expectedRate_CC_night,
        # ES_angular = eventRate_ES_angular,
        BG_ES_tot = BG_ES_tot,
        BG_CC_tot = BG_CC_tot,
    )
end