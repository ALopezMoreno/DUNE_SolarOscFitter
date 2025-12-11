include(joinpath(@__DIR__, "propagation_core.jl"))
include(joinpath(@__DIR__, "propagation_osc.jl"))
include(joinpath(@__DIR__, "propagation_bg.jl"))
include(joinpath(@__DIR__, "propagation_reco.jl"))
include(joinpath(@__DIR__, "propagation_debug.jl"))

using .PropagationDebug

"""
    propagateSamples(unoscillatedSample, responseMatrices, params, solarModel,
                     bin_edges, raw_backgrounds)

Main function to propagate unoscillated neutrino samples through oscillations
and detector response to calculate expected event rates.

Arguments:
- unoscillatedSample: Unoscillated MC event rates by channel and process
- responseMatrices: Detector response matrices for energy reconstruction
- angularResponseMatrices: Angular reconstruction matrices for background discrimination
- params: Oscillation and systematic parameters
- solarModel: Solar neutrino production model
- bin_edges: Energy bin edges for analysis (bin centres computed elsewhere)
- raw_backgrounds: Background event samples

Returns:
- eventRate_ES_day
- eventRate_CC_day
- eventRate_ES_night
- eventRate_CC_night
- BG_ES
- BG_CC
"""

function propagateSamples(unoscillatedSample, responseMatrices, params, solarModel, bin_edges, raw_backgrounds)

    # 1) Mixing parameters
    mixingPars = get_mixing_parameters(params)

    # 2) Earth propagation setup
    oscProbs_1e, earth_norm_vector, lookup =
        setup_earth_propagation(E_calc, mixingPars, params)

    # 3) Backgrounds
    backgrounds, BG_ES, BG_CC = normalize_backgrounds(raw_backgrounds, params)

    # 4) Oscillation probabilities (day/night, 8B/hep, ν_e/ν_other)
    oscProbs = compute_oscillation_probabilities(
        E_calc,
        mixingPars,
        solarModel,
        params,
        oscProbs_1e,
        earth_norm_vector,
        lookup,
    )

    # 5) Oscillated event rates
    oscillatedSample = compute_oscillated_samples(unoscillatedSample, params, oscProbs)

    # 6) ES reco event rates
    if angular_reco
        eventRate_ES_day, eventRate_ES_night =
            compute_ES_angular_event_rates(oscillatedSample.ES, responseMatrices, BG_ES)
    else
        eventRate_ES_day, eventRate_ES_night =
            compute_ES_event_rates(oscillatedSample.ES, responseMatrices, BG_ES)
    end

    # 7) CC reco event rates
    eventRate_CC_day, eventRate_CC_night =
        compute_CC_event_rates(oscillatedSample.CC, responseMatrices, BG_CC)

    # 8) Debug plots (optional) --- HARD CODED FLAG
    DEBUG_PLOTS = false  

    if DEBUG_PLOTS
        PropagationDebug.debug_plot_CC_backgrounds(backgrounds)
        PropagationDebug.debug_heatmap_response_CC(responseMatrices)
        PropagationDebug.debug_heatmap_CC_night(eventRate_CC_night)

        sleep(200) # wait for a while
    end

    return eventRate_ES_day, eventRate_CC_day, eventRate_ES_night, eventRate_CC_night, BG_ES, BG_CC
end
