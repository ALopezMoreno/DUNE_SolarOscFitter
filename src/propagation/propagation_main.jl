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

function propagateSamples(unoscillatedSample, responseMatrices, params, solarModel, bin_edges, raw_backgrounds; verbose=false)

    # 1) Mixing parameters
    mixingPars = get_mixing_parameters(params)

    # 2) Earth propagation setup
    oscProbs_1e, earth_norm_vector, lookup =
        setup_earth_propagation(E_calc, mixingPars, params)

    # 3) Backgrounds
    BG_ES, BG_CC = normalize_backgrounds(raw_backgrounds, params)

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
    if verbose
        println()
        @logmsg Setup "── Unoscillated event rates (true-E, pre-efficiency) ──"
        if ES_mode
            es_nue_8B   = sum(unoscillatedSample.ES_nue_8B)   * params.integrated_8B_flux
            es_nue_hep  = sum(unoscillatedSample.ES_nue_hep)  * params.integrated_HEP_flux
            es_numu_8B  = sum(unoscillatedSample.ES_nuother_8B)  * params.integrated_8B_flux
            es_numu_hep = sum(unoscillatedSample.ES_nuother_hep) * params.integrated_HEP_flux
            @logmsg Setup "  ES  νe   8B : $(sci_notation(es_nue_8B))"
            @logmsg Setup "  ES  νe   hep: $(sci_notation(es_nue_hep))"
            @logmsg Setup "  ES  νμτ  8B : $(sci_notation(es_numu_8B))"
            @logmsg Setup "  ES  νμτ  hep: $(sci_notation(es_numu_hep))"
            @logmsg Setup "  ES  total   : $(sci_notation(es_nue_8B + es_nue_hep + es_numu_8B + es_numu_hep))"
        end
        if CC_mode
            cc_8B  = sum(unoscillatedSample.CC_8B)  * params.integrated_8B_flux
            cc_hep = sum(unoscillatedSample.CC_hep) * params.integrated_HEP_flux
            @logmsg Setup "  CC  νe   8B : $(sci_notation(cc_8B))"
            @logmsg Setup "  CC  νe   hep: $(sci_notation(cc_hep))"
            @logmsg Setup "  CC  total   : $(sci_notation(cc_8B + cc_hep))"
        end
        println()
    end

    oscillatedSample = compute_oscillated_samples(unoscillatedSample, params, oscProbs)

    if ES_mode
        unosc_nue_total = sum(unoscillatedSample.ES_nue_8B) * params.integrated_8B_flux
        osc_total_day   = sum(oscillatedSample.ES.nue_day) + sum(oscillatedSample.ES.nuother_day)
    end

    # 6) ES reco event rates
    if angular_reco
        eventRate_ES_day, eventRate_ES_night =
            compute_ES_angular_event_rates(oscillatedSample.ES, responseMatrices, BG_ES)
    else
        eventRate_ES_day, eventRate_ES_night =
            compute_ES_event_rates(oscillatedSample.ES, responseMatrices, BG_ES)
    end

    # 7) CC reco event rates
    if inclusive_analysis
        cc_incl_day, cc_incl_night =
            compute_CC_inclusive_event_rates(oscillatedSample.CC, responseMatrices)
        # Collapse to 1D Ereco_ES spectrum for diagnostic logging (sum over angle and cosz)
        global CC_incl_spectrum = if angular_reco
            vec(sum(cc_incl_day, dims=1)) .+ vec(sum(cc_incl_night, dims=(1, 3)))
        else
            cc_incl_day .+ vec(sum(cc_incl_night, dims=1))
        end
        eventRate_ES_day   .+= cc_incl_day
        eventRate_ES_night .+= cc_incl_night
        eventRate_CC_day   = fill(eltype(cc_incl_day)(0), Ereco_bins_CC.bin_number)
        eventRate_CC_night = fill(eltype(cc_incl_night)(0), (cosz_bins.bin_number, Ereco_bins_CC.bin_number))
    else
        eventRate_CC_day, eventRate_CC_night =
            compute_CC_event_rates(oscillatedSample.CC, responseMatrices, BG_CC)
    end

    # 8) Debug plots (optional) --- HARD CODED FLAG
    DEBUG_PLOTS = false  

    if DEBUG_PLOTS
        #PropagationDebug.debug_plot_CC_backgrounds(backgrounds)
        #PropagationDebug.debug_heatmap_response_CC(responseMatrices)
        PropagationDebug.debug_heatmap_CC_night(eventRate_ES_night)
        sleep(10)
        myP = plot(eventRate_ES_day,
        seriestype = :bar,
        xlabel = "Bin number",
        ylabel = "Count",
        legend = false)
        display(myP)
        sleep(200) # wait for a while
    end

    return eventRate_ES_day, eventRate_CC_day, eventRate_ES_night, eventRate_CC_night, BG_ES, BG_CC
end
