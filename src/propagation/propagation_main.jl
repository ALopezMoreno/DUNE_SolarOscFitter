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

"""
    propagateSamples(unoscillatedSample, responseMatrices, params, solarModel,
                     bin_edges, raw_backgrounds, det_flags; verbose=false)

Propagate unoscillated samples through oscillations and detector response.
`det_flags` is a named tuple with fields:
  ES_mode, CC_mode, angular_reco, inclusive_analysis, angular_cos_cut, det_name

Returns 7 values:
  eventRate_ES_day, eventRate_CC_day, eventRate_ES_night, eventRate_CC_night,
  BG_ES, BG_CC, CC_incl_spectrum
where `CC_incl_spectrum` is `nothing` when `inclusive_analysis` is false.
"""
function compute_shared_osc_probs(params, solarModel)
    mixingPars = get_mixing_parameters(params)
    oscProbs_1e, earth_norm_vector, lookup =
        setup_earth_propagation(E_calc, mixingPars, params)
    return compute_oscillation_probabilities(
        E_calc, mixingPars, solarModel, params, oscProbs_1e, earth_norm_vector, lookup,
    )
end

function propagateSamples(unoscillatedSample, responseMatrices, params, solarModel,
                          bin_edges, raw_backgrounds, det_flags;
                          precomputed_osc=nothing, verbose=false)

    # Oscillation probabilities — shared across detectors; skip if pre-supplied
    oscProbs = if precomputed_osc === nothing
        compute_shared_osc_probs(params, solarModel)
    else
        precomputed_osc
    end

    # 3) Backgrounds (detector-specific)
    BG_ES, BG_CC = normalize_backgrounds(raw_backgrounds, params, det_flags.det_name)

    # 5) Verbose unoscillated rate logging
    if verbose
        println()
        @logmsg Setup "── [$(det_flags.det_name)] Unoscillated event rates (true-E, pre-efficiency) ──"
        if det_flags.ES_mode
            es_nue_8B   = sum(unoscillatedSample.ES_nue_8B)      * params.integrated_8B_flux
            es_nue_hep  = sum(unoscillatedSample.ES_nue_hep)     * params.integrated_HEP_flux
            es_numu_8B  = sum(unoscillatedSample.ES_nuother_8B)  * params.integrated_8B_flux
            es_numu_hep = sum(unoscillatedSample.ES_nuother_hep) * params.integrated_HEP_flux
            @logmsg Setup "  ES  νe   8B : $(sci_notation(es_nue_8B))"
            @logmsg Setup "  ES  νe   hep: $(sci_notation(es_nue_hep))"
            @logmsg Setup "  ES  νμτ  8B : $(sci_notation(es_numu_8B))"
            @logmsg Setup "  ES  νμτ  hep: $(sci_notation(es_numu_hep))"
            @logmsg Setup "  ES  total   : $(sci_notation(es_nue_8B + es_nue_hep + es_numu_8B + es_numu_hep))"
        end
        if det_flags.CC_mode
            cc_8B  = sum(unoscillatedSample.CC_8B)  * params.integrated_8B_flux
            cc_hep = sum(unoscillatedSample.CC_hep) * params.integrated_HEP_flux
            @logmsg Setup "  CC  νe   8B : $(sci_notation(cc_8B))"
            @logmsg Setup "  CC  νe   hep: $(sci_notation(cc_hep))"
            @logmsg Setup "  CC  total   : $(sci_notation(cc_8B + cc_hep))"
        end
        println()
    end

    oscillatedSample = compute_oscillated_samples(unoscillatedSample, params, oscProbs;
                                                   es_mode=det_flags.ES_mode, cc_mode=det_flags.CC_mode)

    # 6) ES reco event rates
    if det_flags.angular_reco
        eventRate_ES_day, eventRate_ES_night =
            compute_ES_angular_event_rates(oscillatedSample.ES, responseMatrices, BG_ES, det_flags)
    else
        eventRate_ES_day, eventRate_ES_night =
            compute_ES_event_rates(oscillatedSample.ES, responseMatrices, BG_ES, det_flags)
    end

    # 7) CC reco event rates
    CC_incl_spectrum = nothing
    if det_flags.semi_inclusive_analysis
        f_above = responseMatrices.CC_split.above
        f_below = responseMatrices.CC_split.below

        # Forward hemisphere: inclusive CC (scaled to above-cut fraction, no double-counting)
        cc_incl_day, cc_incl_night =
            compute_CC_inclusive_event_rates(oscillatedSample.CC, responseMatrices, det_flags)
        # CC_incl_spectrum reflects the f_above-scaled contribution actually added to ES rates.
        CC_incl_spectrum = f_above .* if det_flags.angular_reco
            vec(sum(cc_incl_day, dims=1)) .+ vec(sum(cc_incl_night, dims=(1, 3)))
        else
            cc_incl_day .+ vec(sum(cc_incl_night, dims=1))
        end
        eventRate_ES_day   .+= f_above .* cc_incl_day
        eventRate_ES_night .+= f_above .* cc_incl_night

        # Backward hemisphere: CC signal + backgrounds (scaled to below-cut fraction) + mis-ID ES.
        # Scaling by f_below applies uniformly to CC signal and CC backgrounds
        # (both assumed isotropically distributed in angle).
        eventRate_CC_day, eventRate_CC_night =
            compute_CC_event_rates(oscillatedSample.CC, responseMatrices, BG_CC, det_flags)
        eventRate_CC_day   .*= f_below
        eventRate_CC_night .*= f_below
        misID_day, misID_night =
            compute_ES_misID_CC_event_rates(oscillatedSample.ES, responseMatrices, det_flags)
        eventRate_CC_day   .+= misID_day
        eventRate_CC_night .+= misID_night

    elseif det_flags.inclusive_analysis
        cc_incl_day, cc_incl_night =
            compute_CC_inclusive_event_rates(oscillatedSample.CC, responseMatrices, det_flags)
        CC_incl_spectrum = if det_flags.angular_reco
            vec(sum(cc_incl_day, dims=1)) .+ vec(sum(cc_incl_night, dims=(1, 3)))
        else
            cc_incl_day .+ vec(sum(cc_incl_night, dims=1))
        end
        eventRate_ES_day   .+= cc_incl_day
        eventRate_ES_night .+= cc_incl_night
        Ereco_bins_CC = responseMatrices.bins.CC
        eventRate_CC_day   = fill(eltype(cc_incl_day)(0),   Ereco_bins_CC.bin_number)
        eventRate_CC_night = fill(eltype(cc_incl_night)(0), (cosz_bins.bin_number, Ereco_bins_CC.bin_number))
    else
        eventRate_CC_day, eventRate_CC_night =
            compute_CC_event_rates(oscillatedSample.CC, responseMatrices, BG_CC, det_flags)
    end

    return eventRate_ES_day, eventRate_CC_day, eventRate_ES_night, eventRate_CC_night,
           BG_ES, BG_CC, CC_incl_spectrum
end
