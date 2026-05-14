"""
    compute_ES_event_rates(oscSamplesES, responseMatrices, BG_ES, det_flags)

Compute ES day and night event rates (including backgrounds), or zero arrays if
`det_flags.ES_mode` is false or `oscSamplesES === nothing`.

`responseMatrices` must carry `.eff.ES_nue`, `.eff.ES_nuother`, `.bins.ES`, `.bins.cos_scatter`.
Uses shared globals: `cosz_bins`, `exposure_weights`.
"""
function compute_ES_event_rates(oscSamplesES, responseMatrices, BG_ES, det_flags)
    Ereco_bins_ES = responseMatrices.bins.ES
    if !det_flags.ES_mode || oscSamplesES === nothing
        return fill(0.0, Ereco_bins_ES.bin_number),
               fill(0.0, (cosz_bins.bin_number, Ereco_bins_ES.bin_number))
    end

    ES_nue_eff    = responseMatrices.eff.ES_nue
    ES_nuother_eff = responseMatrices.eff.ES_nuother

    eventRate_ES_nue_day     = apply_day_response(oscSamplesES.nue_day,     responseMatrices.ES.nue,     ES_nue_eff)
    eventRate_ES_nue_night   = apply_night_response(oscSamplesES.nue_night, responseMatrices.ES.nue,     ES_nue_eff)
    eventRate_ES_nuother_day   = apply_day_response(oscSamplesES.nuother_day,   responseMatrices.ES.nuother, ES_nuother_eff)
    eventRate_ES_nuother_night = apply_night_response(oscSamplesES.nuother_night, responseMatrices.ES.nuother, ES_nuother_eff)

    eventRate_ES_day   = (eventRate_ES_nue_day   .+ eventRate_ES_nuother_day)   .+ 0.5 .* BG_ES
    eventRate_ES_night = (eventRate_ES_nue_night .+ eventRate_ES_nuother_night) .+ 0.5 .* (BG_ES' .* exposure_weights)

    return eventRate_ES_day, eventRate_ES_night
end


function compute_ES_angular_event_rates_conditional(oscSamplesES, responseMatrices, BG_ES, det_flags)
    Ereco_bins_ES    = responseMatrices.bins.ES
    cos_scatter_bins = responseMatrices.bins.cos_scatter
    if !det_flags.ES_mode || oscSamplesES === nothing || !det_flags.angular_reco
        return fill(0.0, (cos_scatter_bins.bin_number, Ereco_bins_ES.bin_number))
    end

    ES_nue_eff    = responseMatrices.eff.ES_nue
    ES_nuother_eff = responseMatrices.eff.ES_nuother

    eventRate_ES_nue_day     = apply_day_response(oscSamplesES.nue_day,     responseMatrices.ES.nue,     ES_nue_eff)
    eventRate_ES_nue_night   = apply_night_response(oscSamplesES.nue_night, responseMatrices.ES.nue,     ES_nue_eff)
    eventRate_ES_nuother_day   = apply_day_response(oscSamplesES.nuother_day,   responseMatrices.ES.nuother, ES_nuother_eff)
    eventRate_ES_nuother_night = apply_night_response(oscSamplesES.nuother_night, responseMatrices.ES.nuother, ES_nuother_eff)

    eventRate_ES_day   = eventRate_ES_nue_day   .+ eventRate_ES_nuother_day
    eventRate_ES_night = eventRate_ES_nue_night .+ eventRate_ES_nuother_night

    eventRate_ES_night_summed = sum(eventRate_ES_night, dims=1)
    eventRate_ES_combined = vec(eventRate_ES_day) .+ vec(eventRate_ES_night_summed)

    eventRate_ES_angular_signal     = responseMatrices.ES.angular .* eventRate_ES_combined'
    eventRate_ES_angular_background = responseMatrices.BG.angular .* BG_ES'

    return eventRate_ES_angular_signal .+ eventRate_ES_angular_background
end


"""
    compute_ES_angular_event_rates(oscSamplesES, responseMatrices, BG_ES, det_flags)

Compute ES day/night angular (cos-scatter × Ereco) event rates including backgrounds.
Returns zero arrays if `det_flags.ES_mode` is false or angular_reco is disabled.
"""
function compute_ES_angular_event_rates(oscSamplesES, responseMatrices, BG_ES, det_flags)
    Ereco_bins_ES    = responseMatrices.bins.ES
    cos_scatter_bins = responseMatrices.bins.cos_scatter
    if !det_flags.ES_mode || oscSamplesES === nothing || !det_flags.angular_reco
        return fill(0.0, (cos_scatter_bins.bin_number, Ereco_bins_ES.bin_number)),
               fill(0.0, (cos_scatter_bins.bin_number, Ereco_bins_ES.bin_number, cosz_bins.bin_number))
    end

    ES_nue_eff    = responseMatrices.eff.ES_nue
    ES_nuother_eff = responseMatrices.eff.ES_nuother

    eventRate_ES_nue_day     = apply_day_response(oscSamplesES.nue_day,     responseMatrices.ES.nue,     ES_nue_eff)
    eventRate_ES_nue_night   = apply_night_response(oscSamplesES.nue_night, responseMatrices.ES.nue,     ES_nue_eff)
    eventRate_ES_nuother_day   = apply_day_response(oscSamplesES.nuother_day,   responseMatrices.ES.nuother, ES_nuother_eff)
    eventRate_ES_nuother_night = apply_night_response(oscSamplesES.nuother_night, responseMatrices.ES.nuother, ES_nuother_eff)

    eventRate_ES_day   = eventRate_ES_nue_day   .+ eventRate_ES_nuother_day
    eventRate_ES_night = eventRate_ES_nue_night .+ eventRate_ES_nuother_night

    eventRate_ES_day_angular_signal     = responseMatrices.ES.angular .* vec(eventRate_ES_day)'
    eventRate_ES_day_angular_background = responseMatrices.BG.angular .* vec(BG_ES)' .* 0.5
    eventRate_ES_day_angular = eventRate_ES_day_angular_signal .+ eventRate_ES_day_angular_background

    ncos  = cos_scatter_bins.bin_number
    nErec = Ereco_bins_ES.bin_number
    ncosz = cosz_bins.bin_number
    eventRate_ES_night_angular_signal     = reshape(responseMatrices.ES.angular, ncos, nErec, 1) .*
                                            reshape(eventRate_ES_night', 1, nErec, ncosz)
    eventRate_ES_night_angular_background = reshape(responseMatrices.BG.angular, ncos, nErec, 1) .*
                                            reshape(exposure_weights' .* BG_ES, 1, nErec, ncosz)
    eventRate_ES_night_angular = eventRate_ES_night_angular_signal .+ eventRate_ES_night_angular_background .* 0.5

    # Angular cut from det_flags
    cos_cut = det_flags.angular_cos_cut
    N = cos_scatter_bins.bin_number
    edges   = range(cos_scatter_bins.min, cos_scatter_bins.max, length=N + 1)
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    amask   = centers .>= cos_cut
    eventRate_ES_day_angular   .*= amask
    eventRate_ES_night_angular .*= reshape(amask, :, 1, 1)

    return eventRate_ES_day_angular, eventRate_ES_night_angular
end


"""
    compute_CC_event_rates(oscSamplesCC, responseMatrices, BG_CC, det_flags)

Compute CC day and night event rates (including backgrounds), or zero arrays if
`det_flags.CC_mode` is false or `oscSamplesCC === nothing`.
"""
function compute_CC_event_rates(oscSamplesCC, responseMatrices, BG_CC, det_flags)
    Ereco_bins_CC = responseMatrices.bins.CC
    if !det_flags.CC_mode || oscSamplesCC === nothing
        return fill(0.0, Ereco_bins_CC.bin_number),
               fill(0.0, (cosz_bins.bin_number, Ereco_bins_CC.bin_number))
    end

    CC_eff = responseMatrices.eff.CC

    eventRate_CC_day   = apply_day_response(oscSamplesCC.day,   responseMatrices.CC, CC_eff) .+ 0.5 .* BG_CC
    eventRate_CC_night = apply_night_response(oscSamplesCC.night, responseMatrices.CC, CC_eff) .+ 0.5 .* (BG_CC' .* exposure_weights)

    return eventRate_CC_day, eventRate_CC_night
end


"""
    compute_CC_inclusive_event_rates(oscSamplesCC, responseMatrices, det_flags)

Compute CC signal event rates for inclusive (ES+CC combined) analysis. Returns arrays
in ES Ereco bins so they can be added directly to ES rates.
"""
function compute_CC_inclusive_event_rates(oscSamplesCC, responseMatrices, det_flags)
    Ereco_bins_ES    = responseMatrices.bins.ES
    Ereco_bins_CC    = responseMatrices.bins.CC
    cos_scatter_bins = responseMatrices.bins.cos_scatter
    if !det_flags.CC_mode || !det_flags.inclusive_analysis || oscSamplesCC === nothing
        if det_flags.angular_reco
            return fill(0.0, (cos_scatter_bins.bin_number, Ereco_bins_ES.bin_number)),
                   fill(0.0, (cos_scatter_bins.bin_number, Ereco_bins_ES.bin_number, cosz_bins.bin_number))
        else
            return fill(0.0, Ereco_bins_ES.bin_number),
                   fill(0.0, (cosz_bins.bin_number, Ereco_bins_ES.bin_number))
        end
    end

    CC_incl_eff = responseMatrices.eff.CC_incl
    cc_day   = apply_day_response(oscSamplesCC.day,   responseMatrices.CC_inclusive, CC_incl_eff)
    cc_night = apply_night_response(oscSamplesCC.night, responseMatrices.CC_inclusive, CC_incl_eff)

    if det_flags.angular_reco
        ncos  = cos_scatter_bins.bin_number
        nErec = Ereco_bins_ES.bin_number
        ncosz = cosz_bins.bin_number
        cc_day_angular   = responseMatrices.BG.angular .* vec(cc_day)'
        cc_night_angular = reshape(responseMatrices.BG.angular, ncos, nErec, 1) .*
                           reshape(cc_night', 1, nErec, ncosz)
        return cc_day_angular, cc_night_angular
    else
        return cc_day, cc_night
    end
end


function compute_CC_angular_event_rates_signal_only(oscSamplesCC, responseMatrices, det_flags)
    Ereco_bins_CC    = responseMatrices.bins.CC
    cos_scatter_bins = responseMatrices.bins.cos_scatter
    if !det_flags.CC_mode || oscSamplesCC === nothing || !det_flags.angular_reco
        return fill(0.0, (cos_scatter_bins.bin_number, Ereco_bins_CC.bin_number))
    end

    CC_eff = responseMatrices.eff.CC
    eventRate_CC_day   = apply_day_response(oscSamplesCC.day,   responseMatrices.CC, CC_eff)
    eventRate_CC_night = apply_night_response(oscSamplesCC.night, responseMatrices.CC, CC_eff)

    ncosz = cosz_bins.bin_number
    eventRate_CC_day_angular   = responseMatrices.CC.angular .* vec(eventRate_CC_day)'
    eventRate_CC_night_angular = responseMatrices.CC.angular .*
                                 reshape(eventRate_CC_night', 1, Ereco_bins_CC.bin_number, ncosz)

    return eventRate_CC_day_angular, eventRate_CC_night_angular
end
