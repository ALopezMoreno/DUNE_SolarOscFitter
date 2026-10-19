"""
    compute_ES_event_rates(oscSamplesES, responseMatrices, BG_ES, det_flags)

Compute ES day and night event rates (including backgrounds), or zero arrays if
`det_flags.ES_mode` is false or `oscSamplesES === nothing`.

`responseMatrices` must carry `.eff.ES_nue`, `.eff.ES_nuother`, `.bins.ES`, `.bins.cos_scatter`.
Uses shared constants: `N_COSZ_BINS`, the signal split `SIGNAL_DAY_FRACTION`/`SIGNAL_NIGHT_FRACTION` (flux-
weighted) and the background split `BG_DAY_FRACTION`/`BG_NIGHT_FRACTION` with `exposure_weights_bg`
(time-weighted), SPEC 4.6.
"""
function compute_ES_event_rates(oscSamplesES, responseMatrices, BG_ES, det_flags)
    Ereco_bins_ES = responseMatrices.bins.ES
    if !det_flags.ES_mode || oscSamplesES === nothing
        return fill(0.0, Ereco_bins_ES.bin_number),
               fill(0.0, (N_COSZ_BINS, Ereco_bins_ES.bin_number))
    end

    ES_nue_eff    = responseMatrices.eff.ES_nue
    ES_nuother_eff = responseMatrices.eff.ES_nuother

    eventRate_ES_nue_day     = apply_day_response(oscSamplesES.nue_day,     responseMatrices.ES.nue,     ES_nue_eff; scale=SIGNAL_DAY_FRACTION)
    eventRate_ES_nue_night   = apply_night_response(oscSamplesES.nue_night, responseMatrices.ES.nue,     ES_nue_eff; scale=SIGNAL_NIGHT_FRACTION)
    eventRate_ES_nuother_day   = apply_day_response(oscSamplesES.nuother_day,   responseMatrices.ES.nuother, ES_nuother_eff; scale=SIGNAL_DAY_FRACTION)
    eventRate_ES_nuother_night = apply_night_response(oscSamplesES.nuother_night, responseMatrices.ES.nuother, ES_nuother_eff; scale=SIGNAL_NIGHT_FRACTION)

    eventRate_ES_day   = (eventRate_ES_nue_day   .+ eventRate_ES_nuother_day)   .+ BG_DAY_FRACTION .* BG_ES
    eventRate_ES_night = (eventRate_ES_nue_night .+ eventRate_ES_nuother_night) .+ BG_NIGHT_FRACTION .* (BG_ES' .* exposure_weights_bg)

    return eventRate_ES_day, eventRate_ES_night
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
               fill(0.0, (cos_scatter_bins.bin_number, Ereco_bins_ES.bin_number, N_COSZ_BINS))
    end

    ES_nue_eff    = responseMatrices.eff.ES_nue
    ES_nuother_eff = responseMatrices.eff.ES_nuother

    eventRate_ES_nue_day     = apply_day_response(oscSamplesES.nue_day,     responseMatrices.ES.nue,     ES_nue_eff; scale=SIGNAL_DAY_FRACTION)
    eventRate_ES_nue_night   = apply_night_response(oscSamplesES.nue_night, responseMatrices.ES.nue,     ES_nue_eff; scale=SIGNAL_NIGHT_FRACTION)
    eventRate_ES_nuother_day   = apply_day_response(oscSamplesES.nuother_day,   responseMatrices.ES.nuother, ES_nuother_eff; scale=SIGNAL_DAY_FRACTION)
    eventRate_ES_nuother_night = apply_night_response(oscSamplesES.nuother_night, responseMatrices.ES.nuother, ES_nuother_eff; scale=SIGNAL_NIGHT_FRACTION)

    eventRate_ES_day   = eventRate_ES_nue_day   .+ eventRate_ES_nuother_day
    eventRate_ES_night = eventRate_ES_nue_night .+ eventRate_ES_nuother_night

    eventRate_ES_day_angular_signal     = responseMatrices.ES.angular .* vec(eventRate_ES_day)'
    eventRate_ES_day_angular_background = responseMatrices.BG.angular .* vec(BG_ES)' .* BG_DAY_FRACTION
    eventRate_ES_day_angular = eventRate_ES_day_angular_signal .+ eventRate_ES_day_angular_background

    ncos  = cos_scatter_bins.bin_number
    nErec = Ereco_bins_ES.bin_number
    ncosz = N_COSZ_BINS
    eventRate_ES_night_angular_signal     = reshape(responseMatrices.ES.angular, ncos, nErec, 1) .*
                                            reshape(eventRate_ES_night', 1, nErec, ncosz)
    eventRate_ES_night_angular_background = reshape(responseMatrices.BG.angular, ncos, nErec, 1) .*
                                            reshape(reshape(BG_ES, nErec, 1) .* reshape(exposure_weights_bg, 1, ncosz), 1, nErec, ncosz)
    eventRate_ES_night_angular = eventRate_ES_night_angular_signal .+ eventRate_ES_night_angular_background .* BG_NIGHT_FRACTION

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
               fill(0.0, (N_COSZ_BINS, Ereco_bins_CC.bin_number))
    end

    CC_eff = responseMatrices.eff.CC

    eventRate_CC_day   = apply_day_response(oscSamplesCC.day,   responseMatrices.CC, CC_eff; scale=SIGNAL_DAY_FRACTION) .+ BG_DAY_FRACTION .* BG_CC
    eventRate_CC_night = apply_night_response(oscSamplesCC.night, responseMatrices.CC, CC_eff; scale=SIGNAL_NIGHT_FRACTION) .+ BG_NIGHT_FRACTION .* (BG_CC' .* exposure_weights_bg)

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
    if !det_flags.CC_mode || !(det_flags.inclusive_analysis || det_flags.semi_inclusive_analysis) || oscSamplesCC === nothing
        if det_flags.angular_reco
            return fill(0.0, (cos_scatter_bins.bin_number, Ereco_bins_ES.bin_number)),
                   fill(0.0, (cos_scatter_bins.bin_number, Ereco_bins_ES.bin_number, N_COSZ_BINS))
        else
            return fill(0.0, Ereco_bins_ES.bin_number),
                   fill(0.0, (N_COSZ_BINS, Ereco_bins_ES.bin_number))
        end
    end

    CC_incl_eff = responseMatrices.eff.CC_incl
    cc_day   = apply_day_response(oscSamplesCC.day,   responseMatrices.CC_inclusive, CC_incl_eff; scale=SIGNAL_DAY_FRACTION)
    cc_night = apply_night_response(oscSamplesCC.night, responseMatrices.CC_inclusive, CC_incl_eff; scale=SIGNAL_NIGHT_FRACTION)

    if det_flags.angular_reco
        ncos  = cos_scatter_bins.bin_number
        nErec = Ereco_bins_ES.bin_number
        ncosz = N_COSZ_BINS
        cc_day_angular   = responseMatrices.BG.angular .* vec(cc_day)'
        cc_night_angular = reshape(responseMatrices.BG.angular, ncos, nErec, 1) .*
                           reshape(cc_night', 1, nErec, ncosz)
        return cc_day_angular, cc_night_angular
    else
        return cc_day, cc_night
    end
end


"""
    compute_ES_misID_CC_event_rates(oscSamplesES, responseMatrices, det_flags)

CC-bin event rates for backward-scattered ES events mis-identified as CC.

`responseMatrices` must carry `.CC_misID_nue` and `.CC_misID_nuother` — precomputed
`(N_Etrue × N_Ereco_CC)` matrices that encode the full two-step chain:
  1. 100%-efficiency ES reco pass (normalised ES response, no `ES_nue_eff` applied).
  2. CC reco pass treating `e_reco_ES` as pseudo-true energy (via CC response reindexed
     to ES reco bin centres — the stated approximation).
Backward angular fraction is already folded into the matrices.

CC efficiency (`eff.CC`) is applied here; ES selection efficiency is NOT.
Only active when `det_flags.semi_inclusive_analysis` is true.
"""
function compute_ES_misID_CC_event_rates(oscSamplesES, responseMatrices, det_flags)
    Ereco_bins_CC = responseMatrices.bins.CC
    if !det_flags.semi_inclusive_analysis || !det_flags.ES_mode || oscSamplesES === nothing
        return fill(0.0, Ereco_bins_CC.bin_number),
               fill(0.0, (N_COSZ_BINS, Ereco_bins_CC.bin_number))
    end
    CC_eff = responseMatrices.eff.CC
    misID_nue_day     = apply_day_response(oscSamplesES.nue_day,       responseMatrices.CC_misID_nue,     CC_eff; scale=SIGNAL_DAY_FRACTION)
    misID_nue_night   = apply_night_response(oscSamplesES.nue_night,   responseMatrices.CC_misID_nue,     CC_eff; scale=SIGNAL_NIGHT_FRACTION)
    misID_nuoth_day   = apply_day_response(oscSamplesES.nuother_day,   responseMatrices.CC_misID_nuother, CC_eff; scale=SIGNAL_DAY_FRACTION)
    misID_nuoth_night = apply_night_response(oscSamplesES.nuother_night, responseMatrices.CC_misID_nuother, CC_eff; scale=SIGNAL_NIGHT_FRACTION)
    return misID_nue_day .+ misID_nuoth_day,
           misID_nue_night .+ misID_nuoth_night
end


