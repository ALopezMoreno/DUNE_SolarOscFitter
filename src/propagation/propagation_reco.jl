"""
    compute_ES_event_rates(oscSamplesES, responseMatrices, BG_ES)

Compute ES day and night event rates (including backgrounds), or zero arrays if
`ES_mode` is false or `oscSamplesES === nothing`.

Inputs:
- oscSamplesES: container with fields `nue_day`, `nue_night`, `nuother_day`, `nuother_night`
- responseMatrices: container with field `ES` having subfields `nue`, `nuother`
- BG_ES: 1D background spectrum in Ereco bins

Returns:
- eventRate_ES_day::Vector{Float64}
- eventRate_ES_night::Matrix{Float64} (cosz x Ereco)

Uses globals:
- `ES_mode`, `Ereco_bins_ES`, `cosz_bins`
- `exposure_weights`, `ES_nue_eff`, `ES_nuother_eff`
- `apply_day_response`, `apply_night_response`
"""
function compute_ES_event_rates(oscSamplesES, responseMatrices, BG_ES)
    if !ES_mode || oscSamplesES === nothing
        return fill(0.0, Ereco_bins_ES.bin_number),
               fill(0.0, (cosz_bins.bin_number, Ereco_bins_ES.bin_number))
    end

    eventRate_ES_nue_day = apply_day_response(oscSamplesES.nue_day, responseMatrices.ES.nue, ES_nue_eff)
    eventRate_ES_nue_night = apply_night_response(oscSamplesES.nue_night, responseMatrices.ES.nue, ES_nue_eff)

    eventRate_ES_nuother_day = apply_day_response(oscSamplesES.nuother_day, responseMatrices.ES.nuother, ES_nuother_eff)
    eventRate_ES_nuother_night = apply_night_response(oscSamplesES.nuother_night, responseMatrices.ES.nuother, ES_nuother_eff)

    # total ES day/night, including backgrounds (0.5 day/night fraction)
    eventRate_ES_day   = (eventRate_ES_nue_day   .+ eventRate_ES_nuother_day)   .+ 0.5 .* BG_ES
    eventRate_ES_night = (eventRate_ES_nue_night .+ eventRate_ES_nuother_night) .+ 0.5 .* (BG_ES' .* exposure_weights)

    return eventRate_ES_day, eventRate_ES_night
end


"""
    compute_ES_angular_event_rates(oscSamplesES, responseMatrices, angularResponseMatrices, BG_ES)

Compute the reco energy vs. electron scattering angle distribution, combining
signal and background. Currently assumes no solar-angle dependence in the
signal; night samples are summed over cos(z) before applying the angular
response.

Returns zero array if `ES_mode` is false or `oscSamplesES === nothing`.

Inputs:
- oscSamplesES: container with fields `nue_day`, `nue_night`, `nuother_day`, `nuother_night`
- responseMatrices: container with field `ES` having subfields `nue`, `nuother` (used to compute 1D energy spectra)
- angularResponseMatrices: container with fields
    - `ES`::Matrix{Float64} mapping Ereco bins -> angular bins (cos_scatter x Ereco)
    - `BG`::Matrix{Float64} mapping Ereco bins -> angular bins (cos_scatter x Ereco)
- BG_ES: 1D background spectrum in Ereco bins

Returns:
- eventRate_ES_angular::Matrix{Float64} (cos_scatter x Ereco)

Uses globals:
- `ES_mode`, `Ereco_bins_ES`, `cos_scatter`
- `ES_nue_eff`, `ES_nuother_eff`
- `apply_day_response`, `apply_night_response`
"""
function compute_ES_angular_event_rates(oscSamplesES, responseMatrices, BG_ES)
    if !ES_mode || oscSamplesES === nothing || !angular_reco
        return fill(0.0, (cos_scatter_bins.bin_number, Ereco_bins_ES.bin_number))
    end

    # AT THE MOMENT WE DO NOT HAVE SOLAR ANGLE DEPENDENCE

    # 1) Calculate the event rates as usual
    eventRate_ES_nue_day = apply_day_response(oscSamplesES.nue_day, responseMatrices.ES.nue, ES_nue_eff)
    eventRate_ES_nue_night = apply_night_response(oscSamplesES.nue_night, responseMatrices.ES.nue, ES_nue_eff)

    eventRate_ES_nuother_day = apply_day_response(oscSamplesES.nuother_day, responseMatrices.ES.nuother, ES_nuother_eff)
    eventRate_ES_nuother_night = apply_night_response(oscSamplesES.nuother_night, responseMatrices.ES.nuother, ES_nuother_eff)

    eventRate_ES_day   = (eventRate_ES_nue_day   .+ eventRate_ES_nuother_day)
    eventRate_ES_night = (eventRate_ES_nue_night .+ eventRate_ES_nuother_night)

    # 2) Collapse the night event rates into 1D arrays of reco energy bins (we ignore solar angle) and sum
    eventRate_ES_night_summed = sum(eventRate_ES_night, dims=1)
    eventRate_ES_combined = vec(eventRate_ES_day) .+ vec(eventRate_ES_night_summed)

    # 3) Rescale the energy slices of the angular response matrices by the event rate in each energy bin
    #    (the angular response matrices are already normalised over the angles)
    eventRate_ES_angular_signal = responseMatrices.ES.angular .*  Diagonal(eventRate_ES_combined)
    eventRate_ES_angular_background = responseMatrices.BG.angular .* Diagonal(BG_ES)

    # 4) Sum total
    eventRate_ES_angular = eventRate_ES_angular_signal + eventRate_ES_angular_background

    return eventRate_ES_angular
end


"""
    compute_CC_event_rates(oscSamplesCC, responseMatrices, BG_CC)

Compute CC day and night event rates (including backgrounds), or zero arrays if
`CC_mode` is false or `oscSamplesCC === nothing`.

Inputs:
- oscSamplesCC: container with fields `day`, `night`
- responseMatrices: container with field `CC`
- BG_CC: 1D background spectrum in Ereco bins

Returns:
- eventRate_CC_day::Vector{Float64}
- eventRate_CC_night::Matrix{Float64} (cosz x Ereco)

Uses globals:
- `CC_mode`, `Ereco_bins_CC`, `cosz_bins`
- `exposure_weights`, `CC_eff`
- `apply_day_response`, `apply_night_response`
"""
function compute_CC_event_rates(oscSamplesCC, responseMatrices, BG_CC)
    if !CC_mode || oscSamplesCC === nothing
        return fill(0.0, Ereco_bins_CC.bin_number),
               fill(0.0, (cosz_bins.bin_number, Ereco_bins_CC.bin_number))
    end

    eventRate_CC_day = apply_day_response(oscSamplesCC.day, responseMatrices.CC, CC_eff) .+ 0.5 .* BG_CC
    eventRate_CC_night = apply_night_response(oscSamplesCC.night, responseMatrices.CC, CC_eff) .+ 0.5 .* (BG_CC' .* exposure_weights)

    return eventRate_CC_day, eventRate_CC_night
end