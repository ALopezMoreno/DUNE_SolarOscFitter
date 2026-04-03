
include(joinpath(@__DIR__, "..", "oscillations", "osc.jl"))


import .Osc: oscPars, osc_prob_both_slow

# Set energy bin centers for calculation
global E_calc = (bin_edges_calc[1:end-1] + bin_edges_calc[2:end]) / 2.0

# Choose fast or slow Earth propagation based on configuration. Choose oscillations calculator (nuFast only works on fast mode)
if nuFast
    include(joinpath(@__DIR__, "..", "oscillations", "nuFast_interface.jl"))
    import .nuFastOsc: osc_prob_both_fast, init_engines
    nuFastOsc.init_engines(E_calc, cosz_calc)
else
  import .Osc: osc_prob_both_fast
  if fast
    import .Osc.NumOsc.Fast: osc_prob_earth
  else
    import .Osc.NumOsc.Slow: osc_prob_earth
  end
end


"""
    get_mixing_parameters(params)

Convert oscillation parameters in `params` to the internal `oscPars` format.
"""
function get_mixing_parameters(params)
    return oscPars(
        params.dm2_21,
        asin(sqrt(params.sin2_th12)),
        asin(sqrt(params.sin2_th13)),
    )
end

"""
    setup_earth_propagation(E_calc, mixingPars, params)

Compute Earth propagation lookup and (optionally) `oscProbs_1e` depending on
`earthUncertainty` and `nuFast`.

Uses globals:
- `earthUncertainty`, `earth_lookup`, `earth_paths`, `nuFast`

Returns
- oscProbs_1e
- earth_norm_vector
- lookup
"""
function setup_earth_propagation(E_calc, mixingPars, params)
    oscProbs_1e = nothing
    earth_norm_vector = Vector{Float64}()
    lookup = earth_lookup   # safe default

    if earthUncertainty
        n = length(earth_lookup)
        earth_norm_vector = [getfield(params, Symbol("earth_norm_", i)) for i in 1:n]
        lookup = earth_norm_vector .* earth_lookup

        if !nuFast
            oscProbs_1e = osc_prob_earth(E_calc, mixingPars, lookup, earth_paths)
        end
    else
        earth_norm_vector = Float64[]
        lookup = earth_lookup
        if !nuFast
            oscProbs_1e = osc_prob_earth(E_calc, mixingPars, earth_lookup, earth_paths)
        end
    end

    return oscProbs_1e, earth_norm_vector, lookup
end


"""
    compute_oscillation_probabilities(E_calc, mixingPars, solarModel, params,
                                      oscProbs_1e, earth_norm_vector, lookup)

Compute day/night oscillation probabilities for 8B and hep, and their
ν_e and ν_other components, binned to the analysis resolution.

Uses globals:
- `nuFast`, `fast`, `earth_paths`
- `block_average`

Returns a NamedTuple with:
- nue_8B_day, nue_8B_night
- nue_hep_day, nue_hep_night
- nuother_8B_day, nuother_8B_night
- nuother_hep_day, nuother_hep_night
"""
function compute_oscillation_probabilities(
    E_calc,
    mixingPars,
    solarModel,
    params,
    oscProbs_1e,
    earth_norm_vector,
    lookup,
)

    # fine-resolution probabilities
    if nuFast
        oscProbs_nue_8B_day_large, oscProbs_nue_8B_night_large =
            osc_prob_both_fast(E_calc, mixingPars, lookup, earth_paths;
                               n_vec = earth_norm_vector)

        # NOTE: still using same function for hep. To fix (?)
        oscProbs_nue_hep_day_large, oscProbs_nue_hep_night_large =
            osc_prob_both_fast(E_calc, mixingPars, lookup, earth_paths;
                               n_vec = earth_norm_vector)
    elseif fast
        oscProbs_nue_8B_day_large, oscProbs_nue_8B_night_large =
            osc_prob_both_fast(E_calc, oscProbs_1e, mixingPars, solarModel;
                               process = "8B")

        oscProbs_nue_hep_day_large, oscProbs_nue_hep_night_large =
            osc_prob_both_fast(E_calc, oscProbs_1e, mixingPars, solarModel;
                               process = "hep")
    else
        oscProbs_nue_8B_day_large, oscProbs_nue_8B_night_large =
            osc_prob_both_slow(E_calc, oscProbs_1e, mixingPars, solarModel;
                               process = "8B")

        oscProbs_nue_hep_day_large, oscProbs_nue_hep_night_large =
            osc_prob_both_slow(E_calc, oscProbs_1e, mixingPars, solarModel;
                               process = "hep")
    end

    # average over fine resolution to desired binning
    oscProbs_nue_8B_day   = block_average(oscProbs_nue_8B_day_large, 2)
    oscProbs_nue_hep_day  = block_average(oscProbs_nue_hep_day_large, 2)

    oscProbs_nue_8B_night = block_average(oscProbs_nue_8B_night_large, (3, 2))
    oscProbs_nue_hep_night = block_average(oscProbs_nue_hep_night_large, (3, 2))

    # ν_other from unitarity
    oscProbs_nuother_8B_day   = 1 .- oscProbs_nue_8B_day
    oscProbs_nuother_hep_day  = 1 .- oscProbs_nue_hep_day

    oscProbs_nuother_8B_night = 1 .- oscProbs_nue_8B_night
    oscProbs_nuother_hep_night = 1 .- oscProbs_nue_hep_night

    return (
        nue_8B_day          = oscProbs_nue_8B_day,
        nue_8B_night        = oscProbs_nue_8B_night,
        nue_hep_day         = oscProbs_nue_hep_day,
        nue_hep_night       = oscProbs_nue_hep_night,
        nuother_8B_day      = oscProbs_nuother_8B_day,
        nuother_8B_night    = oscProbs_nuother_8B_night,
        nuother_hep_day     = oscProbs_nuother_hep_day,
        nuother_hep_night   = oscProbs_nuother_hep_night,
    )
end


function compute_oscillated_samples(unoscillatedSample, params, oscProbs)
    ES = nothing
    CC = nothing

    if ES_mode
        ES = (
            nue_day =
                unoscillatedSample.ES_nue_8B  .* oscProbs.nue_8B_day  .* params.integrated_8B_flux .+
                unoscillatedSample.ES_nue_hep .* oscProbs.nue_hep_day .* params.integrated_HEP_flux,

            nue_night =
                (unoscillatedSample.ES_nue_8B'  .* (params.integrated_8B_flux' .* oscProbs.nue_8B_night  .* exposure_weights)) .+
                (unoscillatedSample.ES_nue_hep' .* (params.integrated_HEP_flux  .* oscProbs.nue_hep_night .* exposure_weights)),

            nuother_day =
                unoscillatedSample.ES_nuother_8B  .* oscProbs.nuother_8B_day  .* params.integrated_8B_flux .+
                unoscillatedSample.ES_nuother_hep .* oscProbs.nuother_hep_day .* params.integrated_HEP_flux,

            nuother_night =
                (unoscillatedSample.ES_nuother_8B'  .* (params.integrated_8B_flux' .* oscProbs.nuother_8B_night  .* exposure_weights)) .+
                (unoscillatedSample.ES_nuother_hep' .* (params.integrated_HEP_flux  .* oscProbs.nuother_hep_night .* exposure_weights)),
        )
    end

    if CC_mode
        CC = (
            day =
                unoscillatedSample.CC_8B  .* oscProbs.nue_8B_day  .* params.integrated_8B_flux .+
                unoscillatedSample.CC_hep .* oscProbs.nue_hep_day .* params.integrated_HEP_flux,

            night =
                (unoscillatedSample.CC_8B'  .* (params.integrated_8B_flux' .* oscProbs.nue_8B_night  .* exposure_weights)) .+
                (unoscillatedSample.CC_hep' .* (params.integrated_HEP_flux  .* oscProbs.nue_hep_night .* exposure_weights)),
        )
    end

    return (ES = ES, CC = CC)
end
