
include(joinpath(@__DIR__, "..", "oscillations", "osc.jl"))
import .Osc: oscPars, osc_prob_day_c2m

include(joinpath(@__DIR__, "..", "oscillations", "osc_night.jl"))
import .NightOsc

include(joinpath(@__DIR__, "osc_setup.jl"))

# Set energy bin centers for calculation
const E_calc = (bin_edges_calc[1:end-1] + bin_edges_calc[2:end]) / 2.0

# Settings read on the likelihood path, frozen as constants after the config is loaded so
# that the likelihood is type-stable (required by Enzyme, and faster under ForwardDiff).
const CC_XSEC_NORM_TRUE  = Float64(cc_xsec_norm_true)

# Which day and night calculations run (see osc_setup.jl).
const OSC = build_osc_setup(
    night_osc         = nightOsc,
    fast              = fast,
    earth_uncertainty = earthUncertainty,
    earth_model_file  = earthModelFile,
    depth_km          = depth,
    cosz_edges        = COARSE_COSZ_EDGES,
    energy_edges      = calculate_bins(Etrue_bins)[1],
    exposure          = exposure_intp,
    exposure_weights  = exposure_weights,
    n_E_calc          = length(E_calc),
    n_earth_norms     = isnothing(earth_normalisation_true) ? nothing : length(earth_normalisation_true),
    legacy_paths      = earth_paths,
    legacy_lookup     = earth_lookup,
    legacy_path       = cz -> make_potential_for_integrand(cz, earth),
)


"""
    get_mixing_parameters(params)

Convert oscillation parameters in `params` to the internal `oscPars` format.
"""
function get_mixing_parameters(params)
    return oscPars(
        params.dm2_21,
        asin(sqrt(clamp(params.sin2_th12, 0.0, 1.0))),
        asin(sqrt(clamp(params.sin2_th13, 0.0, 1.0))),
    )
end

"""
    setup_earth_propagation(E_calc, mixingPars, params)

Night-time P₁ₑ for the configured backend. Returns (oscProbs_1e, earth_norm_vector, lookup).
"""
setup_earth_propagation(E_calc, mixingPars, params) = _earth_crossing(OSC.night, E_calc, mixingPars, params)

function _earth_crossing(night::NightOscNight, E_calc, mixingPars, params)
    earth_norms = earth_uncertainty(OSC) ? _params_by_prefix(params, Val(:earth_norm_)) : nothing
    p1e = NightOsc.binned_p1e(night.engine, mixingPars.Δm²₂₁, mixingPars.θ₁₂, mixingPars.θ₁₃, earth_norms)
    return p1e, earth_norms, nothing
end

function _legacy_lookup(night, params)
    earth_uncertainty(OSC) || return Float64[], night.lookup
    earth_norm_vector = collect(_params_by_prefix(params, Val(:earth_norm_)))
    return earth_norm_vector, earth_norm_vector .* night.lookup
end

function _earth_crossing(night::NumOscNight{Fast}, E_calc, mixingPars, params) where {Fast}
    earth_norm_vector, lookup = _legacy_lookup(night, params)
    osc_prob_earth = Fast ? Osc.NumOsc.Fast.osc_prob_earth : Osc.NumOsc.Slow.osc_prob_earth
    return osc_prob_earth(E_calc, mixingPars, lookup, night.paths), earth_norm_vector, lookup
end


"""
    compute_oscillation_probabilities(E_calc, mixingPars, solarModel, params,
                                      oscProbs_1e, earth_norm_vector, lookup)

Day and night νₑ and ν_other probabilities for 8B and hep at the analysis resolution.
"""
compute_oscillation_probabilities(E_calc, mixingPars, solarModel, params, oscProbs_1e, earth_norm_vector, lookup) =
    _day_night(OSC.night, E_calc, mixingPars, solarModel, oscProbs_1e, earth_norm_vector, lookup)

_day_night(::NightOscNight, E_calc, mixingPars, solarModel, p1e_binned, earth_norm_vector, lookup) =
    binned_oscillation_probabilities(E_calc, mixingPars, solarModel, p1e_binned)

function _day_night(::NumOscNight{Fast}, E_calc, mixingPars, solarModel, oscProbs_1e, earth_norm_vector, lookup) where {Fast}
    both = Fast ? Osc.osc_prob_both_fast : Osc.osc_prob_both_slow
    day_8B,  night_8B  = both(E_calc, oscProbs_1e, mixingPars, solarModel; process = "8B")
    day_hep, night_hep = both(E_calc, oscProbs_1e, mixingPars, solarModel; process = "hep")
    return _legacy_binned(day_8B, night_8B, day_hep, night_hep)
end

# Average the per-path legacy probabilities to the analysis bins
function _legacy_binned(day_8B_large, night_8B_large, day_hep_large, night_hep_large)
    nue_8B_day    = block_average(day_8B_large, 2)
    nue_hep_day   = block_average(day_hep_large, 2)
    nue_8B_night  = block_average(night_8B_large,  (N_COSZ_SUB, 2))
    nue_hep_night = block_average(night_hep_large, (N_COSZ_SUB, 2))

    # ν_other from unitarity
    return (
        nue_8B_day          = nue_8B_day,
        nue_8B_night        = nue_8B_night,
        nue_hep_day         = nue_hep_day,
        nue_hep_night       = nue_hep_night,
        nuother_8B_day      = 1 .- nue_8B_day,
        nuother_8B_night    = 1 .- nue_8B_night,
        nuother_hep_day     = 1 .- nue_hep_day,
        nuother_hep_night   = 1 .- nue_hep_night,
    )
end


"""
    binned_oscillation_probabilities(E_calc, mixingPars, solarModel, p1e_binned)

Day and night probabilities when `p1e_binned` holds bin-averaged P₁ₑ on the analysis
(cos z, E_true) grid. The day probability and ⟨cos2θ₁₂ᵐ⟩ vary slowly and are averaged
over the two `E_calc` points of each bin; the night probability in each bin is
P_day + c₁₃² ⟨cos2θ₁₂ᵐ⟩ (P̄₁ₑ − c₁₂²c₁₃²).
"""
function binned_oscillation_probabilities(E_calc, mixingPars, solarModel, p1e_binned)
    c13sq = cos(mixingPars.θ₁₃)^2
    P_0 = cos(mixingPars.θ₁₂)^2 * c13sq
    nue_8B_day,  nue_8B_night  = _binned_day_night(E_calc, mixingPars, solarModel, p1e_binned, c13sq, P_0, "8B")
    nue_hep_day, nue_hep_night = _binned_day_night(E_calc, mixingPars, solarModel, p1e_binned, c13sq, P_0, "hep")
    return (
        nue_8B_day          = nue_8B_day,
        nue_8B_night        = nue_8B_night,
        nue_hep_day         = nue_hep_day,
        nue_hep_night       = nue_hep_night,
        nuother_8B_day      = 1 .- nue_8B_day,
        nuother_8B_night    = 1 .- nue_8B_night,
        nuother_hep_day     = 1 .- nue_hep_day,
        nuother_hep_night   = 1 .- nue_hep_night,
    )
end


function _binned_day_night(E_calc, mixingPars, solarModel, p1e_binned, c13sq, P_0, process)
    day_large, c2m_large = osc_prob_day_c2m(E_calc, mixingPars, solarModel; process=process, fast=fast_solar(OSC))
    day = block_average(day_large, 2)
    c2m = block_average(c2m_large, 2)
    night = permutedims(day) .+ c13sq .* permutedims(c2m) .* (p1e_binned .- P_0)
    return day, night
end


compute_oscillated_samples(unoscillatedSample, params, oscProbs; es_mode::Bool=true, cc_mode::Bool=true) =
    compute_oscillated_samples(unoscillatedSample, params, oscProbs, Val(es_mode), Val(cc_mode))

# Channel switches as type parameters: ES/CC are `nothing` or NamedTuples decided at compile time.
function compute_oscillated_samples(unoscillatedSample, params, oscProbs, ::Val{es_mode}, ::Val{cc_mode}) where {es_mode,cc_mode}
    ES = nothing
    CC = nothing

    if es_mode
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

    if cc_mode
        _xsec_norm = hasproperty(params, :cc_xsec_norm) ? params.cc_xsec_norm : CC_XSEC_NORM_TRUE
        _xsec_tilt = hasproperty(params, :cc_xsec_tilt) ? params.cc_xsec_tilt : 0.0
        _xsec_curv = hasproperty(params, :cc_xsec_curv) ? params.cc_xsec_curv : 0.0
        xsec_shape = cc_xsec_shape(_xsec_norm, _xsec_tilt, _xsec_curv,
                                   unoscillatedSample.xsec_p1, unoscillatedSample.xsec_p2)

        # Pre-apply xsec_shape to 1D CC samples so the night (cosz×Etrue) broadcast
        # works: (1,n_Etrue) .* (n_cosz,n_Etrue) is fine; (n_cosz,n_Etrue) .* (n_Etrue,)
        # fails because Julia compares axes left-to-right without left-padding.
        CC_8B_s  = unoscillatedSample.CC_8B  .* xsec_shape
        CC_hep_s = unoscillatedSample.CC_hep .* xsec_shape

        CC = (
            day =
                CC_8B_s  .* oscProbs.nue_8B_day  .* params.integrated_8B_flux .+
                CC_hep_s .* oscProbs.nue_hep_day .* params.integrated_HEP_flux,

            night =
                (CC_8B_s'  .* (params.integrated_8B_flux' .* oscProbs.nue_8B_night  .* exposure_weights)) .+
                (CC_hep_s' .* (params.integrated_HEP_flux  .* oscProbs.nue_hep_night .* exposure_weights)),
        )
    end

    return (ES = ES, CC = CC)
end
