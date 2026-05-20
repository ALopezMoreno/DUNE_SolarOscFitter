
#=
backgrounds.jl

Background event processing for the Solar Oscillation Fitter.
This module loads and processes background Monte Carlo samples for both
Elastic Scattering (ES) and Charged Current (CC) detection channels.

Key Features:
- Background MC sample loading and histogram creation
- Detection efficiency calculations
- Systematic uncertainty handling for background normalizations
- Support for weighted and unweighted MC samples
- Automatic parameter setup for MCMC fitting

The backgrounds are normalized to the expected detection time and exposure,
with optional systematic uncertainties treated as nuisance parameters.

Author: [Author name]
=#

include(joinpath(@__DIR__, "histHelpers.jl"))

"""
    build_backgrounds(det, Ereco_bins_ES_extended, Ereco_bins_CC_extended)
        -> (backgrounds, ES_bg_norms_true, CC_bg_norms_true, ES_bg_norms_pars, CC_bg_norms_pars)

Load and process background MC for detector `det`. Returns:
- `backgrounds`: named tuple `(ES, CC, sides, ES_par_counts, CC_par_counts)` passed to
  `propagateSamples` at every likelihood evaluation.
- `ES/CC_bg_norms_true`: true (Asimov) normalization values for nuisance parameters.
- `ES/CC_bg_norms_pars`: prior `Distribution` objects, one per nuisance parameter.

The par_counts arrays are embedded in `backgrounds` so `normalize_backgrounds` does not
need separate globals when running multiple detectors concurrently.
"""
function build_backgrounds(det, Ereco_bins_ES_extended, Ereco_bins_CC_extended)
    ES_normalisation   = det.ES_normalisation
    CC_normalisation   = det.CC_normalisation
    inclusive_analysis      = det.inclusive_analysis
    semi_inclusive_analysis = det.semi_inclusive_analysis
    ES_mode                 = det.ES_mode
    CC_mode                 = det.CC_mode
    ES_bg_norms        = det.ES_bg_norms
    CC_bg_norms        = det.CC_bg_norms
    ES_bg_sys          = det.ES_bg_sys
    CC_bg_sys          = det.CC_bg_sys

    df_ES_list = extract_dataframes(det.ES_filepaths_BG)
    df_CC_list = extract_dataframes(det.CC_filepaths_BG)

    # ── ES backgrounds ──────────────────────────────────────────────────────
    ES_bg    = []
    ES_sides = []
    for df in df_ES_list
        if "weights" in names(df)
            ES_temp, _      = create_histogram(df.Ereco, Ereco_bins_ES_extended, weights=df.weights, normalise=true)
            ES_temp_selec, _ = create_histogram(df.Ereco[df.mask], Ereco_bins_ES_extended, weights=df.weights[df.mask], normalise=false)
            ES_temp_total, _ = create_histogram(df.Ereco,           Ereco_bins_ES_extended,                              normalise=false)
        else
            ES_temp, _      = create_histogram(df.Ereco,            Ereco_bins_ES_extended, normalise=true)
            ES_temp_selec, _ = create_histogram(df.Ereco[df.mask],  Ereco_bins_ES_extended, normalise=false)
            ES_temp_total, _ = create_histogram(df.Ereco,            Ereco_bins_ES_extended, normalise=false)
        end
        side = "side" in names(df) ? coalesce(first(df.side), -1) : -1
        ES_eff_bg   = @. ifelse(ES_temp_total == 0, 0.0, ES_temp_selec / ES_temp_total)
        # ── TEMPORARY efficiency boost ──────────────────────────────────────────
        # Matches the boost applied to ES_nue_eff / ES_nuother_eff / CC_incl_eff in
        # response.jl: raises ES background masking efficiency to a 90% plateau by
        # 11.5 MeV so that ES background rates are consistent with the boosted signal
        # rates in the inclusive likelihood.  No effect in exclusive mode.
        # Remove together with the corresponding blocks in response.jl.
        if inclusive_analysis || semi_inclusive_analysis
            _bg_edges   = collect(range(Ereco_bins_ES_extended.min, Ereco_bins_ES_extended.max,
                                        length=Ereco_bins_ES_extended.bin_number+1))
            _bg_centers = 0.5 .* (_bg_edges[1:end-1] .+ _bg_edges[2:end])
            _t      = clamp.((_bg_centers .- 0.010) ./ (0.0115 - 0.010), 0.0, 1.0)
            _smooth = @. _t^2 * (3 - 2*_t)   # Hermite smooth-step: 0 at 10 MeV, 1 at 11.5 MeV
            ES_eff_bg = @. ES_eff_bg + _smooth * (0.9 - ES_eff_bg)
        end
        attenuation = sum(ES_temp_total) / 50e6
        push!(ES_bg,    ES_temp .* detection_time .* ES_eff_bg .* ES_normalisation .* attenuation)
        push!(ES_sides, side)
    end

    # ── CC backgrounds (disabled in pure inclusive mode; enabled in semi-inclusive) ──────────────────────────
    CC_bg = []
    if !inclusive_analysis || semi_inclusive_analysis
        for df in df_CC_list
            if "weights" in names(df)
                CC_temp_selec, _ = create_histogram(df.Ereco[df.mask], Ereco_bins_CC_extended, weights=df.weights[df.mask], normalise=false)
            else
                CC_temp_selec, _ = create_histogram(df.Ereco[df.mask], Ereco_bins_CC_extended, normalise=false)
            end
            # CC BG MC is normalised to 1 kt-year; signal MC uses detector_nAr40 = 10 kT,
            # so total signal scaling is 10 kT × CC_normalisation. Match with ×10 here.
            push!(CC_bg, CC_temp_selec .* 10 .* CC_normalisation)
        end
    end

    # ── Systematic uncertainties ──────────────────────────────────────────────
    ES_bg_norms_true = Float64[]
    ES_bg_norms_pars = Distribution[]
    ES_bg_par_counts = Int[]

    CC_bg_norms_true = Float64[]
    CC_bg_norms_pars = Distribution[]
    CC_bg_par_counts = Int[]

    if ES_mode
        for (bg, norm, sys) in zip(ES_bg, ES_bg_norms, ES_bg_sys)
            if sys == 0
                bg .*= norm
                push!(ES_bg_par_counts, 0)
            else
                push!(ES_bg_norms_true, norm)
                push!(ES_bg_norms_pars, truncated(Normal(norm, norm * sys), 0.0, norm * 2))
                push!(ES_bg_par_counts, 1)
            end
        end
    end

    if CC_mode && (!inclusive_analysis || semi_inclusive_analysis)
        for (bg, norm, sys) in zip(CC_bg, CC_bg_norms, CC_bg_sys)
            if sys == 0
                bg .*= norm / 2.2e-6
                push!(CC_bg_par_counts, 0)
            else
                push!(CC_bg_norms_true, norm)
                push!(CC_bg_norms_pars, truncated(Normal(norm, norm * sys), 0.0, norm * 2))
                push!(CC_bg_par_counts, 1)
            end
        end
    end

    backgrounds = (ES=ES_bg, CC=CC_bg, sides=ES_sides,
                   ES_par_counts=ES_bg_par_counts, CC_par_counts=CC_bg_par_counts)

    return backgrounds, ES_bg_norms_true, CC_bg_norms_true, ES_bg_norms_pars, CC_bg_norms_pars
end