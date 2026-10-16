
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
    warn_if_bg_arrays_mismatched(channel, filepaths, norms, sys)

`build_backgrounds` zips the background file list against its normalisation/systematic
arrays, so a shorter `norms`/`sys` array is a supported way to use only the first N
background files. Warn when that happens so an accidental length mismatch (e.g. a typo'd
config) doesn't silently drop backgrounds unnoticed.
"""
function warn_if_bg_arrays_mismatched(channel, filepaths, norms, sys)
    if !(length(filepaths) == length(norms) == length(sys))
        n = min(length(filepaths), length(norms), length(sys))
        @warn "$channel background arrays have mismatched lengths " *
              "(files=$(length(filepaths)), norms=$(length(norms)), sys=$(length(sys))); " *
              "only using the first $n."
    end
end

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
    conv               = bg_conventions(det)

    warn_if_bg_arrays_mismatched("ES", det.ES_filepaths_BG, ES_bg_norms, ES_bg_sys)
    warn_if_bg_arrays_mismatched("CC", det.CC_filepaths_BG, CC_bg_norms, CC_bg_sys)

    # Only the first n files are used (see warn_if_bg_arrays_mismatched); truncate here so
    # `backgrounds.ES`/`.CC` stay aligned with their par_counts and conversion factors.
    n_ES = min(length(det.ES_filepaths_BG), length(ES_bg_norms), length(ES_bg_sys))
    n_CC = min(length(det.CC_filepaths_BG), length(CC_bg_norms), length(CC_bg_sys))
    df_ES_list = extract_dataframes(collect(String, det.ES_filepaths_BG[1:n_ES]))
    df_CC_list = extract_dataframes(collect(String, det.CC_filepaths_BG[1:n_CC]))

    # ── ES backgrounds ──────────────────────────────────────────────────────
    # The ES background MC is not flux-normalised: each file holds the events that reached
    # the detector out of `ES_bg_generated_events` generated. Weights (if present) enter the
    # selected and the total counts alike, so the efficiency stays in [0, 1].
    ES_bg    = Vector{Vector{Float64}}()
    ES_sides = Int[]
    ES_relvar = Vector{Vector{Float64}}()     # per-bin relative MC variance Σw²/(Σw)² (selected)
    for df in df_ES_list
        w    = bg_weights(df)
        mask = Bool.(df.mask)
        ES_temp, _       = create_histogram(df.Ereco,       Ereco_bins_ES_extended, weights=w,       normalise=true)
        ES_temp_selec, _ = create_histogram(df.Ereco[mask], Ereco_bins_ES_extended, weights=w[mask], normalise=false)
        ES_temp_total, _ = create_histogram(df.Ereco,       Ereco_bins_ES_extended, weights=w,       normalise=false)
        side = "side" in names(df) ? coalesce(first(df.side), -1) : -1
        ES_eff_bg   = @. ifelse(ES_temp_total == 0, 0.0, ES_temp_selec / ES_temp_total)
        attenuation = sum(ES_temp_total) / conv.ES_generated
        push!(ES_bg,    Float64.(ES_temp .* detection_time .* ES_eff_bg .* ES_normalisation .* attenuation))
        push!(ES_sides, side)
        push!(ES_relvar, mc_relative_variance(df.Ereco[mask], w[mask], Ereco_bins_ES_extended))
    end
    ES_conv = Float64[es_face_area_cm2(s, conv.ES_face_areas_m2) for s in ES_sides]

    # ── CC backgrounds (disabled in pure inclusive mode; enabled in semi-inclusive) ──────────────────────────
    # The CC background MC is normalised per kt-year at the reference flux; scale to the module
    # mass and exposure here, and by norm / reference flux in normalize_backgrounds.
    CC_bg = Vector{Vector{Float64}}()
    CC_relvar = Vector{Vector{Float64}}()
    if !inclusive_analysis || semi_inclusive_analysis
        for df in df_CC_list
            w    = bg_weights(df)
            mask = Bool.(df.mask)
            CC_temp_selec, _ = create_histogram(df.Ereco[mask], Ereco_bins_CC_extended, weights=w[mask], normalise=false)
            push!(CC_bg, Float64.(CC_temp_selec .* conv.module_mass_kt .* CC_normalisation))
            push!(CC_relvar, mc_relative_variance(df.Ereco[mask], w[mask], Ereco_bins_CC_extended))
        end
    end
    CC_conv = fill(1.0 / conv.CC_reference_flux, length(CC_bg))

    # ── Systematic uncertainties ──────────────────────────────────────────────
    ES_bg_norms_true = Float64[]
    ES_bg_norms_pars = Distribution[]
    ES_bg_par_counts = Int[]

    CC_bg_norms_true = Float64[]
    CC_bg_norms_pars = Distribution[]
    CC_bg_par_counts = Int[]

    if ES_mode
        for (i, (norm, sys)) in enumerate(zip(ES_bg_norms, ES_bg_sys))
            i > length(ES_bg) && break
            if sys == 0
                ES_bg[i] .*= ES_conv[i] * norm      # fixed normalisation: apply the conversion now
                push!(ES_bg_par_counts, 0)
            else
                push!(ES_bg_norms_true, norm)
                push!(ES_bg_norms_pars, bg_norm_prior(norm, sys))
                push!(ES_bg_par_counts, 1)
            end
        end
    end

    if CC_mode && (!inclusive_analysis || semi_inclusive_analysis)
        for (i, (norm, sys)) in enumerate(zip(CC_bg_norms, CC_bg_sys))
            i > length(CC_bg) && break
            if sys == 0
                CC_bg[i] .*= CC_conv[i] * norm
                push!(CC_bg_par_counts, 0)
            else
                push!(CC_bg_norms_true, norm)
                push!(CC_bg_norms_pars, bg_norm_prior(norm, sys))
                push!(CC_bg_par_counts, 1)
            end
        end
    end

    backgrounds = (ES=ES_bg, CC=CC_bg, sides=ES_sides, ES_conv=ES_conv, CC_conv=CC_conv,
                   ES_par_counts=ES_bg_par_counts, CC_par_counts=CC_bg_par_counts,
                   ES_relvar=ES_relvar, CC_relvar=CC_relvar)

    return backgrounds, ES_bg_norms_true, CC_bg_norms_true, ES_bg_norms_pars, CC_bg_norms_pars
end

# Normalisation conventions of the background MC, from the detector block (defaults in
# readConfig.jl reproduce the old hard-coded values). `get` keeps hand-built test tuples working.
bg_conventions(det) = (
    module_mass_kt    = Float64(get(det, :module_mass_kt, 10.0)),
    ES_generated      = Float64(get(det, :ES_bg_generated_events, 50e6)),
    ES_face_areas_m2  = Float64.(get(det, :ES_bg_face_areas_m2, [2 * 12.0 * 64.0, 2 * 12.0 * 12.0])),
    CC_reference_flux = Float64(get(det, :CC_bg_reference_flux, 2.2e-6)),
)

# Exposed area [cm²] for an ES background generated on detector side `side`:
# 0 = long faces, 1 = end caps, anything else = all faces.
function es_face_area_cm2(side, face_areas_m2)
    area_m2 = side == 0 ? face_areas_m2[1] :
              side == 1 ? face_areas_m2[2] : sum(face_areas_m2)
    return area_m2 * 1e4
end

bg_weights(df) = "weights" in names(df) ? Float64.(coalesce.(df.weights, 0.0)) : ones(nrow(df))

# Truncated Gaussian prior on a background normalisation. A zero nominal norm with nonzero
# sys gives a zero-width prior, so it is rejected (use sys = 0 to fix a background to zero).
function bg_norm_prior(norm, sys)
    norm > 0 || error("Background normalisation must be > 0 when its systematic is nonzero (got norm=$norm, sys=$sys); set sys = 0 to fix it.")
    return truncated(Normal(norm, norm * sys), 0.0, norm * 2)
end
