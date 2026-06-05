#=
setup.jl

Initializes the analysis environment: loads shared physics inputs (solar model,
Earth profile, oscillation paths), then loops over all detectors in `detector_configs`
to build per-detector unoscillated samples, response matrices, backgrounds, Asimov data,
and LikelihoodInputs.  Results are collected in the global `detector_outputs` dict.
=#

######################################
######## IMPORTS AND INCLUDES ########
######################################

using JLD2, CSV, DataFrames, Plots

include(joinpath(@__DIR__, "core.jl"))

#############################################
######## SHARED INITIALIZATION (once) ########
#############################################

# Load solar production region and flux shapes
include(joinpath(@__DIR__, "solarModel.jl"))

# Load Earth model and build neutrino propagation paths
include(joinpath(@__DIR__, "earthProfile.jl"))
include(joinpath(@__DIR__, "oscillations", "makePaths.jl"))

# Load exposure vs. zenith angle (shared across detectors)
include(joinpath(@__DIR__, "exposure.jl"))

# Include function-definition files (no top-level computation here)
include(joinpath(@__DIR__, "unoscillatedSample.jl"))
include(joinpath(@__DIR__, "response.jl"))
include(joinpath(@__DIR__, "backgrounds.jl"))

# Compute shared Etrue bin edges needed by propagation_osc.jl
# (E_calc is set as a global inside propagation_osc.jl when propagation_main.jl is included)
global bin_edges_calc, energies_calc = calculate_bins(
    (max=Etrue_bins.max, min=Etrue_bins.min, bin_number=Etrue_bins.bin_number * 2)
)

include(joinpath(@__DIR__, "propagation", "propagation_main.jl"))

# Likelihood struct and function definitions — must precede the detector loop
include(joinpath(@__DIR__, "likelihoods", "likelihood_core.jl"))
include(joinpath(@__DIR__, "likelihoods", "likelihood_debug.jl"))
include(joinpath(@__DIR__, "likelihoods", "likelihood_statistical.jl"))
include(joinpath(@__DIR__, "likelihoods", "likelihood_builder.jl"))

# Build Earth propagation paths (shared)
global earth_paths  = [make_potential_for_integrand(z, earth) for z in cosz_calc]
global earth_lookup = let
    base    = get_avg_densities(earth_paths)
    n_exp   = isnothing(earth_normalisation_true) ? length(base) : length(earth_normalisation_true)
    if length(base) < n_exp
        # Some innermost layers (e.g. inner core) lie outside the exposure support
        # and are not traversed by any path in cosz_calc.  Get their nominal
        # densities from the vertical reference path (layers are indexed
        # outermost→innermost, so missing ones are always at the high end).
        ref = get_avg_densities([make_potential_for_integrand(-1.0, earth)])
        [base; ref[length(base)+1:n_exp]]
    else
        base
    end
end

#############################################
######## PER-DETECTOR LOOP ########
#############################################

# Shared Asimov oscillation parameters (detector-independent)
global sin2_th12_true, sin2_th13_true, dm2_21_true
global integrated_8B_flux_true, integrated_HEP_flux_true

# Collect outputs for every detector
global detector_outputs = Dict{String, NamedTuple}()

@logmsg Setup "Running fit with detector(s): $(join(keys(detector_configs), ", "))"

for (det_name, det) in detector_configs

    @logmsg Setup ("\n══ Detector: $det_name ══")
    if det.singleChannel == false
        @logmsg Setup ("  Fitting ES and CC channels.")
    elseif det.singleChannel == "ES"
        @logmsg Setup ("  Fitting ES channel only. CC event rates will appear as zero.")
    elseif det.singleChannel == "CC"
        @logmsg Setup ("  Fitting CC channel only. ES event rates will appear as zero.")
    end
    if det.inclusive_analysis
        @logmsg Setup ("  Inclusive mode: CC signal folded into ES channel. CC backgrounds disabled.")
    end
    if det.semi_inclusive_analysis
        @logmsg Setup ("  Semi-inclusive mode: forward hemisphere inclusive, backward hemisphere CC + mis-ID ES.")
    end

    # ── Build unoscillated sample ──────────────────────────────────────────
    unoscillatedSample_det, bin_edges_det, _ = build_unoscillated_sample(det)

    # ── Build response matrices ────────────────────────────────────────────
    responseMatrices_det, Ereco_bins_ES_ext, Ereco_bins_CC_ext = build_response_matrices(det)

    # ── Build backgrounds ──────────────────────────────────────────────────
    backgrounds_det, ES_bg_norms_true_det, CC_bg_norms_true_det,
        ES_bg_norms_pars_det, CC_bg_norms_pars_det = build_backgrounds(
            det, Ereco_bins_ES_ext, Ereco_bins_CC_ext
        )

    # ── Build true parameters (Asimov) ────────────────────────────────────
    true_parameters_det = Dict{Symbol, Any}(
        :sin2_th12            => sin2_th12_true,
        :sin2_th13            => sin2_th13_true,
        :dm2_21               => dm2_21_true,
        :integrated_HEP_flux  => integrated_HEP_flux_true,
        :integrated_8B_flux   => integrated_8B_flux_true,
        :ES_asymmetry         => 0,
        :CC_asymmetry         => 0,
        :cc_xsec_norm         => cc_xsec_norm_true,
        :cc_xsec_tilt         => 0.0,
        :cc_xsec_curv         => 0.0,
    )
    if earthUncertainty
        for (i, val) in enumerate(earth_normalisation_true)
            true_parameters_det[Symbol("earth_norm_$i")] = val
        end
    end
    if det.ES_mode
        for (i, norm) in enumerate(ES_bg_norms_true_det)
            true_parameters_det[Symbol("$(det_name)_ES_bg_norm_$i")] = norm
        end
    end
    if det.CC_mode && (!det.inclusive_analysis || det.semi_inclusive_analysis)
        for (i, norm) in enumerate(CC_bg_norms_true_det)
            true_parameters_det[Symbol("$(det_name)_CC_bg_norm_$i")] = norm
        end
    end
    true_params_det = (; true_parameters_det...)

    # ── det_flags: per-detector mode flags for propagateSamples ───────────
    det_flags = (
        ES_mode                 = det.ES_mode,
        CC_mode                 = det.CC_mode,
        angular_reco            = det.angular_reco,
        inclusive_analysis      = det.inclusive_analysis,
        semi_inclusive_analysis = det.semi_inclusive_analysis,
        angular_cos_cut         = det.angular_cos_cut,
        det_name                = det_name,
    )

    # ── Propagation closure (captures det_flags) ───────────────────────────
    # This is stored in LikelihoodInputs.f so expected_rates needs no extra args.
    prop_fn = let df = det_flags
        (unosc, resp, pars, ssm, edges, bg; precomputed_osc=nothing) ->
            propagateSamples(unosc, resp, pars, ssm, edges, bg, df;
                             precomputed_osc=precomputed_osc)
    end

    # ── Generate Asimov event rates ────────────────────────────────────────
    (measuredRate_ES_day, measuredRate_CC_day,
     measuredRate_ES_night, measuredRate_CC_night,
     BG_ES_tot_true, BG_CC_tot_true, CC_incl_spectrum_det) = propagateSamples(
        unoscillatedSample_det, responseMatrices_det,
        true_params_det, solarModel,
        bin_edges_det, backgrounds_det, det_flags; verbose=true
    )

    # ── Energy threshold indices ───────────────────────────────────────────
    index_ES_det = findfirst(x -> x > det.E_threshold.ES, Ereco_bins_ES_ext.bins)
    index_CC_det = findfirst(x -> x > det.E_threshold.CC, Ereco_bins_CC_ext.bins)

    if isnothing(index_ES_det)
        error("[$det_name] E_threshold.ES = $(det.E_threshold.ES) exceeds all ES reco-energy bins.")
    end
    if isnothing(index_CC_det) && det.CC_mode && (!det.inclusive_analysis || det.semi_inclusive_analysis)
        error("[$det_name] E_threshold.CC = $(det.E_threshold.CC) exceeds all CC reco-energy bins.")
    end
    index_CC_det = isnothing(index_CC_det) ? 1 : index_CC_det

    # ── Day-night asymmetry logging ────────────────────────────────────────
    CC_bg_aboveThreshold = ((det.inclusive_analysis && !det.semi_inclusive_analysis) || isempty(BG_CC_tot_true)) ?
                            0.0 : sum(BG_CC_tot_true[index_CC_det:end])
    ES_bg_aboveThreshold = sum(BG_ES_tot_true[index_ES_det:end])

    CC_Ntot = sum(@view measuredRate_CC_night[:, index_CC_det:end]) - 0.5 * CC_bg_aboveThreshold
    CC_Dtot = sum(measuredRate_CC_day[index_CC_det:end]) - 0.5 * CC_bg_aboveThreshold

    if det.ES_mode
        if det.angular_reco
            ES_Ntot = sum(@view measuredRate_ES_night[:, index_ES_det:end, :]) - 0.5 * ES_bg_aboveThreshold
            ES_Dtot = sum(measuredRate_ES_day[:, index_ES_det:end]) - 0.5 * ES_bg_aboveThreshold
        else
            ES_Ntot = sum(@view measuredRate_ES_night[:, index_ES_det:end]) - 0.5 * ES_bg_aboveThreshold
            ES_Dtot = sum(measuredRate_ES_day[index_ES_det:end]) - 0.5 * ES_bg_aboveThreshold
        end
    else
        ES_Ntot = 0.0; ES_Dtot = 0.0
    end

    ES_denom = ES_Dtot + ES_Ntot
    CC_denom = CC_Dtot + CC_Ntot

    asymm_ES     = ES_denom == 0 ? 0.0 : 2 * (ES_Dtot - ES_Ntot) / ES_denom
    eff_asymm_ES = ES_denom == 0 ? 0.0 : 2 * (ES_Dtot - ES_Ntot) / (ES_denom + 2 * ES_bg_aboveThreshold)
    asymm_CC     = CC_denom == 0 ? 0.0 : 2 * (CC_Dtot - CC_Ntot) / CC_denom
    eff_asymm_CC = CC_denom == 0 ? 0.0 : 2 * (CC_Dtot - CC_Ntot) / (CC_denom + 2 * CC_bg_aboveThreshold)

    true_parameters_det[:ES_asymmetry] = asymm_ES
    true_parameters_det[:CC_asymmetry] = asymm_CC

    # ── Observed-rate tuple (Asimov data) ─────────────────────────────────
    ereco_data_det = (
        ES_day   = measuredRate_ES_day,
        CC_day   = measuredRate_CC_day,
        ES_night = measuredRate_ES_night,
        CC_night = measuredRate_CC_night,
    )

    # ── Event count logging ────────────────────────────────────────────────
    CC_night_summed = sum(ereco_data_det.CC_night, dims=1)
    CC_combined     = vec(ereco_data_det.CC_day) .+ vec(CC_night_summed)

    if det.ES_mode
        if det.angular_reco
            ES_night_summed = sum(ereco_data_det.ES_night, dims=(1, 3))
            ES_combined     = vec(sum(ereco_data_det.ES_day, dims=1)) .+ vec(ES_night_summed)
        else
            ES_night_summed = sum(ereco_data_det.ES_night, dims=1)
            ES_combined     = vec(ereco_data_det.ES_day) .+ vec(ES_night_summed)
        end
    else
        ES_combined = zeros(size(CC_combined))
    end

    if (det.inclusive_analysis || det.semi_inclusive_analysis) && !isnothing(CC_incl_spectrum_det)
        incl_total = sum(ES_combined[index_ES_det:end])
        CC_above   = sum(CC_incl_spectrum_det[index_ES_det:end])
        BG_above   = ES_bg_aboveThreshold
        ES_above   = incl_total - CC_above - BG_above
        @logmsg Setup "  Total inclusive (ES+CC) data above threshold: $(sci_notation(incl_total))"
        @logmsg Setup "    of which ES signal: $(sci_notation(ES_above))"
        @logmsg Setup "    of which CC signal: $(sci_notation(CC_above))"
        @logmsg Setup "    of which BG: $(sci_notation(BG_above))"
    else
        det.ES_mode && @logmsg Setup "  Total ES data above threshold: $(sci_notation(sum(ES_combined[index_ES_det:end])))"
        if det.CC_mode && (!det.inclusive_analysis || det.semi_inclusive_analysis)
            CC_total_above = sum(CC_combined[index_CC_det:end])
            CC_sig_above   = CC_total_above - CC_bg_aboveThreshold
            @logmsg Setup "  Total CC data above threshold: $(sci_notation(CC_total_above))"
            @logmsg Setup "    of which signal: $(sci_notation(CC_sig_above))"
            @logmsg Setup "    of which BG: $(sci_notation(CC_bg_aboveThreshold))"
        end
    end
    @logmsg Setup @sprintf("  D-N asymmetry ES: %.4f%%", eff_asymm_ES * 100)
    @logmsg Setup @sprintf("  D-N asymmetry CC: %.4f%%", eff_asymm_CC * 100)

    # ── Angular diagnostics (debug) ────────────────────────────────────────
    if det.ES_mode || det.CC_mode
        save_debug_data(
            unoscillatedSample_det, responseMatrices_det, true_params_det,
            solarModel, bin_edges_det, backgrounds_det, det_flags;
            save_path = "$(outFile)_$(det_name)_angular_stacks.jld2",
        )
    end

    # ── Build LikelihoodInputs ─────────────────────────────────────────────
    likelihood_inputs_det = LikelihoodInputs(
        ereco_data_det,
        bin_edges_det,
        responseMatrices_det,
        solarModel,
        unoscillatedSample_det,
        backgrounds_det,
        prop_fn,
        det.ES_mode,
        det.CC_mode,
        index_ES_det,
        index_CC_det,
        det_name,
    )

    detector_outputs[det_name] = (
        likelihood_inputs  = likelihood_inputs_det,
        true_params        = (; true_parameters_det...),
        ES_bg_norms_pars   = ES_bg_norms_pars_det,
        CC_bg_norms_pars   = CC_bg_norms_pars_det,
        det_flags          = det_flags,
    )
end  # detector loop

println(" ")

# Load likelihood-building functions
include(joinpath(@__DIR__, "likelihoods", "likelihood_main.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatibility globals for llhScan.jl / derive_variables_from_chain.jl
# These scripts predate the multi-detector refactor and reference detector
# properties as bare globals.  Expose the first detector's values so they
# continue to work unchanged in the single-detector case.
# ─────────────────────────────────────────────────────────────────────────────
let _det  = first(values(detector_configs)),
    _out  = first(values(detector_outputs))
    global angular_reco            = _det.angular_reco
    global inclusive_analysis      = _det.inclusive_analysis
    global semi_inclusive_analysis = _det.semi_inclusive_analysis
    global ES_mode                 = _det.ES_mode
    global CC_mode            = _det.CC_mode
    global singleChannel      = _det.singleChannel
    global index_ES           = _out.likelihood_inputs.index_ES
    global index_CC           = _out.likelihood_inputs.index_CC
    global Ereco_bins_ES      = _det.Ereco_bins_ES
    global Ereco_bins_CC      = _det.Ereco_bins_CC
    global cos_scatter_bins   = _det.cos_scatter_bins
    # For llhScan.jl: true parameter values and background prior distributions
    global true_params        = _out.true_params
    global true_parameters    = Dict{Symbol,Any}(pairs(_out.true_params)...)
    global ES_bg_norms_pars   = _out.ES_bg_norms_pars
    global CC_bg_norms_pars   = _out.CC_bg_norms_pars
end
