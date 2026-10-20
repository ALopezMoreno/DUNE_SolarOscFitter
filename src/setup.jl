#=
setup.jl — loads the shared inputs (solar model, Earth, exposure), then builds each detector's
samples, response, backgrounds, Asimov data and LikelihoodInputs into `detector_outputs`.
=#

using JLD2, CSV, DataFrames, Plots

include(joinpath(@__DIR__, "core.jl"))

include(joinpath(@__DIR__, "solarModel.jl"))

include(joinpath(@__DIR__, "earthProfile.jl"))
include(joinpath(@__DIR__, "oscillations", "makePaths.jl"))

include(joinpath(@__DIR__, "exposure.jl"))

# Function definitions only
include(joinpath(@__DIR__, "unoscillatedSample.jl"))
include(joinpath(@__DIR__, "response.jl"))
include(joinpath(@__DIR__, "backgrounds.jl"))
include(joinpath(@__DIR__, "inputChecks.jl"))

# Earth paths for the legacy NumOsc night calculation; osc_setup.jl checks NightOsc against them
global earth_paths  = [make_potential_for_integrand(z, earth) for z in cosz_calc]
global earth_lookup = let
    base    = get_avg_densities(earth_paths)
    n_exp   = isnothing(earth_normalisation_true) ? length(base) : length(earth_normalisation_true)
    if length(base) < n_exp
        # Innermost layers no path in cosz_calc crosses: nominal densities from the vertical path
        ref = get_avg_densities([make_potential_for_integrand(-1.0, earth)])
        [base; ref[length(base)+1:n_exp]]
    else
        base
    end
end

# Etrue bin edges for propagation_osc.jl
global bin_edges_calc, energies_calc = calculate_bins(
    (max=Etrue_bins.max, min=Etrue_bins.min, bin_number=Etrue_bins.bin_number * 2)
)

include(joinpath(@__DIR__, "propagation", "propagation_main.jl"))

# Must precede the detector loop
include(joinpath(@__DIR__, "likelihoods", "likelihood_core.jl"))
include(joinpath(@__DIR__, "likelihoods", "likelihood_debug.jl"))
include(joinpath(@__DIR__, "likelihoods", "likelihood_statistical.jl"))
include(joinpath(@__DIR__, "likelihoods", "likelihood_builder.jl"))
include(joinpath(@__DIR__, "likelihoods", "mcstat.jl"))

# Shared Asimov oscillation parameters (detector-independent)
global sin2_th12_true, sin2_th13_true, dm2_21_true
global integrated_8B_flux_true, integrated_HEP_flux_true

global detector_outputs = OrderedDict{String, NamedTuple}()

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

    # ── Input sample checks (per-file table and warnings) ─────────────────
    check_input_files(det)

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
    det_flags = DetFlags(
        ES_mode                 = det.ES_mode,
        CC_mode                 = det.CC_mode,
        angular_reco            = det.angular_reco,
        inclusive_analysis      = det.inclusive_analysis,
        semi_inclusive_analysis = det.semi_inclusive_analysis,
        angular_cos_cut         = det.angular_cos_cut,
        det_name                = det_name,
    )

    # ── Propagation closure (captures det_flags), stored in LikelihoodInputs.f ──
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

    # ── Energy threshold indices: the first fitted bin starts at the threshold (bin 2) ──
    index_ES_det = findfirst(x -> x >= det.E_threshold.ES, Ereco_bins_ES_ext.bins)
    index_CC_det = findfirst(x -> x >= det.E_threshold.CC, Ereco_bins_CC_ext.bins)
    if isnothing(index_ES_det)
        error("[$det_name] E_threshold.ES = $(det.E_threshold.ES) exceeds all ES reco-energy bins.")
    end
    if isnothing(index_CC_det)
        error("[$det_name] E_threshold.CC = $(det.E_threshold.CC) exceeds all CC reco-energy bins.")
    end
    for (ch, idx) in (("ES", index_ES_det), ("CC", index_CC_det))
        idx == 2 || error("[$det_name] the first fitted $ch bin is $idx, expected 2 (threshold edge)")
    end

    # ── Day-night asymmetry logging ────────────────────────────────────────
    # The background the CC prediction carries (f_below·BG in semi-inclusive mode)
    CC_bg_aboveThreshold = ((det.inclusive_analysis && !det.semi_inclusive_analysis) || isempty(BG_CC_tot_true)) ?
                            0.0 : cc_background_fraction(det.semi_inclusive_analysis, responseMatrices_det) *
                                  sum(BG_CC_tot_true[index_CC_det:end])
    ES_bg_aboveThreshold = sum(BG_ES_tot_true[index_ES_det:end])

    # Signal above threshold: the background is removed with its own (time-weighted) split (SPEC 4.6)
    fD, fN = SIGNAL_DAY_FRACTION, SIGNAL_NIGHT_FRACTION
    bD, bN = BG_TOTAL_FACTORS.day, BG_TOTAL_FACTORS.night
    CC_Ntot = sum(@view measuredRate_CC_night[:, index_CC_det:end]) - bN * CC_bg_aboveThreshold
    CC_Dtot = sum(measuredRate_CC_day[index_CC_det:end]) - bD * CC_bg_aboveThreshold

    if det.ES_mode
        if det.angular_reco
            ES_Ntot = sum(@view measuredRate_ES_night[:, index_ES_det:end, :]) - bN * ES_bg_aboveThreshold
            ES_Dtot = sum(measuredRate_ES_day[:, index_ES_det:end]) - bD * ES_bg_aboveThreshold
        else
            ES_Ntot = sum(@view measuredRate_ES_night[:, index_ES_det:end]) - bN * ES_bg_aboveThreshold
            ES_Dtot = sum(measuredRate_ES_day[index_ES_det:end]) - bD * ES_bg_aboveThreshold
        end
    else
        ES_Ntot = 0.0; ES_Dtot = 0.0
    end

    # Signal asymmetries per unit exposure (d = D/f_day, n = N/f_night). Per unit of its own exposure
    # the background is BG by day and by night, so the effective asymmetry is 2(d − n)/(d + n + 2·BG).
    ES_d, ES_n = ES_Dtot / fD, ES_Ntot / fN
    CC_d, CC_n = CC_Dtot / fD, CC_Ntot / fN
    asymm_ES     = dn_asymmetry(ES_Dtot, ES_Ntot, fD, fN)
    eff_asymm_ES = ES_d + ES_n == 0 ? 0.0 : 2 * (ES_d - ES_n) / (ES_d + ES_n + 2 * ES_bg_aboveThreshold)
    asymm_CC     = dn_asymmetry(CC_Dtot, CC_Ntot, fD, fN)
    eff_asymm_CC = CC_d + CC_n == 0 ? 0.0 : 2 * (CC_d - CC_n) / (CC_d + CC_n + 2 * CC_bg_aboveThreshold)

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
            # the background as CC_combined carries it
            _cc_bg_factored = CC_bg_aboveThreshold * (BG_DAY_FRACTION + BG_NIGHT_FRACTION * sum(exposure_weights_bg))
            CC_sig_above    = CC_total_above - _cc_bg_factored
            @logmsg Setup "  Total CC data above threshold: $(sci_notation(CC_total_above))"
            @logmsg Setup "    of which signal: $(sci_notation(CC_sig_above))"
            @logmsg Setup "    of which BG: $(sci_notation(_cc_bg_factored))"
        end
    end
    # ── MC statistics of the predictions: effective MC count per E_reco bin at the truth ──
    osc_samples_true = compute_oscillated_samples(unoscillatedSample_det, true_params_det,
                                                  compute_shared_osc_probs(true_params_det, solarModel),
                                                  Val(det.ES_mode), Val(det.CC_mode))
    mc_counts_det = mc_effective_counts(det, responseMatrices_det, backgrounds_det, osc_samples_true,
                                        ES_bg_norms_true_det, CC_bg_norms_true_det, exposure_weights_bg,
                                        SIGNAL_FRACTIONS, BG_FRACTIONS)
    log_mc_counts(det, mc_counts_det, index_ES_det, index_CC_det)

    # ── Per-bin data / signal / background breakdown at the truth ──────────
    night_factor = BG_DAY_FRACTION + BG_NIGHT_FRACTION * sum(exposure_weights_bg)   # day + night
    if det.ES_mode
        save_rate_breakdown(outFile, det_name, "ES", Ereco_bins_ES_ext.bins, index_ES_det, ES_combined,
                            backgrounds_det.ES, backgrounds_det.ES_conv, backgrounds_det.ES_par_counts,
                            ES_bg_norms_true_det, bg_component_names(det.ES_filepaths_BG, length(backgrounds_det.ES)),
                            night_factor; mc = get(mc_counts_det, :ES, nothing))
    end
    if det.CC_mode && (!det.inclusive_analysis || det.semi_inclusive_analysis)
        # semi-inclusive: the CC channel holds the backward fraction of CC signal and backgrounds
        cc_factor = det.semi_inclusive_analysis ? night_factor * responseMatrices_det.CC_split.below : night_factor
        save_rate_breakdown(outFile, det_name, "CC", Ereco_bins_CC_ext.bins, index_CC_det, CC_combined,
                            backgrounds_det.CC, backgrounds_det.CC_conv, backgrounds_det.CC_par_counts,
                            CC_bg_norms_true_det, bg_component_names(det.CC_filepaths_BG, length(backgrounds_det.CC)),
                            cc_factor; mc = get(mc_counts_det, :CC, nothing))
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
        mc_counts          = mc_counts_det,
        likelihood_inputs  = likelihood_inputs_det,
        true_params        = (; true_parameters_det...),
        ES_bg_norms_pars   = ES_bg_norms_pars_det,
        CC_bg_norms_pars   = CC_bg_norms_pars_det,
        det_flags          = det_flags,
    )
end

# Domain vs used range of every tabulated input
log_tabulated_inputs()

println(" ")

include(joinpath(@__DIR__, "likelihoods", "likelihood_main.jl"))

# Single-detector globals for llhScan.jl and derive_variables_from_chain.jl (the first detector)
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
    # Asimov values of all fit parameters (every detector's background norms); per-detector
    # diagnostics such as :ES_asymmetry are the first detector's.
    global true_params        = foldl((a, o) -> merge(o.true_params, a), values(detector_outputs);
                                      init = NamedTuple())
    global true_parameters    = Dict{Symbol,Any}(pairs(true_params)...)
    global ES_bg_norms_pars   = _out.ES_bg_norms_pars
    global CC_bg_norms_pars   = _out.CC_bg_norms_pars
end
