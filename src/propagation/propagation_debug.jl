module PropagationDebug

using Plots, Printf

export debug_plot_CC_backgrounds,
       debug_heatmap_response_CC,
       debug_heatmap_oscprobs,
       debug_heatmap_CC_night,
       plot_angular_stacks

# --- Vibrant but Balanced Palette (Hex) ---
const ES_COL   = "#ff6b6b" # Vibrant Coral/Red
const ES_LINE  = "#b33939" # Deep Crimson boundary
const CC_COL   = "#70a1ff" # Bright Sky Blue
const CC_LINE  = "#1e3799" # Royal Blue boundary
const BG_COL   = :gray     # Gray
const BG_LINE  = :black    # Black boundary
const LINE_W   = 1.8       # Slightly thicker for visibility

function debug_plot_CC_backgrounds(backgrounds; E_min::Real = 5, E_max::Real = 25)
    @assert !isempty(backgrounds.CC) "backgrounds.CC is empty"
    n_bins = length(backgrounds.CC[1])
    bins   = range(E_min, E_max, length = n_bins)
    bar(bins, backgrounds.CC[1], label = "Gammas", title = "CC Backgrounds", color=BG_COL, lw=LINE_W)
    if length(backgrounds.CC) ≥ 2
        bar!(bins, backgrounds.CC[2], label = "Neutrons", bar_position = :stack, color=CC_COL, lw=LINE_W)
    end
    display(current())
end

function debug_heatmap_response_CC(responseMatrices)
    display(heatmap(responseMatrices.CC', title = "Response Matrices CC'", aspect_ratio = :equal))
end

function debug_heatmap_oscprobs(oscProbs_nue_8B_night_large; title_suffix::AbstractString = "")
    t = "Oscillation probabilities νₑ 8B night" * (isempty(title_suffix) ? "" : " – " * title_suffix)
    display(heatmap(oscProbs_nue_8B_night_large, title=t, aspect_ratio=:equal))
end

function debug_heatmap_CC_night(eventRate_CC_night)
    display(heatmap(eventRate_CC_night[:, 1:end], title="CC Night Event Rate", xlabel="E_reco bin", ylabel="cos(θₙ) bin", aspect_ratio=:equal))
end

function plot_angular_stacks(
    ES_angular::AbstractMatrix,
    CC_angular::AbstractMatrix,
    BG_angular::AbstractMatrix;
    Ereco_min::Real  = 2.0,
    Ereco_max::Real  = 20.0,
    cos_min::Real    = -1.0,
    cos_max::Real    = 1.0,
    energy_slices    = nothing,
    n_panels::Int    = 4, 
    save_path        = nothing,
)
    n_cos, n_E = size(ES_angular)
    cos_edges = range(cos_min, cos_max, length=n_cos + 1)
    E_width   = (Ereco_max - Ereco_min) / n_E
    E_centers = [Ereco_min + E_width * (i - 0.5) for i in 1:n_E]

    if energy_slices === nothing
        step = max(1, n_E ÷ n_panels)
        energy_slices = step:step:n_E
    end

    panel_plots = []
    e_axis = range(Ereco_min, Ereco_max, length=n_E)
    es_1d = vec(sum(ES_angular, dims=1))
    cc_1d = vec(sum(CC_angular, dims=1))
    bg_1d = vec(sum(BG_angular, dims=1))

    # --- PANEL 1: Integrated Stacked Linear Spectrum ---
    p_spec_lin = plot(e_axis, bg_1d .+ cc_1d .+ es_1d, linetype=:steppre, fillrange=0, fillcolor=ES_COL, label="ES", title="Stacked Spectrum (Lin)", linecolor=ES_LINE, lw=LINE_W)
    plot!(p_spec_lin, e_axis, bg_1d .+ cc_1d, linetype=:steppre, fillrange=0, fillcolor=CC_COL, label="CC", linecolor=CC_LINE, lw=LINE_W)
    plot!(p_spec_lin, e_axis, bg_1d, linetype=:steppre, fillrange=0, fillcolor=BG_COL, label="BG", linecolor=BG_LINE, lw=LINE_W)
    plot!(p_spec_lin, xlabel="E_reco [MeV]", ylabel="Events", framestyle=:box, legend=:topright, 
          ylims=(0, 3e5), xlims=(Ereco_min+1, Ereco_max))
    push!(panel_plots, p_spec_lin)

    # --- PANEL 2: Integrated Unstacked Log Spectrum ---
    clamp_log(v) = [x > 0 ? x : 1e-2 for x in v]
    
    p_spec_log = plot(e_axis, clamp_log(es_1d), linetype=:steppre, color=ES_LINE, label="ES", title="Components (Log)", lw=LINE_W, yaxis=:log10)
    plot!(p_spec_log, e_axis, clamp_log(cc_1d), linetype=:steppre, color=CC_LINE, label="CC", lw=LINE_W)
    plot!(p_spec_log, e_axis, clamp_log(bg_1d), linetype=:steppre, color=BG_LINE, label="BG", lw=LINE_W)
    plot!(p_spec_log, xlabel="E_reco [MeV]", ylabel="Events", framestyle=:box, legend=:topright,
          ylims=(1e-1, 1e6), xlims=(Ereco_min+1, Ereco_max))
    push!(panel_plots, p_spec_log)

    # --- SUBSEQUENT PANELS: Angular Slices ---
    for i_E in energy_slices
        bg, cc, es = BG_angular[:, i_E], CC_angular[:, i_E], ES_angular[:, i_E]
        s3, s2, s1 = (bg .+ cc .+ es), (bg .+ cc), bg
        x = cos_edges[1:end-1]
        
        p = plot(x, s3, linetype=:steppre, fillrange=0, fillcolor=ES_COL, label="ES", linecolor=ES_LINE, lw=LINE_W)
        plot!(p, x, s2, linetype=:steppre, fillrange=0, fillcolor=CC_COL, label="CC", linecolor=CC_LINE, lw=LINE_W)
        plot!(p, x, s1, linetype=:steppre, fillrange=0, fillcolor=BG_COL, label="BG", linecolor=BG_LINE, lw=LINE_W)

        y_max = max(10.0, isempty(s3) ? 0.0 : maximum(s3) * 1.5)

        plot!(p, title=@sprintf("E_reco ≈ %.1f MeV", E_centers[i_E]), xlabel="cos θ", ylabel="Events", 
              framestyle=:box, legend=false, xlims=(cos_min, cos_max), 
              ylims=(0, y_max)) 
        push!(panel_plots, p)
    end

    n_total = length(panel_plots)
    ncols, nrows = 3, ceil(Int, n_total / 3)
    plt = plot(panel_plots..., layout=(nrows, ncols), size=(450*ncols, 380*nrows), margin=6Plots.mm)

    display(plt)
    save_path !== nothing && savefig(plt, save_path)
    return plt
end

end # module

# ─────────────────────────────────────────────────────────────────────────────
# Global Scope Functions
# ─────────────────────────────────────────────────────────────────────────────
using JLD2

function compute_angular_components(unoscillatedSample, responseMatrices, params, solarModel,
                                    bin_edges, raw_backgrounds, det_flags)
    Ereco_bins_ES    = responseMatrices.bins.ES
    cos_scatter_bins = responseMatrices.bins.cos_scatter
    n_cos = cos_scatter_bins.bin_number
    n_E   = Ereco_bins_ES.bin_number
    n_z   = cosz_bins.bin_number

    mixingPars = get_mixing_parameters(params)
    oscProbs_1e, earth_norm_vector, lookup = setup_earth_propagation(E_calc, mixingPars, params)
    BG_ES, BG_CC = normalize_backgrounds(raw_backgrounds, params, det_flags.det_name)
    oscProbs  = compute_oscillation_probabilities(E_calc, mixingPars, solarModel, params,
                                                  oscProbs_1e, earth_norm_vector, lookup)
    osc = compute_oscillated_samples(unoscillatedSample, params, oscProbs;
                                     es_mode=det_flags.ES_mode, cc_mode=det_flags.CC_mode)

    es_day_all   = zeros(Float64, n_E)
    es_night_all = zeros(Float64, n_z, n_E)

    if det_flags.ES_mode && osc.ES !== nothing
        ES_nue_eff     = responseMatrices.eff.ES_nue
        ES_nuother_eff = responseMatrices.eff.ES_nuother
        es_nue_day     = apply_day_response(osc.ES.nue_day,     responseMatrices.ES.nue,     ES_nue_eff)
        es_other_day   = apply_day_response(osc.ES.nuother_day, responseMatrices.ES.nuother, ES_nuother_eff)
        es_nue_night   = apply_night_response(osc.ES.nue_night,     responseMatrices.ES.nue,     ES_nue_eff)
        es_other_night = apply_night_response(osc.ES.nuother_night, responseMatrices.ES.nuother, ES_nuother_eff)
        es_day_all   = es_nue_day .+ es_other_day          # (n_Ereco_ES,)
        es_night_all = es_nue_night .+ es_other_night      # (n_cosz, n_Ereco_ES)
        es_total     = vec(es_day_all) .+ vec(sum(es_night_all, dims=1))
        ES_angular        = responseMatrices.ES.angular .* es_total'
        ES_angular_day    = responseMatrices.ES.angular .* es_day_all'
        # 3-D night distribution: (n_cos_scatter, n_Ereco_ES, n_cosz)
        ES_angular_night_3d = reshape(responseMatrices.ES.angular, n_cos, n_E, 1) .*
                              reshape(es_night_all', 1, n_E, n_z)
    else
        ES_angular          = zeros(n_cos, n_E)
        ES_angular_day      = zeros(n_cos, n_E)
        ES_angular_night_3d = zeros(n_cos, n_E, n_z)
    end

    if det_flags.CC_mode && det_flags.inclusive_analysis &&
       osc.CC !== nothing && hasproperty(responseMatrices, :CC_inclusive)
        CC_incl_eff = responseMatrices.eff.CC_incl
        cc_day   = apply_day_response(osc.CC.day,   responseMatrices.CC_inclusive, CC_incl_eff)
        cc_night = apply_night_response(osc.CC.night, responseMatrices.CC_inclusive, CC_incl_eff)
        cc_total   = vec(cc_day) .+ vec(sum(cc_night, dims=1))
        CC_angular          = responseMatrices.BG.angular .* cc_total'
        CC_angular_night_3d = reshape(responseMatrices.BG.angular, n_cos, n_E, 1) .*
                              reshape(cc_night', 1, n_E, n_z)
    else
        CC_angular          = zeros(n_cos, n_E)
        CC_angular_night_3d = zeros(n_cos, n_E, n_z)
    end

    if det_flags.CC_mode && !det_flags.inclusive_analysis && osc.CC !== nothing
        cc_day_ereco, cc_night_ereco =
            compute_CC_event_rates(osc.CC, responseMatrices, BG_CC, det_flags)
    else
        n_CC = responseMatrices.bins.CC.bin_number
        cc_day_ereco   = zeros(Float64, n_CC)
        cc_night_ereco = zeros(Float64, n_z, n_CC)
    end

    bg_ES_day = isempty(BG_ES) ? zeros(Float64, n_E) : 0.5 .* vec(BG_ES)

    if !isempty(BG_ES)
        bg_night_per_z = 0.5 .* (BG_ES' .* exposure_weights)   # (n_z, n_E)
        bg_night_total = vec(sum(bg_night_per_z, dims=1))
        bg_total       = 0.5 .* vec(BG_ES) .+ bg_night_total
        BG_angular          = responseMatrices.BG.angular .* bg_total'
        BG_angular_night_3d = reshape(responseMatrices.BG.angular, n_cos, n_E, 1) .*
                              reshape(bg_night_per_z', 1, n_E, n_z)
    else
        BG_angular          = zeros(n_cos, n_E)
        BG_angular_night_3d = zeros(n_cos, n_E, n_z)
    end

    return (
        ES_angular          = ES_angular,
        CC_angular          = CC_angular,
        BG_angular          = BG_angular,
        ES_angular_day      = ES_angular_day,
        ES_angular_night_3d = ES_angular_night_3d,
        CC_angular_night_3d = CC_angular_night_3d,
        BG_angular_night_3d = BG_angular_night_3d,
        oscProbs            = oscProbs,
        osc                 = osc,
        es_day_1d           = es_day_all,
        es_night_2d         = es_night_all,
        cc_day_1d           = cc_day_ereco,
        cc_night_2d         = cc_night_ereco,
        bg_ES_day           = bg_ES_day,
    )
end

function save_debug_data(unoscillatedSample, responseMatrices, params,
                         solarModel, bin_edges, raw_backgrounds, det_flags;
                         save_path::AbstractString)
    Ereco_bins_ES    = responseMatrices.bins.ES
    Ereco_bins_CC    = responseMatrices.bins.CC
    cos_scatter_bins = responseMatrices.bins.cos_scatter
    has_CC           = det_flags.CC_mode
    has_CC_incl      = has_CC && det_flags.inclusive_analysis && hasproperty(responseMatrices, :CC_inclusive)

    d = compute_angular_components(
        unoscillatedSample, responseMatrices, params, solarModel,
        bin_edges, raw_backgrounds, det_flags)

    jldopen(save_path, "w") do f
        # ── backward-compat top-level keys ──────────────────────────────────
        f["ES_angular"] = det_flags.angular_reco ? d.ES_angular : zeros(0, 0)
        f["CC_angular"] = det_flags.angular_reco ? d.CC_angular : zeros(0, 0)
        f["BG_angular"] = det_flags.angular_reco ? d.BG_angular : zeros(0, 0)
        f["Ereco_min"]  = Float64(Ereco_bins_ES.min * 1e3)
        f["Ereco_max"]  = Float64(Ereco_bins_ES.max * 1e3)
        f["cos_min"]    = det_flags.angular_reco ? Float64(cos_scatter_bins.min) : 0.0
        f["cos_max"]    = det_flags.angular_reco ? Float64(cos_scatter_bins.max) : 1.0

        # ── axis metadata ─────────────────────────────────────────────────────
        f["meta/Etrue_min"]        = Float64(Etrue_bins.min * 1e3)
        f["meta/Etrue_max"]        = Float64(Etrue_bins.max * 1e3)
        f["meta/Etrue_n"]          = Etrue_bins.bin_number
        f["meta/Ereco_ES_min"]     = Float64(Ereco_bins_ES.min * 1e3)
        f["meta/Ereco_ES_max"]     = Float64(Ereco_bins_ES.max * 1e3)
        f["meta/Ereco_ES_n"]       = Ereco_bins_ES.bin_number
        f["meta/Ereco_CC_min"]     = has_CC ? Float64(Ereco_bins_CC.min * 1e3) : 0.0
        f["meta/Ereco_CC_max"]     = has_CC ? Float64(Ereco_bins_CC.max * 1e3) : 0.0
        f["meta/Ereco_CC_n"]       = has_CC ? Ereco_bins_CC.bin_number         : 0
        f["meta/cos_scatter_min"]  = Float64(cos_scatter_bins.min)
        f["meta/cos_scatter_max"]  = Float64(cos_scatter_bins.max)
        f["meta/cos_scatter_n"]    = cos_scatter_bins.bin_number
        f["meta/cosz_min"]         = Float64(cosz_bins.min)
        f["meta/cosz_max"]         = Float64(cosz_bins.max)
        f["meta/cosz_n"]           = cosz_bins.bin_number
        f["meta/cosz_edges"]       = collect(Float64.(COARSE_COSZ_EDGES))   # real (possibly non-uniform) night bin edges
        f["meta/has_CC"]           = has_CC
        f["meta/has_CC_inclusive"] = has_CC_incl
        f["meta/has_angular"]      = det_flags.angular_reco
        f["meta/has_ES_mode"]      = det_flags.ES_mode
        f["meta/has_CC_mode"]      = det_flags.CC_mode
        f["meta/has_CC_separate"]  = det_flags.CC_mode && !det_flags.inclusive_analysis

        # ── unoscillated spectra (true E) ─────────────────────────────────────
        f["unosc/ES_nue_8B"]      = Vector{Float64}(unoscillatedSample.ES_nue_8B)
        f["unosc/ES_nuother_8B"]  = Vector{Float64}(unoscillatedSample.ES_nuother_8B)
        f["unosc/CC_8B"]          = Vector{Float64}(unoscillatedSample.CC_8B)
        f["unosc/ES_nue_hep"]     = Vector{Float64}(unoscillatedSample.ES_nue_hep)
        f["unosc/ES_nuother_hep"] = Vector{Float64}(unoscillatedSample.ES_nuother_hep)
        f["unosc/CC_hep"]         = Vector{Float64}(unoscillatedSample.CC_hep)

        # ── response matrices ─────────────────────────────────────────────────
        det_flags.ES_mode      && (f["resp/ES_nue"]      = Matrix{Float64}(responseMatrices.ES.nue))
        det_flags.ES_mode      && (f["resp/ES_nuother"]  = Matrix{Float64}(responseMatrices.ES.nuother))
        det_flags.angular_reco && (f["resp/ES_angular"]  = Matrix{Float64}(responseMatrices.ES.angular))
        has_CC                 && (f["resp/CC"]          = Matrix{Float64}(responseMatrices.CC))
        has_CC_incl            && (f["resp/CC_inclusive"] = Matrix{Float64}(responseMatrices.CC_inclusive))

        # ── oscillation probabilities (true E) ───────────────────────────────
        f["osc_probs/nue_8B_day"]    = Vector{Float64}(d.oscProbs.nue_8B_day)
        f["osc_probs/nue_8B_night"]  = Matrix{Float64}(d.oscProbs.nue_8B_night)
        f["osc_probs/nue_hep_day"]   = Vector{Float64}(d.oscProbs.nue_hep_day)
        f["osc_probs/nue_hep_night"] = Matrix{Float64}(d.oscProbs.nue_hep_night)

        # ── oscillated spectra (true E, before response) ─────────────────────
        osc = d.osc
        if osc.ES !== nothing
            f["oscillated/ES_nue_day"]       = Vector{Float64}(osc.ES.nue_day)
            f["oscillated/ES_nuother_day"]   = Vector{Float64}(osc.ES.nuother_day)
            f["oscillated/ES_nue_night"]     = Matrix{Float64}(osc.ES.nue_night)
            f["oscillated/ES_nuother_night"] = Matrix{Float64}(osc.ES.nuother_night)
        end
        if osc.CC !== nothing
            f["oscillated/CC_day"]   = Vector{Float64}(osc.CC.day)
            f["oscillated/CC_night"] = Matrix{Float64}(osc.CC.night)
        end

        # ── reco-space spectra (always saved regardless of mode) ──────────────
        f["ereco/ES_day"]    = Vector{Float64}(d.es_day_1d)
        f["ereco/ES_night"]  = Matrix{Float64}(d.es_night_2d)
        f["ereco/CC_day"]    = Vector{Float64}(d.cc_day_1d)
        f["ereco/CC_night"]  = Matrix{Float64}(d.cc_night_2d)
        f["ereco/BG_ES_day"] = Vector{Float64}(d.bg_ES_day)

        # ── angular distributions (reco space, only when angular_reco=true) ───
        if det_flags.angular_reco
            f["angular/ES_combined"]  = Matrix{Float64}(d.ES_angular)
            f["angular/ES_day"]       = Matrix{Float64}(d.ES_angular_day)
            f["angular/ES_night_3d"]  = Array{Float64, 3}(d.ES_angular_night_3d)
            f["angular/CC"]           = Matrix{Float64}(d.CC_angular)
            f["angular/CC_night_3d"]  = Array{Float64, 3}(d.CC_angular_night_3d)
            f["angular/BG"]           = Matrix{Float64}(d.BG_angular)
            f["angular/BG_night_3d"]  = Array{Float64, 3}(d.BG_angular_night_3d)
        end
    end
    @info "Debug pipeline data saved to $save_path"
end