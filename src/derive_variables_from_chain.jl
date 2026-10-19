using ElasticArrays    # For dynamic array handling
using ProgressMeter    # For progress tracking
using JLD2            # For data file operations
using LinearAlgebra, Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using ForwardDiff
import AutoDiffOperators: ADSelector

include(joinpath(@__DIR__, "setup.jl"))



function loadAllBatches(jld2File::String)
    param_data_accum = Dict{Symbol,Vector}()
    weights_accum = Int[]
    stepno_accum  = Int[]
    chainid_accum = Int[]

    jldopen(jld2File, "r") do f
        batch_names = sort(
            [k for k in keys(f) if startswith(k, "batch_")],
            by = k -> parse(Int, split(k, "_")[2])
        )
        for bname in batch_names
            g = f[bname]
            skip_keys = Set(["weights", "stepno", "chainid"])
            for k in keys(g)
                k in skip_keys && continue
                pname = Symbol(k)
                if !haskey(param_data_accum, pname)
                    param_data_accum[pname] = []
                end
                append!(param_data_accum[pname], g[k])
            end
            append!(weights_accum, g["weights"])
            append!(stepno_accum,  g["stepno"])
            append!(chainid_accum, g["chainid"])
        end
    end

    return param_data_accum, weights_accum, stepno_accum, chainid_accum
end


#=
Derived mode (RunMode: derived). Reads a finished chain (prevFile) and, for EVERY active
detector, computes per step the day-night asymmetries and, on thinned steps, the per-bin
log-likelihood diagnostics and posterior-predictive rates. The parameter moments and the
best-fit oscillation map are shared by all detectors.

Output (written into prevFile.jld2, replacing any earlier derived_* keys):
  derived_detectors           "VD,HD": active detectors, in config order
  derived_<DET>_<key>         per-detector quantities (keys as in the single-detector era)
  derived_<key>               shared quantities (cosz_edges, oscmap_*, exposure_weights)
utils/posteriorHelpers.py selects a detector and strips the prefix, so the plotting code reads
the same key names as before, and older single-detector files still load.
=#

# Overwrite all derived_* keys: a re-derive with different detectors must not leave stale keys.
# Errors propagate (a failed write must not look like a successful derive).
function saveDerivedChain(per_det::AbstractDict, shared::AbstractDict)
    fileName = prevFile * ".jld2"
    jldopen(fileName, "a+") do f
        for k in collect(keys(f))
            startswith(k, "derived_") && delete!(f, k)
        end
        f["derived_detectors"] = join(keys(per_det), ",")
        for (k, v) in shared
            f["derived_" * string(k)] = v
        end
        for (dname, derived) in per_det, (k, v) in derived
            f["derived_$(dname)_" * string(k)] = v
        end
    end
end

# Copy of x with the bins below the energy threshold index set to zero. Energy is the only
# axis of a vector and the 2nd axis otherwise (night: cosz × E; angular: cos × E [× night]).
function zero_below(x::AbstractArray, idx::Int)
    y = Array{Float64}(x)
    idx > 1 && (selectdim(y, ndims(y) == 1 ? 1 : 2, 1:idx-1) .= 0)
    return y
end

# Weighted accumulation into a Dict of running sums (arrays or scalars), allocated lazily.
acc_add!(acc::Dict{Symbol,Any}, key::Symbol, x::Number) = (acc[key] = get(acc, key, 0.0) + x)
acc_add!(acc::Dict{Symbol,Any}, key::Symbol, x::AbstractArray) =
    haskey(acc, key) ? (acc[key] .+= x) : (acc[key] = Array{Float64}(x))

# Everything derived mode tracks for one detector.
struct DerivedDetector
    name::String
    li::Any                       # LikelihoodInputs
    perbin::Function              # per-bin likelihood closure
    angular::Bool
    has_CC::Bool                  # CC channel present (not pure inclusive)
    iES::Int
    iCC::Int
    ref::Any                      # per-bin log L at the Asimov truth (see centring note below)
    true_params::Any
    cfg::Any                      # detector config block
    asym_ES::Vector{Float64}
    asym_CC::Vector{Float64}
    acc::Dict{Symbol,Any}         # weighted running sums
    ESday_samples::Vector{Vector{Float64}}
    CCday_samples::Vector{Vector{Float64}}
end

function DerivedDetector(dname)
    out = detector_outputs[dname]
    cfg = detector_configs[dname]
    li  = out.likelihood_inputs
    perbin = detector_perbin_fns[dname]
    # Reference per-bin log-likelihood at the Asimov truth. In background-dominated bins the
    # per-bin LLH is O(1e6), so the two-pass variance Var = E[ℓ²] − E[ℓ]² loses all precision
    # to catastrophic cancellation. Subtracting this per-bin reference makes the accumulated
    # deviations O(1); variance and covariance are invariant under a per-bin constant shift.
    ref = perbin(true_params)
    return DerivedDetector(dname, li, perbin, cfg.angular_reco, !cfg.inclusive_analysis,
                           li.index_ES, li.index_CC, ref, out.true_params, cfg,
                           Float64[], Float64[], Dict{Symbol,Any}(),
                           Vector{Vector{Float64}}(), Vector{Vector{Float64}}())
end

# Day-night asymmetry of the signal above threshold, per unit exposure (background removed: it is
# f_day·BG by day and f_night·BG over all night bins). dn_asymmetry is in propagation_core.jl (SPEC 4.6).
dn_asymmetry(day, night, BG, idx) = dn_asymmetry(day, night, BG, idx, DAY_FRACTION, NIGHT_FRACTION)

# One chain step for one detector. `thinned` steps also feed the diagnostics.
function process_step!(dd::DerivedDetector, pars, osc, w, θ1, θ2, thinned::Bool)
    r = expected_rates(dd.li, pars; precomputed_osc=osc)

    push!(dd.asym_ES, dn_asymmetry(r.ES_day, r.ES_night, r.BG_ES_tot, dd.iES))
    dd.has_CC && push!(dd.asym_CC, dn_asymmetry(r.CC_day, r.CC_night, r.BG_CC_tot, dd.iCC))
    thinned || return

    acc = dd.acc
    # --- posterior predictive moments (bins below threshold stay zero) ---
    for (ch, day, night, idx, on) in (("ES", r.ES_day, r.ES_night, dd.iES, true),
                                      ("CC", r.CC_day, r.CC_night, dd.iCC, dd.has_CC))
        on || continue
        d0, n0 = zero_below(day, idx), zero_below(night, idx)
        acc_add!(acc, Symbol(ch, "day_m1"),   w .* d0)
        acc_add!(acc, Symbol(ch, "day_m2"),   w .* d0 .^ 2)
        acc_add!(acc, Symbol(ch, "night_m1"), w .* n0)
        acc_add!(acc, Symbol(ch, "night_m2"), w .* n0 .^ 2)
    end

    # Total day rates above threshold per sample; background is subtracted after the loop with
    # the posterior-mean background, so the bg-norm spread widens the quantile band.
    push!(dd.ESday_samples, dd.angular ? vec(sum(@view(r.ES_day[:, dd.iES:end]), dims=1)) :
                                         r.ES_day[dd.iES:end])
    acc_add!(acc, :BGESday_m1, w .* (DAY_FRACTION .* r.BG_ES_tot))
    acc_add!(acc, :BGESday_m2, w .* (DAY_FRACTION .* r.BG_ES_tot) .^ 2)
    if dd.has_CC
        push!(dd.CCday_samples, r.CC_day[dd.iCC:end])
        acc_add!(acc, :BGCCday_m1, w .* (DAY_FRACTION .* r.BG_CC_tot))
        acc_add!(acc, :BGCCday_m2, w .* (DAY_FRACTION .* r.BG_CC_tot) .^ 2)
    end

    # --- per-bin and total log-likelihood moments, centred on the Asimov reference ---
    perbin = dd.perbin(pars; rates=r)
    for ch in (dd.has_CC ? (:ESnight, :ESday, :CCnight, :CCday) : (:ESnight, :ESday))
        ℓ = getfield(perbin, Symbol(replace(String(ch), "night" => "_night", "day" => "_day"))) .-
            getfield(dd.ref,  Symbol(replace(String(ch), "night" => "_night", "day" => "_day")))
        ℓt = sum(ℓ)
        acc_add!(acc, Symbol(ch, "_S1"),    w .* ℓ)
        acc_add!(acc, Symbol(ch, "_S2"),    w .* ℓ .^ 2)
        acc_add!(acc, Symbol(ch, "_Ssin2"), w .* ℓ .* θ1)
        acc_add!(acc, Symbol(ch, "_Sdm2"),  w .* ℓ .* θ2)
        acc_add!(acc, Symbol(ch, "_tot_S1"),    w * ℓt)
        acc_add!(acc, Symbol(ch, "_tot_S2"),    w * ℓt^2)
        acc_add!(acc, Symbol(ch, "_tot_Ssin2"), w * ℓt * θ1)
        acc_add!(acc, Symbol(ch, "_tot_Sdm2"),  w * ℓt * θ2)
    end
    return
end

function _wquantile(vals::AbstractVector{Float64}, weights::AbstractVector{Float64}, q::Float64)
    idx  = sortperm(vals)
    cumw = cumsum(weights[idx])
    cumw ./= cumw[end]
    k = searchsortedfirst(cumw, q)
    return vals[idx[clamp(k, 1, length(vals))]]
end

# Mean, variance, covariance and correlation with a parameter θ, from weighted running sums
# (arrays or scalars).
function moment_stats(S1, S2, Sθ; N, meanθ, varθ)
    Eℓ   = S1 ./ N
    Varℓ = max.(S2 ./ N .- Eℓ .^ 2, 0.0)
    varθ = max(varθ, 0.0)
    Cov  = Sθ ./ N .- Eℓ .* meanθ
    Corr = Cov ./ sqrt.(max.(Varℓ, 1e-30) .* max(varθ, 1e-30))
    return (mean=Eℓ, var=Varℓ, cov=Cov, corr=Corr)
end

# 1/2/3σ bands of (total − posterior-mean background) per energy bin, full-length arrays with
# zeros below threshold.
function quantile_bands(samples, bg_mean, idx, n_E, sample_weights)
    bands = Dict(k => zeros(Float64, n_E) for k in (:lo, :hi, :lo2, :hi2, :lo3, :hi3))
    isempty(samples) && return bands
    mat = reduce(hcat, samples)
    for j in axes(mat, 1)
        col = mat[j, :] .- bg_mean[idx + j - 1]
        for (k, q) in ((:lo, 0.1587), (:hi, 0.8413), (:lo2, 0.0228), (:hi2, 0.9772), (:lo3, 0.0013), (:hi3, 0.9987))
            bands[k][idx + j - 1] = _wquantile(col, sample_weights, q)
        end
    end
    return bands
end

function finalize_detector(dd::DerivedDetector, N, mean_sin2, var_sin2, mean_dm2, var_dm2, sample_weights)
    acc, li, cfg = dd.acc, dd.li, dd.cfg
    out = Dict{Symbol,Any}(:ES_asymmetry => dd.asym_ES, :inclusive_mode => cfg.inclusive_analysis)

    for ch in (dd.has_CC ? (:ESnight, :ESday, :CCnight, :CCday) : (:ESnight, :ESday))
        reff = getfield(dd.ref, Symbol(replace(String(ch), "night" => "_night", "day" => "_day")))
        s  = moment_stats(acc[Symbol(ch, "_S1")], acc[Symbol(ch, "_S2")], acc[Symbol(ch, "_Ssin2")];
                          N=N, meanθ=mean_sin2, varθ=var_sin2)
        sd = moment_stats(acc[Symbol(ch, "_S1")], acc[Symbol(ch, "_S2")], acc[Symbol(ch, "_Sdm2")];
                          N=N, meanθ=mean_dm2, varθ=var_dm2)
        t  = moment_stats(acc[Symbol(ch, "_tot_S1")], acc[Symbol(ch, "_tot_S2")], acc[Symbol(ch, "_tot_Ssin2")];
                          N=N, meanθ=mean_sin2, varθ=var_sin2)
        td = moment_stats(acc[Symbol(ch, "_tot_S1")], acc[Symbol(ch, "_tot_S2")], acc[Symbol(ch, "_tot_Sdm2")];
                          N=N, meanθ=mean_dm2, varθ=var_dm2)
        # means restored to the absolute scale by adding the reference back
        out[Symbol(ch, "_mean_llh")]              = s.mean .+ reff
        out[Symbol(ch, "_var_llh")]               = s.var
        out[Symbol(ch, "_corr_llh_sin2_th12")]    = s.corr
        out[Symbol(ch, "_corr_llh_dm2_21")]       = sd.corr
        out[Symbol(ch, "_tot_mean_llh")]          = [t.mean + sum(reff)]
        out[Symbol(ch, "_tot_var_llh")]           = [t.var]
        out[Symbol(ch, "_tot_corr_llh_sin2_th12")] = [t.corr]
        out[Symbol(ch, "_tot_corr_llh_dm2_21")]    = [td.corr]
        out[Symbol(ch, "_shape")]                 = collect(size(s.mean))
    end

    r_true = expected_rates(li, dd.true_params)
    for (ch, on, idx) in (("ES", true, dd.iES), ("CC", dd.has_CC, dd.iCC))
        on || continue
        day_mean   = acc[Symbol(ch, "day_m1")] ./ N
        night_mean = acc[Symbol(ch, "night_m1")] ./ N
        bg_mean    = acc[Symbol("BG", ch, "day_m1")] ./ N
        out[Symbol(ch, "day_pp_mean")]   = day_mean
        out[Symbol(ch, "day_pp_var")]    = max.(acc[Symbol(ch, "day_m2")] ./ N .- day_mean .^ 2, 0.0)
        out[Symbol(ch, "night_pp_mean")] = night_mean
        out[Symbol(ch, "night_pp_var")]  = max.(acc[Symbol(ch, "night_m2")] ./ N .- night_mean .^ 2, 0.0)
        out[Symbol(ch, "day_pp_shape")]   = collect(size(day_mean))
        out[Symbol(ch, "night_pp_shape")] = collect(size(night_mean))
        out[Symbol("BG", ch, "day_pp_mean")] = bg_mean
        out[Symbol("BG", ch, "day_pp_var")]  = max.(acc[Symbol("BG", ch, "day_m2")] ./ N .- bg_mean .^ 2, 0.0)
        # Night bg per (cosz, Ereco) = (f_night/f_day)·exposure_weights ⊗ day bg (day bg is f_day·BG,
        # night bg f_night·BG·exposure_weights; SPEC 4.6).
        out[Symbol("BG", ch, "night_pp_mean")] = (NIGHT_FRACTION / DAY_FRACTION) .* (exposure_weights .* bg_mean')
        n_E = ch == "ES" && dd.angular ? size(day_mean, 2) : length(day_mean)
        samples = ch == "ES" ? dd.ESday_samples : dd.CCday_samples
        for (k, v) in quantile_bands(samples, bg_mean, idx, n_E, sample_weights)
            out[Symbol(ch, "day_pp_", k)] = v
        end
        # Clean signal-only (oscillated, no bkg) rates at the Asimov truth, full reco range.
        # (data − bkg is unusable: bkg can be ~1e11 below threshold.) Angular ES keeps the bkg.
        BG = getfield(r_true, Symbol("BG_", ch, "_tot"))
        day_t, night_t = getfield(r_true, Symbol(ch, "_day")), getfield(r_true, Symbol(ch, "_night"))
        if ch == "ES" && dd.angular
            out[:signal_ESday], out[:signal_ESnight] = day_t, night_t
        else
            out[Symbol("signal_", ch, "day")]   = day_t .- DAY_FRACTION .* BG
            out[Symbol("signal_", ch, "night")] = night_t .- NIGHT_FRACTION .* (BG' .* exposure_weights)
        end
        out[Symbol("data_", ch, "day")]   = getfield(li.nObserved, Symbol(ch, "_day"))
        out[Symbol("data_", ch, "night")] = getfield(li.nObserved, Symbol(ch, "_night"))
        bins = ch == "ES" ? cfg.Ereco_bins_ES : cfg.Ereco_bins_CC
        # finite edges in MeV; the underflow bin is drawn from max(threshold − width, 0)
        edges = ereco_edges(bins); edges[1] = max(edges[1], 0.0)
        out[Symbol("Ereco_edges_", ch)] = edges .* 1e3
    end
    out[:index_ES] = dd.iES
    out[:index_CC] = dd.iCC

    if dd.has_CC
        out[:CC_asymmetry] = dd.asym_CC
        # Per-component CC background day spectra at the Asimov truth, for stacked plots,
        # mirroring normalize_backgrounds (f_day day factor, CC_conv × norm for free norms).
        comps = Vector{Vector{Float64}}()
        ni = 1
        for (i, beh) in enumerate(li.BG.CC_par_counts)
            if beh != 0
                nrm = dd.true_params[Symbol("$(dd.name)_CC_bg_norm_$ni")]
                push!(comps, DAY_FRACTION .* li.BG.CC[i] .* (li.BG.CC_conv[i] * nrm))
                ni += 1
            else
                push!(comps, DAY_FRACTION .* li.BG.CC[i])
            end
        end
        out[:CCbg_comp_day]   = isempty(comps) ? zeros(Float64, length(out[:signal_CCday]), 0) : reduce(hcat, comps)
        out[:CCbg_comp_names] = join([replace(split(basename(f), "_")[end], ".csv" => "")
                                      for f in cfg.CC_filepaths_BG[1:length(li.BG.CC)]], ",")
    end
    return out
end


######## BEGIN EXECUTION ########

println(" ")
prevFile === nothing && error("Please specify an input chain (prevFile)")
@logmsg Setup ("Calculating derived quantities for chain *$(prevFile)*, detectors: $(join(keys(detector_outputs), ", "))")

dets = [DerivedDetector(dname) for dname in keys(detector_outputs)]

param_data, weights, stepno, chainid = loadAllBatches(prevFile * ".jld2")
param_fields = collect(keys(param_data))
n_total = length(param_data[param_fields[1]])
@logmsg Setup ("Processing $(n_total) parameter values")
println(" ")

# Shared parameter moments (thinned, weighted)
N = 0.0
sum_sin2 = 0.0; sum_sin2_2 = 0.0
sum_dm2  = 0.0; sum_dm2_2  = 0.0
pp_sample_weights = Float64[]

p = Progress(n_total, dt = 0.1, showspeed = true, color = :yellow)
for i in 1:n_total
    global N, sum_sin2, sum_sin2_2, sum_dm2, sum_dm2_2
    pars = NamedTuple{Tuple(param_fields)}((param_data[field][i] for field in param_fields))
    # Fixed parameters (e.g. backgrounds with sys = 0) are not in the chain: fill from the truth.
    pars = merge(true_params, pars)
    osc  = compute_shared_osc_probs(pars, solarModel)    # once per step, shared by all detectors
    thinned = i % thinning == 0
    w = float(weights[i])
    θ1, θ2 = pars.sin2_th12, pars.dm2_21
    for dd in dets
        process_step!(dd, pars, osc, w, θ1, θ2, thinned)
    end
    if thinned
        N += w
        sum_sin2 += w * θ1; sum_sin2_2 += w * θ1^2
        sum_dm2  += w * θ2; sum_dm2_2  += w * θ2^2
        push!(pp_sample_weights, w)
    end
    ProgressMeter.update!(p, i, showvalues = [(:Progress, @sprintf("%.2f%%", 100*i/n_total))])
end

mean_sin2 = sum_sin2 / N; var_sin2 = sum_sin2_2 / N - mean_sin2^2
mean_dm2  = sum_dm2 / N;  var_dm2  = sum_dm2_2 / N - mean_dm2^2

per_det = OrderedDict(dd.name => finalize_detector(dd, N, mean_sin2, var_sin2, mean_dm2, var_dm2, pp_sample_weights)
                      for dd in dets)

# --- Shared: best-fit (posterior-mean) oscillation map P(νe→νe) over (cos z, E_true) ---
_wtot          = sum(weights)
_bestfit_means = NamedTuple{Tuple(param_fields)}(
    Tuple(sum(param_data[f][i] * weights[i] for i in 1:n_total) / _wtot for f in param_fields))
_oscmap        = compute_shared_osc_probs(merge(true_params, _bestfit_means), solarModel)
shared = Dict{Symbol,Any}(
    # cosz grid is piecewise-uniform (PREM segments) over the exposure support, not uniform [-1, 0]
    :cosz_edges         => collect(COARSE_COSZ_EDGES),
    :oscmap_8B_day      => collect(Float64.(_oscmap.nue_8B_day)),
    :oscmap_8B_night    => Matrix{Float64}(_oscmap.nue_8B_night),
    :oscmap_Etrue_edges => collect(range(Etrue_bins.min, Etrue_bins.max, length=Etrue_bins.bin_number + 1)) .* 1e3,
    # per-cos z exposure fraction, to turn night signal into a day-night survival ratio
    :exposure_weights   => collect(Float64.(vec(exposure_weights))),
)

println(" ")
for dd in dets
    tp = dd.true_params
    @logmsg Output ("[$(dd.name)] true D-N asymmetry: ES $(tp[:ES_asymmetry])" *
                    (dd.has_CC ? ", CC $(tp[:CC_asymmetry])" : " (CC not applicable in inclusive mode)"))
end
println(" ")

saveDerivedChain(per_det, shared)
@logmsg Output ("Saved derived quantities for $(length(dets)) detector(s) to $(prevFile).jld2")
