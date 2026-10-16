#=
mcstat.jl — MC statistical uncertainty of the predictions in the likelihood.

The predicted rates come from finite MC samples (signal response matrices, background
templates). Following Dembinski & Abdelmotteleb (arXiv:2206.12346) and Conway
(arXiv:1103.0354), as in MaCh3's `GetTestStatLLH`, each prediction μ is allowed to move by a
factor β constrained by the MC: the MC is summarised per E_reco bin by its effective count

    a = M² / Var_MC(M)        (M = predicted events in that E_reco bin, day + night)

and β is profiled analytically.

  DembinskiAbdelmotteleb:  −log L = Σ_b P(n_b; β μ_b) + P(a; β a),        β = (N + a)/(M + a)
  BarlowBeeston (Conway):  −log L = Σ_b P(n_b; β μ_b) + a (β − 1)² / 2,   a β² + (M − a) β − N = 0

with P(n; μ) = μ − n + n ln(n/μ) (Baker–Cousins), N = Σ_b n_b and M = Σ_b μ_b. The day bin and
all night (cos z) bins of one E_reco bin come from the same MC events (exactly for backgrounds,
night = day template × exposure; approximately for signal, whose night bins weight the same
response with different oscillated E_true rates), so by default ("shared") one β per E_reco bin
multiplies all of them. "independent" gives every bin its own β with the same a, as MaCh3 does
per bin: it treats the night copies of one MC fluctuation as independent, which understates the
normalisation uncertainty but lets the MC absorb day/night shape differences, diluting the
day-night (Δm²₂₁) information.

What is frozen: the effective count a, i.e. the RELATIVE MC uncertainty 1/√a, computed once at the
Asimov truth. MaCh3 (UpdateW2 = false) instead freezes Σw², the absolute uncertainty, and
recomputes a = μ²/Σw² from the current prediction; the two agree at the truth only.
Not modelled: correlations of one MC sample across channels (the CC MC feeds both semi-inclusive
channels; the ES MC also feeds the mis-ID term) and per-component shape fluctuations within one
E_reco bin (one β covers all cos θ bins). Notes: docs/mcstat/mcstat.pdf.
=#
using Printf
using Statistics: median

# ── effective MC counts per E_reco bin, at setup ─────────────────────────────────────────────

# True-energy rate entering the reco folding, day + all night bins (the 0.5 day/night split of
# apply_day_response / apply_night_response).
_true_total(day::AbstractVector, night::AbstractMatrix) = 0.5 .* (Float64.(day) .+ vec(sum(Float64.(night), dims=1)))

"""
    signal_column(T, counts, eff, tot, g) -> (c, v)

Contribution c_j = Σ_i T_i R_ij eff_j g_j of one signal component to E_reco bin j and its MC
variance. The response R_ij = n_ij / N_i is built from MC counts n_ij, normalised per row
(N_i = Σ_j n_ij), so given N_i a row is multinomial: Var(n_ij) = n_ij (1 − n_ij / N_i). An MC
event of row i stands for T_i / N_i events, so the response part of the variance is
Σ_i n_ij (1 − n_ij / N_i) (T_i / N_i)² (eff_j g_j)². The efficiency eff_j = sel_j / tot_j adds
the binomial term c_j² (1 − eff_j) / (eff_j tot_j). `g` is a fixed per-bin factor (angular
forward fraction, semi-inclusive split).
"""
function signal_column(T, counts, eff, tot, g)
    N = vec(sum(counts, dims=2))
    w = [N[i] > 0 ? T[i] / N[i] : 0.0 for i in eachindex(T)]          # events per MC event, row i
    c = vec(sum(counts .* w, dims=1)) .* eff .* g
    p = counts ./ max.(N, 1)                                            # row fractions n_ij / N_i
    v = vec(sum(counts .* (1 .- p) .* w .^ 2, dims=1)) .* (eff .* g) .^ 2
    for j in eachindex(c)
        if eff[j] > 0 && tot[j] > 0
            v[j] += c[j]^2 * (1 - eff[j]) / (eff[j] * tot[j])
        end
    end
    return c, v
end

# Day + night total of each background template at the given (true) normalisations, as in
# normalize_backgrounds, times `factor`.
function bg_components(raw, conv, par_counts, norms_true, factor)
    comps = Vector{Vector{Float64}}()
    ni = 1
    for (i, beh) in enumerate(par_counts)
        if beh != 0
            push!(comps, raw[i] .* (conv[i] * norms_true[ni]) .* factor); ni += 1
        else
            push!(comps, raw[i] .* factor)
        end
    end
    return comps
end

effective_count(M, V) = [V[j] > 0 && M[j] > 0 ? M[j]^2 / V[j] : Inf for j in eachindex(M)]

"""
    mc_effective_counts(det, responseMatrices, backgrounds, osc_samples, ES_norms_true,
                        CC_norms_true, exposure_weights) -> (ES = (a, M, V), CC = (a, M, V))

Per E_reco bin: the predicted events M (day + night) at the truth, rebuilt from its signal and
background components, their total MC variance V and the effective MC count a = M²/V (Inf when
V = 0). Covers plain, angular (forward fraction), inclusive and semi-inclusive modes; the
semi-inclusive mis-ID term enters M with no MC variance (its response is a product of two
already-smoothed matrices).
"""
function mc_effective_counts(det, R, bg, osc, ES_norms_true, CC_norms_true, expo)
    nf  = 0.5 * (1 + sum(expo))
    mc  = R.mc
    semi, incl = det.semi_inclusive_analysis, det.inclusive_analysis
    f_above = semi ? R.CC_split.above : 1.0
    f_below = semi ? R.CC_split.below : 1.0

    out = Dict{Symbol,Any}()
    if det.ES_mode
        n = R.bins.ES.bin_number
        g = det.angular_reco ? vec(sum(R.ES.angular[mc.ang_mask, :], dims=1)) : ones(n)
        M, V = zeros(n), zeros(n)
        for (T, m, eff) in ((_true_total(osc.ES.nue_day, osc.ES.nue_night), mc.ES_nue, R.eff.ES_nue),
                            (_true_total(osc.ES.nuother_day, osc.ES.nuother_night), mc.ES_nuother, R.eff.ES_nuother))
            c, v = signal_column(T, m.counts, eff, m.tot, g)
            M .+= c; V .+= v
        end
        if (incl || semi) && det.CC_mode
            c, v = signal_column(_true_total(osc.CC.day, osc.CC.night), mc.CC_incl.counts, R.eff.CC_incl,
                                 mc.CC_incl.tot, fill(f_above, n))
            M .+= c; V .+= v
        end
        for (c, rv) in zip(bg_components(bg.ES, bg.ES_conv, bg.ES_par_counts, ES_norms_true, nf), bg.ES_relvar)
            M .+= c; V .+= c .^ 2 .* rv
        end
        out[:ES] = (a = effective_count(M, V), M = M, V = V)
    end
    if det.CC_mode && (!incl || semi)
        n = R.bins.CC.bin_number
        M, V = zeros(n), zeros(n)
        c, v = signal_column(_true_total(osc.CC.day, osc.CC.night), mc.CC.counts, R.eff.CC, mc.CC.tot, fill(f_below, n))
        M .+= c; V .+= v
        for (c, rv) in zip(bg_components(bg.CC, bg.CC_conv, bg.CC_par_counts, CC_norms_true, nf * f_below), bg.CC_relvar)
            M .+= c; V .+= c .^ 2 .* rv
        end
        if semi && det.ES_mode   # backward ES mis-identified as CC (no MC variance attributed)
            for (T, Rm) in ((_true_total(osc.ES.nue_day, osc.ES.nue_night), R.CC_misID_nue),
                            (_true_total(osc.ES.nuother_day, osc.ES.nuother_night), R.CC_misID_nuother))
                M .+= vec(sum(T .* Rm, dims=1)) .* R.eff.CC
            end
        end
        out[:CC] = (a = effective_count(M, V), M = M, V = V)
    end
    return out
end


# ── test statistics ──────────────────────────────────────────────────────────────────────────

abstract type MCStatMethod end
struct DembinskiAbdelmotteleb <: MCStatMethod end
struct BarlowBeeston <: MCStatMethod end          # Conway's approximation

# The positive-floor used by poissonLogLikelihood (softplus at 1e-9). As there, a prediction that
# is negative beyond numerical noise is an error, not something to floor away.
_mu_floor(e) = 0.5 * (e + 1e-9 + sqrt((e - 1e-9)^2 + 1e-18))
@noinline _negative_prediction(e) = throw(ArgumentError("MC-stat likelihood: prediction significantly negative (μ = $e)"))
@inline _checked(e) = (e < -1e-3 && _negative_prediction(e); e)

# Baker–Cousins P(n; μ) = μ − n + n ln(n/μ) for a data count n and prediction μ > 0.
_bc(n, μ) = n > 0 ? μ - n + n * (log(n) - log(μ)) : μ

# Profiled scale β for totals (N data, M prediction) and effective MC count a, and the MC
# penalty at that β. Both are written in x = β − 1, solved directly: for large a, β → 1 and
# forms such as β a − a − a ln β lose every digit to cancellation.
function _beta_penalty(::DembinskiAbdelmotteleb, N, M, a)
    x = (N - M) / (M + a)                                # β = (N + a)/(M + a)
    return 1 + x, a * (x - log1p(x))                     # P(a; β a) = a (β − 1 − ln β)
end
function _beta_penalty(::BarlowBeeston, N, M, a)
    # a β² + (M − a) β − N = 0  ⇔  a x² + (a + M) x + (M − N) = 0; the root with β ≥ 0, in the
    # form without cancellation (the discriminant equals (a − M)² + 4 a N ≥ 0)
    s = a + M
    D = (a - M)^2 + 4 * a * N
    # max(): at N = 0 and M = a exactly, sqrt'(0) would give a NaN gradient (dD vanishes there)
    x = -2 * (M - N) / (s + sqrt(max(D, 1e-300)))
    return 1 + x, a * x^2 / 2
end

# −log L of one group of bins (day + night of one E_reco bin, or a single bin) sharing one β.
# `μ` and `n` are iterables of the group's predictions and data. Plain Poisson when a = Inf.
function _group_negllh(method, μs, ns, a)
    if !isfinite(a)
        acc = zero(eltype(μs))
        for k in eachindex(μs)
            acc += _bc(ns[k], _mu_floor(_checked(μs[k])))
        end
        return acc
    end
    M = zero(eltype(μs)); N = 0.0
    for k in eachindex(μs)
        M += _mu_floor(_checked(μs[k])); N += ns[k]
    end
    β, pen = _beta_penalty(method, N, M, a)
    acc = pen
    for k in eachindex(μs)
        acc += _bc(ns[k], β * _mu_floor(μs[k]))
    end
    return acc
end

# One bin with its own β (independent mode); scalar form of _group_negllh.
function _bin_negllh(method, μ, n, a)
    m = _mu_floor(_checked(μ))
    isfinite(a) || return _bc(n, m)
    β, pen = _beta_penalty(method, n, m, a)
    return _bc(n, β * m) + pen
end

# Allocation-free sums over the bins of E_reco column j (the day bin(s) and every night bin), for
# the likelihood path: reverse-mode AD pays for every temporary array.
function _col_MN(day::AbstractVector, night::AbstractMatrix, od, on, j)
    M = _mu_floor(_checked(day[j])); N = Float64(od[j])
    for k in axes(night, 1)
        M += _mu_floor(_checked(night[k, j])); N += on[k, j]
    end
    return M, N
end
function _col_MN(day::AbstractMatrix, night::AbstractArray{<:Any,3}, od, on, j)
    M = zero(promote_type(eltype(day), eltype(night))); N = 0.0
    for c in axes(day, 1)
        M += _mu_floor(_checked(day[c, j])); N += od[c, j]
    end
    for k in axes(night, 3), c in axes(night, 1)
        M += _mu_floor(_checked(night[c, j, k])); N += on[c, j, k]
    end
    return M, N
end
function _col_bc(day::AbstractVector, night::AbstractMatrix, od, on, j, β)
    acc = _bc(od[j], β * _mu_floor(_checked(day[j])))
    for k in axes(night, 1)
        acc += _bc(on[k, j], β * _mu_floor(_checked(night[k, j])))
    end
    return acc
end
function _col_bc(day::AbstractMatrix, night::AbstractArray{<:Any,3}, od, on, j, β)
    acc = zero(promote_type(eltype(day), eltype(night), typeof(β)))
    for c in axes(day, 1)
        acc += _bc(od[c, j], β * _mu_floor(_checked(day[c, j])))
    end
    for k in axes(night, 3), c in axes(night, 1)
        acc += _bc(on[c, j, k], β * _mu_floor(_checked(night[c, j, k])))
    end
    return acc
end
function _col_indep(method, day::AbstractVector, night::AbstractMatrix, od, on, j, a)
    acc = _bin_negllh(method, day[j], od[j], a)
    for k in axes(night, 1)
        acc += _bin_negllh(method, night[k, j], on[k, j], a)
    end
    return acc
end
function _col_indep(method, day::AbstractMatrix, night::AbstractArray{<:Any,3}, od, on, j, a)
    acc = zero(promote_type(eltype(day), eltype(night)))
    for c in axes(day, 1)
        acc += _bin_negllh(method, day[c, j], od[c, j], a)
    end
    for k in axes(night, 3), c in axes(night, 1)
        acc += _bin_negllh(method, night[c, j, k], on[c, j, k], a)
    end
    return acc
end

# Bins of E_reco column j: the day bin(s) and every night bin, as flat vectors. Day is (n_E,)
# or (n_cos, n_E) (angular); night is (n_cosz, n_E) or (n_cos, n_E, n_cosz).
_column(day::AbstractVector, night::AbstractMatrix, j) = (vcat(day[j], night[:, j]))
_column(day::AbstractMatrix, night::AbstractArray{<:Any,3}, j) = vcat(day[:, j], vec(night[:, j, :]))

"""
    mcstat_loglikelihood(method, shared, day, night, obs_day, obs_night, a, idx) -> log L

Log-likelihood of one channel from E_reco bin `idx` on, with the MC uncertainty of each
E_reco bin j summarised by its effective count a[j]. `shared = true`: one β per E_reco bin
for its day and night bins; `false`: one β per bin.
"""
function mcstat_loglikelihood(method::MCStatMethod, shared::Bool, day, night, obs_day, obs_night, a, idx)
    acc = zero(promote_type(eltype(day), eltype(night)))
    for j in idx:length(a)
        if !shared
            acc += _col_indep(method, day, night, obs_day, obs_night, j, a[j])
        elseif !isfinite(a[j])
            acc += _col_bc(day, night, obs_day, obs_night, j, 1.0)
        else
            M, N = _col_MN(day, night, obs_day, obs_night, j)
            β, pen = _beta_penalty(method, N, M, a[j])
            acc += pen + _col_bc(day, night, obs_day, obs_night, j, β)
        end
    end
    return -acc
end

"""
    mcstat_perbin(method, shared, day, night, obs_day, obs_night, a, idx, n_fit)

Per-bin log-likelihood with the same profiled β as `mcstat_loglikelihood` (same shapes as day
and night). A shared column's MC penalty is split over its bins in proportion to their
predictions. Bins outside idx:n_fit are 0 (the per-bin diagnostic leaves out the overflow bin).
"""
function mcstat_perbin(method::MCStatMethod, shared::Bool, day, night, obs_day, obs_night, a, idx, n_fit)
    out_d = zeros(size(day)); out_n = zeros(size(night))
    for j in idx:n_fit
        μs = _checked.(Float64.(_column(day, night, j)))
        ns = Float64.(_column(obs_day, obs_night, j))
        vals = zeros(length(μs))
        if !isfinite(a[j])
            vals .= _bc.(ns, _mu_floor.(μs))
        elseif shared
            mu = _mu_floor.(μs); M = sum(mu)
            β, pen = _beta_penalty(method, sum(ns), M, a[j])
            vals .= _bc.(ns, β .* mu) .+ pen .* mu ./ M
        else
            for k in eachindex(μs)
                vals[k] = _group_negllh(method, μs[k:k], ns[k:k], a[j])
            end
        end
        nd = ndims(day) == 1 ? 1 : size(day, 1)
        if ndims(day) == 1
            out_d[j] = -vals[1]; out_n[:, j] .= .-vals[2:end]
        else
            out_d[:, j] .= .-vals[1:nd]; out_n[:, j, :] .= reshape(.-vals[nd+1:end], size(night, 1), size(night, 3))
        end
    end
    return out_d, out_n
end

mcstat_method(name::AbstractString) =
    name == "DembinskiAbdelmotteleb" ? DembinskiAbdelmotteleb() :
    name == "BarlowBeeston"          ? BarlowBeeston() :
    error("unknown mc_stat_likelihood '$name' (Poisson, BarlowBeeston or DembinskiAbdelmotteleb)")

# Setup summary: the MC statistical uncertainty 1/√a in the fit window, per channel.
function log_mc_counts(det, mc, idx_ES, idx_CC)
    for (ch, idx) in ((:ES, idx_ES), (:CC, idx_CC))
        haskey(mc, ch) || continue
        a = mc[ch].a[idx:end]
        fin = filter(isfinite, a)
        isempty(fin) && continue
        rel = 1 ./ sqrt.(fin)
        @logmsg Setup @sprintf("  %s MC statistics in the fit window: relative uncertainty median %.2g%%, max %.2g%% (effective MC counts min %.3g)%s",
                               ch, 100 * median(rel), 100 * maximum(rel), minimum(fin),
                               det.mc_stat_likelihood == "Poisson" ? "; not in the likelihood (mc_stat_likelihood: Poisson)" :
                               "; in the likelihood ($(det.mc_stat_likelihood), $(det.mc_stat_day_night))")
    end
end

# Channel log-likelihood closures with the same call signature as llh_ES_poisson / llh_CC_poisson
# (and their per-bin forms), for make_likelihood / make_perbin_likelihood. `a` is the channel's
# effective-count vector from mc_effective_counts, fixed at setup.
llh_ES_mcstat(method::MCStatMethod, shared::Bool, a::Vector{Float64}) =
    (d, parameters, rates) -> mcstat_loglikelihood(method, shared, rates.ES_day, rates.ES_night,
                                                   d.nObserved.ES_day, d.nObserved.ES_night, a, d.index_ES)
llh_CC_mcstat(method::MCStatMethod, shared::Bool, a::Vector{Float64}) =
    (d, parameters, rates) -> mcstat_loglikelihood(method, shared, rates.CC_day, rates.CC_night,
                                                   d.nObserved.CC_day, d.nObserved.CC_night, a, d.index_CC)

function llh_ES_mcstat_perbin(method::MCStatMethod, shared::Bool, a::Vector{Float64})
    return (d, parameters, rates) -> begin
        pd, pn = mcstat_perbin(method, shared, rates.ES_day, rates.ES_night, d.nObserved.ES_day,
                               d.nObserved.ES_night, a, d.index_ES, length(a) - 1)
        (ES_day = pd, ES_night = pn)
    end
end
function llh_CC_mcstat_perbin(method::MCStatMethod, shared::Bool, a::Vector{Float64})
    return (d, parameters, rates) -> begin
        pd, pn = mcstat_perbin(method, shared, rates.CC_day, rates.CC_night, d.nObserved.CC_day,
                               d.nObserved.CC_night, a, d.index_CC, length(a) - 1)
        (CC_day = pd, CC_night = pn)
    end
end
