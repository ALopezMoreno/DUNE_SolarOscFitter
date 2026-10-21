using Logging
using LinearAlgebra, Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using DelimitedFiles, Random, Printf

include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "priors.jl"))

#=
2D likelihood scans of the oscillation parameters (RunMode: LLH).

`nuisance: false`: each cell is 2 log L with every other parameter at its Asimov true value (a
conditional scan). `nuisance: N`: each cell is 2 log of the likelihood averaged over N throws of
the nuisances from their priors, the same throws for every cell: a Monte Carlo marginalisation,
a quick picture of the posterior. Thrown: sin²θ₁₃ when not scanned, fluxes, CC cross section,
backgrounds, Earth layers (jointly, with their covariance). `nuisance: {top: k, min_share: c, points: 3}`: at
most k nuisances, those carrying a share ≥ c (default 0.1) of the sin²θ₁₂ or Δm²₂₁ variance in the Fisher matrix at the truth
(numerical Hessian of log L + log π) are averaged by a 3-point rule per nuisance (prior mean and
± √3 σ, weights 2/3, 1/6, 1/6; `points: 2`: ± 1 σ), all combinations; the rest stay at their true
values. With `method: laplace` (the default for this form) the top k are instead marginalised by
the Laplace approximation at each cell (laplace_cell). Deterministic, no throw noise. In every mode the other solar oscillation parameter stays
at its true value. Plot χ² = chi2_ref − cell, chi2_ref the value at the truth; the
2-parameter 1/2/3σ contours are at Δχ² = 2.30, 6.18, 11.83 (utils/plotLLH.py).
=#

const SCAN_PRIORS = build_priors()

lim_th12 = llh_range_sin2_th12
lim_th13 = llh_range_sin2_th13
lim_dm21 = llh_range_dm2_21

vals_12 = range(lim_th12[1], stop=lim_th12[2], length=llhBins)
vals_13 = range(lim_th13[1], stop=lim_th13[2], length=llhBins)
vals_dm = range(lim_dm21[1], stop=lim_dm21[2], length=llhBins)

# One throw of the nuisances `nuis` from their priors; Earth layers jointly from their MvNormal,
# redrawn until inside the (0, 2) bounds the MCMC uses.
function throw_nuisances(rng, nuis)
    vals = Dict{Symbol,Float64}(k => rand(rng, SCAN_PRIORS[k]) for k in nuis if !startswith(String(k), "earth_norm_"))
    earth = [k for k in nuis if startswith(String(k), "earth_norm_")]
    if !isempty(earth)
        e = rand(rng, earth_normalisation_prior)
        while !all(0 .< e .< 2)
            e = rand(rng, earth_normalisation_prior)
        end
        for k in earth
            vals[k] = e[parse(Int, String(k)[length("earth_norm_")+1:end])]
        end
    end
    return NamedTuple{Tuple(nuis)}(Tuple(vals[k] for k in nuis))
end

loglik(p) = logdensityof(likelihood_all_samples, p)

# 2 log of the likelihood at `base`, averaged over `throws` with `weights` (none: the conditional value)
function cell(base, throws, weights)
    isempty(throws) && return 2 * loglik(base)
    v = [loglik(merge(base, t)) for t in throws]
    m = maximum(v)
    return 2 * (m + log(sum(weights .* exp.(v .- m))))
end

const OSC_KEYS = (:sin2_th12, :sin2_th13, :dm2_21)

"""
    nuisance_ranking() -> Vector{Pair{Symbol,Float64}}

Every free parameter but sin²θ₁₂ and Δm²₂₁, ranked by the largest fraction of their variance it
carries: max_p C_pk² / (C_kk C_pp), p ∈ (sin²θ₁₂, Δm²₂₁), with C the inverse of minus the numerical
Hessian of log L + log π at the truth (steps of 5% of each prior's standard deviation).
"""
function nuisance_ranking()
    ks = collect(keys(SCAN_PRIORS))
    x0 = Float64[true_params[k] for k in ks]
    h = [0.05 * std(SCAN_PRIORS[k]) for k in ks]
    f(x) = (p = merge(true_params, NamedTuple{Tuple(ks)}(Tuple(x)));
            loglik(p) + sum(logpdf(SCAN_PRIORS[k], p[k]) for k in ks))
    n = length(ks)
    pairs = [(a, b) for a in 1:n for b in a:n]
    H = zeros(n, n)
    fc = f(x0)
    Threads.@threads for (a, b) in pairs
        ea, eb = zeros(n), zeros(n); ea[a] = h[a]; eb[b] = h[b]
        H[a, b] = H[b, a] = a == b ?
            (f(x0 + ea) - 2 * fc + f(x0 - ea)) / h[a]^2 :
            (f(x0 + ea + eb) - f(x0 + ea - eb) - f(x0 - ea + eb) + f(x0 - ea - eb)) / (4h[a] * h[b])
    end
    C = inv(Symmetric(-H))
    pois = [findfirst(==(k), ks) for k in (:sin2_th12, :dm2_21)]
    score = Dict(ks[k] => maximum(C[p, k]^2 / (C[k, k] * C[p, p]) for p in pois)
                 for k in 1:n if ks[k] ∉ (:sin2_th12, :dm2_21))
    return sort(collect(score), by = last, rev = true)
end

# Tensor-product quadrature over the nuisances `ks`: (points, weights)
function quadrature(ks, npoints)
    nodes, w = npoints == 3 ? ((-sqrt(3.0), 0.0, sqrt(3.0)), (1 / 6, 2 / 3, 1 / 6)) : ((-1.0, 1.0), (0.5, 0.5))
    throws, weights = NamedTuple[], Float64[]
    for idx in Iterators.product(ntuple(_ -> eachindex(nodes), length(ks))...)
        vals = Tuple(mean(SCAN_PRIORS[k]) + nodes[i] * std(SCAN_PRIORS[k]) for (k, i) in zip(ks, idx))
        push!(throws, NamedTuple{Tuple(ks)}(vals))
        push!(weights, prod(w[i] for i in idx))
    end
    return throws, weights
end

const RANKING = llh_nuisance isa AbstractDict ? nuisance_ranking() : nothing
if RANKING !== nothing
    @logmsg MCMC "Nuisances ranked by their largest share of the sin²θ₁₂ or Δm²₂₁ variance (Fisher at the truth; cutoff $(get(llh_nuisance, "min_share", 0.1))):"
    for (k, v) in RANKING[1:min(end, 8)]
        @logmsg MCMC @sprintf("  %-24s %.3f", k, v)
    end
end

laplace_mode() = llh_nuisance isa AbstractDict && get(llh_nuisance, "method", "laplace") == "laplace"
# The nuisances to marginalise: those with a Fisher share ≥ min_share, at most `top`, scanned axes excluded
function top_nuisances(xkey, ykey)
    cut = Float64(get(llh_nuisance, "min_share", 0.1))
    ks = [k for (k, v) in RANKING if k ∉ (xkey, ykey) && v >= cut]
    return ks[1:min(end, Int(llh_nuisance["top"]))]
end

const NUISANCE_LABEL = !(llh_nuisance isa AbstractDict) ? string(llh_nuisance) :
    laplace_mode() ? "top $(llh_nuisance["top"]), share >= $(get(llh_nuisance, "min_share", 0.1)), Laplace" :
    "top $(llh_nuisance["top"]), $(get(llh_nuisance, "points", 3)) points"

function scan_throws(xkey, ykey)
    llh_nuisance === false && return NamedTuple[], Float64[]
    if llh_nuisance isa AbstractDict
        top = top_nuisances(xkey, ykey)
        @logmsg MCMC "  averaged over $(join(top, ", ")) ($(get(llh_nuisance, "points", 3)) points each)"
        return quadrature(top, Int(get(llh_nuisance, "points", 3)))
    end
    nuis = [k for k in keys(SCAN_PRIORS) if k ∉ (xkey, ykey, :sin2_th12, :dm2_21)]
    rng = Xoshiro(hash((xkey, ykey)))
    n = Int(llh_nuisance)
    return [throw_nuisances(rng, nuis) for _ in 1:n], fill(1 / n, n)
end

"""
    laplace_cell(base, ks, u0) -> (2 log L_marg, u)

Laplace marginalisation over the nuisances `ks` at `base`: 2[max_u f(u) − ½ log det(−∇²f)],
f = log L + log π, u the nuisances in units of their prior σ from their true values. Damped
Newton steps from `u0` (the neighbouring cell's optimum), finite-difference derivatives.
"""
function laplace_cell(base, ks, u0; maxiter = 6, h = 0.02)
    μ = [true_params[k] for k in ks]
    σ = [std(SCAN_PRIORS[k]) for k in ks]
    f(u) = (p = merge(base, NamedTuple{Tuple(ks)}(Tuple(μ .+ σ .* u)));
            loglik(p) + sum(logpdf(SCAN_PRIORS[k], p[k]) for k in ks))
    n = length(ks)
    u, fu = copy(u0), f(u0)
    g, H = zeros(n), zeros(n, n)
    e(i) = (v = zeros(n); v[i] = h; v)
    for it in 1:maxiter
        for a in 1:n
            fp, fm = f(u + e(a)), f(u - e(a))
            g[a], H[a, a] = (fp - fm) / (2h), (fp - 2 * fu + fm) / h^2
            for b in a+1:n
                H[a, b] = H[b, a] = (f(u + e(a) + e(b)) - f(u + e(a) - e(b)) -
                                     f(u - e(a) + e(b)) + f(u - e(a) - e(b))) / (4h^2)
            end
        end
        it == maxiter && break
        A, λ = Symmetric(-H), 0.0
        while !isposdef(A + λ * I)
            λ = max(2λ, 1e-3)
        end
        step = (A + λ * I) \ g
        norm(step) > 2 && (step .*= 2 / norm(step))
        norm(step) < 1e-3 && break
        tries = 0
        unew = u + step; fnew = f(unew)
        while fnew < fu && tries < 5
            step ./= 2; unew = u + step; fnew = f(unew); tries += 1
        end
        fnew < fu && break
        u, fu = unew, fnew
    end
    A = Symmetric(-H)
    return 2 * (fu - 0.5 * (isposdef(A) ? logdet(A) : 0.0)), u
end

function scan2d(xkey, xvals, ykey, yvals)
    lap = laplace_mode()
    ks = lap ? top_nuisances(xkey, ykey) : Symbol[]
    lap && @logmsg MCMC "  Laplace over $(join(ks, ", "))"
    throws, weights = lap ? (NamedTuple[], Float64[]) : scan_throws(xkey, ykey)
    out = zeros(Float64, length(xvals), length(yvals))
    done = Threads.Atomic{Int}(0)
    Threads.@threads for i in eachindex(xvals)
        u = zeros(length(ks))
        for (j, y) in enumerate(yvals)
            base = merge(true_params, NamedTuple{(xkey, ykey)}((xvals[i], y)))
            if lap
                out[i, j], u = laplace_cell(base, ks, u)
            else
                out[i, j] = cell(base, throws, weights)
            end
        end
        k = Threads.atomic_add!(done, 1) + 1
        k % 5 == 0 && @logmsg MCMC "Completed $k rows out of $(length(xvals))"
    end
    ref = lap ? first(laplace_cell(true_params, ks, zeros(length(ks)))) : cell(true_params, throws, weights)
    return out, ref
end

# Fitted bins of all detectors (E_reco from the threshold on; day, every night cos z bin, cos θ)
function n_fit_bins()
    count(A, idx) = (d = ndims(A) == 1 ? 1 : 2; length(A) ÷ size(A, d) * (size(A, d) - idx + 1))
    n = 0
    for (dname, out) in detector_outputs
        li, det, obs = out.likelihood_inputs, detector_configs[dname], out.likelihood_inputs.nObserved
        li.ES_mode && (n += count(obs.ES_day, li.index_ES) + count(obs.ES_night, li.index_ES))
        if det.CC_mode && (!det.inclusive_analysis || det.semi_inclusive_analysis)
            n += count(obs.CC_day, li.index_CC) + count(obs.CC_night, li.index_CC)
        end
    end
    return n
end

const SCAN_NDOF = n_fit_bins() - length(SCAN_PRIORS)

function write_scan(suffix, xname, xlim, yname, ylim, (matrix, chi2_ref), xkey, ykey)
    open(outFile * suffix, "w") do file
        println(file, "$xname: $(xlim[1]), $(xlim[2])")
        println(file, "$yname: $(ylim[1]), $(ylim[2])")
        println(file, "truth: $(true_params[xkey]), $(true_params[ykey])")   # the scan's conditioning point
        println(file, "chi2_ref: $chi2_ref")                                  # cell value at the truth
        println(file, "ndof: $SCAN_NDOF")
        println(file, "nuisance: $NUISANCE_LABEL")
        writedlm(file, matrix, ',')
    end
end

@logmsg MCMC "LLH scan: $llhBins × $llhBins per pair, nuisance = $NUISANCE_LABEL, ndof = $SCAN_NDOF, $(Threads.nthreads()) threads"

@logmsg MCMC ("Scanning sin2th12 vs sin2th13")
write_scan("_llh_sin2th12_sin2th13.csv", "lim_th12", lim_th12, "lim_th13", lim_th13,
           scan2d(:sin2_th12, vals_12, :sin2_th13, vals_13), :sin2_th12, :sin2_th13)

@logmsg MCMC ("Scanning sin2th12 vs dm2_21")
write_scan("_llh_sin2th12_delt2m21.csv", "lim_th12", lim_th12, "lim_dm21", lim_dm21,
           scan2d(:sin2_th12, vals_12, :dm2_21, vals_dm), :sin2_th12, :dm2_21)

@logmsg MCMC ("Scanning sin2th13 vs dm2_21")
write_scan("_llh_sin2th13_delt2m21.csv", "lim_th13", lim_th13, "lim_dm21", lim_dm21,
           scan2d(:sin2_th13, vals_13, :dm2_21, vals_dm), :sin2_th13, :dm2_21)
