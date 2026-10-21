using Logging
using LinearAlgebra, Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using DelimitedFiles, Random

include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "priors.jl"))

#=
2D likelihood scans of the oscillation parameters (RunMode: LLH).

`nuisance: false`: each cell is 2 log L with every other parameter at its Asimov true value (a
conditional scan). `nuisance: N`: each cell is 2 log of the likelihood averaged over N throws of
the nuisances from their priors, the same throws for every cell: a Monte Carlo marginalisation,
a quick picture of the posterior. Thrown: sin²θ₁₃ when not scanned, fluxes, CC cross section,
backgrounds, Earth layers (jointly, with their covariance); the other solar oscillation parameter
stays at its true value. Plot χ² = chi2_ref − cell, chi2_ref the value at the truth; the
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

# 2 log of the likelihood at `base`, averaged over `throws` (none: the conditional value)
function cell(base, throws)
    isempty(throws) && return 2 * loglik(base)
    v = [loglik(merge(base, t)) for t in throws]
    m = maximum(v)
    return 2 * (m + log(sum(exp.(v .- m)) / length(v)))
end

function scan2d(xkey, xvals, ykey, yvals)
    nuis = [k for k in keys(SCAN_PRIORS) if k ∉ (xkey, ykey, :sin2_th12, :dm2_21)]
    n_throws = llh_nuisance === false ? 0 : Int(llh_nuisance)
    rng = Xoshiro(hash((xkey, ykey)))
    throws = [throw_nuisances(rng, nuis) for _ in 1:n_throws]
    out = zeros(Float64, length(xvals), length(yvals))
    done = Threads.Atomic{Int}(0)
    Threads.@threads for i in eachindex(xvals)
        for (j, y) in enumerate(yvals)
            out[i, j] = cell(merge(true_params, NamedTuple{(xkey, ykey)}((xvals[i], y))), throws)
        end
        k = Threads.atomic_add!(done, 1) + 1
        k % 5 == 0 && @logmsg MCMC "Completed $k rows out of $(length(xvals))"
    end
    return out, cell(true_params, throws)
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
        println(file, "nuisance: $llh_nuisance")
        writedlm(file, matrix, ',')
    end
end

@logmsg MCMC "LLH scan: $llhBins × $llhBins per pair, nuisance = $llh_nuisance, ndof = $SCAN_NDOF, $(Threads.nthreads()) threads"

@logmsg MCMC ("Scanning sin2th12 vs sin2th13")
write_scan("_llh_sin2th12_sin2th13.csv", "lim_th12", lim_th12, "lim_th13", lim_th13,
           scan2d(:sin2_th12, vals_12, :sin2_th13, vals_13), :sin2_th12, :sin2_th13)

@logmsg MCMC ("Scanning sin2th12 vs dm2_21")
write_scan("_llh_sin2th12_delt2m21.csv", "lim_th12", lim_th12, "lim_dm21", lim_dm21,
           scan2d(:sin2_th12, vals_12, :dm2_21, vals_dm), :sin2_th12, :dm2_21)

@logmsg MCMC ("Scanning sin2th13 vs dm2_21")
write_scan("_llh_sin2th13_delt2m21.csv", "lim_th13", lim_th13, "lim_dm21", lim_dm21,
           scan2d(:sin2_th13, vals_13, :dm2_21, vals_dm), :sin2_th13, :dm2_21)
