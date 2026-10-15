
using Logging
# DEBUGGING AND TESTING: Set the logging level to Warn to suppress Info messages
# global_logger(ConsoleLogger(stderr, Logging.Warn))

using LinearAlgebra, Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using DelimitedFiles  # For CSV output

# Set up the analysis environment
include(joinpath(@__DIR__, "setup.jl"))

#=
2D likelihood scans of the oscillation parameters (RunMode: LLH).

Each cell holds 2·log L of all active detectors and channels (total_llh), with every other
parameter fixed at its Asimov true value: a conditional scan, not a marginalised or profiled
one, so its contours are narrower than the posterior's. Plot 2·log L − max: the
2-parameter 1/2/3σ contours are at −2.30, −6.18, −11.83 (utils/plotLLH.py).

The earlier version summed ad-hoc ±FWHM shifts of selected systematics divided by a
degrees-of-freedom count, and evaluated the full likelihood once per active channel, which
counted it twice when both ES and CC were on.
=#

nbins = llhBins  # grid points per axis

# Parameter ranges for scanning
lim_th12 = [0.2, 0.4]      # sin²θ₁₂ range
lim_th13 = [1e-4, 0.25]    # sin²θ₁₃ range
lim_dm21 = [4e-5, 1.5e-4]  # Δm²₂₁ range (eV²)

vals_12 = range(lim_th12[1], stop=lim_th12[2], length=nbins)
vals_13 = range(lim_th13[1], stop=lim_th13[2], length=nbins)
vals_dm = range(lim_dm21[1], stop=lim_dm21[2], length=nbins)

function scan2d(xkey, xvals, ykey, yvals)
    out = zeros(Float64, length(xvals), length(yvals))
    for (i, x) in enumerate(xvals)
        for (j, y) in enumerate(yvals)
            p = merge(true_params, NamedTuple{(xkey, ykey)}((x, y)))
            out[i, j] = 2 * logdensityof(likelihood_all_samples, p)
        end
        i % 5 == 0 && @logmsg MCMC "Completed $i rows out of $(length(xvals))"
    end
    return out
end

function write_scan(suffix, xname, xlim, yname, ylim, matrix)
    open(outFile * suffix, "w") do file
        println(file, "$xname: $(xlim[1]), $(xlim[2])")
        println(file, "$yname: $(ylim[1]), $(ylim[2])")
        writedlm(file, matrix, ',')
    end
end

@logmsg MCMC ("Scanning sin2th12 vs sin2th13")
write_scan("_llh_sin2th12_sin2th13.csv", "lim_th12", lim_th12, "lim_th13", lim_th13,
           scan2d(:sin2_th12, vals_12, :sin2_th13, vals_13))

println("")
@logmsg MCMC ("Scanning sin2th12 vs dm2_21")
write_scan("_llh_sin2th12_delt2m21.csv", "lim_th12", lim_th12, "lim_dm21", lim_dm21,
           scan2d(:sin2_th12, vals_12, :dm2_21, vals_dm))

println("")
@logmsg MCMC ("Scanning sin2th13 vs dm2_21")
write_scan("_llh_sin2th13_delt2m21.csv", "lim_th13", lim_th13, "lim_dm21", lim_dm21,
           scan2d(:sin2_th13, vals_13, :dm2_21, vals_dm))
