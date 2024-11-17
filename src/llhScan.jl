using Logging

# DEBUGGING AND TESTING: Set the logging level to Warn to suppress Info messages
# global_logger(ConsoleLogger(stderr, Logging.Warn))

using LinearAlgebra, Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using DelimitedFiles

include("../src/oscCalc.jl")
include("../src/reweighting.jl")
include("../src/statsLikelihood.jl")

true_vals = (sin2_th12=0.307, sin2_th13=0.022,dm2_21=7.53e-5)

nbins = 40

lim_th12 = [0.307-0.15, 0.307+0.15]
lim_th13 = [0.001, .1]
lim_dm21 = [1e-8, 1.5e-4]

vals_12 = range(lim_th12[1], stop=lim_th12[2], length=nbins)
vals_13 = range(lim_th13[1], stop=lim_th13[2], length=nbins)
vals_dm = range(lim_dm21[1], stop=lim_dm21[2], length=nbins)

# Initialize a matrix to store the llh scans
llh_sin2th12_sin2th13 = Matrix{Float64}(undef, length(vals_12), length(vals_13))

println(logdensityof(likelihood, true_vals))

# Loop over each combination of vals_12 and vals_13
for i in 1:length(vals_12)
    for j in 1:length(vals_13)
        temp_params = (sin2_th12=vals_12[i], sin2_th13=vals_13[j], dm2_21=true_vals.dm2_21)
        # Call the function with the current values and store the result
        llh_sin2th12_sin2th13[i, j] = logdensityof(likelihood, temp_params)
    end
end

# Prepare the header with axis limits
header = [
    "lim_th12: $(lim_th12[1]), $(lim_th12[2])"
    "lim_th13: $(lim_th13[1]), $(lim_th13[2])"
]

# Open the file and write the header and matrix
open("outputs/llh_sin2th12_sin2th13.csv", "w") do file
    # Write the header
    for line in header
        println(file, line)
    end
    # Write the matrix
    writedlm(file, llh_sin2th12_sin2th13, ',')
end

# Initialize a matrix to store the llh scans for sin2_th12 and dm2_21
llh_sin2th12_dm2_21 = Matrix{Float64}(undef, length(vals_12), length(vals_dm))

# Loop over each combination of vals_12 and vals_dm
for i in 1:length(vals_12)
    for j in 1:length(vals_dm)
        temp_params = (sin2_th12=vals_12[i], sin2_th13=true_vals.sin2_th13, dm2_21=vals_dm[j])
        # Call the function with the current values and store the result
        llh_sin2th12_dm2_21[i, j] = logdensityof(likelihood, temp_params)
    end
end

# Prepare the header with axis limits for sin2_th12 and dm2_21
header_dm = [
    "lim_th12: $(lim_th12[1]), $(lim_th12[2])"
    "lim_dm21: $(lim_dm21[1]), $(lim_dm21[2])"
]

# Open the file and write the header and matrix for sin2_th12 and dm2_21
open("outputs/llh_sin2th12_delt2m21.csv", "w") do file
    # Write the header
    for line in header_dm
        println(file, line)
    end
    # Write the matrix
    writedlm(file, llh_sin2th12_dm2_21, ',')
end