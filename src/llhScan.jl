"""
This script performs a likelihood scan over neutrino oscillation parameters using a grid search approach. 
It calculates the log-likelihood values for different combinations of the squared sine of mixing angles 
(`sin²θ₁₂`, `sin²θ₁₃`) and the squared mass difference (`Δm²₂₁`) and stores the results in CSV files.

Modules and Libraries:
- Utilizes various Julia packages such as `LinearAlgebra`, `Statistics`, `Distributions`, `StatsBase`, 
  `BAT`, `DensityInterface`, `IntervalSets`, and `DelimitedFiles` for mathematical operations, statistical 
  distributions, and file handling.
- Includes a setup script from `../src/setup.jl` to configure the fitting process.

Parameters:
- `nbins`: Number of bins for the grid search, defined by `llhBins`.
- `lim_th12`, `lim_th13`, `lim_dm21`: Arrays defining the parameter limits for `sin²θ₁₂`, `sin²θ₁₃`, and 
  `Δm²₂₁` respectively.

Process:
1. Initializes parameter ranges for `sin²θ₁₂`, `sin²θ₁₃`, and `Δm²₂₁` based on specified limits and number of bins.
2. Computes log-likelihood values for each parameter combination using either `likelihood_all_samples_avg` 
   or `likelihood_all_samples_ctr` based on the `fast` flag.
3. Stores the results in matrices and writes them to CSV files with appropriate headers indicating the 
   parameter limits.

Output:
- Three CSV files containing the log-likelihood scans for:
  1. `sin²θ₁₂` vs `sin²θ₁₃`
  2. `sin²θ₁₂` vs `Δm²₂₁`
  3. `sin²θ₁₃` vs `Δm²₂₁`

Note:
- The script assumes the existence of certain global variables such as `true_params`, `fast`, and `outFile`.
- Logging is set up but commented out; adjust the logging level as needed for debugging.
"""



using Logging
# DEBUGGING AND TESTING: Set the logging level to Warn to suppress Info messages
# global_logger(ConsoleLogger(stderr, Logging.Warn))

using LinearAlgebra, Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using DelimitedFiles

# Set up the fit
include("../src/setup.jl")

function get_n_freedom(index)
    if index == 1
        return n_obs_ES - (2 + length(Δ_FWHM))
    elseif index == 2
        return n_obs_CC - (2 + length(Δ_FWHM))
    else
        error("Invalid index: must be 1 or 2 (got $index)")
    end
end


nbins = llhBins

lim_th12 = [0.2, 0.4]
lim_th13 = [1e-4, 0.25]
lim_dm21 = [4e-5, 1.5e-4]

vals_12 = range(lim_th12[1], stop=lim_th12[2], length=nbins)
vals_13 = range(lim_th13[1], stop=lim_th13[2], length=nbins)
vals_dm = range(lim_dm21[1], stop=lim_dm21[2], length=nbins)
flux_8B = true_params.integrated_8B_flux



# Get Δ_FWHM for the priors of the dominant systematics (8B flux and background systematics)
Δ_FWHM = Dict{Symbol,Any}(
    :integrated_8B_flux => sqrt(2 * log(2)) * std(prior_8B_flux),
)

if !isempty(ES_bg_norms_pars)
    for (i, norm) in enumerate(ES_bg_norms_pars)
        if std(norm) > 0
            Δ_FWHM[Symbol("ES_bg_norm_$i")] = sqrt(2 * log(2)) * std(norm)
        end
    end
end

if !isempty(CC_bg_norms_pars)
    for (i, norm) in enumerate(CC_bg_norms_pars)
        if std(norm) > 0
            Δ_FWHM[Symbol("CC_bg_norm_$i")] = sqrt(2 * log(2)) * std(norm)
        end
    end
end

# get degrees of freedom:
global n_obs_CC = (Ereco_bins_CC.bin_number - index_CC) * (cosz_bins.bin_number + 1) # night bins + day bins
global n_obs_ES = (Ereco_bins_ES.bin_number - index_ES) * (cosz_bins.bin_number + 1) # night bins + day bins 


# Initialize a matrix to store the llh scans
llh_sin2th12_sin2th13 = zeros(Float64, length(vals_12), length(vals_13))
@logmsg MCMC ("Scanning sin2th12 vs sin2th13")

# println(logdensityof(likelihood_all_samples_ctr, true_params))
temp_parameters = deepcopy(true_parameters)

for (i, mode) in enumerate([ES_mode, CC_mode])
    if mode
        n_freedom = get_n_freedom(i)
        # Loop over each combination of vals_12 and vals_13
        for i in 1:length(vals_12)
            for j in 1:length(vals_13)
                temp_parameters[:sin2_th12] = vals_12[i]
                temp_parameters[:sin2_th13] = vals_13[j]

                # Loop over the systematics for a rough integral
                for (name, val) in pairs(Δ_FWHM)
                    temp_parameters[name] -= val
                    temp_params = (; temp_parameters...)
                    min_llh = logdensityof(likelihood_all_samples, temp_params)

                    temp_parameters[name] += 2 * val
                    temp_params = (; temp_parameters...)
                    max_llh = logdensityof(likelihood_all_samples, temp_params)

                    temp_parameters[name] -= val
                    temp_params = (; temp_parameters...)
                    mid_llh = logdensityof(likelihood_all_samples, temp_params)

                    # Call the function with the current values + (normalised) integral accross the relevant dimension and store the result
                    llh_sin2th12_sin2th13[i, j] += ((min_llh + max_llh) / 2 + mid_llh) / n_freedom
                end
            end
            if i % 5 == 0
                @logmsg MCMC "Completed $i rows out of $(length(vals_12))"
            end
        end
    end
end

# Prepare the header with axis limits
header = [
    "lim_th12: $(lim_th12[1]), $(lim_th12[2])"
    "lim_th13: $(lim_th13[1]), $(lim_th13[2])"
]

# Open the file and write the header and matrix
open(outFile * "_llh_sin2th12_sin2th13.csv", "w") do file
    # Write the header
    for line in header
        println(file, line)
    end
    # Write the matrix
    writedlm(file, llh_sin2th12_sin2th13, ',')
end

println("")
@logmsg MCMC ("Scanning sin2th12 vs dm2_21")

# Initialize a matrix to store the llh scans for sin2_th12 and dm2_21
llh_sin2th12_dm2_21 = zeros(Float64, length(vals_12), length(vals_13))
temp_parameters = deepcopy(true_parameters)

for (i, mode) in enumerate([ES_mode, CC_mode])
    if mode
        n_freedom = get_n_freedom(i)
        # Loop over each combination of vals_12 and vals_dm
        for i in 1:length(vals_12)
            for j in 1:length(vals_dm)
                temp_parameters[:sin2_th12] = vals_12[i]
                temp_parameters[:dm2_21] = vals_dm[j]
                # Loop over the systematics for a rough integral
                for (name, val) in pairs(Δ_FWHM)
                    temp_parameters[name] -= val
                    temp_params = (; temp_parameters...)
                    min_llh = logdensityof(likelihood_all_samples, temp_params)

                    temp_parameters[name] += 2 * val
                    temp_params = (; temp_parameters...)
                    max_llh = logdensityof(likelihood_all_samples, temp_params)

                    temp_parameters[name] -= val
                    temp_params = (; temp_parameters...)
                    mid_llh = logdensityof(likelihood_all_samples, temp_params)

                    # Call the function with the current values + (normalised) integral accross the relevant dimension and store the result
                    llh_sin2th12_dm2_21[i, j] += ((min_llh + max_llh) / 2 + mid_llh ) / n_freedom 
                end
            end
            if i % 5 == 0
                @logmsg MCMC ("Completed $i rows out of $(length(vals_12))")
            end
        end
    end
end


# Prepare the header with axis limits for sin2_th12 and dm2_21
header_dm = [
    "lim_th12: $(lim_th12[1]), $(lim_th12[2])"
    "lim_dm21: $(lim_dm21[1]), $(lim_dm21[2])"
]

# Open the file and write the header and matrix for sin2_th12 and dm2_21
open(outFile * "_llh_sin2th12_delt2m21.csv", "w") do file
    # Write the header
    for line in header_dm
        println(file, line)
    end
    # Write the matrix
    writedlm(file, llh_sin2th12_dm2_21, ',')
end

println("")
@logmsg MCMC ("Scanning sin2th13 vs dm2_21")

# Initialize a matrix to store the llh scans for sin2_th13 and dm2_21
llh_sin2th13_dm2_21 = zeros(Float64, length(vals_12), length(vals_13))

temp_parameters = deepcopy(true_parameters)

for (i, mode) in enumerate([ES_mode, CC_mode])
    if mode
        n_freedom = get_n_freedom(i)
        for i in 1:length(vals_13)
            for j in 1:length(vals_dm)
                temp_parameters[:sin2_th13] = vals_13[i]
                temp_parameters[:dm2_21] = vals_dm[j]
                # Loop over the systematics for a rough integral
                for (name, val) in pairs(Δ_FWHM)
                    temp_parameters[name] -= val
                    temp_params = (; temp_parameters...)
                    min_llh = logdensityof(likelihood_all_samples, temp_params)

                    temp_parameters[name] += 2 * val
                    temp_params = (; temp_parameters...)
                    max_llh = logdensityof(likelihood_all_samples, temp_params)

                    temp_parameters[name] -= val
                    temp_params = (; temp_parameters...)
                    mid_llh = logdensityof(likelihood_all_samples, temp_params)

                    # Call the function with the current values + (normalised) integral accross the relevant dimension and store the result
                    llh_sin2th13_dm2_21[i, j] += ((min_llh + max_llh) / 2 + mid_llh) / n_freedom
                end
            end
            if i % 5 == 0
                @logmsg MCMC ("Completed $i rows out of $(length(vals_13))")
            end
        end
    end
end


# Prepare the header with axis limits for sin2_th13 and dm2_21
header_dm = [
    "lim_th13: $(lim_th13[1]), $(lim_th13[2])"
    "lim_dm21: $(lim_dm21[1]), $(lim_dm21[2])"
]

# Open the file and write the header and matrix for sin2_th13 and dm2_21
open(outFile * "_llh_sin2th13_delt2m21.csv", "w") do file
    # Write the header
    for line in header_dm
        println(file, line)
    end
    # Write the matrix
    writedlm(file, llh_sin2th13_dm2_21, ',')
end