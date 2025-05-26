######################################
######## IMPORTS AND INCLUDES ########
######################################

using JLD2
using CSV
using DataFrames
using Plots

include("../src/objects.jl")

#############################
######## LOAD INPUTS ########
#############################

# Load solar production region and flux shapes
include("../src/solarModel.jl")

# Load earth model and neutrino paths
include("../src/earthProfile.jl")
include("../src/oscillations/makePaths.jl")

# Load exposure to cosz in the earth
include("../src/exposure.jl")

# Load unoscillated MC
include("../src/unoscillatedSample.jl")
include("../src/propagateSample.jl")
include("../src/response.jl")
include("../src/backgrounds.jl")
include("../src/efficiency.jl")

# Initialise parameters and set oservations to Asimov parameter values (PDG)
true_parameters = Dict{Symbol, Any}(
    # mixing parameters
    :sin2_th12 => sin2_th12_true,
    :sin2_th13 => sin2_th13_true,
    :dm2_21   => dm2_21_true,

    # HEP discovery
    :integrated_HEP_flux => integrated_HEP_flux_true,

    # systematic parameters
    :integrated_8B_flux => integrated_8B_flux_true,

    # dependent variables
    :ES_asymmetry => 0,
    :CC_asymmetry => 0
)

# Conditionally add nuisance parameters
if earthUncertainty
    true_parameters[:earth_norm] = earth_normalisation_true
end

if !isempty(ES_bg_norms_true)
    for (i, norm) in enumerate(ES_bg_norms_true)
        true_parameters[Symbol("ES_bg_norm_$i")] = norm
    end
end
  
if !isempty(CC_bg_norms_true)
    for (i, norm) in enumerate(CC_bg_norms_true)
        true_parameters[Symbol("CC_bg_norm_$i")] = norm
    end
end


# create tuple
true_params = (; true_parameters...)


# Generate neutrino paths through the earth
global earth_paths = [make_potential_for_integrand(z, earth) for z in cosz_calc]

# Get average layer densities for fast calculations
global earth_lookup = get_avg_densities(earth_paths)

###################### TEST ######################
# include("../src/oscillations/numPropagation.jl")
##################################################

##################################
######## CREATE FAKE DATA ########
##################################
backgrounds = (ES=ES_bg, CC=CC_bg)

# Propagate Asimov point to generate Asimov event rates
measuredRate_ES_nue_day, measuredRate_ES_nuother_day, measuredRate_CC_day, measuredRate_ES_nue_night, measuredRate_ES_nuother_night, measuredRate_CC_night, BG_ES_tot_true, BG_CC_tot_true = propagateSamplesCtr(unoscillatedSample, responseMatrices, true_params, solarModel, bin_edges, backgrounds)

# Find the first index where energy is greater than Emin
global index_ES = findfirst(x -> x > E_threshold.ES, Ereco_bins_ES_extended.bins)
global index_CC = findfirst(x -> x > E_threshold.CC, Ereco_bins_CC_extended.bins)

# Check if indices were found 
if isnothing(index_ES)
    @logmsg Setup ("No energy bins above the threshold for ES.")
end

if isnothing(index_CC)
    @logmsg Setup ("No energy bins above the threshold for CC.")
end

### -- Calculate D-N asymmetry -- ###
CC_bg_aboveThreshold = sum(BG_CC_tot_true[index_CC:end])
ES_bg_aboveThreshold = sum(BG_ES_tot_true[index_ES:end])

CC_Ntot = sum(@view measuredRate_CC_night[:, index_CC:end]) - 0.5 * CC_bg_aboveThreshold
CC_Dtot = sum(measuredRate_CC_day[index_CC:end]) - 0.5 * CC_bg_aboveThreshold

ES_Ntot = sum(@view measuredRate_ES_nue_night[:, index_ES:end]) + sum(@view measuredRate_ES_nuother_night[:, index_ES:end]) - 0.5 * ES_bg_aboveThreshold
ES_Dtot = sum(measuredRate_ES_nue_day[index_ES:end]) + sum(measuredRate_ES_nuother_day[index_ES:end]) - 0.5 * ES_bg_aboveThreshold

asymm_CC = 2 * (CC_Dtot - CC_Ntot) / (CC_Dtot + CC_Ntot)
asymm_ES = 2 * (ES_Dtot - ES_Ntot) / (ES_Dtot + ES_Ntot)

# save Asimov asymmetries
true_parameters[:ES_asymmetry] = asymm_ES
true_parameters[:CC_asymmetry] = asymm_CC


# Group for feeding to likelihood
ereco_data = (
    ES_nue_day=measuredRate_ES_nue_day,
    ES_nuother_day=measuredRate_ES_nuother_day,
    CC_day=measuredRate_CC_day,
    
    ES_nue_night=measuredRate_ES_nue_night,
    ES_nuother_night=measuredRate_ES_nuother_night,
    CC_night=measuredRate_CC_night
    )


ereco_data_mergedES = (
    ES_day=measuredRate_ES_nue_day .+ measuredRate_ES_nuother_day,
    CC_day=measuredRate_CC_day,
    
    ES_night=measuredRate_ES_nue_night .+ measuredRate_ES_nuother_night,
    CC_night=measuredRate_CC_night
    )

# Sum over the second dimension of CC_night
CC_night_summed = sum(ereco_data_mergedES.CC_night, dims=1)

# Add the summed CC_night to CC_day
CC_combined = ereco_data_mergedES.CC_day .+ vec(CC_night_summed)


###################
#### DEBUGGING ####
#=

using Plots


# Get dimensions of the matrix
nrows, ncols = size(ereco_data_mergedES.CC_night)

# Calculate the adjusted CC_night by subtracting CC_day from each row.
#### THIS DOESN'T WORK BECAUSE I AM NOT CALCULATING THE DAYTIME exposure
adjusted_CC_night = 2 .* (ereco_data_mergedES.CC_night .- ereco_data_mergedES.CC_day ./ 40) ./ (ereco_data_mergedES.CC_night .+ ereco_data_mergedES.CC_day ./ 40)
adjusted_CC_night = ereco_data_mergedES.CC_night
# Define transformation functions for tick labels:
# Define transformation functions for tick labels:
# x automatic coordinates (1 to ncols) should appear as 5 to 30.
xlab(x) = 5 + (30 - 5) * (x - 1) / (ncols - 1)
# y automatic coordinates (1 to nrows) should appear as -1 to 0.
ylab(y) = -1 + (0 - (-1)) * (y - 1) / (nrows - 1)

# Choose how many ticks you would like (for example, 5 ticks along each axis)
num_xticks = 5
num_yticks = 5

# Compute tick positions and corresponding labels
xtick_positions = range(1, ncols; length = num_xticks)
xtick_labels = string.(round.(xlab.(xtick_positions), digits=1))

ytick_positions = range(1, nrows; length = num_yticks)
ytick_labels = string.(round.(ylab.(ytick_positions), digits=1))

# Ensure the data is strictly positive for logarithmic scaling
adjusted_CC_night_pos = copy(adjusted_CC_night)
# Replace non-positive values with a small fraction of the smallest positive value
min_positive = minimum(adjusted_CC_night_pos[adjusted_CC_night_pos .> 0])
adjusted_CC_night_pos[adjusted_CC_night_pos .<= 0] .= min_positive * 1e-3

# Determine color limits based on the adjusted data
clims = (minimum(adjusted_CC_night_pos), maximum(adjusted_CC_night_pos))
# Manually compute logarithmic tick marks for the colorbar (5 ticks in this example)
logticks = 10 .^ range(log10(clims[1]), log10(clims[2]), length=5)

log_data = log10.(adjusted_CC_night_pos)
diff_data = adjusted_CC_night_pos .- ereco_data_mergedES.CC_day' .* exposure_weights

# Set color limits for log_data (should now be linear in log scale)
clims_log = (minimum(log_data), maximum(log_data))

using ColorSchemes

parulas = ColorScheme([RGB(0.2422, 0.1504, 0.6603),
        RGB(0.2504, 0.1650, 0.7076),
        RGB(0.2578, 0.1818, 0.7511),
        RGB(0.2647, 0.1978, 0.7952),
        RGB(0.2706, 0.2147, 0.8364),
        RGB(0.2751, 0.2342, 0.8710),
        RGB(0.2783, 0.2559, 0.8991),
        RGB(0.2803, 0.2782, 0.9221),
        RGB(0.2813, 0.3006, 0.9414),
        RGB(0.2810, 0.3228, 0.9579),
        RGB(0.2795, 0.3447, 0.9717),
        RGB(0.2760, 0.3667, 0.9829),
        RGB(0.2699, 0.3892, 0.9906),
        RGB(0.2602, 0.4123, 0.9952),
        RGB(0.2440, 0.4358, 0.9988),
        RGB(0.2206, 0.4603, 0.9973),
        RGB(0.1963, 0.4847, 0.9892),
        RGB(0.1834, 0.5074, 0.9798),
        RGB(0.1786, 0.5289, 0.9682),
        RGB(0.1764, 0.5499, 0.9520),
        RGB(0.1687, 0.5703, 0.9359),
        RGB(0.1540, 0.5902, 0.92, integrated_8B_flux=flux_8B)18),
        RGB(0.1460, 0.6091, 0.9079),
        RGB(0.1380, 0.6276, 0.8973),
        RGB(0.1248, 0.6459, 0.8883),
        RGB(0.1113, 0.6635, 0.8763),
        RGB(0.0952, 0.6798, 0.8598),
        RGB(0.0689, 0.6948, 0.8394),
        RGB(0.0297, 0.7082, 0.8163),
        RGB(0.0036, 0.7203, 0.7917),
        RGB(0.0067, 0.7312, 0.7660),
        RGB(0.0433, 0.7411, 0.7394),
        RGB(0.0964, 0.7500, 0.7120),
        RGB(0.1408, 0.7584, 0.6842),
        RGB(0.1717, 0.7670, 0.6554),
        RGB(0.1938, 0.7758, 0.6251),
        RGB(0.2161, 0.7843, 0.5923),
        RGB(0.2470, 0.7918, 0.5567),
        RGB(0.2906, 0.7973, 0.5188),
        RGB(0.3406, 0.8008, 0.4789),
        RGB(0.3909, 0.8029, 0.4354),
        RGB(0.4456, 0.8024, 0.3909),
        RGB(0.5044, 0.7993, 0.3480),
        RGB(0.5616, 0.7942, 0.3045),
        RGB(0.6174, 0.7876, 0.2612),
        RGB(0.6720, 0.7793, 0.2227),
        RGB(0.7242, 0.7698, 0.1910),
        RGB(0.7738, 0.7598, 0.1646),
        RGB(0.8203, 0.7498, 0.1535),
        RGB(0.8634, 0.7406, 0.1596),
        RGB(0.9035, 0.7330, 0.1774),
        RGB(0.9393, 0.7288, 0.2100),
        RGB(0.9728, 0.7298, 0.2394),
        RGB(0.9956, 0.7434, 0.2371),
        RGB(0.9970, 0.7659, 0.2199),
        RGB(0.9952, 0.7893, 0.2028),
        RGB(0.9892, 0.8136, 0.1885),
        RGB(0.9786, 0.8386, 0.1766),
        RGB(0.9676, 0.8639, 0.1643),
        RGB(0.9610, 0.8890, 0.1537),
        RGB(0.9597, 0.9135, 0.1423),
        RGB(0.9628, 0.9373, 0.1265),
        RGB(0.9691, 0.9606, 0.1064),
        RGB(0.9769, 0.9839, 0.0805)],
    "Parula",
    "From MATLAB")

heatmap(
    diff_data,
    color = :inferno, #cgrad(parulas),
    clim = (0, 50),
    # colorbar_title = "log10(counts)",
    title = "Night - Day",
    xlabel = "Energy",
    ylabel = "cos(z)",
    xticks = (collect(xtick_positions), xtick_labels),
    yticks = (collect(ytick_positions), ytick_labels)
)


# Optionally save the plot to a file
savefig("cc_night_asymm_linear_heatmap.png")
# Print the result to the console
using Printf
formatted_output = join([@sprintf("%.2f", x) for x in CC_combined], ", ")
println(formatted_output)
println(" ")
sleep(100)
exit()
=#
###################
###################

# Check if index_above_threshold is not nothing

# Sum the elements from index_above_threshold to the end
es_nue_data_above_threshold = 2 * sum(ereco_data.ES_nue_day[index_ES:end])
es_nuother_data_above_threshold = 2 * sum(ereco_data.ES_nuother_day[index_ES:end])
total_es_data_above_threshold = es_nue_data_above_threshold .+ es_nuother_data_above_threshold

@logmsg Setup ("Total number of ES data above threshold: ", total_es_data_above_threshold)
@logmsg Setup ("Number of ES nue data above threshold: ", es_nue_data_above_threshold)
@logmsg Setup ("Number of ES nuother data above threshold: ", es_nuother_data_above_threshold)
@logmsg Setup ("Total number of CC data above threshold: ", sum(CC_combined[index_CC:end]))

# load likelihood
include("../src/statsLikelihood.jl")