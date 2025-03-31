######################################
######## IMPORTS AND INCLUDES ########
######################################

using JLD2
using CSV
using DataFrames
using Plots

include("../src/objects.jl")
include("../src/oscillations/earthPropagation.jl")

#############################
######## LOAD INPUTS ########
#############################

# Load solar production region and flux shapes
include("../src/solarModel.jl")

# Load earth model
include("../src/earthProfile.jl")

# Load exposure to cosz in the earth
include("../src/exposure.jl")

# Load unoscillated MC
include("../src/unoscillatedSample.jl")
include("../src/propagateSample.jl")
include("../src/backgrounds.jl")

# Initialise parameters and set oservations to Asimov parameter values (PDG)
true_parameters = Dict{Symbol, Any}(
    # miximg parameters
    :sin2_th12 => sin2_th12_true,
    :sin2_th13 => sin2_th13_true,
    :dm2_21   => dm2_21_true,

    # systematic parameters
    :integrated_8B_flux => integrated_8B_flux_true
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

##################################
######## CREATE FAKE DATA ########
##################################
backgrounds = (ES=ES_bg, CC=CC_bg)

# Propagate Asimov point to generate Asimov event rates
measuredRate_ES_nue_day, measuredRate_ES_nuother_day, measuredRate_CC_day, measuredRate_ES_nue_night, measuredRate_ES_nuother_night, measuredRate_CC_night = propagateSamplesCtr(unoscillatedSample, responseMatrices, true_params, solarModel, bin_edges, backgrounds)

index_above_threshold = findfirst(x -> x > E_threshold.ES, energies_GeV)

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
using Plots

# Get dimensions of the matrix
nrows, ncols = size(ereco_data_mergedES.CC_night)

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

heatmap(ereco_data_mergedES.CC_night,
    color=:viridis,                      # Color scheme (adjust as needed)
    colorbar_title="counts",             # Colorbar label for raw counts
    title="CC_night: Linear Color Scale",
    xlabel="Column Index",
    ylabel="Row Index",
    xticks = (collect(xtick_positions), xtick_labels),
    yticks = (collect(ytick_positions), ytick_labels)
)


# Optionally save the plot to a file
savefig("cc_night_linear_heatmap.png")
# Print the result to the console
using Printf
formatted_output = join([@sprintf("%.2f", x) for x in CC_combined], ", ")
println(formatted_output)
println(" ")
###################
###################

# Check if index_above_threshold is not nothing
if index_above_threshold !== nothing
    # Sum the elements from index_above_threshold to the end
    total_es_data_above_threshold = 2 * sum(ereco_data.ES_nue_day[index_above_threshold:end] .+ ereco_data.ES_nuother_day[index_above_threshold:end])
    es_nue_data_above_threshold = 2 * sum(ereco_data.ES_nue_day[index_above_threshold:end])
    es_nuother_data_above_threshold = 2 * sum(ereco_data.ES_nuother_day[index_above_threshold:end])

    @logmsg Setup ("Total number of ES data above threshold: ", total_es_data_above_threshold)
    @logmsg Setup ("Number of ES nue data above threshold: ", es_nue_data_above_threshold)
    @logmsg Setup ("Number of ES nuother data above threshold: ", es_nuother_data_above_threshold)
    @logmsg Setup ("Total number of CC data above threshold: ", sum(CC_combined[index_above_threshold:end]))
else
    @logmsg Setup ("No energies above the threshold.")
end
println(" ")

# load likelihood
include("../src/statsLikelihood.jl")