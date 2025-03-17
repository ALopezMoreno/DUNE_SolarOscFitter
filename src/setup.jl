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
include("../src/response.jl")

# Set up backgrounds
df_ES_nue = CSV.File(ES_nue_filepath_BG) |> DataFrame
df_ES_nuother = CSV.File(ES_nuother_filepath_BG) |> DataFrame
df_CC = CSV.File(CC_filepath_BG) |> DataFrame

# Get bin heights and central bin energies for the backgrounds
ES_nue_bg, ES_nue_bg_etrue = create_histogram(df_ES_nue.Ereco, Ereco_bins_ES_extended)
ES_nuother_bg, ES_nuother_bg_etrue = create_histogram(df_ES_nuother.Ereco, Ereco_bins_ES_extended)
CC_bg, CC_bg_etrue = create_histogram(df_CC.Ereco .* 1e-6, Ereco_bins_CC_extended)

# Initialise parameters and set oservations to Asimov parameter values (PDG)
true_params = (sin2_th12=sin2_th12_true,
    sin2_th13=sin2_th13_true,
    dm2_21=dm2_21_true,
    integrated_8B_flux=integrated_8B_flux_true)

# Generate neutrino paths through the earth
global earth_paths = [make_potential_for_integrand(z, earth) for z in cosz_calc]

##################################
######## CREATE FAKE DATA ########
##################################


# Propagate Asimov point to generate Asimov event rates
measuredRate_ES_nue_day, measuredRate_ES_nuother_day, measuredRate_CC_day, measuredRate_ES_nue_night, measuredRate_ES_nuother_night, measuredRate_CC_night = propagateSamplesCtr(unoscillatedSample, responseMatrices, true_params, solarModel, bin_edges, CC_bg)

# Generate covariance matrices
include("../src/earthUncertainty.jl")

index_above_threshold = findfirst(x -> x > E_threshold.ES, energies_GeV)

# Group for feeding to likelihood
backgrounds = (ES=(nue=ES_nue_bg, nuother=ES_nuother_bg), CC=CC_bg)
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

# Check if index_above_threshold is not nothing
if index_above_threshold !== nothing
    # Sum the elements from index_above_threshold to the end
    total_es_data_above_threshold = 2 * sum(ereco_data.ES_nue_day[index_above_threshold:end] .+ ereco_data.ES_nuother_day[index_above_threshold:end])
    es_nue_data_above_threshold = 2 * sum(ereco_data.ES_nue_day[index_above_threshold:end])
    es_nuother_data_above_threshold = 2 * sum(ereco_data.ES_nuother_day[index_above_threshold:end])

    @logmsg Setup ("Total number of ES data above threshold: ", total_es_data_above_threshold)
    @logmsg Setup ("Number of ES nue data above threshold: ", es_nue_data_above_threshold)
    @logmsg Setup ("Number of ES nuother data above threshold: ", es_nuother_data_above_threshold)
    @logmsg Setup ("Total number of CC data above threshold: ", 2 * sum(ereco_data.CC_day[index_above_threshold:end]))
else
    @logmsg Setup ("No energies above the threshold.")
end
println(" ")

# load likelihood
include("../src/statsLikelihood.jl")