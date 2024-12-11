using JLD2
using CSV
using DataFrames
using Plots

include("../src/objects.jl")

# Check if the solar model file exists and read the datasets
if isfile(SolarModelFile)
    solarModel = jldopen(SolarModelFile, "r") do file
        # Load the necessary datasets from the file
        radii = file["radii"]
        prodFractionBoron = file["prodFractionBoron"]
        prodFractionHep = file["prodFractionHep"]
        n_e = file["n_e"]
        # Calculate weighted averages
        avgNeBoron = sum(prodFractionBoron .* n_e) / sum(prodFractionBoron)
        avgNeHep = sum(prodFractionHep .* n_e) / sum(prodFractionHep)

        # Return a named tuple with both prodFractionBoron and prodFractionHep
        return (radii=radii, prodFractionBoron=prodFractionBoron, prodFractionHep=prodFractionHep, n_e=n_e, avgNeBoron=avgNeBoron, avgNeHep=avgNeHep)
    end

else
    error("File not found: $SolarModelFile")
end

#  Check if the solar flux file exists and read the datasets
if isfile(flux_file_path)
    energies, flux8B, fluxHep = jldopen(flux_file_path, "r") do file
        energies = file["energies"]
        flux8B = file["flux8B"]
        fluxHep = file["fluxHep"]

        return energies, flux8B, fluxHep
    end
else
    error("File not found: $flux_file_path")
end


# Load unoscillated MC
include("../src/unoscillatedSample.jl")
include("../src/propagateSample.jl")
include("../src/response.jl")

# Set up backgrounds
df_ES_nue = CSV.File(ES_nue_filepath_BG) |> DataFrame
df_ES_nuother = CSV.File(ES_nuother_filepath_BG) |> DataFrame
df_CC = CSV.File(CC_filepath_BG) |> DataFrame

ES_nue_bg, ES_nue_bg_etrue = create_histogram(df_ES_nue.Ereco, bins)
ES_nuother_bg, ES_nuother_bg_etrue = create_histogram(df_ES_nuother.Ereco, bins)
CC_bg, CC_bg_etrue = create_histogram(df_CC.Ereco .* 1e-6, bins)

# println(df_CC.Ereco .* 1e-6)
# println(CC_bg)
# println(sum(CC_bg.*200))

# Initialise parameters and set oservations to Asimov parameter values (PDG)
true_params = (sin2_th12=sin2_th12_true,
    sin2_th13=sin2_th13_true,
    dm2_21=dm2_21_true)

# Propagate Asimov point to generate Asimov event rates
measuredRate_ES_nue, measuredRate_ES_nuother, measuredRate_CC = propagateSamplesAvg(unoscillatedSample, responseMatrices, true_params, solarModel, bin_edges, CC_bg)

index_above_threshold = findfirst(x -> x > E_threshold.ES, energies_GeV)

backgrounds = (ES=(nue=ES_nue_bg, nuother=ES_nuother_bg), CC=CC_bg)
ereco_data = (ES_nue=measuredRate_ES_nue, ES_nuother=measuredRate_ES_nuother, CC=measuredRate_CC)
ereco_data_mergedES = (ES=measuredRate_ES_nue .+ measuredRate_ES_nuother, CC=measuredRate_CC)

# Check if index_above_threshold is not nothing
if index_above_threshold !== nothing
    # Sum the elements from index_above_threshold to the end
    total_es_data_above_threshold = sum(ereco_data.ES_nue[index_above_threshold:end] .+ ereco_data.ES_nuother[index_above_threshold:end])
    es_nue_data_above_threshold = sum(ereco_data.ES_nue[index_above_threshold:end])
    es_nuother_data_above_threshold = sum(ereco_data.ES_nuother[index_above_threshold:end])

    println("Total number of ES data above threshold: ", total_es_data_above_threshold)
    println("Number of ES nue data above threshold: ", es_nue_data_above_threshold)
    println("Number of ES nuother data above threshold: ", es_nuother_data_above_threshold)
    println("Total number of CC data: ", sum(ereco_data.CC[index_above_threshold:end]))
else
    println("No energies above the threshold.")
end

# load likelihood
include("../src/statsLikelihood.jl")