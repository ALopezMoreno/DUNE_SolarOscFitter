# Calculate oscillated event rate for each channel in the detector
include("../src/oscCalc.jl")


function propagateSamplesAvg(unoscillatedSample, responseMatrices, oscParams, solarModel, bin_edges)
    oscProbs_nue_8B = averageProbOverBins(bin_edges::Vector{Float64}, oscParams, solarModel, process="8B")
    oscProbs_nue_hep = averageProbOverBins(bin_edges::Vector{Float64}, oscParams, solarModel, process="hep")

    oscProbs_nuother_8B = 1 .- oscProbs_nue_8B
    oscProbs_nuother_hep = 1 .- oscProbs_nue_hep

    oscillated_sample_ES_nue = unoscillatedSample.ES_nue_8B .* oscProbs_nue_8B .+ unoscillatedSample.ES_nue_hep .* oscProbs_nue_hep
    oscillated_sample_ES_nuother = unoscillatedSample.ES_nuother_8B .* oscProbs_nuother_8B .+ unoscillatedSample.ES_nuother_hep .* oscProbs_nuother_hep
    oscillated_sample_CC = unoscillatedSample.CC_8B .* oscProbs_nue_8B .+ unoscillatedSample.CC_hep .* oscProbs_nue_hep

    eventRate_ES_nue = responseMatrices.ES.nue' * oscillated_sample_ES_nue
    eventRate_ES_nuother = responseMatrices.ES.nuother' * oscillated_sample_ES_nuother
    eventRate_CC = responseMatrices.CC' * oscillated_sample_CC

    return eventRate_ES_nue, eventRate_ES_nuother, eventRate_CC
end


function propagateSamplesCtr(unoscillatedSample, responseMatrices, oscParams, solarModel, bin_edges)
    oscProbs_nue_8B = centralProbOverBins(bin_edges::Vector{Float64}, oscParams, solarModel, process="8B")
    oscProbs_nue_hep = centralProbOverBins(bin_edges::Vector{Float64}, oscParams, solarModel, process="hep")

    oscProbs_nuother_8B = 1 .- oscProbs_nue_8B
    oscProbs_nuother_hep = 1 .- oscProbs_nue_hep

    oscillated_sample_ES_nue = unoscillatedSample.ES_nue_8B .* oscProbs_nue_8B .+ unoscillatedSample.ES_nue_hep .* oscProbs_nue_hep
    oscillated_sample_ES_nuother = unoscillatedSample.ES_nuother_8B .* oscProbs_nuother_8B .+ unoscillatedSample.ES_nuother_hep .* oscProbs_nuother_hep
    oscillated_sample_CC = unoscillatedSample.CC_8B .* oscProbs_nue_8B .+ unoscillatedSample.CC_hep .* oscProbs_nue_hep

    eventRate_ES_nue = responseMatrices.ES.nue' * oscillated_sample_ES_nue
    eventRate_ES_nuother = responseMatrices.ES.nuother' * oscillated_sample_ES_nuother
    eventRate_CC = responseMatrices.CC' * oscillated_sample_CC

    return eventRate_ES_nue, eventRate_ES_nuother, eventRate_CC
end