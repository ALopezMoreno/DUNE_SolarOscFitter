"""
This module provides functionality for propagating neutrino samples through a detector simulation, 
calculating the oscillated event rates for different interaction channels. It utilizes pre-defined 
oscillation probabilities and response matrices to transform unoscillated neutrino samples into 
oscillated event rates.

Dependencies:
- Includes the `oscCalc.jl` module, which contains functions for calculating neutrino oscillation 
  probabilities.
- Assumes the existence of organization-specific data structures for handling neutrino samples and 
  response matrices.

Functions:
- `propagateSamplesAvg`: Computes the bin-average oscillated event rates for electron neutrinos and other 
  neutrino flavors across different solar processes (e.g., 8B and hep). It applies the response matrices 
  to the oscillated samples to obtain the event rates for elastic scattering (ES) and charged current (CC) 
  interactions.
- `propagateSamplesCtr`: Computes the oscillated event rates for electron neutrinos and other 
  neutrino flavors across different solar processes (e.g., 8B and hep) at the bin centers. It applies the response matrices 
  to the oscillated samples to obtain the event rates for elastic scattering (ES) and charged current (CC) 
  interactions.

Parameters:
- `unoscillatedSample`: A data structure containing unoscillated neutrino event samples for different 
  processes and interaction channels.
- `responseMatrices`: A data structure containing response matrices for ES and CC interactions, used to 
  transform oscillated samples into event rates.
- `oscParams`: Oscillation parameters used to calculate the oscillation probabilities.
- `solarModel`: A model specifying solar neutrino fluxes, used in conjunction with oscillation parameters.
- `bin_edges`: A vector of bin edges for averaging oscillation probabilities over energy bins.
- `BG_CC`: Background event rate for charged current interactions, added to the calculated event rate.

Process:
1. Calculates the average oscillation probabilities for electron neutrinos (`nue`) and other neutrino 
   flavors (`nuother`) for specified solar processes.
2. Computes the oscillated samples by applying the oscillation probabilities to the unoscillated samples.
3. Transforms the oscillated samples into event rates using the response matrices, accounting for 
   background contributions in the CC channel.

Output:
- Returns the event rates for ES interactions with electron neutrinos and other neutrino flavors, as well 
  as the event rate for CC interactions.

Note:
- The function assumes that the input data structures are compatible with the organization's internal 
  formats and that the constants and models are appropriately defined.
"""


include("../src/oscCalc.jl")


function propagateSamplesAvg(unoscillatedSample, responseMatrices, oscParams, solarModel, bin_edges, BG_CC)
    oscProbs_nue_8B = averageProbOverBins(bin_edges::Vector{Float64}, oscParams, solarModel, process="8B")
    oscProbs_nue_hep = averageProbOverBins(bin_edges::Vector{Float64}, oscParams, solarModel, process="hep")

    oscProbs_nuother_8B = 1 .- oscProbs_nue_8B
    oscProbs_nuother_hep = 1 .- oscProbs_nue_hep

    oscillated_sample_ES_nue = unoscillatedSample.ES_nue_8B .* oscProbs_nue_8B .+ unoscillatedSample.ES_nue_hep .* oscProbs_nue_hep
    oscillated_sample_ES_nuother = unoscillatedSample.ES_nuother_8B .* oscProbs_nuother_8B .+ unoscillatedSample.ES_nuother_hep .* oscProbs_nuother_hep
    oscillated_sample_CC = unoscillatedSample.CC_8B .* oscProbs_nue_8B .+ unoscillatedSample.CC_hep .* oscProbs_nue_hep

    eventRate_ES_nue = responseMatrices.ES.nue' * oscillated_sample_ES_nue
    eventRate_ES_nuother = responseMatrices.ES.nuother' * oscillated_sample_ES_nuother
    eventRate_CC = responseMatrices.CC' * oscillated_sample_CC + 2e8 * BG_CC

    return eventRate_ES_nue, eventRate_ES_nuother, eventRate_CC
end


function propagateSamplesCtr(unoscillatedSample, responseMatrices, oscParams, solarModel, bin_edges, BG_CC)
    oscProbs_nue_8B = centralProbOverBins(bin_edges::Vector{Float64}, oscParams, solarModel, process="8B")
    oscProbs_nue_hep = centralProbOverBins(bin_edges::Vector{Float64}, oscParams, solarModel, process="hep")

    oscProbs_nuother_8B = 1 .- oscProbs_nue_8B
    oscProbs_nuother_hep = 1 .- oscProbs_nue_hep

    oscillated_sample_ES_nue = unoscillatedSample.ES_nue_8B .* oscProbs_nue_8B .+ unoscillatedSample.ES_nue_hep .* oscProbs_nue_hep
    oscillated_sample_ES_nuother = unoscillatedSample.ES_nuother_8B .* oscProbs_nuother_8B .+ unoscillatedSample.ES_nuother_hep .* oscProbs_nuother_hep
    oscillated_sample_CC = unoscillatedSample.CC_8B .* oscProbs_nue_8B .+ unoscillatedSample.CC_hep .* oscProbs_nue_hep

    eventRate_ES_nue = responseMatrices.ES.nue' * oscillated_sample_ES_nue
    eventRate_ES_nuother = responseMatrices.ES.nuother' * oscillated_sample_ES_nuother
    eventRate_CC = responseMatrices.CC' * oscillated_sample_CC + 2e8 * BG_CC

    return eventRate_ES_nue, eventRate_ES_nuother, eventRate_CC
end