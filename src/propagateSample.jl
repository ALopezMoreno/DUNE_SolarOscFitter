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
include("../src/oscillations/earthPropagation.jl")
using LinearAlgebra
using Plots

function block_average(mat::AbstractMatrix, block_dims::Tuple{Int,Int}=(5, 3))
  block_rows, block_cols = block_dims
  n_rows, n_cols = size(mat)
  
  # Ensure that matrix dimensions are multiples of block dimensions
  if n_rows % block_rows != 0 || n_cols % block_cols != 0
      error("Matrix dimensions must be multiples of block dimensions: got $(n_rows)x$(n_cols) for blocks of size $(block_rows)x$(block_cols)")
  end

  out_n = n_rows ÷ block_rows
  out_m = n_cols ÷ block_cols
  result = Array{eltype(mat)}(undef, out_n, out_m)

  for i in 1:out_n
      for j in 1:out_m
          rows_range = ((i - 1) * block_rows + 1):(i * block_rows)
          cols_range = ((j - 1) * block_cols + 1):(j * block_cols)
          block = view(mat, rows_range, cols_range)
          # Calculate average over the block
          result[i, j] = sum(block) / (block_rows * block_cols)
      end
  end

  return result
end


# DEPRECATED !!!!
function propagateSamplesAvg(unoscillatedSample, responseMatrices, params, solarModel, bin_edges, BG_CC)
    oscProbs_nue_8B = averageProbOverBins(bin_edges::Vector{Float64}, params, solarModel, process="8B") 
    oscProbs_nue_hep = averageProbOverBins(bin_edges::Vector{Float64}, params, solarModel, process="hep")

    oscProbs_nuother_8B = 1 .- oscProbs_nue_8B
    oscProbs_nuother_hep = 1 .- oscProbs_nue_hep

    oscillated_sample_ES_nue = unoscillatedSample.ES_nue_8B .* oscProbs_nue_8B .* params.integrated_8B_flux .+ unoscillatedSample.ES_nue_hep .* oscProbs_nue_hep .* params.integrated_8B_flux * 2e-4 #ESTIMATE OF HEP FLUX
    oscillated_sample_ES_nuother = unoscillatedSample.ES_nuother_8B .* oscProbs_nuother_8B  .* params.integrated_8B_flux .+ unoscillatedSample.ES_nuother_hep .* oscProbs_nuother_hep .* params.integrated_8B_flux * 2e-4 #ESTIMATE OF HEP FLUX
    oscillated_sample_CC = unoscillatedSample.CC_8B .* oscProbs_nue_8B .* params.integrated_8B_flux .+ unoscillatedSample.CC_hep .* oscProbs_nue_hep .* params.integrated_8B_flux * 2e-4 #ESTIMATE OF HEP FLUX

    eventRate_ES_nue = responseMatrices.ES.nue' * oscillated_sample_ES_nue
    eventRate_ES_nuother = responseMatrices.ES.nuother' * oscillated_sample_ES_nuother
    eventRate_CC = responseMatrices.CC' * oscillated_sample_CC + 9e2 * BG_CC

    return eventRate_ES_nue, eventRate_ES_nuother, eventRate_CC
end


function propagateSamplesCtr(unoscillatedSample, responseMatrices, params, solarModel, bin_edges, raw_backgrounds)
    mixingPars = oscPars(params.dm2_21, asin(sqrt(params.sin2_th12)), asin(sqrt(params.sin2_th13)))

    bin_centers = (bin_edges[1:end-1] + bin_edges[2:end]) / 2.0
    bin_centers_calc = (bin_edges_calc[1:end-1] + bin_edges_calc[2:end]) / 2.0

    # get oscillation probabilities
    oscProbs_nue_8B_day = centralProbOverBins(bin_centers::Vector{Float64}, params, solarModel, process="8B")
    oscProbs_nue_hep_day = centralProbOverBins(bin_centers::Vector{Float64}, params, solarModel, process="hep") 

    oscProbs_nuother_8B_day = 1 .- oscProbs_nue_8B_day
    oscProbs_nuother_hep_day = 1 .- oscProbs_nue_hep_day

    # call the appropriate function for getting the earth propagation matrix
    if earthUncertainty
        oscProbs_1e = get_1e(bin_centers_calc, mixingPars, params.earth_norm, earth_paths)
    else
        oscProbs_1e = get_1e(bin_centers_calc, mixingPars, earth_paths)
    end

    # Treat backgrounds accordingly
    backgrounds = deepcopy(raw_backgrounds)

    norm_index = 1
    for (i, behaviour) in enumerate(ES_bg_par_counts)
        if behaviour != 0
            # Access field named "ES_bg_norm_<index>" from params
            backgrounds.ES[i] .*= getfield(params, Symbol("ES_bg_norm_", norm_index))
            norm_index += 1
        end
    end
  
    norm_index = 1
    for (i, behaviour) in enumerate(CC_bg_par_counts)
        if behaviour != 0
            # Access field named "CC_bg_norm_<index>" from params
            backgrounds.CC[i] .*= getfield(params, Symbol("CC_bg_norm_", norm_index))
            norm_index += 1
        end
    end

    BG_ES = reduce(+, backgrounds.ES)
    BG_CC = reduce(+, backgrounds.CC)

    oscProbs_nue_8B_night = block_average(get_probs(bin_centers_calc, oscProbs_1e, mixingPars, solarModel.avgNeBoron)) 
    oscProbs_nue_hep_night = block_average(get_probs(bin_centers_calc, oscProbs_1e, mixingPars, solarModel.avgNeHep))

    # oscProbs_nuother_8B_night = 1 .- oscProbs_nue_8B_night
    # oscProbs_nuother_hep_night = 1 .- oscProbs_nue_hep_night

    # Apply weights
    # oscillated_sample_ES_nue_day = unoscillatedSample.ES_nue_8B .* oscProbs_nue_8B_day .* params.integrated_8B_flux .+ unoscillatedSample.ES_nue_hep .* oscProbs_nue_hep_day .* params.integrated_8B_flux * 2e-4 #ESTIMATE OF HEP FLUX
    # oscillated_sample_ES_nuother_day = unoscillatedSample.ES_nuother_8B .* oscProbs_nuother_8B_day  .* params.integrated_8B_flux .+ unoscillatedSample.ES_nuother_hep .* oscProbs_nuother_hep_day .* params.integrated_8B_flux * 2e-4 #ESTIMATE OF HEP FLUX
    oscillated_sample_CC_day = unoscillatedSample.CC_8B .* oscProbs_nue_8B_day .* params.integrated_8B_flux .+ unoscillatedSample.CC_hep .* oscProbs_nue_hep_day .* params.integrated_8B_flux * 2e-4 #ESTIMATE OF HEP FLUX
    
    # oscillated_sample_ES_nue_night = (unoscillatedSample.ES_nue_8B' .* (params.integrated_8B_flux' .* oscProbs_nue_8B_night .* exposure_weights)) .+
    #                                  (unoscillatedSample.ES_nue_hep' .* (2e-4 .* params.integrated_8B_flux' .* oscProbs_nue_hep_night .* exposure_weights)) # ESTIMATE OF HEP FLUX

    # oscillated_sample_ES_nuother_night = (unoscillatedSample.ES_nuother_8B' .* (params.integrated_8B_flux' .* oscProbs_nuother_8B_night .* exposure_weights)) .+
    #                                      (unoscillatedSample.ES_nuother_hep' .* (2e-4 .* params.integrated_8B_flux' .* oscProbs_nuother_hep_night .* exposure_weights)) # ESTIMATE OF HEP FLUX

    oscillated_sample_CC_night = (unoscillatedSample.CC_8B' .* (params.integrated_8B_flux' .* oscProbs_nue_8B_night .* exposure_weights)) .+
                                 (unoscillatedSample.CC_hep' .* (2e-4 .* params.integrated_8B_flux' .* oscProbs_nue_hep_night .* exposure_weights)) # ESTIMATE OF HEP FLUX

    # Apply detector responses and split event rates 50/50 between day and night samples
    # eventRate_ES_nue_day = 0.5 .* (responseMatrices.ES.nue' * oscillated_sample_ES_nue_day)
    # eventRate_ES_nuother_day = 0.5 .* (responseMatrices.ES.nuother' * oscillated_sample_ES_nuother_day)

    #pt1 = heatmap(responseMatrices.CC', title="Response Matrices CC'", xlabel="Columns", ylabel="Rows", size=(1500,1500))
    #display(pt1)
    #println("Shape of responseMatrices.CC': ", size(responseMatrices.CC'))
    #println("Shape of oscillated_sample_CC_day: ", size(oscillated_sample_CC_day))
    #println("Shape of BG_CC: ", size(BG_CC))
    #sleep(10)

    eventRate_CC_day = 0.5 .* (responseMatrices.CC' * oscillated_sample_CC_day  .+ BG_CC )

    # eventRate_ES_nue_night = vcat([0.5 .* (row' * responseMatrices.ES.nue) for row in eachrow(oscillated_sample_ES_nue_night)]...)
    # eventRate_ES_nuother_night = vcat([0.5 .* (row' * responseMatrices.ES.nuother) for row in eachrow(oscillated_sample_ES_nuother_night)]...)
    eventRate_CC_night = vcat([0.5 .* (row' * responseMatrices.CC .+ (BG_CC' ./ cosz_bins.bin_number) ) for row in eachrow(oscillated_sample_CC_night)]...)

    eventRate_ES_nue_day = eventRate_CC_day
    eventRate_ES_nuother_day = eventRate_CC_day

    eventRate_ES_nue_night = eventRate_CC_night
    eventRate_ES_nuother_night = eventRate_CC_night

    # myPlot = heatmap(eventRate_CC_night[:, 1:end], colormap=:viridis, aspect_ratio=:equal, size=(1500,1500))
    # display(myPlot)
    # sleep(30)

    return eventRate_ES_nue_day, eventRate_ES_nuother_day, eventRate_CC_day, eventRate_ES_nue_night, eventRate_ES_nuother_night, eventRate_CC_night
end


function propagateSamplesUncertainty(unoscillatedSample, responseMatrices, params, solarModel, bin_edges, BG_CC, probMatrix)
  mixingPars = oscPars(params.dm2_21, asin(sqrt(params.sin2_th12)), asin(sqrt(params.sin2_th13)))

  bin_centers = (bin_edges[1:end-1] + bin_edges[2:end]) / 2.0
  bin_centers_calc = (bin_edges_calc[1:end-1] + bin_edges_calc[2:end]) / 2.0

  # get oscillation probabilities
  oscProbs_nue_8B_day = centralProbOverBins(bin_centers::Vector{Float64}, params, solarModel, process="8B")
  oscProbs_nue_hep_day = centralProbOverBins(bin_centers::Vector{Float64}, params, solarModel, process="hep") 

  oscProbs_nuother_8B_day = 1 .- oscProbs_nue_8B_day
  oscProbs_nuother_hep_day = 1 .- oscProbs_nue_hep_day

  oscProbs_1e = get_1e(bin_centers_calc, mixingPars, earth_paths)

  oscProbs_nue_8B_night = probMatrix
  oscProbs_nue_hep_night = probMatrix

  # oscProbs_nuother_8B_night = 1 .- oscProbs_nue_8B_night
  # oscProbs_nuother_hep_night = 1 .- oscProbs_nue_hep_night

  # Apply weights
  # oscillated_sample_ES_nue_day = unoscillatedSample.ES_nue_8B .* oscProbs_nue_8B_day .* params.integrated_8B_flux .+ unoscillatedSample.ES_nue_hep .* oscProbs_nue_hep_day .* params.integrated_8B_flux * 2e-4 #ESTIMATE OF HEP FLUX
  # oscillated_sample_ES_nuother_day = unoscillatedSample.ES_nuother_8B .* oscProbs_nuother_8B_day  .* params.integrated_8B_flux .+ unoscillatedSample.ES_nuother_hep .* oscProbs_nuother_hep_day .* params.integrated_8B_flux * 2e-4 #ESTIMATE OF HEP FLUX
  oscillated_sample_CC_day = unoscillatedSample.CC_8B .* oscProbs_nue_8B_day .* params.integrated_8B_flux .+ unoscillatedSample.CC_hep .* oscProbs_nue_hep_day .* params.integrated_8B_flux * 2e-4 #ESTIMATE OF HEP FLUX

  # oscillated_sample_ES_nue_night = (unoscillatedSample.ES_nue_8B' .* (params.integrated_8B_flux' .* oscProbs_nue_8B_night .* exposure_weights)) .+
  #                                  (unoscillatedSample.ES_nue_hep' .* (2e-4 .* params.integrated_8B_flux' .* oscProbs_nue_hep_night .* exposure_weights)) # ESTIMATE OF HEP FLUX

  # oscillated_sample_ES_nuother_night = (unoscillatedSample.ES_nuother_8B' .* (params.integrated_8B_flux' .* oscProbs_nuother_8B_night .* exposure_weights)) .+
  #                                      (unoscillatedSample.ES_nuother_hep' .* (2e-4 .* params.integrated_8B_flux' .* oscProbs_nuother_hep_night .* exposure_weights)) # ESTIMATE OF HEP FLUX

  oscillated_sample_CC_night = (unoscillatedSample.CC_8B' .* (params.integrated_8B_flux' .* oscProbs_nue_8B_night .* exposure_weights)) .+
                               (unoscillatedSample.CC_hep' .* (2e-4 .* params.integrated_8B_flux' .* oscProbs_nue_hep_night .* exposure_weights)) # ESTIMATE OF HEP FLUX
  

  # Apply detector responses and split event rates 50/50 between day and night samples
  # eventRate_ES_nue_day = 0.5 .* (responseMatrices.ES.nue' * oscillated_sample_ES_nue_day)
  # eventRate_ES_nuother_day = 0.5 .* (responseMatrices.ES.nuother' * oscillated_sample_ES_nuother_day)

  eventRate_CC_day = 0.5 .* (responseMatrices.CC * oscillated_sample_CC_day  + BG_CC .* CC_bg_norm)

  # eventRate_ES_nue_night = vcat([0.5 .* (row' * responseMatrices.ES.nue) for row in eachrow(oscillated_sample_ES_nue_night)]...)
  # eventRate_ES_nuother_night = vcat([0.5 .* (row' * responseMatrices.ES.nuother) for row in eachrow(oscillated_sample_ES_nuother_night)]...)
  eventRate_CC_night = vcat([0.5 .* (row' * responseMatrices.CC' .+ BG_CC' .* CC_bg_norm) for row in eachrow(oscillated_sample_CC_night)]...)

  eventRate_ES_nue_day = eventRate_CC_day
  eventRate_ES_nuother_day = eventRate_CC_day

  eventRate_ES_nue_night = eventRate_CC_night
  eventRate_ES_nuother_night = eventRate_CC_night

  # myPlot2 = heatmap(oscillated_sample_CC_night[:, 10:end], colormap=:viridis)
  # display(myPlot2)
  # sleep(10)

  return eventRate_ES_nue_day, eventRate_ES_nuother_day, eventRate_CC_day, eventRate_ES_nue_night, eventRate_ES_nuother_night, eventRate_CC_night
end