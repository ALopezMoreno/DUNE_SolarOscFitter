#=
propagateSample.jl

Neutrino sample propagation and event rate calculation for the Solar Oscillation Fitter.
This module takes unoscillated Monte Carlo samples and applies oscillation probabilities
to calculate expected event rates in the detector for different channels and time periods.

Key Features:
- Oscillation probability calculation for day and night periods
- Earth matter effect propagation for nighttime neutrinos
- Detector response matrix application
- Background event handling with systematic uncertainties
- Block averaging for energy binning
- Support for both fast and slow calculation modes

The main function propagateSamples() is the core of the likelihood calculation,
converting theoretical predictions into observable event rates.

Author: [Author name]
=#

include("../src/oscillations/osc.jl")

# Import oscillation calculation functions
using .Osc: oscPars, osc_prob_both_slow

# Set energy bin centers for calculation
global E_calc = (bin_edges_calc[1:end-1] + bin_edges_calc[2:end]) / 2.0

# Choose fast or slow Earth propagation based on configuration. Choose oscillations calculator (nuFast only works on fast mode)
if nuFast
    include("../src/oscillations/nuFast_interface.jl")
    using .nuFastOsc: osc_prob_both_fast, init_engines
    nuFastOsc.init_engines(E_calc, cosz_calc)
else
  using .Osc: osc_prob_both_fast
  if fast
    using .Osc.NumOsc.Fast: osc_prob_earth
  else
    using .Osc.NumOsc.Slow: osc_prob_earth
  end
end

using LinearAlgebra  # For matrix operations
using Plots          # For debugging plots (optional)

# Block averaging utilities for energy binning
# These functions reduce high-resolution calculations to analysis binning

# Block averaging for 2D matrices (e.g., zenith vs energy oscillation probabilities)
function block_average(mat::AbstractMatrix, block_dims::Tuple{Int,Int}=(5, 3))
    """Average matrix elements over rectangular blocks"""
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

# Block averaging for 1D vectors (e.g., energy-dependent oscillation probabilities)
function block_average(vec::AbstractVector, block_size::Int=5)
    """Average vector elements over consecutive blocks"""
    n = length(vec)
    
    # Ensure that vector length is a multiple of block size
    if n % block_size != 0
        error("Vector length must be a multiple of block size: got length $(n) for blocks of size $(block_size)")
    end
    
    out_n = n ÷ block_size
    result = Array{eltype(vec)}(undef, out_n)
    
    for i in 1:out_n
        range = ((i - 1) * block_size + 1):(i * block_size)
        block = view(vec, range)
        result[i] = sum(block) / block_size
    end
    
    return result
end

# Generic wrapper function for block averaging
function block_average(arr::AbstractArray, block_dims...)
    """Dispatch to appropriate block averaging function based on array dimension"""
    if ndims(arr) == 1
        return block_average(arr, block_dims...)
    elseif ndims(arr) == 2
        return block_average(arr, block_dims...)
    else
        error("Unsupported array dimension: $(ndims(arr)). Only 1D and 2D arrays are supported.")
    end
end


function propagateSamples(unoscillatedSample, responseMatrices, params, solarModel, bin_edges, raw_backgrounds)
    """
    Main function to propagate unoscillated neutrino samples through oscillations
    and detector response to calculate expected event rates.
    
    Arguments:
    - unoscillatedSample: Unoscillated MC event rates by channel and process
    - responseMatrices: Detector response matrices for energy reconstruction
    - params: Oscillation and systematic parameters
    - solarModel: Solar neutrino production model
    - bin_edges: Energy bin edges for analysis
    - raw_backgrounds: Background event samples
    
    Returns:
    - Event rates for ES/CC channels in day/night periods
    - Background totals for each channel
    """
    
    # Convert oscillation parameters to internal format
    mixingPars = oscPars(params.dm2_21, asin(sqrt(params.sin2_th12)), asin(sqrt(params.sin2_th13)))

    # Calculate bin centers for oscillation probability evaluation #### we already do this outside!!!!!
    # bin_centers = (bin_edges[1:end-1] + bin_edges[2:end]) / 2.0
    # bin_centers_calc = (bin_edges_calc[1:end-1] + bin_edges_calc[2:end]) / 2.0

    # call the appropriate function for getting the earth propagation matrix ##########################################
    if earthUncertainty
        n = length(earth_lookup)
        earth_norm_vector = [getfield(params, Symbol("earth_norm_", i)) for i in 1:n]
        lookup = earth_norm_vector .* earth_lookup
        if !nuFast
          oscProbs_1e = osc_prob_earth(E_calc, mixingPars, lookup, earth_paths)
        end
        else
          earth_norm_vector = []
          if !nuFast
            oscProbs_1e = osc_prob_earth(E_calc, mixingPars, earth_lookup, earth_paths)
          end
    end

    # Add backgrounds and normalise according to uncertainty parameters (if any) #######################################
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
    
    # get oscillation probabilities with fine resolution ################################################################
    if nuFast
      oscProbs_nue_8B_day_large, oscProbs_nue_8B_night_large = osc_prob_both_fast(E_calc, mixingPars, lookup, earth_paths, n_vec=earth_norm_vector)
      oscProbs_nue_hep_day_large, oscProbs_nue_hep_night_large = osc_prob_both_fast(E_calc, mixingPars, lookup, earth_paths, n_vec=earth_norm_vector) ## NOT YET IMPLEMENTED
    elseif fast
      oscProbs_nue_8B_day_large, oscProbs_nue_8B_night_large = osc_prob_both_fast(E_calc, oscProbs_1e, mixingPars, solarModel, process="8B")
      oscProbs_nue_hep_day_large, oscProbs_nue_hep_night_large = osc_prob_both_fast(E_calc, oscProbs_1e, mixingPars, solarModel, process="hep")
    else
      oscProbs_nue_8B_day_large, oscProbs_nue_8B_night_large = osc_prob_both_slow(E_calc, oscProbs_1e, mixingPars, solarModel, process="8B")
      oscProbs_nue_hep_day_large, oscProbs_nue_hep_night_large = osc_prob_both_slow(E_calc, oscProbs_1e, mixingPars, solarModel, process="hep")
    end

    # average over fine resolution to desired binning ###################################################################
    oscProbs_nue_8B_day = block_average(oscProbs_nue_8B_day_large, 2)
    oscProbs_nue_hep_day = block_average(oscProbs_nue_hep_day_large, 2)

    oscProbs_nue_8B_night = block_average(oscProbs_nue_8B_night_large, (3, 2)) 
    oscProbs_nue_hep_night = block_average(oscProbs_nue_hep_night_large, (3, 2))

    # get nu_other probabilities from unitarity relation ################################################################
    oscProbs_nuother_8B_day = 1 .- oscProbs_nue_8B_day
    oscProbs_nuother_hep_day = 1 .- oscProbs_nue_hep_day

    oscProbs_nuother_8B_night = 1 .- oscProbs_nue_8B_night
    oscProbs_nuother_hep_night = 1 .- oscProbs_nue_hep_night

    # Apply weights to true energy and propagate through reco matrix. Set samples to zero if channel is not being fit ###
    if ES_mode
      oscillated_sample_ES_nue_day = unoscillatedSample.ES_nue_8B .* oscProbs_nue_8B_day .* params.integrated_8B_flux .+ unoscillatedSample.ES_nue_hep .* oscProbs_nue_hep_day .* params.integrated_HEP_flux
      oscillated_sample_ES_nue_night = (unoscillatedSample.ES_nue_8B' .* (params.integrated_8B_flux' .* oscProbs_nue_8B_night .* exposure_weights)) .+
                                       (unoscillatedSample.ES_nue_hep' .* (params.integrated_HEP_flux .* oscProbs_nue_hep_night .* exposure_weights))

      eventRate_ES_nue_day = 0.5 .* ((responseMatrices.ES.nue' * oscillated_sample_ES_nue_day) .* ES_nue_eff)  # 0.5 corresponds to the yearly daytime fraction (#CHECK!)
      eventRate_ES_nue_night = vcat([0.5 .* ((row' * responseMatrices.ES.nue) .* ES_nue_eff' ) for row in eachrow(oscillated_sample_ES_nue_night)]...)  # FACTOR OUT EFFICIENCIES!!


      oscillated_sample_ES_nuother_day = unoscillatedSample.ES_nuother_8B .* oscProbs_nuother_8B_day  .* params.integrated_8B_flux .+ unoscillatedSample.ES_nuother_hep .* oscProbs_nuother_hep_day .* params.integrated_HEP_flux
      oscillated_sample_ES_nuother_night = (unoscillatedSample.ES_nuother_8B' .* (params.integrated_8B_flux' .* oscProbs_nuother_8B_night .* exposure_weights)) .+
                                           (unoscillatedSample.ES_nuother_hep' .* (params.integrated_HEP_flux .* oscProbs_nuother_hep_night .* exposure_weights))

      eventRate_ES_nuother_day = 0.5 .* ((responseMatrices.ES.nuother' * oscillated_sample_ES_nuother_day) .* ES_nuother_eff)  # 0.5 corresponds to the yearly daytime fraction (#CHECK!)
      eventRate_ES_nuother_night = vcat([0.5 .* ((row' * responseMatrices.ES.nuother) .* ES_nuother_eff' ) for row in eachrow(oscillated_sample_ES_nuother_night)]...)  # FACTOR OUT EFFICIENCIES!!

      eventRate_ES_day = (eventRate_ES_nue_day .+ eventRate_ES_nuother_day) .+ 0.5 .* BG_ES
      eventRate_ES_night = (eventRate_ES_nue_night .+ eventRate_ES_nuother_night) .+ 0.5 .* (BG_ES' .* exposure_weights)

    else
      eventRate_ES_day = fill(0., Ereco_bins_ES.bin_number)
      eventRate_ES_night = fill(0., (cosz_bins.bin_number, Ereco_bins_ES.bin_number))
    end


    if CC_mode 
      oscillated_sample_CC_day = unoscillatedSample.CC_8B .* oscProbs_nue_8B_day .* params.integrated_8B_flux .+ unoscillatedSample.CC_hep .* oscProbs_nue_hep_day .* params.integrated_HEP_flux
      oscillated_sample_CC_night = (unoscillatedSample.CC_8B' .* (params.integrated_8B_flux' .* oscProbs_nue_8B_night .* exposure_weights)) .+
                                   (unoscillatedSample.CC_hep' .* (params.integrated_HEP_flux .* oscProbs_nue_hep_night .* exposure_weights))

      eventRate_CC_day = 0.5 .* ((responseMatrices.CC' * oscillated_sample_CC_day) .* CC_eff  .+ BG_CC ) # 0.5 corresponds to the yearly daytime fraction (#CHECK!)
      eventRate_CC_night = vcat([0.5 .* ((row' * responseMatrices.CC) .* CC_eff' ) for row in eachrow(oscillated_sample_CC_night)]...) .+ 0.5 .* (BG_CC' .* exposure_weights)
    
    else
      eventRate_CC_day = fill(0., Ereco_bins_CC.bin_number)
      eventRate_CC_night = fill(0., (cosz_bins.bin_number, Ereco_bins_CC.bin_number))
    end

    
    # oscillated_sample_ES_nue_night = (unoscillatedSample.ES_nue_8B' .* (params.integrated_8B_flux' .* oscProbs_nue_8B_night .* exposure_weights)) .+
    #                                  (unoscillatedSample.ES_nue_hep' .* (2e-4 .* params.integrated_8B_flux' .* oscProbs_nue_hep_night .* exposure_weights)) # ESTIMATE OF HEP FLUX

    # oscillated_sample_ES_nuother_night = (unoscillatedSample.ES_nuother_8B' .* (params.integrated_8B_flux' .* oscProbs_nuother_8B_night .* exposure_weights)) .+
    #                                      (unoscillatedSample.ES_nuother_hep' .* (2e-4 .* params.integrated_8B_flux' .* oscProbs_nuother_hep_night .* exposure_weights)) # ESTIMATE OF HEP FLUX

     # ESTIMATE OF HEP FLUX

    # Apply detector responses and split event rates 50/50 between day and night samples
    # eventRate_ES_nue_day = 0.5 .* (responseMatrices.ES.nue' * oscillated_sample_ES_nue_day)
    # eventRate_ES_nuother_day = 0.5 .* (responseMatrices.ES.nuother' * oscillated_sample_ES_nuother_day)


    

    # eventRate_ES_nue_night = vcat([0.5 .* (row' * responseMatrices.ES.nue) for row in eachrow(oscillated_sample_ES_nue_night)]...)
    # eventRate_ES_nuother_night = vcat([0.5 .* (row' * responseMatrices.ES.nuother) for row in eachrow(oscillated_sample_ES_nuother_night)]...)

    # bins = range(5, 25, length=length(backgrounds.CC[1]))

    # Plot the stacked bar chart with the two contributions.
    # bar(bins,
    # backgrounds.CC[1],
    # label = "Gammas",
    # title = "Stacked Bar Chart of CC Backgrounds",
    # xlabel = "Bins",
    # ylabel = "Counts")

    # myP = heatmap(oscprobs_boron_large,
    # xlabel = "Column Index",
    # ylabel = "Row Index",
    # title = "Oscprobs",
    # color = :viridis)
    # display(myP)
    # sleep(5)

    # Add the neutrons counts on top of the gammas counts with stacking enabled.
    # bar!(bins,
    #     backgrounds.CC[2],
    #     label = "Neutrons",
    #     bar_position = :stack)

    # display(current())  # Display the current figure

    # display(oscProbs_nue_8B_day_large)

    #pt1 = heatmap(responseMatrices.CC', title="Response Matrices CC'", xlabel="Columns", ylabel="Rows", size=(1500,1500))
    #display(pt1)
    #println("Shape of responseMatrices.CC': ", size(responseMatrices.CC'))
    #println("Shape of oscillated_sample_CC_day: ", size(oscillated_sample_CC_day))
    #println("Shape of BG_CC: ", size(BG_CC))
    #sleep(10)

    # myP2 = heatmap([oscProbs_nue_8B_night_large],
    # xlabel = "Column Index",
    # ylabel = "Row Index",
    # title = "Oscprobs",
    # color = :viridis)
    # display(myP2)
    # sleep(200)

    # sleep(20)

    # myPlot = heatmap(eventRate_CC_night[:, 1:end], colormap=:viridis, aspect_ratio=:equal, size=(1500,1500))
    # display(myPlot)
    # sleep(30)

    return eventRate_ES_day, eventRate_CC_day, eventRate_ES_night, eventRate_CC_night, BG_ES, BG_CC
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