"""
This script defines functions for calculating the Poisson log-likelihood of neutrino event data, using 
Monte Carlo (MC) propagated samples to evaluate expected event rates against observed data. It is part of 
a larger framework for neutrino physics analysis, focusing on elastic scattering (ES) and charged current 
(CC) interactions.

Dependencies:
- Utilizes Julia packages such as `Random`, `LinearAlgebra`, `Statistics`, `Distributions`, `StatsBase`, 
  `BAT`, `DensityInterface`, and `IntervalSets` for statistical and mathematical operations.
- Assumes the existence of organization-specific data structures and modules, including `ereco_data`, 
  `responseMatrices`, `solarModel`, `unoscillatedSample`, and `backgrounds`.

Functions:
- `poissonLogLikelihood`: Computes the Poisson log-likelihood for vectors of expected and measured event 
  counts, handling edge cases where counts are zero to avoid mathematical errors.
- `likelihood_all_samples_avg`: Calculates the total log-likelihood for neutrino events using average 
  propagation of samples, considering ES and CC interactions.
- `likelihood_all_samples_ctr`: Faster than `likelihood_all_samples_avg`, but uses a different propagation 
  method for samples, where the propagation is done only at the bin centers instead of integrating over bins.

Parameters:
- `nExpected`: A vector of expected event counts for each interaction channel.
- `nMeasured`: A vector of measured event counts for each interaction channel.
- `parameters`: Model parameters used in the propagation of samples and likelihood calculations.

Process:
1. Validates input vectors for non-negativity and equal length in `poissonLogLikelihood`.
2. Propagates MC samples through the detector simulation using specified functions (`propagateSamplesAvg` or `propagateSamplesCtr`).
3. Computes expected event rates for different interaction channels and calculates the log-likelihood 
   using the `poissonLogLikelihood` function.
4. Handles energy thresholding by identifying the appropriate indices for analysis based on the energy 
   threshold `Emin`.

Output:
- Returns the total negative log-likelihood for the given parameters, which can be used in optimization 
  routines or further statistical analyses.

Testing:
- Includes a test function `test_log_likelihood_execution_time` to measure the execution time of the 
  `poissonLogLikelihood` function and verify its correctness against expected results.

Note:
- Ensure that all required data structures and constants are defined and accessible in the working 
  environment before executing the script.
- The script assumes that the input data is pre-processed and compatible with the organization's internal 
  formats.
"""


using Random, LinearAlgebra, Statistics, Distributions, StatsBase
using BAT, DensityInterface, IntervalSets


likelihood_all_samples_avg = let nObserved = ereco_data,
    energies = bin_edges,
    Mreco = responseMatrices,
    SSM = solarModel,
    MC_no_osc = unoscillatedSample,
    BG = backgrounds,
    f = propagateSamplesAvg
    Emin = E_threshold

    logfuncdensity(function (parameters)
        function poissonLogLikelihood(nExpected::Vector{Float64}, nMeasured::Vector{Float64})::Float64
            """
            Calculate the Poisson log likelihood given expected and measured counts.

            # Arguments
            - `nExpected::Vector{Float64}`: A vector of expected counts.
            - `nMeasured::Vector{Float64}`: A vector of measured counts.

            # Returns
            - `Float64`: The calculated log-likelihood.

            # Errors
            - Throws an error if any input is negative or if the vectors have different lengths.
            """
            if !all(x -> x >= 0, nExpected) || !all(x -> x >= 0, nMeasured)
                throw(ArgumentError("Inputs must be non-negative"))
            end
            if length(nExpected) != length(nMeasured)
                throw(ArgumentError("Inputs must have the same length"))
            end

            llh = 0.0
            @inbounds for i in eachindex(nExpected)
                e = nExpected[i]
                m = nMeasured[i]
                if m > 0
                    if e > 0
                        llh += (e - m + m * log(m / e)) 
                    elseif e == 0
                        e = 0.5
                        llh += (e - m + m * log(m / e)) 
                    end

                else
                    if e == 0
                        llh += 0
                    else
                        m = 0.5
                        llh += (e - m + m * log(m / e)) 
                    end
                end
            end
            return -llh
        end

        # Propagate MC
        expectedRate_ES_nue, expectedRate_ES_nuother, expectedRate_CC = f(MC_no_osc, Mreco, parameters, SSM, energies, backgrounds.CC)
        expectedRate_ES = expectedRate_ES_nue .+ expectedRate_ES_nuother

        # THIS SHOULD GO OUTSIDE EVENTUALLY
        # Find the first index where energy is greater than Emin.ES
        index_ES = findfirst(x -> x > Emin.ES, energies)

        # Check if the index was found
        if isnothing(index_ES)
            error("No energies greater than Emin found for ES.")
        end

        # Assuming index_CC should be the same as index_ES
        index_CC = index_ES

        # Check if indices were found
        if isnothing(index_ES)
            error("No energies greater than Emin found for ES.")
        end

        if isnothing(index_CC)
            error("No energies greater than Emin found for CC.")
        end

        loglh_ES = poissonLogLikelihood(expectedRate_ES[index_ES:end], nObserved.ES[index_ES:end])
        loglh_CC = poissonLogLikelihood(expectedRate_CC[index_CC:end], nObserved.CC[index_CC:end])

        loglh = loglh_ES + loglh_CC

        return loglh
    end)
end


likelihood_all_samples_ctr = let nObserved = ereco_data_mergedES,
    energies = bin_edges,
    Mreco = responseMatrices,
    SSM = solarModel,
    MC_no_osc = unoscillatedSample,
    BG = backgrounds,
    f = propagateSamplesCtr
    Emin = E_threshold

    logfuncdensity(function (parameters)
        function poissonLogLikelihood(nExpected::Vector{Float64}, nMeasured::Vector{Float64})::Float64
            """
            Calculate the Poisson log likelihood given expected and measured counts.

            # Arguments
            - `nExpected::Vector{Float64}`: A vector of expected counts.
            - `nMeasured::Vector{Float64}`: A vector of measured counts.

            # Returns
            - `Float64`: The calculated log-likelihood.

            # Errors
            - Throws an error if any input is negative or if the vectors have different lengths.
            """
            if !all(x -> x >= 0, nExpected) || !all(x -> x >= 0, nMeasured)
                throw(ArgumentError("Inputs must be non-negative"))
            end
            if length(nExpected) != length(nMeasured)
                throw(ArgumentError("Inputs must have the same length"))
            end

            llh = 0.0
            @inbounds for i in eachindex(nExpected)
                e = nExpected[i]
                m = nMeasured[i]
                if m > 0
                    if e > 0
                        llh += (e - m + m * log(m / e)) 
                    elseif e == 0
                        e = 0.5
                        llh += (e - m + m * log(m / e)) 
                    end

                else
                    if e == 0
                        llh += 0
                    else
                        m = 0.5
                        llh += (e - m + m * log(m / e)) 
                    end
                end
            end
            return -llh
        end

        # Propagate MC
        expectedRate_ES_nue, expectedRate_ES_nuother, expectedRate_CC = f(MC_no_osc, Mreco, parameters, SSM, energies, backgrounds.CC)
        expectedRate_ES = expectedRate_ES_nue .+ expectedRate_ES_nuother

        # THIS SHOULD GO OUTSIDE EVENTUALLY
        # Find the first index where energy is greater than Emin.ES
        index_ES = findfirst(x -> x > Emin.ES, energies)

        # Check if the index was found
        if isnothing(index_ES)
            error("No energies greater than Emin found for ES.")
        end

        # Assuming index_CC should be the same as index_ES
        index_CC = index_ES

        # Check if indices were found
        if isnothing(index_ES)
            error("No energies greater than Emin found for ES.")
        end

        if isnothing(index_CC)
            error("No energies greater than Emin found for CC.")
        end

        loglh_ES = poissonLogLikelihood(expectedRate_ES[index_ES:end], nObserved.ES[index_ES:end])
        loglh_CC = poissonLogLikelihood(expectedRate_CC[index_CC:end], nObserved.CC[index_CC:end])

        loglh = loglh_ES + loglh_CC

        return loglh_CC
    end)
end



# Measure the execution time of the poissonLogLikelihood function with positive expected and measured counts.
function test_log_likelihood_execution_time()
    nExpected = [10.0, 20.0, 30.0, 15.0, 25.0, 35.0, 12.0, 22.0, 32.0, 18.0, 28.0, 38.0, 14.0, 24.0, 34.0, 16.0, 26.0, 36.0, 19.0, 29.0]
    nMeasured = [12.0, 18.0, 29.0, 14.0, 26.0, 33.0, 11.0, 21.0, 31.0, 17.0, 27.0, 37.0, 13.0, 23.0, 33.0, 15.0, 25.0, 35.0, 18.0, 28.0]
    @time result = poissonLogLikelihood(nExpected, nMeasured)
    expected_result = sum((e - m) + ifelse(m > 0, e > 0 ? m * log1p(m / e - 1) : -Inf, 0) for (e, m) in zip(nExpected, nMeasured))
    @assert result ≈ expected_result
end

