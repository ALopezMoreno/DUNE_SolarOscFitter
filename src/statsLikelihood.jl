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


likelihood_all_samples = let nObserved = ereco_data,
    energies = bin_edges,
    Mreco = responseMatrices,
    SSM = solarModel,
    MC_no_osc = unoscillatedSample,
    BG = backgrounds,
    f = propagateSamples

    logfuncdensity(function (parameters)
        # first initialise llh to zero
        loglh = 0.

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

            if llh < 0
                throw(ArgumentError("Seems like we got the sign wrong!"))
            end
                
            return -llh
        end

        # LLH FOR BARLOW BEESTON
        function systematicLogLikelihood(nExpected, nMeasured, sigmaVar)
        """
            Calculate the Poisson log likelihood given expected and measured counts with uncertainties in the expectation.

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
                s = sigmaVar[i]
                if e < 0
                    println("model predicted negative event rate: ", e)
                    e = 0
                end
                
                if m > 0
                    if e > 0
                        if s == 0
                            llh += (e - m + m * log(m / e)) 
                        else
                            beta = 0.5 * ( 1 - e*s^2 + sqrt( (e*s^2 - 1)^2 + 4*m*s^2 ) )
                            llh += beta * e - m + m * log(m / (beta*e)) + (beta - 1)^2 / (2*s^2) 
                        end
                    elseif e == 0
                        e = 0.5
                        if s == 0
                            llh += (e - m + m * log(m / e)) 
                        else
                            beta = 0.5 * ( 1 - e*s^2 + sqrt( (e*s^2 - 1)^2 + 4*m*s^2 ) )
                            llh += beta * e - m + m * log(m / (beta*e)) + (beta - 1)^2 / (2*s^2) 
                        end
                    end

                else
                    if e == 0
                        llh += 0
                    else
                        m = 0.5
                        if s == 0
                            llh += (e - m + m * log(m / e)) 
                        else
                            beta = 0.5 * ( 1 - e*s^2 + sqrt( (e*s^2 - 1)^2 + 4*m*s^2 ) )
                            llh += beta * e - m + m * log(m / (beta*e)) + (beta - 1)^2 / (2*s^2) 
                        end
                    end
                end
            end
            return -llh
        end

        # Ensure that the earth normalisation parametrs are within bounds [0, 2]
        earth_norm_keys = filter(k -> startswith(String(k), "earth_norm"), keys(parameters))

        # Only proceed if there is at least one such parameter
        if !isempty(earth_norm_keys)
            # Gather all values into a vector
            earth_norms = [parameters[k] for k in earth_norm_keys]
            # Find out-of-bounds values
            out_of_bounds = filter(x -> x < 0 || x > 2, earth_norms)
            if !isempty(out_of_bounds)
                @warn "Earth normalisation trying to leave bounds"
                return -Inf
            end
        end

        # Propagate MC
        expectedRate_ES_day, expectedRate_CC_day, expectedRate_ES_night, expectedRate_CC_night, BG_ES_tot, BG_CC_tot = f(MC_no_osc, Mreco, parameters, SSM, energies, backgrounds)
        

        # loglh_ES_day = poissonLogLikelihood(expectedRate_ES_day[index_ES:end], nObserved.ES_day[index_ES:end])

        ## CHECK FOR NEGATIVE VALUES
        ## WE WILL HAVE TO REMOVE THIS EVENTUALLY
        function print_negatives_1d(arr, parameters)
            @inbounds for i in eachindex(arr)
                x = arr[i]
                x < 0 && @warn ("Index $i: $x")
                x < 0 && @warn ("Parameter values:")
                x < 0 && @show (parameters)
            end
        end

        function print_negatives_2d(arr, parameters)
            @inbounds for j in axes(arr, 2), i in axes(arr, 1)
                x = arr[i, j]
                x < 0 && @warn ("Position ($i, $j): $x")
                x < 0 && @warn ("Parameter values:")
                x < 0 && @show (parameters)
            end
        end

        print_negatives_1d(expectedRate_CC_day, parameters)
        print_negatives_2d(expectedRate_CC_night, parameters)


        if ES_mode
            loglh_ES_day = poissonLogLikelihood(expectedRate_ES_day[index_ES:end], nObserved.ES_day[index_ES:end])
            loglh_ES_night = sum([poissonLogLikelihood(row[index_ES:end], obs_row[index_ES:end])
            for (row, obs_row) in zip(eachrow(expectedRate_ES_night), eachrow(nObserved.ES_night))])

            loglh += loglh_ES_day + loglh_ES_night
        end

        if CC_mode
            loglh_CC_day = poissonLogLikelihood(expectedRate_CC_day[index_CC:end], nObserved.CC_day[index_CC:end])
            loglh_CC_night = sum([poissonLogLikelihood(row[index_CC:end], obs_row[index_CC:end])
            for (row, obs_row) in zip(eachrow(expectedRate_CC_night), eachrow(nObserved.CC_night))])

            loglh += loglh_CC_day + loglh_CC_night
        end

        # EXAMPLE FOR BARLOW BEESTON
        # loglh_CC_night = sum([systematicLogLikelihood(row[index_CC:end], obs_row[index_CC:end], sys_row[index_CC:end])
        # for (row, obs_row, sys_row) in zip(eachrow(expectedRate_CC_night), eachrow(nObserved.CC_night), eachrow(uncertainty_ratio_matrix_CC_night))])
        return loglh
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

