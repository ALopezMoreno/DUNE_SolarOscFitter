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
        expectedRate_ES_nue, expectedRate_ES_nuother, expectedRate_CC = f(MC_no_osc, Mreco, parameters, SSM, energies)

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

        loglh_ES_nue = poissonLogLikelihood(expectedRate_ES_nue[index_ES:end], nObserved.ES_nue[index_ES:end])
        loglh_ES_nuother = poissonLogLikelihood(expectedRate_ES_nuother[index_ES:end], nObserved.ES_nuother[index_ES:end])
        loglh_CC = poissonLogLikelihood(expectedRate_CC[index_CC:end], nObserved.CC[index_CC:end])

        loglh = loglh_ES_nue + loglh_ES_nuother + loglh_CC

        return loglh_CC
    end)
end


likelihood_all_samples_ctr = let nObserved = ereco_data,
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
        expectedRate_ES_nue, expectedRate_ES_nuother, expectedRate_CC = f(MC_no_osc, Mreco, parameters, SSM, energies)

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

        loglh_ES_nue = poissonLogLikelihood(expectedRate_ES_nue[index_ES:end], nObserved.ES_nue[index_ES:end])
        loglh_ES_nuother = poissonLogLikelihood(expectedRate_ES_nuother[index_ES:end], nObserved.ES_nuother[index_ES:end])
        loglh_CC = poissonLogLikelihood(expectedRate_CC[index_CC:end], nObserved.CC[index_CC:end])

        loglh = loglh_ES_nue + loglh_ES_nuother + loglh_CC

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

