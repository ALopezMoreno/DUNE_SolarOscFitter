using Random, LinearAlgebra, Statistics, Distributions, StatsBase
using BAT, DensityInterface, IntervalSets

include("../src/setup.jl")
include("../src/reweighting.jl")


likelihood = let dat = data, mc = monteCarlo, SSM = solarModel, f = oscReweight!

    nCC_measured = dat.events_cc_oscillated
    nES_nue_measured = dat.events_es_oscillated_nue
    nES_nuother_measured = dat.events_es_oscillated_other

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
                        llh += (e - m) + m * log(m / e)
                    else
                        llh += Inf
                    end
                else
                    llh += Inf
                end
            end
            return -llh
        end

        #Set weights
        f(mc, parameters)

        nCC_expected = mc.events_cc_oscillated
        nES_nue_expected = mc.events_es_oscillated_nue
        nES_nuother_expected = mc.events_es_oscillated_other

        loglh = poissonLogLikelihood(nCC_expected, nCC_measured)
        loglh += poissonLogLikelihood(nES_nue_expected, nES_nue_measured)
        loglh += poissonLogLikelihood(nES_nuother_expected, nES_nuother_measured)
        return loglh
    end)
end



# THIS WAS THE ORIGINAL FUNCTION
function poissonLogLikelihoodV1(nExpected::Vector{Float64}, nMeasured::Vector{Float64})::Float64
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
    llh = @inbounds sum((e - m) + ifelse(m > 0, e > 0 ? m * log1p(m / e - 1) : -Inf, 0) for (e, m) in zip(nExpected, nMeasured))
    return llh
end


function poissonLogLikelihoodV2(nExpected::Vector{Float64}, nMeasured::Vector{Float64})::Float64
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
                llh += (e - m) + m * log1p(m / e - 1)
            else
                llh += -Inf
            end
        else
            llh += e
        end
    end
    return llh
end

# Measure the execution time of the poissonLogLikelihood function with positive expected and measured counts.
function test_log_likelihood_execution_time()
    nExpected = [10.0, 20.0, 30.0, 15.0, 25.0, 35.0, 12.0, 22.0, 32.0, 18.0, 28.0, 38.0, 14.0, 24.0, 34.0, 16.0, 26.0, 36.0, 19.0, 29.0]
    nMeasured = [12.0, 18.0, 29.0, 14.0, 26.0, 33.0, 11.0, 21.0, 31.0, 17.0, 27.0, 37.0, 13.0, 23.0, 33.0, 15.0, 25.0, 35.0, 18.0, 28.0]
    @time result = poissonLogLikelihoodV2(nExpected, nMeasured)
    expected_result = sum((e - m) + ifelse(m > 0, e > 0 ? m * log1p(m / e - 1) : -Inf, 0) for (e, m) in zip(nExpected, nMeasured))
    @assert result ≈ expected_result
end

