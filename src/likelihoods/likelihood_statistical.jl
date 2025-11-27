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


function barlowBeestonLogLikelihood(nExpected, nMeasured, sigmaVar)
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
    if length(nExpected) != length(nMeasured) || length(nExpected) != length(sigmaVar)
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