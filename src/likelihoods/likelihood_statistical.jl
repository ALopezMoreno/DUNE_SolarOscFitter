function poissonLogLikelihood(nExpected::Vector{Float64}, nMeasured::Vector{Float64})::Float64
    """
    Calculate the Poisson log likelihood given expected and measured counts (seen as a KL divergence)

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
                llh += 1e9
            end

        elseif m == 0
            if e == 0
                llh += 0
            elseif e > 0
                llh += e 
            end
        end
    end
        
    return -llh
end


function perbin_poissonLogLikelihood(nExpected::AbstractArray,
                                     nMeasured::AbstractArray)
    size(nExpected) == size(nMeasured) ||
        throw(DimensionMismatch("Inputs must have the same size"))

    out = Array{Float64}(undef, size(nExpected))

    @inbounds for I in eachindex(nExpected)
        e = Float64(nExpected[I])
        m = Float64(nMeasured[I])

        if m > 0
            if e > 0
                out[I] = - (e - m + m * log(m / e))
            else
                out[I] = -1e9
            end
        else
            if e == 0
                out[I] = 0.0
            else
                out[I] = -e
            end
        end
    end

    return out
end


function conditional_poissonLogLikelihood(nExpected::AbstractVector{<:Float64}, nMeasured::AbstractVector{<:Float64})::Float64
    """
    conditional_poissonLogLikelihood(nExpected, nMeasured) -> Float64

    Compute the conditional log-likelihood contribution for P(c | b) in a single
    b-bin, using the multinomial (shape-only) likelihood induced by a Poisson model.

    Given expected bin contents `nExpected[j] = μ_j` and observed counts
    `nMeasured[j] = n_j` for categories c within a fixed b-bin, we define:

        p_j = μ_j / sum(μ_j)          # model conditional probabilities P(c=j | b)
        q_j = n_j / sum(n_j)          # empirical conditional probabilities

    The conditional deviance (i.e. multinomial KL divergence) is

        D_cond = 2 * sum_j n_j * log(q_j / p_j).

    This function returns

        llh = -0.5 * D_cond

    which is equal to the true conditional log-likelihood

        log L_cond = sum_j n_j * log(p_j)

    up to an additive constant that depends only on the data.  
    Such constants cancel in Metropolis–Hastings and do not affect the posterior.

    Behavior:
    - Bins with n_j == 0 contribute zero.
    - If n_j > 0 but μ_j == 0, a large negative penalty is added.
    - If sum(nExpected) == 0:
        - returns 0.0 if also sum(nMeasured) == 0,
        - otherwise returns a large negative penalty.

    This function is meant to be *added* to the log-likelihood for P(a,b)
    to form the total joint log-likelihood for a factorized model P(a,b)*P(c|b).
    """

    if !all(x -> x >= 0, nExpected) || !all(x -> x >= 0, nMeasured)
        throw(ArgumentError("Inputs must be non-negative"))
    end
    if length(nExpected) != length(nMeasured)
        throw(ArgumentError("Inputs must have the same length"))
    end

    row_sum_expected = sum(nExpected)
    row_sum_measured = sum(nMeasured)

    if row_sum_expected < 0
        throw(ArgumentError("Row sum of expected counts must be non-negative for a conditional likelihood"))
    elseif row_sum_expected == 0
        return 0.0  # The condition on the probability is not met -- no likelihood contribution
    end

    llh = 0.0
    @inbounds for i in eachindex(nExpected)
        μ = nExpected[i]
        n = nMeasured[i]
        if n > 0
            if μ > 0
                # n * log( μ / sum μ )
                # llh += n * (log(μ) - log(row_sum_expected))
                llh -= n * (log(n / row_sum_measured) - log(μ / row_sum_expected))
            else
                # model assigns zero intensity where we observed counts
                llh += -1e9 
            end
        end
        # if n == 0, this bin contributes only a constant term we can ignore
    end

    return llh
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