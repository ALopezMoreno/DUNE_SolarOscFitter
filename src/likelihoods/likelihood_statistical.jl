function poissonLogLikelihood(nExpected::AbstractVector{<:Real}, nMeasured::AbstractVector{<:Real})
    if !all(x -> x >= 0, nExpected) || !all(x -> x >= 0, nMeasured)
        throw(ArgumentError("Inputs must be non-negative"))
    end
    if length(nExpected) != length(nMeasured)
        throw(ArgumentError("Inputs must have the same length"))
    end

    llh = zero(eltype(nExpected))
    ε = 1e-12  # Small constant to avoid log(0)
    @inbounds for i in eachindex(nExpected)
        e = nExpected[i]
        m = Float64(nMeasured[i])

        # Regularize e and m to avoid log(0)
        e_reg = max(e, ε)
        m_reg = max(m, ε)

        # Differentiable approximation of the original logic
        # For m > 0:
        #   if e > 0: e - m + m * log(m / e)
        #   if e == 0: 1e9 (penalty)
        # For m == 0:
        #   if e == 0: 0
        #   if e > 0: e
        #
        # We approximate the penalty for e == 0 and m > 0 as 1e9 * (1 - e / ε)^2
        # This is smooth and large when e ≈ 0, but differentiable.
        penalty = 1e9 * (1 - e / ε)^2  # Large when e ≈ 0, smooth everywhere

        # Differentiable expression for all cases
        term = ifelse(
            m > 0,
            ifelse(e > 0, e - m + m * (log(m_reg) - log(e_reg)), penalty),
            ifelse(e > 0, e, 0.0)
        )

        llh += term
    end
    return -llh
end


function perbin_poissonLogLikelihood(nExpected::AbstractArray,
                                     nMeasured::AbstractArray)
    size(nExpected) == size(nMeasured) ||
        throw(DimensionMismatch("Inputs must have the same size"))

    out = similar(nExpected)

    @inbounds for I in eachindex(nExpected)
        e = nExpected[I]
        m = Float64(nMeasured[I])

        if m > 0
            out[I] = -(e - m * log(max(e, 1e-300)))
        else
            out[I] = -e
        end
    end

    return out
end


function conditional_poissonLogLikelihood(nExpected::AbstractVector{<:Real}, nMeasured::AbstractVector{<:Real})
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