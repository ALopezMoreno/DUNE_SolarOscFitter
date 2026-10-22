# Out of line: a logging macro's try/catch would box variables Enzyme rejects.
@noinline _warn_negative_bin(i, e) =
    (@warn "poissonLogLikelihood: bin $i negative within numerical noise (e=$e)"; nothing)

function poissonLogLikelihood(nExpected::AbstractVector{<:Real}, nMeasured::AbstractVector{<:Real})
    if length(nExpected) != length(nMeasured)
        throw(ArgumentError("Inputs must have the same length"))
    end

    llh = zero(eltype(nExpected))
    @inbounds for i in eachindex(nExpected)
        raw_e = nExpected[i]
        if raw_e < 0
            if raw_e > -1e-3
                _warn_negative_bin(i, raw_e)
            else
                throw(ArgumentError("poissonLogLikelihood: bin $i significantly negative (e=$raw_e)"))
            end
        end
        e = 0.5 * (raw_e + 1e-9 + sqrt((raw_e - 1e-9)^2 + 1e-18))
        m = Float64(nMeasured[i])
        m < 0 && throw(ArgumentError("poissonLogLikelihood: bin $i measured count negative (m=$m)"))

        term = ifelse(
            m > 0,
            e - m + m * (log(m) - log(e)),
            e
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
        m < 0 && throw(ArgumentError("perbin_poissonLogLikelihood: bin $I measured count negative (m=$m)"))

        if m > 0
            # same floor as poissonLogLikelihood, so per-bin values match the total
            e_safe = 0.5 * (e + 1e-9 + sqrt((e - 1e-9)^2 + 1e-18))
            out[I] = -(e - m * log(e_safe))
        else
            out[I] = -e
        end
    end

    return out
end


# Poisson log-likelihood with Barlow–Beeston treatment of the uncertainty of the expectation
function barlowBeestonLogLikelihood(nExpected, nMeasured, sigmaVar)
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
        end
        e = 0.5 * (e + 1e-9 + sqrt((e - 1e-9)^2 + 1e-18))
        
        if m > 0
            if e > 0
                if s == 0
                    llh += (e - m + m * log(m / e)) 
                else
                    beta = 0.5 * ( 1 - e*s^2 + sqrt( (e*s^2 - 1)^2 + 4*m*s^2 ) )
                    llh += beta * e - m + m * log(m / (beta*e)) + (beta - 1)^2 / (2*s^2) 
                end
            elseif e == 0
                # undefined at e = 0: half a count keeps the penalty finite (until a background floor exists)
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
                # undefined at m = 0: half a count, as above
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