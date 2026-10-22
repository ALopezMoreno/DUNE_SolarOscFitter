function check_earth_norm_bounds(parameters)::Bool
    earth_norms = _params_by_prefix(parameters, Val(:earth_norm_))
    return all(x -> 0 ≤ x ≤ 2, earth_norms)
end

function print_negatives_1d(arr, parameters)
    @inbounds for i in eachindex(arr)
        x = arr[i]
        if x < 0
            @warn "Index $i: $x"
            @warn "Parameter values:"
            @show parameters
        end
    end
end

function print_negatives_2d(arr, parameters)
    @inbounds for j in axes(arr, 2), i in axes(arr, 1)
        x = arr[i, j]
        if x < 0
            @warn "Position ($i, $j): $x"
            @warn "Parameter values:"
            @show parameters
        end
    end
end
# Correlation term −½ Δxᵀ M Δx, M = C⁻¹ − diag(C)⁻¹, that turns the independent earth_norm_i priors
# into the full MvNormal(μ, C). Plain loops for Enzyme.
function earth_prior_correlation(parameters, M::AbstractMatrix, μ::AbstractVector)
    x = _params_by_prefix(parameters, Val(:earth_norm_))
    acc = zero(eltype(x))
    @inbounds for j in eachindex(μ), i in eachindex(μ)
        acc += (x[i] - μ[i]) * M[i, j] * (x[j] - μ[j])
    end
    return -acc / 2
end
