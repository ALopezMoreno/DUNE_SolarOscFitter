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