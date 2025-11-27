function check_earth_norm_bounds(parameters)::Bool
    earth_norm_keys = filter(k -> startswith(String(k), "earth_norm"), keys(parameters))

    if !isempty(earth_norm_keys)
        earth_norms = [parameters[k] for k in earth_norm_keys]
        out_of_bounds = filter(x -> x < 0 || x > 2, earth_norms)
        if !isempty(out_of_bounds)
            @warn "Earth normalisation trying to leave bounds"
            return false
        end
    end

    return true
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