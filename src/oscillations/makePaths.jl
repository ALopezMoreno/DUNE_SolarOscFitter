using Statistics
using StaticArrays
include(joinpath(@__DIR__, "..", "tabulated.jl"))
using QuadGK: gauss

global EARTH_RADIUS = 6371  # km
global RESOLUTION = 12000   # points along a path

# Represents a discontinuous jump in Earth's density profile
struct Jump
    x::Float64      # Position along path where jump occurs (km)
    b::Float64      # Density value before the jump (g/cm³)
    a::Float64      # Density value after the jump (g/cm³)
end

# Represents a continuous segment of the neutrino path through Earth
struct Segment
    index::Int16                # Layer index for systematic uncertainties
    start::Float64              # Starting position along path (km)
    finish::Float64             # Ending position along path (km)
    nodes::Vector{Float64}      # Integration nodes within segment (km)
    values::Vector{Float64}     # Density values at nodes (g/cm³)
    length::Float64             # Segment length (km)
    avgRho::Float64             # Average density in segment (g/cm³)
end

# Complete neutrino path through Earth with jumps and segments
struct Path
    jumps::Vector{Jump}         # Density discontinuities along path
    segments::Vector{Segment}   # Continuous path segments for integration
end

global depth = 0.001  # detector depth below sea level (km)

function find_jumps(x, f; threshold_factor::Float64=5.0)
    if length(x) != length(f)
        throw(ArgumentError("x and f must have the same length"))
    end
    if length(x) < 1
        throw(ArgumentError("x must have more than 1 point"))
    end

    diff_f = diff(f)
    abs_diff = abs.(diff_f)
    median_diff = median(abs_diff)
    threshold = threshold_factor * median_diff

    # indices into diff(f) where the step exceeds the threshold
    jump_indices = findall(x_val -> x_val > threshold, abs_diff)

    if isempty(jump_indices)
        return [Dict("x" => x[1], "b" => 0, "a" => f[end]),
            Dict("x" => x[end], "b" => f[1], "a" => f[1])]
    end

    groups = []
    current_group = [jump_indices[1]]
    for idx in jump_indices[2:end]
        if idx == current_group[end] + 1
            push!(current_group, idx)
        else
            push!(groups, copy(current_group))
            current_group = [idx]
        end
    end
    push!(groups, current_group)

    jumps = [Dict("x" => x[1], "b" => 0, "a" => f[end])]

    for group in groups
        start_idx, end_idx = group[1], group[end]   # the jump lies between f[end_idx] and f[end_idx+1]
        x_start = x[start_idx]
        x_end = x[end_idx+1]
        x_pos = (x_start + x_end) / 2
        before = f[start_idx]
        after = f[end_idx+1]
        push!(jumps, Dict("x" => x_pos, "b" => before, "a" => after))
    end

    push!(jumps, Dict("x" => x[end], "b" => jumps[end]["a"], "a" => jumps[end]["a"]))
    return jumps
end

function get_track_length(cosz)
    total_length = sqrt.((EARTH_RADIUS - depth) .^ 2 .* cosz .^ 2 .+ 2 * EARTH_RADIUS * depth .- depth^2) .- (EARTH_RADIUS - depth) .* cosz
    return total_length
end

function get_track(cosz)
    # Straight track from the detector, y = y_det + x cot z (vertical when sin z = 0)
    track_length = get_track_length(cosz)

    if cosz != -1
        sinz = sqrt(1 .- cosz .^ 2)
        cotz = cosz / sinz

        x_end = track_length / sqrt(1 .+ cotz .^ 2)

        x_coords = collect(range(0, stop=x_end, length=RESOLUTION))

        y_coords = (EARTH_RADIUS - depth) .+ x_coords .* cotz

        distances = x_coords * sqrt(1 .+ cotz^2)

    else
        x_coords = zeros(RESOLUTION)
        y_coords = collect(range(EARTH_RADIUS - depth, stop=-EARTH_RADIUS, length=RESOLUTION))
        distances = EARTH_RADIUS - depth .- y_coords
    end

    r_x = sqrt.(x_coords .^ 2 + y_coords .^ 2)
    return r_x, distances
end


function get_potential(cosz, earth_potential_function)
    track_depths, distances = get_track(cosz)
    xvals = collect(range(0, stop=EARTH_RADIUS, length=RESOLUTION))
    jumps = find_jumps(xvals, earth_potential_function(xvals))
    matter_potential = earth_potential_function(track_depths)
    track_jumps = []

    for k in 1:length(jumps) - 1
        jump = jumps[k]
        jump_depth = jump["x"]
        for i in 1:length(track_depths)-1
            if (track_depths[i] <= jump_depth && track_depths[i+1] >= jump_depth)
                t = (jump_depth - track_depths[i]) / (track_depths[i+1] - track_depths[i])
                intersection_distance = distances[i] + t * (distances[i+1] - distances[i])
                push!(track_jumps, Jump(intersection_distance, jump["a"], jump["b"]))
            elseif (track_depths[i] >= jump_depth && track_depths[i+1] <= jump_depth)
                t = (jump_depth - track_depths[i]) / (track_depths[i+1] - track_depths[i])
                intersection_distance = distances[i] + t * (distances[i+1] - distances[i])
                push!(track_jumps, Jump(intersection_distance, jump["b"], jump["a"]))
            end
        end
    end
    push!(track_jumps, Jump(0.0, matter_potential[1], matter_potential[1]))
    push!(track_jumps, Jump(distances[end], 0.0, matter_potential[RESOLUTION]))
    sort!(track_jumps, by=x -> x.x)

    return distances, matter_potential, track_jumps
end


function make_potential_for_integrand(cosz, earth_potential_function, n=3)
    # Layers from the vertical path: the first half of its jumps (rounded up)
    _, _, ref_jumps = get_potential(-1, earth_potential_function)

    half_count = ceil(Int, length(ref_jumps) / 2)
    collected_jumps = ref_jumps[1:half_count]
    
    # Each layer is (previous jump, following jump); the innermost reflects its jump
    layers = Vector{Tuple{typeof(collected_jumps[1]), typeof(collected_jumps[1])}}(undef, length(collected_jumps))
    for i in 1:length(collected_jumps)-1
        layers[i] = (collected_jumps[i], collected_jumps[i+1])
    end
    layers[end] = (collected_jumps[end], collected_jumps[end])

    if !isnothing(earth_normalisation_true) && length(layers) != length(earth_normalisation_true)
        error("Mismatch in length: layers has ", length(layers), " elements, but earth_normalisation_true has ", length(earth_normalisation_true), " elements. Check your earth matter potential uncertainties prior covariance file")
    end

    distances, matter_potential, jumps = get_potential(cosz, earth_potential_function)
    jump_positions = [jump.x for jump in jumps]

    # Potential along the path; the Gauss–Legendre nodes lie strictly inside it (not registered for
    # the setup log: one table per path).
    f = TabulatedInput(distances, matter_potential; name = "Earth potential along a path (cos z = $cosz)",
                       units = "km", extrapolation = :error, nonnegative = true, register = false)

    segments = []

    @views for i in 1:(length(jump_positions)-1)
        a = jump_positions[i]
        b = jump_positions[i+1]

        # Segment-average density by Gauss–Legendre (interior nodes never land on a jump)
        nodes, gl_weights = gauss(n + 2, a, b)
        values = f(nodes)
        avgRho = sum(gl_weights .* values) / (b - a)

        # Match the segment's boundary densities (left jump's b, right jump's a) to a layer
        candidate = sort([jumps[i].b, jumps[i+1].a])
        
        seg_index = 0
        for (layer_idx, layer) in enumerate(layers)
            if isapprox(candidate[1], candidate[2]; atol=1e-7)
                if isapprox(candidate[1], layer[1].b; atol=1e-7)
                    seg_index = layer_idx
                    break
                end
            else
                layer_candidate = sort([layer[1].b, layer[2].a])
                if isapprox(candidate[1], layer_candidate[1]; atol=1e-7) && isapprox(candidate[2], layer_candidate[2]; atol=1e-10)
                    seg_index = layer_idx
                    break
                end
            end
        end

        if seg_index == 0
            layer_candidates = ""
            for (layer_idx, layer) in enumerate(layers)
                if isapprox(layer[1].b, layer[2].a; atol=1e-10)
                    candidate_layer = string(layer[1].b)
                else
                    candidate_layer = string(([layer[1].b, layer[2].a]))
                end
                layer_candidates *= "Layer " * string(layer_idx) * ": " * candidate_layer * "\n"
            end
            error("No matching layer found for segment ", i, " with candidate jumps: ", string(candidate), "\nLayer candidates:\n", layer_candidates)
        end

        push!(segments, Segment(seg_index, a, b, nodes, values, b - a, avgRho))
    end

    return Path(jumps, segments)
end

function get_avg_densities(path_list)
    tmp = []
    for path in path_list
        for seg in path.segments
            layer_info = [seg.index, seg.avgRho]
            push!(tmp, layer_info)
        end
    end

    data_tuples = [(item[1], item[2]) for item in tmp]

    sorted_data = sort(data_tuples, by=first)

    indices = Int.([item[1] for item in sorted_data])
    densities = [item[2] for item in sorted_data]

    unique_indices = sort(unique(indices))
    
    starts = [findfirst(isequal(idx), indices) for idx in unique_indices]
    ends = [findlast(isequal(idx), indices) for idx in unique_indices]

    # Average density per layer index
    average_densities = [sum(densities[starts[idx]:ends[idx]]) / (ends[idx] - starts[idx] + 1) for idx in unique_indices]
    return average_densities
end