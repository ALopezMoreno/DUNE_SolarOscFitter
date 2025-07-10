#=
makePaths.jl

Neutrino path generation through Earth for matter effect calculations.
This module creates detailed paths for neutrinos traveling through Earth's
varying density profile, accounting for different zenith angles.

Key Features:
- Earth geometry calculations for neutrino trajectories
- Density profile segmentation and layer identification
- Jump detection for discontinuous density changes
- Path optimization for numerical integration
- Support for systematic uncertainties in Earth matter density

The paths are used in oscillation calculations to account for matter effects
as neutrinos propagate through Earth's layers with different densities.

Author: [Author name]
=#

################################################################################
# Global Constants and Imports
################################################################################

using Statistics      # For statistical calculations
using StaticArrays    # For efficient static arrays
using Interpolations  # For density profile interpolation

# Earth model parameters
global EARTH_RADIUS = 6371  # Earth radius in km
global RESOLUTION = 12000   # Number of points along neutrino path

################################################################################
# Struct Definitions
################################################################################

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

# Detector depth below sea level (km) - essentially at surface
global depth = 0.001 # basically sea level

################################################################################
# Track Computation Functions
################################################################################

function find_jumps(x, f; threshold_factor::Float64=5.0)
    # Convert inputs to arrays and check length consistency.
    if length(x) != length(f)
        throw(ArgumentError("x and f must have the same length"))
    end
    if length(x) < 1
        throw(ArgumentError("x must have more than 1 point"))
    end

    # Compute differences and absolute differences between consecutive f values.
    diff_f = diff(f)
    abs_diff = abs.(diff_f)
    median_diff = median(abs_diff)
    threshold = threshold_factor * median_diff

    # Find indices where the jump coressponding to diff exceeds the threshold.
    # Note: jump_indices are with respect to the diff array (1-indexed, covering elements 1 to length(f)-1).
    jump_indices = findall(x_val -> x_val > threshold, abs_diff)

    # If there are no jumps, return a default set of dictionaries.
    if isempty(jump_indices)
        return [Dict("x" => x[1], "b" => 0, "a" => f[end]),
            Dict("x" => x[end], "b" => f[1], "a" => f[1])]
    end

    # Group consecutive indices.
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

    # Initialize the jumps list with an initial jump dictionary.
    jumps = [Dict("x" => x[1], "b" => 0, "a" => f[end])]

    # Process each group to determine the jump position and corresponding function values.
    for group in groups
        start_idx = group[1]      # jump from f[start_idx] -> f[start_idx+1]
        end_idx = group[end]    # last index in this group
        # The jump happens between f[end_idx] and f[end_idx+1]
        x_start = x[start_idx]
        x_end = x[end_idx+1]
        x_pos = (x_start + x_end) / 2
        before = f[start_idx]
        after = f[end_idx+1]
        push!(jumps, Dict("x" => x_pos, "b" => before, "a" => after))
    end

    # Append a final dictionary entry using the x[end] value,
    # with "b" and "a" set to the last jump's output value.
    push!(jumps, Dict("x" => x[end], "b" => jumps[end]["a"], "a" => jumps[end]["a"]))
    return jumps
end

function get_track_length(cosz)
    total_length = sqrt.((EARTH_RADIUS - depth) .^ 2 .* cosz .^ 2 .+ 2 * EARTH_RADIUS * depth .- depth^2) .- (EARTH_RADIUS - depth) .* cosz
    return total_length
end

function get_track(cosz)
    # Track begins (ends) at detector and stops when it reaches its path length.
    # Track equation is therefore y = y_detector + x*cot(z) if sin(z) =/= 0. Otherwise it is just a vertical trajectory
    track_length = get_track_length(cosz)

    if cosz != -1
        # Calculate track slope
        sinz = sqrt(1 .- cosz .^ 2)
        cotz = cosz / sinz

        # x-position when the track reaches the surface
        x_end = track_length / sqrt(1 .+ cotz .^ 2)

        # divide x into the resolution and log the points.
        x_coords = collect(range(0, stop=x_end, length=RESOLUTION))

        # calculate the corresponding y-coordinates
        y_coords = (EARTH_RADIUS - depth) .+ x_coords .* cotz

        # calculate distances
        distances = x_coords * sqrt(1 .+ cotz^2)

    else
        # Track is just a vertical line
        x_coords = zeros(RESOLUTION)
        y_coords = collect(range(EARTH_RADIUS - depth, stop=-EARTH_RADIUS, length=RESOLUTION))
        distances = EARTH_RADIUS - depth .- y_coords
    end

    # We now get the distances to the centre of the earth at every point in the track
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
                # Linear interpolation to find the exact intersection distance
                t = (jump_depth - track_depths[i]) / (track_depths[i+1] - track_depths[i])
                intersection_distance = distances[i] + t * (distances[i+1] - distances[i])
                push!(track_jumps, Jump(intersection_distance, jump["a"], jump["b"]))
            elseif (track_depths[i] >= jump_depth && track_depths[i+1] <= jump_depth)
                # Linear interpolation to find the exact intersection distance
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
    # generate a vertical (reference) path to identify layers
    _, _, ref_jumps = get_potential(-1, earth_potential_function)

    # generate a list of which jump values correspond to each layer
    # Take the first half of the ref_jumps (rounded up)
    half_count = ceil(Int, length(ref_jumps) / 2)
    collected_jumps = ref_jumps[1:half_count]
    
    # Generate a layers array where each element is a tuple of (previous_jump, following_jump)
    layers = Vector{Tuple{typeof(collected_jumps[1]), typeof(collected_jumps[1])}}(undef, length(collected_jumps))
    for i in 1:length(collected_jumps)-1
        layers[i] = (collected_jumps[i], collected_jumps[i+1])
    end
    # For the final layer (near the core) we reflect the final jump
    layers[end] = (collected_jumps[end], collected_jumps[end])

    # Check if the number of layers is equal to the length of the earth normalisation vector
    if length(layers) != length(earth_normalisation_true)
        error("Mismatch in length: layers has ", length(layers), " elements, but earth_normalisation_true has ", length(earth_normalisation_true), " elements. Check your earth matter potential uncertainties prior covariance file")
    end

    distances, matter_potential, jumps = get_potential(cosz, earth_potential_function)
    jump_positions = [jump.x for jump in jumps]

    # create interpolated function for finding optimal integration points
    f = LinearInterpolation(distances, matter_potential; extrapolation_bc=Flat())

    segments = []

    @views for i in 1:(length(jump_positions)-1)
        a = jump_positions[i]
        b = jump_positions[i+1]

        # nodes = optimal_integration_nodes(f, a, b) # NOT SURE THIS IS WORKING
        nodes = collect(range(a, stop=b, length=n + 2))
        values = f(nodes)

        # set the pre and post jump values
        values[1]   = jumps[i].b  
        values[end] = jumps[i+1].a

        # Create a candidate pair from the segment boundaries.
        # Using the left jump's "b" and the right jump's "a".
        candidate = sort([jumps[i].b, jumps[i+1].a])
        
        # Find a matching layer in the layers array.
        seg_index = 0  # default value if no matching layer is found
        for (layer_idx, layer) in enumerate(layers)
            if isapprox(candidate[1], candidate[2]; atol=1e-7)
                # Only search for a match on the beginning value (left jump's b)
                if isapprox(candidate[1], layer[1].b; atol=1e-7)
                    seg_index = layer_idx
                    break
                end
            else
                # For the layer, consider its two jump values:
                # we compare the left jump's b and the right jump's a.
                layer_candidate = sort([layer[1].b, layer[2].a])
                if isapprox(candidate[1], layer_candidate[1]; atol=1e-7) && isapprox(candidate[2], layer_candidate[2]; atol=1e-10)
                    seg_index = layer_idx
                    break
                end
            end
        end

        if seg_index == 0
            # Prepare error message with all layer candidate pairs.
            layer_candidates = ""
            for (layer_idx, layer) in enumerate(layers)
                # Determine candidate for each layer. If both jump values are nearly identical,
                # only include one value; else, include both values sorted.
                if isapprox(layer[1].b, layer[2].a; atol=1e-10)
                    candidate_layer = string(layer[1].b)
                else
                    candidate_layer = string(([layer[1].b, layer[2].a]))
                end
                layer_candidates *= "Layer " * string(layer_idx) * ": " * candidate_layer * "\n"
            end
            error("No matching layer found for segment ", i, " with candidate jumps: ", string(candidate), "\nLayer candidates:\n", layer_candidates)
        end

        push!(segments, Segment(seg_index, a, b, nodes, values, b - a, mean(values)))
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

    # Sort the vector of tuples based on the 'idx' (the first element of the tuple)
    sorted_data = sort(data_tuples, by=first)

    # Extract the 'idx' and 'density' values from the sorted tuples
    indices = Int.([item[1] for item in sorted_data])
    densities = [item[2] for item in sorted_data]

    # Find the unique indices and the start/end points of their occurrences
    unique_indices = sort(unique(indices))
    
    starts = [findfirst(isequal(idx), indices) for idx in unique_indices]
    ends = [findlast(isequal(idx), indices) for idx in unique_indices]

    # Calculate the average density for each unique index and structure the output
    average_densities = [sum(densities[starts[idx]:ends[idx]]) / (ends[idx] - starts[idx] + 1) for idx in unique_indices]
    return average_densities
end