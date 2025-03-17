################################################################################
# Global Constants and Imports
################################################################################

using Statistics
using StaticArrays
using Interpolations

global EARTH_RADIUS = 6371
global RESOLUTION = 12000

################################################################################
# Struct Definitions
################################################################################

struct Jump
    x::Float64
    b::Float64
    a::Float64
end

struct Segment
    start::Float64
    finish::Float64
    nodes::Vector{Float64}
    values::Vector{Float64}
end

struct Path
    jumps::Vector{Jump}
    segments::Vector{Segment}
end

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
    distances, matter_potential, jumps = get_potential(cosz, earth_potential_function)
    jump_positions = [jump.x for jump in jumps]

    # create interpolated function for finding optimal integration points
    f = LinearInterpolation(distances, matter_potential; extrapolation_bc=Flat())

    # find optimal integration points for each interval between jumps
    segments = []

    @views for i in 1:(length(jump_positions)-1)
        a = jump_positions[i]
        b = jump_positions[i+1]

        # nodes = optimal_integration_nodes(f, a, b) #NOT SURE THIS IS WORKING
        nodes = collect(range(a, stop=b, length=n + 2))
        values = f(nodes)

        # set the pre and post jump values
        values[1]   = jumps[i].b  
        values[end] = jumps[i+1].a


        push!(segments, Segment(a, b, nodes, values))
    end

    return Path(jumps, segments)
end