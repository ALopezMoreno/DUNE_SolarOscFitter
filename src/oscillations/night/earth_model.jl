#################################################################################
# Earth model
#################################################################################

"""
    EarthModel(file; potential_const, jump_threshold=LAYER_JUMP_FACTOR)

Radial matter potential [eV²/GeV] from a table of (r/R⊕, density, electron fraction).
Layers are numbered from 1 (outermost), as the `earth_norm_i` parameters.
"""
struct EarthModel
    radius::Vector{Float64}             # km, ascending
    potential_grid::Vector{Float64}     # potential at `radius`
    layer_boundaries::Vector{Float64}   # km, ascending
end

function EarthModel(file::AbstractString; potential_const::Real, jump_threshold::Real=LAYER_JUMP_FACTOR)
    data = readdlm(file)
    _validate_earth_table(data, file)
    radius = data[:, 1] .* EARTH_RADIUS_KM
    density = Float64.(data[:, 2])
    potential = density .* data[:, 3] .* potential_const

    steps = abs.(diff(density))
    jumps = findall(steps .> jump_threshold * median(steps))
    boundaries = [0.5 * (radius[j] + radius[j+1]) for j in jumps]

    # Enter each jump as a sharp step at the boundary
    eps = JUMP_HALF_WIDTH_KM
    r_out, v_out = Float64[], Float64[]
    jset = Set(jumps)
    for i in eachindex(radius)
        push!(r_out, radius[i]); push!(v_out, potential[i])
        if i in jset
            b = 0.5 * (radius[i] + radius[i+1])
            push!(r_out, b - eps, b + eps); push!(v_out, potential[i], potential[i+1])
        end
    end
    return EarthModel(r_out, v_out, sort(boundaries))
end

# The input-table contract of SPEC 8.1 (src/tabulated.jl's validate_table, which this module
# does not load): a numeric table of 3 columns and at least 2 rows, finite values, strictly
# increasing radii in [0, 1], density ≥ 0 and electron fraction ≥ 0.
function _validate_earth_table(data, file)
    w = "EarthModel ($file)"
    (data isa AbstractMatrix{<:Real} && size(data, 2) == 3) ||
        error("$w: expected a headerless numeric table of 3 columns (r/R⊕, density, electron fraction)")
    size(data, 1) >= 2 || error("$w: a table needs at least 2 points, it has $(size(data, 1))")
    i = findfirst(!isfinite, data)
    i === nothing || error("$w: non-finite value in row $(i[1]), column $(i[2])")
    r = data[:, 1]
    i = findfirst(k -> !(r[k+1] > r[k]), 1:length(r)-1)
    i === nothing || error("$w: radii must be strictly increasing (sorted, no duplicates); rows $i and $(i+1): $(r[i]), $(r[i+1])")
    (r[1] >= 0 && r[end] <= 1) || error("$w: radii must lie in [0, 1] (fractions of the Earth radius)")
    i = findfirst(<(0), data[:, 2:3])
    i === nothing || error("$w: negative $(i[2] == 1 ? "density" : "electron fraction") in row $(i[1])")
    return nothing
end

n_layers(earth::EarthModel) = length(earth.layer_boundaries) + 1

# Matter potential at radius r [km], linear interpolation clamped at the ends. Kept as is
# rather than ported to TabulatedInput: the jump steps entered above are load-bearing and this
# formula, y[i] + t (y[i+1] - y[i]), is not bit-identical to Interpolations.jl's. The clamp is
# not reached: chord radii lie in [0, r_det] with r_det < radius[end] (measured: 225–6370.9994 km on
# the shipped cos z grids, 2026-10-18).
function potential(earth::EarthModel, r::Real)
    x, y = earth.radius, earth.potential_grid
    r <= x[1] && return y[1]
    r >= x[end] && return y[end]
    i = searchsortedlast(x, r)
    t = (r - x[i]) / (x[i+1] - x[i])
    return y[i] + t * (y[i+1] - y[i])
end

#################################################################################
# Chords
#################################################################################

# One step of a chord, inside a single layer
struct PathSegment
    layer::Int              # 1 = outermost
    length::Float64         # km
    avg_potential::Float64  # eV²/GeV
    slope::Float64          # eV²/GeV/km, along the direction of travel
end

chord_radius(l, r_det, cosz) = sqrt(r_det^2 + l^2 + 2r_det * cosz * l)

"""
    chord_length(r_det, cosz, r_surface=EARTH_RADIUS_KM)

Chord length [km] from a detector at radius `r_det` to the sphere of radius `r_surface`.
"""
function chord_length(r_det, cosz, r_surface=EARTH_RADIUS_KM)
    disc = r_surface^2 - r_det^2 * (1 - cosz^2)
    return -r_det * cosz + sqrt(max(disc, 0.0))
end

function _layer_crossings(r_det, cosz, r_layer, l_max)
    disc = r_det^2 * cosz^2 - r_det^2 + r_layer^2
    disc < 0 && return Float64[]
    root = sqrt(disc)
    return sort([l for l in (-r_det * cosz - root, -r_det * cosz + root) if 0 < l < l_max])
end

"""
    build_earth_path(cosz, earth; depth_km=DEFAULT_DEPTH_KM, max_step_km=nothing, n_quad=SEGMENT_GL_NODES, steps_per_layer=nothing)

Chord to a detector at `depth_km`, as `PathSegment`s ordered from the detector outward.
It is split at each layer boundary and, optionally, into steps of at most `max_step_km`
(or `steps_per_layer[layer]` steps per crossing).
"""
function build_earth_path(cosz::Real, earth::EarthModel; depth_km=DEFAULT_DEPTH_KM, max_step_km=nothing,
                          n_quad::Int=SEGMENT_GL_NODES, steps_per_layer=nothing)
    cosz >= 0 && return PathSegment[]
    r_det = EARTH_RADIUS_KM - depth_km
    l_max = chord_length(r_det, cosz)

    edges = Set([0.0, l_max])
    for r_layer in earth.layer_boundaries
        union!(edges, _layer_crossings(r_det, cosz, r_layer, l_max))
    end
    edges = sort(collect(edges))

    gl_x, gl_w = gauss(n_quad)
    nb = length(earth.layer_boundaries)
    segments = PathSegment[]
    for (lo, hi) in zip(edges[1:end-1], edges[2:end])
        hi <= lo && continue
        # Both crossings of a shell get the same layer index
        r_mid = chord_radius(0.5 * (lo + hi), r_det, cosz)
        layer = nb - (searchsortedfirst(earth.layer_boundaries, r_mid) - 1) + 1
        n_steps = if steps_per_layer !== nothing
            steps_per_layer[layer]
        elseif max_step_km === nothing
            1
        else
            ceil(Int, (hi - lo) / max_step_km)
        end
        cuts = range(lo, hi; length=n_steps + 1)
        for (a, b) in zip(cuts[1:end-1], cuts[2:end])
            len = b - a
            l_nodes = 0.5 * len .* gl_x .+ 0.5 * (a + b)
            w = 0.5 * len .* gl_w
            v = [potential(earth, chord_radius(l, r_det, cosz)) for l in l_nodes]
            avg = sum(w .* v) / len
            # Neutrinos travel towards the detector at l = 0
            slope = -12.0 * sum(w .* (l_nodes .- 0.5 * (a + b)) .* v) / len^3
            push!(segments, PathSegment(layer, len, avg, slope))
        end
    end
    return segments
end

# Paths as zero-padded (n_paths, max_steps) arrays, ordered from Earth entry to the detector
function _stack_steps(paths::Vector{Vector{PathSegment}})
    max_steps = maximum(length, paths; init=0)
    n = length(paths)
    L, V, S = zeros(n, max_steps), zeros(n, max_steps), zeros(n, max_steps)
    layer = ones(Int, n, max_steps)
    for (ip, path) in enumerate(paths)
        for (s, seg) in enumerate(reverse(path))
            L[ip, s], V[ip, s], S[ip, s], layer[ip, s] = seg.length, seg.avg_potential, seg.slope, seg.layer
        end
    end
    return L, V, S, layer
end
