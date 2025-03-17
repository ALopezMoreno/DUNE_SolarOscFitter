################################################################################
# Global Constants and Imports
################################################################################

using Statistics
using Interpolations
using Trapz

include("makePaths.jl")

################################################################################
# Struct Definitions
################################################################################

struct oscPars
    dm21::Float64
    th12::Float64
    th13::Float64
end

################################################################################
# Calculation of the oscillation probability
################################################################################

function get_matter_angle(potential_val::Vector{<:Real}, mixingPars::oscPars, E::Vector{<:Real})
    # Extract mixing parameters
    dm21 = mixingPars.dm21
    th12 = mixingPars.th12
    th13 = mixingPars.th13

    # Calculate the numerator (scalar)
    numerator = sin(2 * th12) * cos(th13)

    # Create the E_potential_matrix using the outer product (E is a column vector, potential_val' is a row vector)
    E_potential_matrix = E * transpose(potential_val)

    # Calculate the denominator for each element with element-wise operations
    # (cos(2*th12) is a scalar, and E_potential_matrix/dm21 scales the matrix)
    denominator = sqrt.((cos(2 * th12) .- (E_potential_matrix ./ dm21)).^2 .+ numerator^2)

    # Calculate sin(2*th12)_matter element-wise
    sin_2th12_matter = numerator ./ denominator

    # Calculate th12_matter element-wise
    th12_matter = 0.5 .* asin.(sin_2th12_matter)

    return transpose(th12_matter)
end

# Overload for when both potential_val and E are scalars.
function get_matter_angle(potential_val::Real, mixingPars::oscPars, E::Real)
    # Convert scalars to single-element vectors, call the vector version, and return the single result.
    result = get_matter_angle([potential_val], mixingPars, [E])
    return result[1, 1]
end

# Overload for when potential_val is scalar and E is a vector.
function get_matter_angle(potential_val::Real, mixingPars::oscPars, E::Vector{<:Real})
    # Convert potential_val to a single-element vector.
    result = get_matter_angle([potential_val], mixingPars, E)
    # Since potential_val was scalar, return a vector of results.
    return reverse(result[1, :]')
end

# Overload for when potential_val is a vector and E is scalar.
function get_matter_angle(potential_val::Vector{<:Real}, mixingPars::oscPars, E::Real)
    # Convert E to a single-element vector.
    result = get_matter_angle(potential_val, mixingPars, [E])
    # Since E was scalar, return a vector of results.
    return result[:, 1]
end


function phase_integrand(segment::Segment, mixingPars::oscPars, E)
    # Extract mixing parameters
    dm21 = mixingPars.dm21
    th12 = mixingPars.th12
    th13 = mixingPars.th13

    #get the values of the potential at the nodes
    potential = segment.values
    
    # Calculate the outer product of E and pot_vals
    E_pot_matrix = E .* transpose(potential)  # Equivalent to np.outer(E, pot_vals) in Python

    # Calculate the term for each element in the matrix
    term_matrix = cos(2 * th12) .- (cos(th13)^2) .* E_pot_matrix ./ dm21

    # Calculate the integrand for each element in the matrix
    integrand_matrix = 1.26693281 * (2 * dm21 ./ E) .* sqrt.(term_matrix .^ 2 .+ sin(2 * th12)^2)

    return integrand_matrix
end


function post_jump_phases(potential::Path, mixingPars::oscPars, E::AbstractVector{<:Real})


    # If there is only one jump or none, no integration is needed.
    if length(potential.jumps) <= 1
        return zeros(length(E))
    end

    N_E = length(E)
    num_segments = length(potential.segments)

    # Preallocate matrix for integrals: rows corwrespond to segments, columns to energies.
    integrals = fill(0.0, num_segments, N_E)

    # Iterate over each segment.
    for i in 1:num_segments
        # Compute the integrand on this segment.
        integrand_vals = phase_integrand(potential.segments[i], mixingPars, E)
    
        # Obtain the limits of integration
        x_start = potential.segments[i].start
        x_end = potential.segments[i].finish
    
        # Vectorized trapezoidal integration for each energy. Clean to 1e-12 if very very small
        computed_integrals = sum(integrand_vals, dims = 2)[:] * abs(x_end-x_start) / size(integrand_vals)[2]
        integrals[i, :] = max.(computed_integrals, 1e-12)
    end

    # Compute cumulative sum of the integrals from the last segment to the first.
    # then reverse so that each column corresponds to cumulative phase starting from the post-jump region.
    integrals_from_end = reverse(integrals, dims=1)
    cumsum_flipped = cumsum(integrals_from_end, dims=1)
    cum_phases = reverse(cumsum_flipped, dims=1)

    return cum_phases
end

function jump_angle_change(jumps, mixingPars::oscPars, E_input)
    # Ensure E is a vector.
    E = (E_input isa Real) ? [E_input] : E_input
    jumps = (jumps isa Jump) ? [jumps] : jumps

    # Build vectors of potential values from the jump dictionaries.
    bef = [j.b for j in jumps]
    aft = [j.a for j in jumps]

    # Compute matter angles at the boundaries.
    # get_matter_angle returns a matrix of size (length(E) x length(potential_val)).
    th_aft = get_matter_angle(aft, mixingPars, E)  # shape: (length(E), n_jumps)
    th_bef = get_matter_angle(bef, mixingPars, E)  # shape: (length(E), n_jumps)

    # Compute the difference (aft - bef) elementwise.
    angle_diff = th_aft .- th_bef

    return angle_diff
end


function propagate_path(potential, mixingPars::oscPars, E_input)
    # Assume E_input can be scalar or vector. If scalar, wrap it in an array.
    E = (E_input isa Real) ? [E_input] : E_input


    # Compute cumulative phases from post_jump_phases.
    phi_a = post_jump_phases(potential, mixingPars, E)
    
    # Compute jump angle changes.
    del_th_full = jump_angle_change(potential.jumps, mixingPars, E)
    
    # Exclude the last jump from the phase integration (there is 1 more jump than there are segments)
    del_th =  reverse(del_th_full, dims=1)[1:end-1, :]

    # display(del_th)
    # display(phi_a)

    # Compute the summation term
    # If there is no angle change (empty del_th), summation is set to 0.
    if isempty(del_th)
        summation = 0.0
    else
        # Element-wise compute sin(del_th) * cos(phi_a) then sum over the jumps dimension.
        summation = vec(sum(sin.(del_th) .* cos.(phi_a), dims=1))
    end

    # Compute final matter angle at the last jump's "a" boundary.
    # get_matter_angle accepts a potential value (or vector thereof). Here, we pass the scalar value.
    th12_f = get_matter_angle(first(potential.jumps).b, mixingPars, E)

    # Ensure th12_f is a vector.
    th12_f = vec(th12_f)
    
    # Compute probability.
    # The formula is:
    #    prob = (cos(th13)^2) * (cos(th12_f)^2 + sin(2*th12_f) * summation)
    prob = (cos(mixingPars.th13)^2) .* (cos.(th12_f).^2 .+ sin.(2 .* th12_f) .* summation)
    
    return prob
end

function LMA_solution(energy, dm21, th12, th13, N_e)
    beta = (2 .* sqrt(2) .* 5.4489e-5 .* cos(th13) .* 2 .* N_e .* energy) / dm21
    matterAngle = (cos(2 * th12) .- beta) / sqrt.((cos(2 * th12) .- beta).^2 .+ sin(2 * th12)^2)
    probLMA = cos(th13)^4 .* (1 / 2 .+ 1 ./ 2 .* matterAngle * cos(2 * th12)) .+ sin(th13)^4
    return matterAngle, probLMA
end

function LMA_angle(energy, mixingPars, N_e)
    th12 = mixingPars.th12
    beta = (2 .* sqrt(2) .* 5.4489e-5 .* cos(mixingPars.th13)^2 .* N_e .* energy) ./ mixingPars.dm21
    matterAngle = (cos(2 * th12) .- beta) ./ sqrt.((cos(2 * th12) .- beta).^2 .+ sin(2 * th12)^2)
    return 0.5 .* acos.(matterAngle)
end

function get_1e(E, mixingPars::oscPars, paths)
    p_1e = [propagate_path(path, mixingPars, E) for path in paths]
    matrix_p_1e = reduce(vcat, [vec' for vec in p_1e])

    return matrix_p_1e
end

function get_probs(E, mixingPars::oscPars, paths, prod_density=90)
    th13 = mixingPars.th13

    p_1e = [propagate_path(path, mixingPars, E) for path in paths]
    matrix_p_1e = reduce(vcat, [vec' for vec in p_1e])

    solarAngle = permutedims(LMA_angle(E, mixingPars, prod_density))
    probs = cos(th13)^2 * ((cos.(2 .* solarAngle) .* matrix_p_1e) .+ cos(th13)^2 .* sin.(solarAngle).^2) .+ sin(th13)^4

    return matrix_p_1e
end

function get_probs(E::Vector{Float64}, matrix_p_1e::Matrix{Float64}, mixingPars::oscPars, prod_density::Float64)
    th13 = mixingPars.th13

    solarAngle = permutedims(LMA_angle(E, mixingPars, prod_density))
    probs = cos(th13)^2 * ((cos.(2 .* solarAngle) .* matrix_p_1e) .+ cos(th13)^2 .* sin.(solarAngle).^2) .+ sin(th13)^4

    return probs
end