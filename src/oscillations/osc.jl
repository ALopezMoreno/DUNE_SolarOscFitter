#=
osc.jl

Neutrino Oscillation Calculations for the Solar Oscillation Fitter.

This module implements comprehensive neutrino oscillation calculations for solar
neutrinos, including both daytime (solar matter effects) and nighttime (Earth
matter effects) propagation.

Contents:
- Day: adiabatic MSW survival probability in the Sun (`mswProb`), at the mean
  production density or averaged over the production region (`osc_prob_day_c2m`)
- Night: the IYSW relation P_N = P_D + c₁₃² cos2θ₁₂ᵐ (P₁ₑ − c₁₂²c₁₃²)
  (`osc_prob_both_fast`, `osc_prob_both_slow`)
- NumOsc: 3-flavour numerical Earth propagation per path (the `nightOsc: false`
  calculation, and the reference that NightOsc is tested against)
- nu4NumOsc: 4-state (one sterile) numerical propagation, not wired into the fit

The default night-time Earth crossing is NightOsc (osc_night.jl).

Oscillation framework adapted from Newthrino by Philipp Eller (special thanks
for his support and openness when sharing his code):
(add_github_repo_when_public)

Modifications to calculate the solar night-time propagation added by Andres Lopez Moreno

Author: Andres Lopez Moreno, based on Philipp Eller's Newthrino
=#
module Osc
using LinearAlgebra
using StaticArrays

#################################################################################
# Struct and const Definitions
#################################################################################

struct oscPars{T<:Real}
    Δm²₂₁::T
    θ₁₂::T
    θ₁₃::T
    Δm²₃₁::T
    m₀::T
    θ₂₃::T
    δCP::T
end

# Constructor with default values
function oscPars(Δm²₂₁, θ₁₂, θ₁₃; Δm²₃₁=2.5e-3, m₀=1e-9, θ₂₃=asin(sqrt(0.5)), δCP=-1.611)
    T = promote_type(typeof.((Δm²₂₁, θ₁₂, θ₁₃, Δm²₃₁, m₀, θ₂₃, δCP))...)
    oscPars{T}(T(Δm²₂₁), T(θ₁₂), T(θ₁₃), T(Δm²₃₁), T(m₀), T(θ₂₃), T(δCP))
end


# Fermi constant in units of Ne/Na
const G_f = 1.5264932436e-4 / (2 * sqrt(2))
const PHASE_CONST = 2.5338653580781976

#################################################################################
# Common functions in the calculation of the oscillation probability
#################################################################################

function get_PMNS(params)
    T = typeof(params.θ₂₃)

    U1 = SMatrix{3,3}(one(T), zero(T), zero(T), zero(T), cos(params.θ₂₃), -sin(params.θ₂₃), zero(T), sin(params.θ₂₃), cos(params.θ₂₃))
    U2 = SMatrix{3,3}(cos(params.θ₁₃), zero(T), -sin(params.θ₁₃) * exp(1im * params.δCP), zero(T), one(T), zero(T), sin(params.θ₁₃) * exp(-1im * params.δCP), zero(T), cos(params.θ₁₃))
    U3 = SMatrix{3,3}(cos(params.θ₁₂), -sin(params.θ₁₂), zero(T), sin(params.θ₁₂), cos(params.θ₁₂), zero(T), zero(T), zero(T), one(T))
    U = U1 * U2 * U3
end


function get_matrices(params)
    U = get_PMNS(params)
    H = Diagonal(SVector(zero(typeof(params.θ₂₃)), params.Δm²₂₁, params.Δm²₃₁))
    return U, H
end


# MSW survival probability P(νe→νe) averaged over the solar production region.
# Wolfenstein (1978), Mikheyev & Smirnov (1985); adiabatic two-flavour approximation
# with θ₁₃ correction from Barger et al. See also eq. (2) of SNO solar analysis.
function mswProb(energy, mixingPars, n_e)
    s2th13 = sin(mixingPars.θ₁₃)^2
    @inbounds begin
        c2th12 = cos.(2 .* mixingPars.θ₁₂)
        Acc = 2 .* sqrt(2) .* n_e .* G_f .* energy .* (1 .- s2th13)  
        beta = Acc ./ mixingPars.Δm²₂₁ 
        # Calculate the modified cosine of twice the angle theta_12 in matter
        c2th12m = (c2th12 .- beta) ./ sqrt.((c2th12 .- beta).^2 .+ (1 .- c2th12.^2))
        # Matter effect is weak enough to ignore the matter contribution to θ₁₃
        s13m = sqrt.(s2th13)
        probs = 1 ./ 2 .* (1 .- s2th13) .* (1 .- s13m.^2) .* (1 .+ c2th12 .* c2th12m) .+ s2th13 .* s13m.^2 
    end
    return probs, c2th12m
end


# Use relationship between P_1e and P_day from Ioannisian, Yu, Smirnov and Wyler:
# P_night = P_day + DeltaP where
# DeltaP = c_13^2*cos(2th12_sol)*(P_1e - P_0) and
# P_0 = c12^2cos^2th12

function osc_prob_both_slow(E::AbstractVector{<:Real}, matrix_p_1e::AbstractMatrix, mixingPars, solarModel; process="8B")
    # Get daytime probability at each n_e over the production region
    results = mswProb.(E, Ref(mixingPars), solarModel.n_e')
    n_energies = length(E)
    n_densities = length(solarModel.n_e)

    T = typeof(real(one(typeof(mixingPars.θ₁₂))))
    prob_day = Matrix{T}(undef, n_energies, n_densities)
    cos2θ₁₂_sol = Matrix{T}(undef, n_energies, n_densities)
    
    @inbounds for i in 1:n_energies, j in 1:n_densities
        prob_day[i, j] = results[i, j][1]
        cos2θ₁₂_sol[i, j] = results[i, j][2]
    end

    # Select the appropriate production fraction based on the process
    prodFraction = process == "8B" ? solarModel.prodFractionBoron :
                   process == "hep" ? solarModel.prodFractionHep :
                   error("Invalid process specified. Please use '8B' or 'hep'.")

    # Precompute constants
    cos2_θ₁₃ = cos(mixingPars.θ₁₃)^2
    P_0 = cos(mixingPars.θ₁₂)^2 * cos2_θ₁₃
    n_paths = size(matrix_p_1e, 1)
    
    # Pre-allocate for the final integrated night probabilities (same shape as matrix_p_1e)
    prob_night_integrated = zeros(T, n_paths, n_energies)

    @inbounds for i in 1:n_densities
        prob_day_i = @view prob_day[:, i]
        cos2θ₁₂_sol_i = @view cos2θ₁₂_sol[:, i]
        ΔP_i = cos2_θ₁₃ .* cos2θ₁₂_sol_i' .* (matrix_p_1e .- P_0)
        prob_night_full_i = prob_day_i' .+ ΔP_i
        prob_night_integrated .+= prodFraction[i] .* prob_night_full_i
    end

    weighted_sum_day = prob_day * prodFraction
    total_weight = sum(prodFraction)

    prob_day_integrated = weighted_sum_day ./ total_weight
    prob_night_integrated ./= total_weight 

    return prob_day_integrated, prob_night_integrated
end


function osc_prob_both_fast(E::AbstractVector{<:Real}, matrix_p_1e::AbstractMatrix, mixingPars, solarModel; process="8B")
    # Get average density over production region
    n_e = process == "8B" ? solarModel.avgNeBoron :
          process == "hep" ? solarModel.avgNeHep :
          error("Invalid process specified. Please use '8B' or 'hep'.")

    # Get daytime probability at average n_e
    prob_day, cos2θ₁₂_sol = mswProb(E, mixingPars, n_e)

    # Precompute constants
    cos2_θ₁₃ = cos(mixingPars.θ₁₃)^2
    P_0 = cos(mixingPars.θ₁₂)^2 * cos2_θ₁₃
    
    # Use transpose for efficient broadcasting (avoids reshape allocation)
    # This creates a view, not a copy
    cos2θ₁₂_sol_t = cos2θ₁₂_sol'
    prob_day_t = prob_day'
    
    # Fused broadcasting operation
    @inbounds prob_night = prob_day_t .+ cos2_θ₁₃ .* cos2θ₁₂_sol_t .* (matrix_p_1e .- P_0)

    return prob_day, prob_night
end


# Day-time P(νe→νe) and the production-averaged cos2θ₁₂ᵐ that multiplies the night
# regeneration term, at energies E. fast=true evaluates both at the production-weighted mean
# density; fast=false averages them over the production region (exact, since the night term is
# linear in cos2θ₁₂ᵐ).
function osc_prob_day_c2m(E::AbstractVector{<:Real}, mixingPars, solarModel; process="8B", fast::Bool=true)
    if fast
        n_e = process == "8B" ? solarModel.avgNeBoron :
              process == "hep" ? solarModel.avgNeHep :
              error("Invalid process specified. Please use '8B' or 'hep'.")
        return mswProb(E, mixingPars, n_e)
    end
    prodFraction = process == "8B" ? solarModel.prodFractionBoron :
                   process == "hep" ? solarModel.prodFractionHep :
                   error("Invalid process specified. Please use '8B' or 'hep'.")
    w = prodFraction ./ sum(prodFraction)
    probs, c2m = mswProb(E, mixingPars, permutedims(solarModel.n_e))   # (n_E, n_radii)
    return probs * w, c2m * w
end


################################################################################
# Numerical propagation
################################################################################

module NumOsc
    using LinearAlgebra, StaticArrays
    import ..PHASE_CONST

    @inline function osc_kernel(U::SMatrix{3,3,<:Complex}, H::SVector{3,<:Real}, e::Real, l::Real)
        phase_factors = exp.(PHASE_CONST * 1im * (l / e) .* H) 
        tmp = U .* phase_factors'  
        return tmp * adjoint(U)
    end

    @inline function get_eigen(H::Hermitian{<:Complex, <:SMatrix{3,3}})
        tmp = eigen(H)
    end

    function get_H(H_vac, e, rho, anti::Bool=false)
        H = MMatrix{3,3}(H_vac)

        if anti
            H[1, 1] -= rho * e
            for i in 1:3
                H[i, i] += rho * e
            end
        else
            H[1, 1] += rho * e
            for i in 1:3
                H[i, i] -= rho * e
            end
        end

        H = Hermitian(SMatrix(H))
        tmp = get_eigen(H)
        tmp.vectors, tmp.values
    end

    function osc_reduce(Mix_dagger, matter_matrices, path, e, anti::Bool)
        @inbounds begin
            first_p, first_mn = path[1], matter_matrices[1]
            X = osc_kernel(first_mn[1], first_mn[2], e, first_p)  # 3x3 SMatrix
            for i in 2:length(path)
                p = path[i]
                m, n = matter_matrices[i]
                X = X * osc_kernel(m, n, e, p)  # SMatrix * SMatrix
            end
            A = abs2.(Mix_dagger * X)
        end
        return A
    end

    module Slow
        using LinearAlgebra
        using ...Osc: oscPars, get_matrices
        using ..NumOsc: get_H, osc_reduce

        function matter_osc_per_e(Mix_dagger, H_eff, e, rho_Array, l_Array, anti)
            matter_matrices = map(rho_vec -> (get_H.(Ref(H_eff), e, rho_vec, anti)), rho_Array)
            stack(map((path, matter) -> (osc_reduce(Mix_dagger, matter, path, e, anti)), l_Array, matter_matrices))
        end

        function osc_prob_earth(E::AbstractVector{<:Real}, params::oscPars, lookup_density, paths; anti=false)
            U, H = get_matrices(params)
            Udag = adjoint(U)
            Uc = anti ? conj.(U) : U
            H_eff = Uc * Diagonal{Complex{eltype(H)}}(H) * adjoint(Uc)

            rho_Array = [reverse([s.avgRho for s in path.segments]) for path in paths]
            l_Array = [reverse([s.length for s in path.segments]) for path in paths]

            p1e = stack(map(e -> matter_osc_per_e(Udag, H_eff, e, rho_Array, l_Array, anti), E))[1, 1, :, :]
        end

    end

    module Fast
        using LinearAlgebra
        using ...Osc: oscPars, get_matrices
        using ..NumOsc: get_H, osc_kernel

        # Like osc_reduce but indexes directly into a pre-allocated lookup buffer,
        # avoiding the allocation from lookup_matrices[indices] in the original map path.
        # Returns only the [1,1] element (ν_e→ν_e probability).
        @inline function osc_reduce_from_lookup(Mix_dagger, lookup_buf, indices, path, e)
            @inbounds begin
                first_mn = lookup_buf[indices[1]]
                X = osc_kernel(first_mn[1], first_mn[2], e, path[1])
                for i in 2:length(path)
                    mn = lookup_buf[indices[i]]
                    X = X * osc_kernel(mn[1], mn[2], e, path[i])
                end
                return abs2.(Mix_dagger * X)[1, 1]
            end
        end

        function osc_prob_earth(E::AbstractVector{<:Real}, params::oscPars, lookup_density, paths; anti=false)
            U, H = get_matrices(params)
            Udag = adjoint(U)
            Uc = anti ? conj.(U) : U
            H_eff = Uc * Diagonal{Complex{eltype(H)}}(H) * adjoint(Uc)

            # Reverse here so osc_reduce_from_lookup can walk front-to-back
            index_array = [reverse([s.index for s in path.segments]) for path in paths]
            l_Array     = [reverse([s.length for s in path.segments]) for path in paths]

            n_E      = length(E)
            n_paths  = length(paths)
            n_lookup = length(lookup_density)

            # Pre-allocate output and reusable eigendecomposition buffer
            p1e        = Matrix{real(eltype(H_eff))}(undef, n_paths, n_E)
            lookup_buf = Vector{typeof(get_H(H_eff, E[1], lookup_density[1], anti))}(undef, n_lookup)

            @inbounds for ie in eachindex(E)
                e = E[ie]
                for j in 1:n_lookup
                    lookup_buf[j] = get_H(H_eff, e, lookup_density[j], anti)
                end
                for ip in 1:n_paths
                    p1e[ip, ie] = osc_reduce_from_lookup(Udag, lookup_buf, index_array[ip], l_Array[ip], e)
                end
            end

            return p1e
        end

    end
end


################################################################################
# 4-state numerical propagation
################################################################################
module nu4NumOsc
    using LinearAlgebra, StaticArrays
    import ..G_f, ..PHASE_CONST

    # we need to redefine the mixing matrix an oscpars to accept new parameters:
    struct oscPars
        Δm²₂₁::Float64
        θ₁₂::Float64
        θ₁₃::Float64
        Δm²₃₁::Float64
        m₀::Float64
        θ₂₃::Float64
        δCP::Float64

        Δm²₄₁::Float64
        θ₁₄::Float64
        θ₂₄::Float64
        θ₃₄::Float64
        # We are ignoring the sterile complex phases because we are not sensitive to them
    end

    # Constructor with default values
    oscPars(Δm²₂₁, θ₁₂, θ₁₃, θ₁₄, θ₂₄, θ₃₄, Δm²₄₁; Δm²₃₁=2.5e-3, m₀=1e-9, θ₂₃=asin(sqrt(0.56)), δCP=-1.6) = 
        oscPars(Δm²₂₁, θ₁₂, θ₁₃, Δm²₃₁, m₀, θ₂₃, δCP, Δm²₄₁, θ₁₄, θ₂₄, θ₃₄)


    @inline function osc_kernel(U::SMatrix{4,4,<:Complex}, H::SVector{4,<:Real}, e::Real, l::Real)
        phase_factors = exp.(PHASE_CONST * 1im * (l / e) .* H) 
        tmp = U .* phase_factors'  
        return tmp * adjoint(U)
    end

    @inline function decoherent_osc_kernel(U::Matrix{ComplexF64}, U_end::SMatrix{4,4,ComplexF64})
        P_decoherent = abs2.(U)
    end

    @inline function get_eigen(H::Hermitian{<:Complex, <:SMatrix{4,4}})
        eigen(H)
    end
    
    function get_PMNS(params)
        T = typeof(params.θ₂₃)

        U23 = SMatrix{4,4}(one(T), zero(T), zero(T), zero(T), zero(T), cos(params.θ₂₃), -sin(params.θ₂₃), zero(T), zero(T), sin(params.θ₂₃), cos(params.θ₂₃), zero(T), zero(T), zero(T), zero(T), one(T))

        U13 = SMatrix{4,4}(cos(params.θ₁₃), zero(T), -sin(params.θ₁₃) * exp(1im * params.δCP), zero(T), zero(T), one(T), zero(T), zero(T), sin(params.θ₁₃) * exp(-1im * params.δCP), zero(T), cos(params.θ₁₃), zero(T), zero(T), zero(T), zero(T), one(T))

        U12 = SMatrix{4,4}(cos(params.θ₁₂), -sin(params.θ₁₂), zero(T), zero(T), sin(params.θ₁₂), cos(params.θ₁₂), zero(T), zero(T), zero(T), zero(T), one(T), zero(T), zero(T), zero(T), zero(T), one(T))

        U14 = SMatrix{4,4}(cos(params.θ₁₄), zero(T), zero(T), -sin(params.θ₁₄), zero(T), one(T), zero(T), zero(T), zero(T), zero(T), one(T), zero(T), sin(params.θ₁₄), zero(T), zero(T), cos(params.θ₁₄))

        U24 = @SMatrix [
            one(T)            zero(T)         zero(T)         zero(T)
            zero(T)           cos(params.θ₂₄) zero(T)        -sin(params.θ₂₄)
            zero(T)           zero(T)         one(T)          zero(T)
            zero(T)           sin(params.θ₂₄) zero(T)         cos(params.θ₂₄)
        ]

        U34 = @SMatrix [
            one(T)            zero(T)         zero(T)         zero(T)
            zero(T)           one(T)          zero(T)         zero(T)
            zero(T)           zero(T)         cos(params.θ₃₄) -sin(params.θ₃₄)
            zero(T)           zero(T)         sin(params.θ₃₄)  cos(params.θ₃₄)
        ]

        U = U34 * U24 * U14 * U23 * U13 * U12 
        return U
    end

    function get_matrices(params)
        U = get_PMNS(params)
        H = Diagonal(SVector(zero(typeof(params.θ₂₃)), params.Δm²₂₁, params.Δm²₃₁, params.Δm²₄₁))
        return U, H
    end

    @inline function get_H(H_vac, e, rho, rho_n, massMatrix, anti::Bool=false)
        # Start from vacuum Hamiltonian and apply matter potentials only on the diagonal
        Hm = MMatrix{4,4}(H_vac)
        ee = e
        rr = rho
        rn = rho_n

        @inbounds begin
            if anti
                # Antineutrinos: sign flip for charged current potential
                Hm[1, 1] -= (rr - rn/2) * ee
                Hm[2, 2] -= rn/2 * ee
                Hm[3, 3] -= rn/2 * ee
                # Hm[4,4] unchanged
            else
                Hm[1, 1] += (rr - rn/2) * ee 
                Hm[2, 2] -= rn/2 * ee
                Hm[3, 3] -= rn/2 * ee
                # Hm[4,4] unchanged
            end
        end

        Hh = Hermitian(SMatrix(Hm))
        tmp = get_eigen(Hh)

        # Ensure correct ordering of eigenvalues: if mixing
        # non-zero then crossings are allowed: we order by same as vacuum
        # otherwise, match the sterile to its constant mass
        tol = 1e-10
        reference_masses = diag(massMatrix)
        
        # Step 1: Get sorting indices for reference masses
        ref_sort_indices = sortperm(reference_masses)
        rank_order = invperm(ref_sort_indices)
        
        # Step 2: Sort eigenvalues and eigenvectors
        eig_sort_indices = sortperm(tmp.values)
        sorted_vals = tmp.values[eig_sort_indices]
        sorted_vecs = tmp.vectors[:, eig_sort_indices]
        
        # Step 3: Apply rank ordering
        final_vals = sorted_vals[rank_order]
        final_vecs = sorted_vecs[:, rank_order]
        
        # Step 4: Check for matches and swap (optimized)
        final_vals_arr = Vector(final_vals)
        final_vecs_arr = Matrix(final_vecs)
        
        n = length(final_vals_arr)
        matched_indices = falses(n)  # Track which indices have been matched
        
        # Precompute differences for efficiency
        for j in 1:n
            ref_mass_j = reference_masses[j]
            # Find the best match for reference_masses[j]
            best_match_idx = 0
            best_match_diff = Inf
            
            for i in 1:n
                if !matched_indices[i]
                    diff = abs(final_vals_arr[i] - ref_mass_j)
                    if diff < tol && diff < best_match_diff
                        best_match_idx = i
                        best_match_diff = diff
                    end
                end
            end
            
            if best_match_idx > 0 && best_match_idx != j
                # Swap eigenvalues
                final_vals_arr[j], final_vals_arr[best_match_idx] = final_vals_arr[best_match_idx], final_vals_arr[j]
                
                # Swap eigenvectors
                for k in 1:size(final_vecs_arr, 1)
                    final_vecs_arr[k, j], final_vecs_arr[k, best_match_idx] = final_vecs_arr[k, best_match_idx], final_vecs_arr[k, j]
                end
                
                matched_indices[best_match_idx] = true
                matched_indices[j] = true
            elseif best_match_idx == j
                matched_indices[j] = true
            end
        end
        
        return final_vecs_arr, final_vals_arr
    end

    function osc_prob_day(E::AbstractVector{<:Real}, params, solarModel; process="8B")
        n_e = process == "8B" ? solarModel.avgNeBoron :
        process == "hep" ? solarModel.avgNeHep :
        error("Invalid process specified. Please use '8B' or 'hep'.")
        
        n_n = n_e * 0.7

        rho_e =  n_e .* G_f .* 2 .* sqrt(2)
        rho_n =  n_n .* G_f .* 2 .* sqrt(2)

        U, H = get_matrices(params)

        H_eff = U * Diagonal{Complex{eltype(H)}}(H) * adjoint(U)
        U_sol = [get_H(H_eff, e, rho_e, rho_n, H)[1] for e in E]

        enuOscProb = stack(map(U_eff -> decoherent_osc_kernel(U_eff, U), U_sol))

        return enuOscProb
    end

    function osc_reduce(Mix, Mix_dagger, matter_matrices, path, e, anti::Bool)
        @inbounds begin
            first_p, first_mn = path[1], matter_matrices[1]
            X = osc_kernel(first_mn[1], first_mn[2], e, first_p)
            for i in 2:length(path)
                p = path[i]
                m, n = matter_matrices[i]
                X = X * osc_kernel(m, n, e, p)
            end
            A = abs2.(Mix_dagger * X)
        end
        return A
    end

    function matter_osc_per_e(lookup_density, Mix, Mix_dagger, H_eff, massMatrix, e, index_array, l_Array, anti)
        lookup_matrices = map(rho_vec -> get_H.(Ref(H_eff), e, rho_vec, (rho_vec .* 0.7), Ref(massMatrix), anti), lookup_density)
        matter_matrices = map(indices -> lookup_matrices[indices], index_array)

        stack(map((path, matter) -> osc_reduce(Mix, Mix_dagger, matter, path, e, anti), l_Array, matter_matrices))
    end

    function osc_prob_earth(E::AbstractVector{<:Real}, params::oscPars, lookup_density, paths; anti=false)
        U, H = get_matrices(params)
        Udag = adjoint(U)
        Uc = anti ? conj.(U) : U
        H_eff = Uc * Diagonal{Complex{eltype(H)}}(H) * adjoint(Uc)

        # We reverse the indices here to avoid reversing in the computation of the amplitude
        index_array = [reverse([s.index for s in path.segments]) for path in paths]
        l_Array = [reverse([s.length for s in path.segments]) for path in paths]

        p1e = stack(map(e -> matter_osc_per_e(lookup_density, U, Udag, H_eff, H, e, index_array, l_Array, anti), E)) # [1, 1, :, :]
    end
end

## MODULE END ##
end
################

