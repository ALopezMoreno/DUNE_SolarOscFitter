#=
osc.jl

Neutrino Oscillation Calculations for the Solar Oscillation Fitter.

This module implements comprehensive neutrino oscillation calculations for solar
neutrinos, including both daytime (solar matter effects) and nighttime (Earth
matter effects) propagation.

Key Features:
- MSW (Mikheyev-Smirnov-Wolfenstein) effect in solar matter
- Earth matter effects for nighttime propagation  
- Both analytical (Barger) and numerical propagation methods
- Fast and slow calculation modes for different precision needs

Oscillation framework adapted from Newthrino by Philipp Eller (special thanks
for his support and openness when sharing his code):
(add_github_repo_when_public)

Barger solution and modifications to calculate the solar night-time propagation
added by Andres Lopez Moreno

Author: Andres Lopez Moreno, based on Philipp Eller's Newthrino
=#
module Osc
using LinearAlgebra
using StaticArrays

#################################################################################
# Struct and const Definitions
#################################################################################

struct oscPars
    Δm²₂₁::Float64
    θ₁₂::Float64
    θ₁₃::Float64
    Δm²₃₁::Float64
    m₀::Float64
    θ₂₃::Float64
    δCP::Float64
end

# Constructor with default values
oscPars(Δm²₂₁, θ₁₂, θ₁₃; Δm²₃₁=2.5e-3, m₀=1e-9, θ₂₃=asin(sqrt(0.5)), δCP=-1.611) = 
    oscPars(Δm²₂₁, θ₁₂, θ₁₃, Δm²₃₁, m₀, θ₂₃, δCP)


# Fermi constant in units of Ne/Na
const G_f = 5.4489e-5

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


function LMA_angle(energy, mixingPars, N_e)
    th12 = mixingPars.θ₁₂
    beta = (2 .* sqrt(2) .* 5.4489e-5 .* cos(mixingPars.θ₁₃)^2 .* N_e .* energy) ./ mixingPars.Δm²₂₁
    matterAngle = (cos(2 * th12) .- beta) ./ sqrt.((cos(2 * th12) .- beta) .^ 2 .+ sin(2 * th12)^2)
    return 0.5 .* acos.(matterAngle)
end


function osc_prob_day(energy::Float64, params, solarModel; process="8B")
    # Calculate the neutrino oscillation probabilities integrating over the production region
    enuOscProb, _ = mswProb(energy, params, solarModel.n_e)

    # Select the appropriate production fraction based on the process
    prodFraction = process == "8B" ? solarModel.prodFractionBoron :
                   process == "hep" ? solarModel.prodFractionHep :
                   error("Invalid process specified. Please use '8B' or 'hep'.")

    # integrate over production region
    weighted_sum = sum(prodFraction .* enuOscProb)
    total_weight = sum(prodFraction)

    prob_nue = weighted_sum / total_weight

    return prob_nue
end


function osc_prob_day_fast(energy::Float64, params, solarModel; process="8B")
    # Calculate the neutrino oscillation probabilities averaging over the production region
    n_e = process == "8B" ? solarModel.avgNeBoron :
          process == "hep" ? solarModel.avgNeHep :
          error("Invalid process specified. Please use '8B' or 'hep'.")

    enuOscProb = mswProb(params, energy, n_e)

    return enuOscProb
end

# Use relationship between P_1e and P_day from Ioannisian, Yu, Smirnov and Wyler:
# P_night = P_day + DeltaP where
# DeltaP = c_13^2*cos(2th12_sol)*(P_1e - P_0) and
# P_0 = c12^2cos^2th12

function osc_prob_both_slow(E::Vector{Float64}, matrix_p_1e::Matrix{Float64}, mixingPars, solarModel; process="8B")
    # Get daytime probability at each n_e over the production region
    results = mswProb.(E, Ref(mixingPars), solarModel.n_e')
    n_energies = length(E)
    n_densities = length(solarModel.n_e)
    
    prob_day = Matrix{Float64}(undef, n_energies, n_densities)
    cos2θ₁₂_sol = Matrix{Float64}(undef, n_energies, n_densities)
    
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
    prob_night_integrated = zeros(Float64, n_paths, n_energies)

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


function osc_prob_both_fast(E::Vector{Float64}, matrix_p_1e::Matrix{Float64}, mixingPars, solarModel; process="8B")
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


function osc_prob_night(E::Vector{Float64}, matrix_p_1e::Matrix{Float64}, mixingPars, prod_density::Float64)
    th13 = mixingPars.θ₁₃

    solarAngle = permutedims(LMA_angle(E, mixingPars, prod_density))
    probs = cos(th13)^2 * ((cos.(2 .* solarAngle) .* matrix_p_1e) .+ cos(th13)^2 .* sin.(solarAngle) .^ 2) .+ sin(th13)^4

    return probs
end

################################################################################
# Barger propagation
################################################################################

module BargerOsc
    using LinearAlgebra, StaticArrays

    @inline function osc_kernel2(U::AbstractMatrix{<:Number}, P::AbstractVector, H::AbstractVector, e::Real, l::Real)
        phase_factors = exp.(2.5338653580781976 * 1im * (l / e) .* H)
        p = U * sum(P[i] * phase_factors[i] for i in eachindex(P)) * U'
    end


    function get_eigen(U, H_vac, H, e, rho)
        m²₁ = H_vac[1, 1]
        Δm²₂₁ = -H_vac[2, 2]
        Δm²₃₁ = -H_vac[3, 3]
        dVac = [m²₁, Δm²₂₁, Δm²₃₁]

        dmVacVac = MMatrix{3,3}(H_vac)
        for i in 1:3
            for j in 1:3
                dmVacVac[i, j] = dVac[i] - dVac[j]
            end
        end

        fac = -e * rho

        # Get eigenvalues from Barger's Formula

        α = fac + Δm²₂₁ + Δm²₃₁
        β = Δm²₂₁ * Δm²₃₁ + fac * (Δm²₂₁ * (1 - abs2(U[1, 2])) + Δm²₃₁ * (1 - abs2(U[1, 3])))
        γ = fac * Δm²₂₁ * Δm²₃₁ * abs2(U[1, 1])

        brac = sqrt(α^2 - 3 * β)

        arg = (2 * α^3 - 9 * α * β + 27 * γ) / (2 * brac^3)

        θ⁰ = acos.(clamp(arg, -1.0, 1.0))
        θ⁺ = θ⁰ + 2π
        θ⁻ = θ⁰ - 2π

        λ₁ = -(2 / 3) * brac * cos(θ⁰ / 3) + m²₁ - α / 3
        λ₂ = -(2 / 3) * brac * cos(θ⁻ / 3) + m²₁ - α / 3
        λ₃ = -(2 / 3) * brac * cos(θ⁺ / 3) + m²₁ - α / 3

        # WE'RE WELL BELOW THE EIGENVALUE CROSSING SO WE DON'T NEED TO WORRY ABOUT ORDERING THE EIGENVALUES. FIX OTHERWISE
        # Get eigenvector matrix from Lagrange's formula

        H₁ = H - Diagonal(fill(λ₁, 3))
        H₂ = H - Diagonal(fill(λ₂, 3))
        H₃ = H - Diagonal(fill(λ₃, 3))

        P₁ = (H₂ * H₃) / ((λ₁ - λ₂) * (λ₁ - λ₃))
        P₂ = (H₃ * H₁) / ((λ₂ - λ₁) * (λ₂ - λ₃))
        P₃ = (H₁ * H₂) / ((λ₃ - λ₁) * (λ₃ - λ₂))

        return [P₁, P₂, P₃], [λ₁, λ₂, λ₃]
    end


    function get_H(U, H_vac, e, rho, anti::Bool=false)
        H = MMatrix{3,3}(H_vac)

        if anti
            fac = -rho * e
        else
            fac = +rho * e
        end

        v = U[1, :]
        H += fac * v * v'

        H = Hermitian(SMatrix(H))

        tmpMat, tempVec = get_eigen(U, H_vac, H, e, rho) # --- test speed wrt num ---
        return tmpMat, tempVec
    end


    function osc_reduce(Mix, matter_matrices, path, e, anti::Bool)
        X = mapreduce((p, (m, n)) -> osc_kernel(Mix, m, n, e, p), *, path, reverse(matter_matrices))
        A = abs2.(Mix' * X)
        return A
    end

    module Slow
        using LinearAlgebra
        using ...Osc: oscPars, get_matrices
        using ..BargerOsc: get_H, osc_reduce

        function matter_osc_per_e(U, H_eff, e, rho_Array, l_Array, anti)
            matter_matrices = map(rho_vec -> (get_H.(Ref(U), Ref(H_eff), e, rho_vec, anti)), rho_Array)
            stack(map((path, matter) -> (osc_reduce(U, matter, path, e, anti)), l_Array, matter_matrices))
        end

        function osc_prob_earth(E::AbstractVector{<:Real}, params::oscPars, paths; anti=false)
            U, H = get_matrices(params)
            # We work on the mass basis

            rho_Array = [[s.avgRho for s in path.segments] for path in paths]
            l_Array = [[s.length for s in path.segments] for path in paths]

            p1e = stack(map(e -> matter_osc_per_e(U, H, e, rho_Array, l_Array, anti), E))[1, 1, :, :]
        end

    end

    module Fast
        using LinearAlgebra
        using ...Osc: oscPars, get_matrices
        using ..BargerOsc: get_H, osc_reduce

        function matter_osc_per_e(lookup_density, U, H_eff, e, index_array, l_Array, anti)
            lookup_matrices = map(rho_vec -> (get_H.(Ref(U), Ref(H_eff), e, rho_vec, anti)), lookup_density)
            matter_matrices = map(indices -> lookup_matrices[indices], index_array)

            stack(map((path, matter) -> (osc_reduce(U, matter, path, e, anti)), l_Array, matter_matrices))
        end

        function osc_prob_earth(E::AbstractVector{<:Real}, params::oscPars, lookup_density, paths; anti=false)
            U, H = get_matrices(params)
            # We work on the mass basis

            index_array = [[s.index for s in path.segments] for path in paths]
            l_Array = [[s.length for s in path.segments] for path in paths]

            p1e = stack(map(e -> matter_osc_per_e(lookup_density, U, H, e, index_array, l_Array, anti), E))[1, 1, :, :]
        end

    end
end


################################################################################
# Numerical propagation
################################################################################

module NumOsc
    using LinearAlgebra, StaticArrays

    @inline function osc_kernel(U::SMatrix{3,3,ComplexF64}, H::SVector{3,Float64}, e::Float64, l::Float64)
        phase_factors = exp.(2.5338653580781976im * (l / e) .* H) 
        tmp = U .* phase_factors'  
        return tmp * adjoint(U)
    end

    @inline function get_eigen(H::Hermitian{ComplexF64,SMatrix{3,3,ComplexF64,9}})
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

    function osc_reduce(Mix_dagger::SMatrix{3,3,ComplexF64}, matter_matrices, path, e::Float64, anti::Bool)
        @inbounds begin
            first_p, first_mn = path[1], matter_matrices[end]
            X = osc_kernel(first_mn[1], first_mn[2], e, first_p)  # 3x3 SMatrix
            for i in 2:length(path)
                p = path[i]
                m, n = matter_matrices[end - i + 1]
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

        function osc_prob_earth(E::AbstractVector{<:Real}, params::oscPars, paths; anti=false)
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
        using ..NumOsc: get_H, osc_reduce

        function matter_osc_per_e(lookup_density, Mix_dagger, H_eff, e, index_array, l_Array, anti)

            lookup_matrices = map(rho_vec -> get_H.(Ref(H_eff), e, rho_vec, anti), lookup_density)
            matter_matrices = map(indices -> lookup_matrices[indices], index_array)

            stack(map((path, matter) -> osc_reduce(Mix_dagger, matter, path, e, anti), l_Array, matter_matrices))
        end

        function osc_prob_earth(E::AbstractVector{<:Real}, params::oscPars, lookup_density, paths; anti=false)
            U, H = get_matrices(params)
            Udag = adjoint(U)
            Uc = anti ? conj.(U) : U
            H_eff = Uc * Diagonal{Complex{eltype(H)}}(H) * adjoint(Uc)

            # We reverse the indices here to avoid reversing in the computation of the amplitude
            index_array = [reverse([s.index for s in path.segments]) for path in paths]
            l_Array = [reverse([s.length for s in path.segments]) for path in paths]

            p1e = stack(map(e -> matter_osc_per_e(lookup_density, Udag, H_eff, e, index_array, l_Array, anti), E))[1, 1, :, :]
        end

    end
end


################################################################################
# 4-state numerical propagation
################################################################################
module nu4NumOsc
    using LinearAlgebra, StaticArrays

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


    @inline function osc_kernel(U::Matrix{ComplexF64}, H::Vector{Float64}, e::Float64, l::Float64)
        phase_factors = exp.(2.5338653580781976im * (l / e) .* H) 
        tmp = U .* phase_factors'  
        return tmp * adjoint(U)
    end

    @inline function decoherent_osc_kernel(U::Matrix{ComplexF64}, U_end::SMatrix{4,4,ComplexF64})
        P_decoherent = abs2.(U)
    end

    @inline function get_eigen(H::Hermitian{ComplexF64,SMatrix{4,4,ComplexF64,16}})
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
        ee = Float64(e)
        rr = Float64(rho)
        rn = Float64(rho_n)

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

        display(n_e)

        rho_e =  n_e .* 5.4489e-5 .* 2 .* sqrt(2)
        rho_n =  n_n .* 5.4489e-5 .* 2 .* sqrt(2)

        U, H = get_matrices(params)

        H_eff = U * Diagonal{Complex{eltype(H)}}(H) * adjoint(U)
        U_sol = [get_H(H_eff, e, rho_e, rho_n, H)[1] for e in E]

        enuOscProb = stack(map(U_eff -> decoherent_osc_kernel(U_eff, U), U_sol))

        return enuOscProb
    end

    function osc_reduce(Mix::SMatrix{4,4,ComplexF64}, Mix_dagger::SMatrix{4,4,ComplexF64}, matter_matrices, path, e::Float64, anti::Bool)
        @inbounds begin
            first_p, first_mn = path[1], matter_matrices[end]
            X = osc_kernel(first_mn[1], first_mn[2], e, first_p) 
            for i in 2:length(path)
                p = path[i]
                m, n = matter_matrices[end - i + 1]
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


####################################################
#####################
###### TESTING ######
#####################
#=
using ProgressMeter
using Printf
using Measures
using ColorTypes
using ColorSchemes
using LaTeXStrings
using .Osc: nu4NumOsc
using .Osc: oscPars, osc_prob_night

#theme(:dracula)
scalefontsizes(1)

# Helper functions for diagnostics
function format_sig_figs(x, sig_figs)
    if x == 0
        return "0"
    end
    magnitude = floor(log10(abs(x)))
    factor = 10^(sig_figs - 1 - magnitude)
    rounded = round(x * factor) / factor
    return string(rounded)
end

function remove_outliers(data, n_sigma=3)
    μ = mean(data)
    σ = std(data)
    return filter(x -> abs(x - μ) <= n_sigma * σ, data)
end


mixingPars_dict = (
    sin2_th12=0.303,
    sin2_th13=0.022,
    θ₁₃=asin(sqrt(0.022)),
    θ₁₂=asin(sqrt(0.303)),
    θ₂₃=asin(sqrt(0.5)), # Example value
    δCP=0.0,   # Example value
    Δm²₂₁=7.5e-5,
    Δm²₃₁=2.5e-3,
    m₀=1e-9 # Example value
)

# mixingPars = nu4NumOsc.oscPars(mixingPars_dict.Δm²₂₁, asin(sqrt(mixingPars_dict.sin2_th12)), asin(sqrt(mixingPars_dict.sin2_th13)), 0, 0, 0, 1)
mixingPars = oscPars(mixingPars_dict.Δm²₂₁, asin(sqrt(mixingPars_dict.sin2_th12)), asin(sqrt(mixingPars_dict.sin2_th13)))

energies = collect(range(0.1, stop=18, length=300)) * 1e-3

solarModel = (avgNeBoron = 100,)

# oscProbs = nu4NumOsc.osc_prob_day(energies, mixingPars, solarModel)
# p1 = plot(energies, oscProbs)
# display(p1)


#=
num_runs = 100

elapsed_times = zeros(num_runs)

# Run once to compile
p_1e_num = osc_prob_earth_num_fast(energies, mixingPars, earth_lookup, earth_paths, anti=false)[:, :, 1, 1] # Get the 1e element only
p_1e_bar = osc_prob_earth_barger_fast(energies, earth_paths, mixingPars_dict, anti=false)[:, :, 1, 1] # Get the 1e element only

prob_num = osc_prob_night(energies, p_1e_num, mixingPars, solarModel.avgNeBoron)
prob_bar = osc_prob_night(energies, p_1e_bar, mixingPars, solarModel.avgNeBoron)


@showprogress 0.02 "Propagating num in loop..." for i in 1:num_runs
    elapsed = @elapsed begin
        p_1e_time = osc_prob_earth_num_fast(energies, mixingPars, earth_lookup, earth_paths, anti=false)[:, :, 1, 1] # Get the 1e element only
        prob_time = osc_prob_night(energies, p_1e_time, mixingPars, solarModel.avgNeBoron)
    end
    elapsed_times[i] = elapsed
    # println("Run $i: Time to generate probs_matrix of size $(size(prob_time)) was $(elapsed) seconds.")
end

filtered_times = remove_outliers(elapsed_times)

average_time = mean(filtered_times)
uncertainty = std(filtered_times)

# Determine the number of significant figures based on the uncertainty
num_sig_figs_base = max(1, floor(Int, -log10(uncertainty)))
num_sig_figs = num_sig_figs_base + 1 # Request one more significant figure


formatted_average = format_sig_figs(average_time * 1e3, num_sig_figs) 
formatted_uncertainty = format_sig_figs(uncertainty * 1e3, num_sig_figs - 1)

println("\nAverage time over $num_runs iterations was $(formatted_average) ± $(formatted_uncertainty) miliseconds.")
println(" ")


using Plots

p1 = Plots.histogram(filtered_times,
    bins=50, # Adjust the number of bins as needed
    xlabel="Elapsed Time (seconds)",
    ylabel="Frequency",
    label="Numerical",
    title="Distribution of Computation Times",
    size=(1200, 900))

display(p1)

@showprogress 0.02 "Propagating num in loop..." for i in 1:num_runs
    elapsed = @elapsed begin
        p_1e_time = osc_prob_earth_barger_fast(energies, earth_paths, mixingPars_dict, anti=false)[:, :, 1, 1] # Get the 1e element only
        prob_time = osc_prob_night(energies, p_1e_time, mixingPars, solarModel.avgNeBoron)
    end
    elapsed_times[i] = elapsed
    # println("Run $i: Time to generate probs_matrix of size $(size(prob_time)) was $(elapsed) seconds.")
end


using Profile, ProfileSVG

sleep(3)
println(" beginning runs ")
ProfileSVG.@profview for i in 1:num_runs
    p_1e_time = osc_prob_earth_num_fast(energies, mixingPars, earth_lookup, earth_paths, anti=false)[:, :, 1, 1] # Get the 1e element only
    prob_time = osc_prob_night(energies, p_1e_time, mixingPars, solarModel.avgNeBoron)
end

println("Plotting profile data")
ProfileSVG.save("profile_large_v2.svg"; width=4000, height=1200)
println("saved data")


filtered_times = remove_outliers(elapsed_times)

average_time = mean(filtered_times)
uncertainty = std(filtered_times)

# Determine the number of significant figures based on the uncertainty
num_sig_figs_base = max(1, floor(Int, -log10(uncertainty)))
num_sig_figs = num_sig_figs_base + 1 # Request one more significant figure


formatted_average = format_sig_figs(average_time, num_sig_figs) * 1e3
formatted_uncertainty = format_sig_figs(uncertainty, num_sig_figs - 1) * 1e3

println("\nAverage time over $num_runs iterations was $(formatted_average) ± $(formatted_uncertainty) miliseconds.")
println(" ")

using Plots

Plots.histogram!(p1, filtered_times,
    bins=50, # Adjust the number of bins as needed
    xlabel="Computation time (s)",
    ylabel="Frequency",
    label="Barger",
    title="300x300 oscillogram. 12 layers",
    size=(1200, 900),
    margin=10mm,
    xlim=(0, 1))


display(p1)

=#
# -----------------------------------------------------------------------------#

using .Osc: osc_prob_both_fast
using .Osc.NumOsc: Fast

using Profile, ProfileSVG
num_runs = 1

p_1e_fast = Fast.osc_prob_earth(energies, mixingPars, earth_lookup, earth_paths, anti=false)[:, :, 1, 1] # Get the 1e element only
prob_day, prob_num_fast_fast = osc_prob_both_fast(energies, p_1e_fast, mixingPars, solarModel, process="8B")

println(" beginning runs ")
ProfileSVG.@profview for i in 1:num_runs
    p_1e_fast = Fast.osc_prob_earth(energies, mixingPars, earth_lookup, earth_paths, anti=false)[:, :, 1, 1] # Get the 1e element only
    prob_day, prob_num_fast_fast = osc_prob_both_fast(energies, p_1e_fast, mixingPars, solarModel, process="8B")
end

using BenchmarkTools

result1 = @benchmark Fast.osc_prob_earth(energies, mixingPars, earth_lookup, earth_paths, anti=false)[:, :, 1, 1] # Get the 1e element only
result2 = @benchmark osc_prob_both_fast(energies, p_1e_fast, mixingPars, solarModel, process="8B")

# Extract specific timing information:
println("Fast version:")
println("  Minimum time: $(minimum(result1.times) / 1e6) ms")
println("  Median time: $(median(result1.times) / 1e6) ms") 
println("  Mean time: $(mean(result1.times) / 1e6) ms")
println("  Maximum time: $(maximum(result1.times) / 1e6) ms")
println("  Memory allocated: $(result1.memory) bytes")
println("  Allocations: $(result1.allocs)")

println("Solar propagation:")
println("  Minimum time: $(minimum(result2.times) / 1e6) ms")
println("  Median time: $(median(result2.times) / 1e6) ms") 
println("  Mean time: $(mean(result2.times) / 1e6) ms")
println("  Maximum time: $(maximum(result2.times) / 1e6) ms")
println("  Memory allocated: $(result2.memory) bytes")
println("  Allocations: $(result2.allocs)")

println("Plotting profile data")
ProfileSVG.save("profile_large_v2.svg"; width=2000, height=1000)
println("saved data")

n_paths = size(p_1e_fast, 1)
y_coords = range(-1, stop=0, length=n_paths)


parulas = ColorScheme([RGB(0.2422, 0.1504, 0.6603),
        RGB(0.2504, 0.1650, 0.7076),
        RGB(0.2578, 0.1818, 0.7511),
        RGB(0.2647, 0.1978, 0.7952),
        RGB(0.2706, 0.2147, 0.8364),
        RGB(0.2751, 0.2342, 0.8710),
        RGB(0.2783, 0.2559, 0.8991),
        RGB(0.2803, 0.2782, 0.9221),
        RGB(0.2813, 0.3006, 0.9414),
        RGB(0.2810, 0.3228, 0.9579),
        RGB(0.2795, 0.3447, 0.9717),
        RGB(0.2760, 0.3667, 0.9829),
        RGB(0.2699, 0.3892, 0.9906),
        RGB(0.2602, 0.4123, 0.9952),
        RGB(0.2440, 0.4358, 0.9988),
        RGB(0.2206, 0.4603, 0.9973),
        RGB(0.1963, 0.4847, 0.9892),
        RGB(0.1834, 0.5074, 0.9798),
        RGB(0.1786, 0.5289, 0.9682),
        RGB(0.1764, 0.5499, 0.9520),
        RGB(0.1687, 0.5703, 0.9359),
        RGB(0.1540, 0.5902, 0.9218),
        RGB(0.1460, 0.6091, 0.9079),
        RGB(0.1380, 0.6276, 0.8973),
        RGB(0.1248, 0.6459, 0.8883),
        RGB(0.1113, 0.6635, 0.8763),
        RGB(0.0952, 0.6798, 0.8598),
        RGB(0.0689, 0.6948, 0.8394),
        RGB(0.0297, 0.7082, 0.8163),
        RGB(0.0036, 0.7203, 0.7917),
        RGB(0.0067, 0.7312, 0.7660),
        RGB(0.0433, 0.7411, 0.7394),
        RGB(0.0964, 0.7500, 0.7120),
        RGB(0.1408, 0.7584, 0.6842),
        RGB(0.1717, 0.7670, 0.6554),
        RGB(0.1938, 0.7758, 0.6251),
        RGB(0.2161, 0.7843, 0.5923),
        RGB(0.2470, 0.7918, 0.5567),
        RGB(0.2906, 0.7973, 0.5188),
        RGB(0.3406, 0.8008, 0.4789),
        RGB(0.3909, 0.8029, 0.4354),
        RGB(0.4456, 0.8024, 0.3909),
        RGB(0.5044, 0.7993, 0.3480),
        RGB(0.5616, 0.7942, 0.3045),
        RGB(0.6174, 0.7876, 0.2612),
        RGB(0.6720, 0.7793, 0.2227),
        RGB(0.7242, 0.7698, 0.1910),
        RGB(0.7738, 0.7598, 0.1646),
        RGB(0.8203, 0.7498, 0.1535),
        RGB(0.8634, 0.7406, 0.1596),
        RGB(0.9035, 0.7330, 0.1774),
        RGB(0.9393, 0.7288, 0.2100),
        RGB(0.9728, 0.7298, 0.2394),
        RGB(0.9956, 0.7434, 0.2371),
        RGB(0.9970, 0.7659, 0.2199),
        RGB(0.9952, 0.7893, 0.2028),
        RGB(0.9892, 0.8136, 0.1885),
        RGB(0.9786, 0.8386, 0.1766),
        RGB(0.9676, 0.8639, 0.1643),
        RGB(0.9610, 0.8890, 0.1537),
        RGB(0.9597, 0.9135, 0.1423),
        RGB(0.9628, 0.9373, 0.1265),
        RGB(0.9691, 0.9606, 0.1064),
        RGB(0.9769, 0.9839, 0.0805)],
    "Parula",
    "From MATLAB")



p2 = heatmap(
    energies*1e3,
    y_coords,
    prob_num_fast_fast,
    xlabel=L"E_{\nu} \, (\mathrm{MeV})",
    ylabel=L"\cos(z)",
    title="Night-time probability",
    colormap=cgrad(parulas),
    #colormap=:berlin,
    size=(1200, 900),
    margin=10mm,
    #clim=(0.20, 0.55)
)

# Display the plot
display(p2)
sleep(100)
####################################################################
=#