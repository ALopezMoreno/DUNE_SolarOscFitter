using LinearAlgebra
using DataStructures
using StaticArrays
using Distributions
using ArraysOfArrays, StructArrays

include("makePaths.jl")

################################################################################
# Struct Definitions
################################################################################

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

################################################################################
# Calculation of the oscillation probability
################################################################################


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

@inline function osc_kernel(U::SMatrix{3,3,ComplexF64}, H::SVector{3,Float64}, e::Float64, l::Float64)
    phase_factors = exp.(2.5338653580781976im * (l / e) .* H) # returns SVector
    # Avoid creating Diagonal(phase_factors): broadcast instead
    tmp = U .* phase_factors'  # broadcasting phase_factors onto columns
    return tmp * adjoint(U)    # U'
end

function osc_kernel_barger(U::AbstractMatrix{<:Number}, P::AbstractVector, H::AbstractVector, e::Real, l::Real)
    phase_factors = exp.(2.5338653580781976 * 1im * (l / e) .* H)
    p = U * sum(P[i] * phase_factors[i] for i in eachindex(P)) * U'
end

function LMA_angle(energy, mixingPars, N_e)
    th12 = mixingPars.θ₁₂
    beta = (2 .* sqrt(2) .* 5.4489e-5 .* cos(mixingPars.θ₁₃)^2 .* N_e .* energy) ./ mixingPars.Δm²₂₁
    matterAngle = (cos(2 * th12) .- beta) ./ sqrt.((cos(2 * th12) .- beta) .^ 2 .+ sin(2 * th12)^2)
    return 0.5 .* acos.(matterAngle)
end

function get_eigen_barger(U, H_vac, H, e, rho)
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

    # α = fac + dmVacVac[1, 2] + dmVacVac[1, 3]
    # β = dmVacVac[1, 2] * dmVacVac[1, 3] + fac * (dmVacVac[1, 2] * (1 - real(U[1, 2])^2 - imag(U[1, 2])^2) + Δm²₃₁ * (1 - real(U[1, 3])^2 - imag(U[1, 3])^2))
    # γ = fac * dmVacVac[1, 2] * dmVacVac[1, 3] * (real(U[1, 1])^2 + imag(U[1, 1])^2)

    brac = sqrt(α^2 - 3 * β)

    arg = (2 * α^3 - 9 * α * β + 27 * γ) / (2 * brac^3)

    θ⁰ = acos.(clamp(arg, -1.0, 1.0))
    θ⁺ = θ⁰ + 2π
    θ⁻ = θ⁰ - 2π

    λ₁ = -(2 / 3) * brac * cos(θ⁰ / 3) + m²₁ - α / 3
    λ₂ = -(2 / 3) * brac * cos(θ⁻ / 3) + m²₁ - α / 3
    λ₃ = -(2 / 3) * brac * cos(θ⁺ / 3) + m²₁ - α / 3

    # WE'RE WELL BELOW THE EIGENVALUE CROSSING SO WE DON'T NEED TO WORRY ABOUT ORDERING
    # Get eigenvector matrix from Lagrange's formula

    H₁ = H - Diagonal(fill(λ₁, 3))
    H₂ = H - Diagonal(fill(λ₂, 3))
    H₃ = H - Diagonal(fill(λ₃, 3))

    P₁ = (H₂ * H₃) / ((λ₁ - λ₂) * (λ₁ - λ₃))
    P₂ = (H₃ * H₁) / ((λ₂ - λ₁) * (λ₂ - λ₃))
    P₃ = (H₁ * H₂) / ((λ₃ - λ₁) * (λ₃ - λ₂))

    return [P₁, P₂, P₃], [λ₁, λ₂, λ₃]
end

function get_eigen_num(H::Hermitian{ComplexF64,SMatrix{3,3,ComplexF64,9}})
    tmp = eigen(H)
end

function get_H_barger(U, H_vac, e, rho, anti::Bool=false)
    H = MMatrix{3,3}(H_vac)

    if anti
        fac = -rho * e
    else
        fac = +rho * e
    end

    v = U[1, :]
    H += fac * v * v'

    H = Hermitian(SMatrix(H))

    tmpMat, tempVec = get_eigen_barger(U, H_vac, H, e, rho) # --- test speed wrt num ---
    return tmpMat, tempVec
end


function get_H_num(H_vac, e, rho, anti::Bool=false)
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
    tmp = get_eigen_num(H) # --- test speed wrt barger ---
    tmp.vectors, tmp.values
end

function osc_reduce_barger(Mix, matter_matrices, path, e, anti::Bool)
    X = mapreduce((p, (m, n)) -> osc_kernel_barger(Mix, m, n, e, p), *, path, reverse(matter_matrices))
    A = abs2.(Mix' * X)
    return A
end

function osc_reduce_num(Mix_dagger::SMatrix{3,3,ComplexF64}, matter_matrices, path, e::Float64, anti::Bool)
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

function matter_osc_per_e_barger(U, H_eff, e, rho_Array, l_Array, anti)
    matter_matrices = map(rho_vec -> (get_H_barger.(Ref(U), Ref(H_eff), e, rho_vec, anti)), rho_Array)
    stack(map((path, matter) -> (osc_reduce_barger(U, matter, path, e, anti)), l_Array, matter_matrices))
end

function matter_osc_per_e_num(U, H_eff, e, rho_Array, l_Array, anti)
    matter_matrices = map(rho_vec -> (get_H_num.(Ref(H_eff), e, rho_vec, anti)), rho_Array)
    stack(map((path, matter) -> (osc_reduce_num(U, matter, path, e, anti)), l_Array, matter_matrices))
end

function matter_osc_per_e_barger_fast(U, H_eff, e, index_array, l_Array, anti)

    lookup_matrices = map(rho_vec -> (get_H_barger.(Ref(U), Ref(H_eff), e, rho_vec, anti)), earth_lookup)
    matter_matrices = map(indices -> lookup_matrices[indices], index_array)

    stack(map((path, matter) -> (osc_reduce_barger(U, matter, path, e, anti)), l_Array, matter_matrices))
end

function matter_osc_per_e_num_fast(lookup_density, Mix_dagger, H_eff, e, index_array, l_Array, anti)

    lookup_matrices = map(rho_vec -> get_H_num.(Ref(H_eff), e, rho_vec, anti), lookup_density)
    matter_matrices = map(indices -> lookup_matrices[indices], index_array)

    stack(map((path, matter) -> osc_reduce_num(Mix_dagger, matter, path, e, anti), l_Array, matter_matrices))
end


function osc_prob_earth_barger(E::AbstractVector{<:Real}, paths, params::NamedTuple; anti=false)
    U, H = get_matrices(params)
    # We work on the mass basis

    rho_Array = [[s.avgRho for s in path.segments] for path in paths]
    l_Array = [[s.length for s in path.segments] for path in paths]

    p1e = stack(map(e -> matter_osc_per_e_barger(U, H, e, rho_Array, l_Array, anti), E))[1, 1, :, :]
end

function osc_prob_earth_num(E::AbstractVector{<:Real}, paths, params::NamedTuple; anti=false)
    U, H = get_matrices(params)
    Uc = anti ? conj.(U) : U
    H_eff = Uc * Diagonal{Complex{eltype(H)}}(H) * adjoint(Uc)

    rho_Array = [[s.avgRho for s in path.segments] for path in paths]
    l_Array = [[s.length for s in path.segments] for path in paths]

    p1e = stack(map(e -> matter_osc_per_e_num(U, H_eff, e, rho_Array, l_Array, anti), E))[1, 1, :, :]
end

function osc_prob_earth_barger_fast(E::AbstractVector{<:Real}, paths, params::NamedTuple; anti=false)
    U, H = get_matrices(params)
    # We work on the mass basis

    index_array = [[s.index for s in path.segments] for path in paths]
    l_Array = [[s.length for s in path.segments] for path in paths]

    p1e = stack(map(e -> matter_osc_per_e_barger_fast(U, H, e, index_array, l_Array, anti), E))[1, 1, :, :]
end

function osc_prob_earth_num_fast(E::AbstractVector{<:Real}, params::oscPars, lookup_density, paths; anti=false)
    U, H = get_matrices(params)
    Udag = adjoint(U)
    Uc = anti ? conj.(U) : U
    H_eff = Uc * Diagonal{Complex{eltype(H)}}(H) * adjoint(Uc)

    # We reverse the indices here to avoid reversing in the computation of the amplitude
    index_array = [reverse([s.index for s in path.segments]) for path in paths]
    l_Array = [reverse([s.length for s in path.segments]) for path in paths]

    p1e = stack(map(e -> matter_osc_per_e_num_fast(lookup_density, Udag, H_eff, e, index_array, l_Array, anti), E))[1, 1, :, :]
end


function osc_prob_night(E::Vector{Float64}, matrix_p_1e::Matrix{Float64}, mixingPars, prod_density::Float64)
    th13 = mixingPars.θ₁₃

    solarAngle = permutedims(LMA_angle(E, mixingPars, prod_density))
    probs = cos(th13)^2 * ((cos.(2 .* solarAngle) .* matrix_p_1e) .+ cos(th13)^2 .* sin.(solarAngle) .^ 2) .+ sin(th13)^4

    return probs
end

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

theme(:dracula)
scalefontsizes(2)

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
    sin2_th12=sin2_th12_true,
    sin2_th13=sin2_th13_true,
    θ₁₃=asin(sqrt(sin2_th13_true)),
    θ₁₂=asin(sqrt(sin2_th12_true)),
    θ₂₃=asin(sqrt(0.5)), # Example value
    δCP=0.0,   # Example value
    Δm²₂₁=7.5e-5,
    Δm²₃₁=2.5e-3,
    m₀=1e-9 # Example value
)

mixingPars = oscPars(mixingPars_dict.Δm²₂₁, asin(sqrt(mixingPars_dict.sin2_th12)), asin(sqrt(mixingPars_dict.sin2_th13)))


energies = collect(range(3, stop=20, length=300)) * 1e-3


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


# -----------------------------------------------------------------------------#


p_1e_num = osc_prob_earth_num_fast(energies, mixingPars, earth_lookup, earth_paths, anti=false)[:, :, 1, 1] # Get the 1e element only
p_1e_bar = osc_prob_earth_barger_fast(energies, earth_paths, mixingPars_dict, anti=false)[:, :, 1, 1] # Get the 1e element only

prob_num = osc_prob_night(energies, p_1e_num, mixingPars, solarModel.avgNeBoron)
prob_bar = osc_prob_night(energies, p_1e_bar, mixingPars, solarModel.avgNeBoron)

n_paths = size(p_1e_bar, 1)
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
    energies,
    y_coords,
    prob_num,
    xlabel=L"E_{\nu} \, (\mathrm{GeV})",
    ylabel=L"\cos(z)",
    title=L"P_{\nu_1 \to \nu_e}\mathrm{ (Earth)}",
    colormap=cgrad(parulas),
    # colormap=:berlin,
    size=(1200, 900),
    margin=10mm,
    #clim=(0.20, 0.55)
)

# Display the plot
display(p2)
sleep(100)

=#