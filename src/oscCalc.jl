using StaticArrays
using JLD2

# Fermi constant
const G_f = 5.4489e-5
# For now, we keep the remaining mass splitting constant and in NO
const m32 = 2.43e-3
# Define a mutable struct to hold the oscillation parameters with default values
mutable struct OscillationParameters
    oscpars::SVector{3,Float64}
    function OscillationParameters()
        new(SVector{3,Float64}(0.307, 0.022, 7.53e-5))  # Default values
    end
end

# Function to update the oscillation parameters
function update_oscpars!(params::OscillationParameters, new_oscpars::SVector{3,Float64})
    @assert 0 <= new_oscpars[1] <= 1 "Error: `new_oscpars[1]` is out of bounds. Expected between 0 and 1, got $(new_oscpars[1])."
    @assert 0 <= new_oscpars[2] <= 1 "Error: `new_oscpars[2]` is out of bounds. Expected between 0 and 1, got $(new_oscpars[2])."
    @assert new_oscpars[3] != 0 "Error: `new_oscpars[3]` is zero, which would cause division by zero."
    params.oscpars = new_oscpars
end



function mswProb(params::OscillationParameters, E_true::Float64, n_e::Float64, allFlavours=false)::SVector{3,Float64}
    """
    Calculate the Mikheyev-Smirnov-Wolfenstein (MSW) oscillation probabilities.

    # Arguments
    - `oscpars::SVector{3, Float64}`: A static vector containing oscillation parameters as sin^2(th12), sin^2(th13), Dm^2_{21}. 
    The first two elements must be between 0 and 1, and the third must be non-zero.
    - `E_true::Float64`: The true energy value, which must be positive.
    - `n_e::Float64`: The electron density, which must be non-negative.

    # Returns
    - `SVector{3, Float64}`: A static vector containing the calculated oscillation probabilities.

    # Throws
    - `AssertionError`: If any of the input parameters are out of their expected bounds.
    """

    oscpars = params.oscpars
    @assert n_e >= 0 "n_e must be non-negative"
    @assert E_true > 0 "E_true must be positive"

    @fastmath @inbounds begin
        # Calculate the cosine of twice the angle theta_12 using the oscillation parameter
        c2th12 = cos(2 * asin(sqrt(oscpars[1])))

        # Calculate the beta parameter, which is a function of electron density, Fermi constant, energy, and oscillation parameters
        Acc = 2 * sqrt(2) * n_e * G_f * E_true * (1 - oscpars[2])
        # dm21m = sqrt((oscpars[3] * c2th12 - Acc)^2 + (oscpars[3]^2 * (1 - c2th12^2))) do not use
        beta = Acc / oscpars[3]

        # Calculate the modified cosine of twice the angle theta_12 in matter
        c2th12m = (c2th12 - beta) / sqrt((c2th12 - beta)^2 + (1 - c2th12^2))

        # Calculate the modified sine squared of theta_13 in matter
        s13m = sqrt(oscpars[2]) # so far we leave it at vacuum value

        # Initialize a mutable vector to store the oscillation probabilities
        @views Probs = MVector{3,Float64}(0.0, 0.0, 0.0)

        # Calculate the probability for the electron neutrino flavor
        Probs[1] = 1 / 2 * (1 - oscpars[2]) * (1 - s13m^2) * (1 + c2th12 * c2th12m) + oscpars[2] * s13m^2

        if allFlavours
            # Calculate the probability for the muon neutrino flavor
            Probs[2] = 1 / 2 * (1 - oscpars[2]) * (1 - c2th12 * c2th12m)

            # Calculate the probability for the tau neutrino flavor
            Probs[3] = 1 / 2 * (1 - oscpars[2]) * s13m^2 * (1 + c2th12 * c2th12m) + oscpars[2] * (1 - s13m^2)
        end
    end

    # Convert the mutable vector of probabilities to an immutable static vector and return it
    return SVector{3}(Probs)
end



@inbounds function peanutsProb(params::OscillationParameters, E_true::Float64, n_e::Float64, allFlavours::Bool=false)::SVector{3,Float64}
    """
    Slightly more precise calculation of the oscillation probabilities using the PEANuTS (Probabilistic Estimation of Atmospheric Neutrino Transitions) model.
    The ordering has been set to NO with dm32 set to the PDG central value. This should not affect solar oscillations to the precision achievable by DUNE

    # Arguments
    - `params::OscillationParameters`: An object containing the current oscillation parameters.
    - `E_true::Float64`: The true energy of the neutrino, which must be positive.
    - `n_e::Float64`: The electron density, which must be non-negative.
    - `allFlavours::Bool`: A flag indicating whether to calculate probabilities for all neutrino flavors. Defaults to `false`.

    # Returns
    - `SVector{3, Float64}`: A static vector containing the calculated oscillation probabilities for electron neutrinos. If `allFlavours` is `true`, the vector will also include probabilities for muon and tau neutrinos.

    # Throws
    - `AssertionError`: If `E_true` is not positive or `n_e` is negative.
    - `Error`: If `m_ee` is zero, leading to division by zero.
    """

    # Extract oscillation parameters from the input struct
    oscpars = params.oscpars

    # Calculate the effective potential due to electron density
    Acc = 2 * sqrt(2) * E_true * G_f * n_e

    # Calculate the mixing angle theta_13 and its cosine of twice the angle
    th13 = asin(sqrt(oscpars[2]))
    c2th13 = cos(2 * th13)

    # Calculate the cosine of twice the angle theta_12
    c2th12 = cos(2 * asin(sqrt(oscpars[1])))

    # Calculate the effective mass term
    m_ee = (1 - oscpars[1]) * (oscpars[3] + m32) + oscpars[1] * m32

    # Function to calculate the modified cosine of twice the angle theta_13 in matter
    function calculate_c2th13m(c2th13::Float64, Acc::Float64, m_ee::Float64)::Float64
        if m_ee == 0
            error("Division by zero: m_ee cannot be zero")
        end
        return (c2th13 - Acc / m_ee) / sqrt((c2th13 - Acc / m_ee)^2 + 1 - c2th13^2)
    end

    # Function to calculate the modified cosine of twice the angle theta_12 in matter
    function calculate_c2th12m(c2th12::Float64, Accm::Float64, oscpar3::Float64, th13m::Float64, th13::Float64)::Float64
        return (c2th12 - Accm / oscpar3) / sqrt((c2th12 - Accm / oscpar3)^2 + (1 - c2th12^2) * cos(th13m - th13)^2)
    end

    # Calculate the modified cosine of twice the angle theta_13 in matter
    c2th13m = calculate_c2th13m(c2th13, Acc, m_ee)

    # Calculate the modified mixing angle theta_13 in matter
    th13m = 1 / 2 * acos(c2th13m)

    # Calculate the modified effective potential
    Accm = Acc * cos(th13m)^2 + m_ee * sin(th13m - th13)^2

    # Calculate the modified cosine of twice the angle theta_12 in matter
    c2th12m = calculate_c2th12m(c2th12, Accm, oscpars[3], th13m, th13)

    # Initialize a mutable vector to store the oscillation probabilities
    @views Probs = MVector{3,Float64}(0.0, 0.0, 0.0)

    # Calculate the probability for the electron neutrino flavor
    Probs[1] = 1 / 2 * (1 - oscpars[2]) * (1 - sin(th13m)^2) * (1 + c2th12 * c2th12m) + oscpars[2] * sin(th13m)^2

    # If allFlavours is true, calculate probabilities for muon and tau neutrinos (not implemented yet)
    # Probs[2] and Probs[3] would be calculated here if needed

    # Convert the mutable vector of probabilities to an immutable static vector and return it
    return SVector{3}(Probs)
end

function peanutsProb(params::OscillationParameters, E_true::Float64, n_e_array::Vector{Float64})::Vector{Float64}
    """
    Calculate the electron neutrino flavor probability for each electron density in `n_e_array` using the PEANuTS model.

    # Arguments
    - `params::OscillationParameters`: An object containing the current oscillation parameters.
    - `E_true::Float64`: The true energy of the neutrino, which must be positive.
    - `n_e_array::Vector{Float64}`: A vector of electron densities, each of which must be non-negative.

    # Returns
    - `Vector{Float64}`: A vector containing the calculated oscillation probabilities for electron neutrinos for each `n_e` in `n_e_array`.

    # Throws
    - `AssertionError`: If `E_true` is not positive or any `n_e` is negative.
    - `Error`: If `m_ee` is zero, leading to division by zero.
    """

    # Extract oscillation parameters from the input struct
    oscpars = SVector{3,Float64}(params.oscpars)

    # Precompute quantities that do not depend on n_e
    th13 = asin(sqrt(oscpars[2]))
    c2th13 = cos(2 * th13)
    c2th12 = cos(2 * asin(sqrt(oscpars[1])))
    m_ee = (1 - oscpars[1]) * (oscpars[3] + m32) + oscpars[1] * m32

    # Check for division by zero
    if m_ee == 0
        throw(Error("m_ee is zero, leading to division by zero."))
    end

    # Calculate Acc for each n_e using broadcasting
    @inbounds Acc = 2 * sqrt(2) * E_true * G_f .* n_e_array

    # Calculate the modified cosine of twice the angle theta_13 in matter
    c2th13m = (c2th13 .- Acc ./ m_ee) ./ sqrt.((c2th13 .- Acc ./ m_ee) .^ 2 .+ 1 .- c2th13^2)

    # Calculate the modified mixing angle theta_13 in matter
    th13m = 1 / 2 .* acos.(c2th13m)

    # Calculate the modified effective potential
    Accm = Acc .* cos.(th13m) .^ 2 .+ m_ee .* sin.(th13m .- th13) .^ 2

    # Calculate the modified cosine of twice the angle theta_12 in matter
    c2th12m = (c2th12 .- Accm ./ oscpars[3]) ./ sqrt.((c2th12 .- Accm ./ oscpars[3]) .^ 2 .+ (1 .- c2th12^2) .* cos.(th13m .- th13) .^ 2)

    # Calculate the probability for the electron neutrino flavor
    electron_probs = 1 / 2 .* (1 .- oscpars[2]) .* (1 .- sin.(th13m) .^ 2) .* (1 .+ c2th12 .* c2th12m) .+ oscpars[2] .* sin.(th13m) .^ 2

    return electron_probs
end



function solarSurfaceProbs(params::OscillationParameters, E_true::Float64, solarModel="inputs/AGSS09_high_z")
    """
    Calculate the neutrino oscillation probabilities at the solar surface for a given energy and solar model.

    # Arguments
    - `params::OscillationParameters`: The oscillation parameters used for the calculation.
    - `E_true::Float64`: The true energy of the neutrino.
    - `solarModel::String`: The name of the solar model file (default is "inputs/AGSS09_high_z").

    # Returns
    - A tuple `(pBoron, pHep)` where:
    - `pBoron`: The integrated probability for Boron neutrinos.
    - `pHep`: The integrated probability for Hep neutrinos.

    # Throws
    - An error if the specified solar model file does not exist.
    - An error if there is a dimension mismatch between the vectors used in integration.
    """

    filePath = solarModel * ".jld2"

    # Check if the file exists and read the datasets
    if isfile(filePath)
        radii, prodFractionBoron, prodFractionHep, n_e = jldopen(filePath, "r") do file
            # Load the necessary datasets from the file
            radii = file["radii"]
            prodFractionBoron = file["prodFractionBoron"]
            prodFractionHep = file["prodFractionHep"]
            n_e = file["n_e"]
            return (radii, prodFractionBoron, prodFractionHep, n_e)
        end
    else
        error("File not found: $filePath")
    end

    # Calculate the neutrino oscillation probabilities using the electron density
    enuOscProb = peanutsProb(params, E_true, n_e)

    # Define a local function to integrate probabilities over the solar radius
    function integrateProb(radii::Vector{Float64}, prodFractionBoron::Vector{Float64}, prodFractionHep::Vector{Float64}, enuOscProb::Vector{Float64})
        # Ensure all vectors have the same length
        if length(prodFractionBoron) != length(enuOscProb) || length(prodFractionHep) != length(enuOscProb)
            error("Dimension mismatch: Vectors must be of the same length.")
        end
        # Calculate the bin widths for integration
        b_edges = [0; radii]
        @views bin_widths = diff(b_edges)
        # Integrate the probabilities for Boron and Hep neutrinos
        intBoron = sum(@. prodFractionBoron * bin_widths * enuOscProb)
        intHep = sum(@. prodFractionHep * bin_widths * enuOscProb)
        return (intBoron, intHep)
    end

    # Call the integration function with the loaded data
    pBoron, pHep = integrateProb(radii, prodFractionBoron, prodFractionHep, enuOscProb)

    return pBoron, pHep
end