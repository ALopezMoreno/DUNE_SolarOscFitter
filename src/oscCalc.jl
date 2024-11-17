"""
Neutrino Oscillation Module
Functions and data structures for calculating neutrino oscillation probabilities using the MSW effect

# Main Components

- `OscillationParameters`: A mutable struct to hold the oscillation parameters with default values.
- `update_oscpars!`: A function to update the oscillation parameters with new values.
- `mswProb`: A function to calculate MSW oscillation probabilities.
- `peanutsProb`: A function to calculate oscillation probabilities using the PEANuTS model.
- `solarSurfaceProbs`: A function to calculate integrated neutrino oscillation probabilities at the solar surface using a specified solar model.

# Constants

- `G_f`: The Fermi constant.
- `m32`: The mass splitting constant, currently set to the normal ordering (NO) value.
"""

include("../src/objects.jl")

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



function solarSurfaceProbs(params::OscillationParameters, E_true::Float64, solarModel; process="8B")
    """
    Calculate the neutrino oscillation probabilities at the solar surface for a given energy and solar model.

    # Arguments
    - `params::OscillationParameters`: The oscillation parameters used for the calculation.
    - `E_true::Float64`: The true energy of the neutrino.
    - `solarModel: the solar model object (radius, electron density, and production regions).

    # Returns
    - A tuple `(pBoron, pHep)` where:
    - `pBoron`: The integrated probability for Boron neutrinos.
    - `pHep`: The integrated probability for Hep neutrinos.
    """

    # Calculate the neutrino oscillation probabilities using the electron density
    enuOscProb = peanutsProb(params, E_true, solarModel.n_e)

    # Define a local function to integrate probabilities over the solar radius
    function integrateProb(radii::Vector{Float64}, prodFraction::Vector{Float64}, enuOscProb::Vector{Float64})
        # Calculate the bin widths for integration
        b_edges = [0; radii]
        @views bin_widths = diff(b_edges)
        # Integrate the probabilities for Boron and Hep neutrinos
        integral = @inbounds sum(@. prodFraction * bin_widths * enuOscProb)
        return integral
    end

    if process == "8B"
        prodFraction = solarModel.prodFractionBoron
    elseif process == "hep"
        prodFraction = solarModel.prodFractionHep
    else
        error("Invalid process specified. Please use '8B' or 'hep'.")
    end

    # Call the integration function with the loaded data
    prob_nue = integrateProb(solarModel.radii, prodFraction, enuOscProb)

    return prob_nue
end


# TESTING
using Test

function test_solar_surface_probs_8b_valid_inputs()
    # Assuming OscillationParameters and a mock solarModel are defined elsewhere
    params = OscillationParameters()  # Fill with appropriate values
    E_true = 10.0  # Example energy value

    # Call the function with "8B" process
    @time prob_nue = solarSurfaceProbs(params, E_true, solarModel, process="8B")


end
