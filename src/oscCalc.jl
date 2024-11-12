using StaticArrays

# Fermi constant
const G_f = 5.4489e-5
# Default oscillation parametres (PDG global fit 2024). Ordering: (sin^2(th12), sin^2(th13), Dm^2_{21})
oscpars = SVector{3}([0.303, 0.022, 7.41e-5])


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
function mswProb(oscpars::SVector{3, Float64}, E_true::Float64, n_e::Float64)::SVector{3, Float64}
    @assert 0 <= oscpars[1] <= 1 "Error: `oscpars[1]` is out of bounds. Expected between 0 and 1, got $(oscpars[1])."
    @assert 0 <= oscpars[2] <= 1 "Error: `oscpars[2]` is out of bounds. Expected between 0 and 1, got $(oscpars[2])."
    @assert oscpars[3] != 0 "Error: `oscpars[3]` is zero, which would cause division by zero."
    @assert n_e >= 0 "n_e must be non-negative"
    @assert E_true > 0 "E_true must be positive"

    @fastmath @inbounds begin
        # Calculate the beta parameter, which is a function of electron density, Fermi constant, energy, and oscillation parameters
        beta = (2 * sqrt(2) * n_e * G_f * E_true * (1 - oscpars[2])) / oscpars[3]
        
        # Calculate the cosine of twice the angle theta_12 using the oscillation parameter
        c2th12 = cos(2 * acos(sqrt(oscpars[1])))
        
        # Calculate the modified cosine of twice the angle theta_12 in matter
        c2th12m = (c2th12 - beta) / sqrt((c2th12 - beta)^2 + (1 - c2th12^2))sin^2(th12), sin^2(th13), Dm^2_{21}
        
        # Calculate the modified sine squared of theta_13 in matter
        s13m = oscpars[2] * (1 + beta)
        
        # Initialize a mutable vector to store the oscillation probabilities
        @views Probs = MVector{3, Float64}(0.0, 0.0, 0.0)
        
        # Calculate the probability for the electron neutrino flavor
        Probs[1] = 1 / 2 * (1 - oscpars[2]) * (1 - s13m^2) * (1 + c2th12 * c2th12m) + oscpars[2] * s13m^2
        
        # Calculate the probability for the muon neutrino flavor
        Probs[2] = 1 / 2 * (1 - oscpars[2]) * (1 - c2th12 * c2th12m)
        
        # Calculate the probability for the tau neutrino flavor
        Probs[3] = 1 / 2 * (1 - oscpars[2]) * s13m^2 * (1 + c2th12 * c2th12m) + oscpars[2] * (1 - s13m^2)
    end
    
    # Convert the mutable vector of probabilities to an immutable static vector and return it
    return SVector{3}(Probs)
end


