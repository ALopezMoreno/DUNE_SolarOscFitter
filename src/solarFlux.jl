# Loads solar fluxes
using Interpolations
using QuadGK
include("../src/objects.jl")

# Add zero and e_max to the fluxes for Interpolations
extended_energies = [0.0; energies]
extended_flux8B = [0.0; flux8B]
extended_fluxHep = [0.0; fluxHep]

if Etrue_bins.max > energies[end]
    extended_energies = [0.0; energies; Etrue_bins.max]
    extended_flux8B = [0.0; flux8B; 0.0]
    extended_fluxHep = [0.0; fluxHep; 0.0]
end

# Interpolate fluxes
flux_8B_noNormalisation = LinearInterpolation(extended_energies, extended_flux8B)
flux_hep_noNormalisation = LinearInterpolation(extended_energies, extended_fluxHep)

# Define the integration limits
lower_limit = extended_energies[1]
upper_limit = extended_energies[end]

# Integrate the functions using QuadGK
integral_flux_8B, _ = quadgk(flux_8B_noNormalisation, lower_limit, upper_limit)
integral_flux_hep, _ = quadgk(flux_hep_noNormalisation, lower_limit, upper_limit)

# Normalise accordingly
flux_8B = LinearInterpolation(extended_energies, extended_flux8B / integral_flux_8B)
flux_hep = LinearInterpolation(extended_energies, extended_fluxHep / integral_flux_hep)
