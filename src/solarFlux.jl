# Loads solar fluxes
using Interpolations
include("../src/objects.jl")

# Add zero and e_max to the fluxes for Interpolations
extended_energies = [0.0; energies]
extended_flux8B = [0.0; flux8B]
extended_fluxHep = [0.0; fluxHep]

if bins.max > energies[end]
    extended_energies = [0.0; energies; bins.max]
    extended_flux8B = [0.0; flux8B; 0.0]
    extended_fluxHep = [0.0; fluxHep; 0.0]
end


# Interpolate fluxes
flux_8B = LinearInterpolation(extended_energies, extended_flux8B)
flux_hep = LinearInterpolation(extended_energies, extended_fluxHep)
