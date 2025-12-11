#=
solarFlux.jl

Solar neutrino flux processing for the Solar Oscillation Fitter.
This module loads and processes solar neutrino flux data for different
nuclear processes in the Sun, creating interpolated and normalized flux functions.

Key Features:
- Loading of 8B and HEP neutrino flux data
- Energy range extension for proper interpolation
- Flux normalization for integration over energy
- Linear interpolation for continuous flux functions

The processed fluxes are used in unoscillated event rate calculations
and are combined with oscillation probabilities to predict observed rates.

Author: [Author name]
=#

# Load solar neutrino fluxes and create interpolated functions
using Interpolations  # For flux interpolation
using QuadGK          # For flux normalization integrals
include("../src/core.jl")

# Extend energy range for proper interpolation
# Add zero energy point and ensure coverage up to maximum analysis energy
extended_energies = [0.0; energies]  # Add zero energy point
extended_flux8B = [0.0; flux8B]      # 8B flux with zero at E=0
extended_fluxHep = [0.0; fluxHep]    # HEP flux with zero at E=0

# Extend to maximum energy if needed for analysis range
if Etrue_bins.max > energies[end]
    extended_energies = [0.0; energies; Etrue_bins.max]
    extended_flux8B = [0.0; flux8B; 0.0]    # Zero flux beyond data range
    extended_fluxHep = [0.0; fluxHep; 0.0]  # Zero flux beyond data range
end

# Create interpolated flux functions (unnormalized)
flux_8B_noNormalisation = LinearInterpolation(extended_energies, extended_flux8B)
flux_hep_noNormalisation = LinearInterpolation(extended_energies, extended_fluxHep)

# Calculate normalization integrals
lower_limit = extended_energies[1]    # Start from zero energy
upper_limit = extended_energies[end]  # End at maximum energy

# Integrate flux functions over full energy range
integral_flux_8B, _ = quadgk(flux_8B_noNormalisation, lower_limit, upper_limit)
integral_flux_hep, _ = quadgk(flux_hep_noNormalisation, lower_limit, upper_limit)

# Create normalized flux functions
# These give flux per unit energy, normalized to unit integral
flux_8B = LinearInterpolation(extended_energies, extended_flux8B / integral_flux_8B)
flux_hep = LinearInterpolation(extended_energies, extended_fluxHep / integral_flux_hep)
