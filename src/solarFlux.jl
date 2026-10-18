# Load solar neutrino fluxes and create interpolated functions (solar_flux_spectra in
# modelInputs.jl: zero flux added at E = 0 and, if needed, at the top of the analysis range;
# normalised to unit integral; an error outside the table)
include(joinpath(@__DIR__, "core.jl"))
include(joinpath(@__DIR__, "modelInputs.jl"))

_spectra = solar_flux_spectra(energies, flux8B, fluxHep, Etrue_bins.max; source = flux_file_path)
extended_energies = _spectra.energies
flux_8B_noNormalisation, flux_hep_noNormalisation = _spectra.raw_8B, _spectra.raw_hep
integral_flux_8B, integral_flux_hep = _spectra.integral_8B, _spectra.integral_hep

# Flux per unit energy, normalised to unit integral
flux_8B, flux_hep = _spectra.flux_8B, _spectra.flux_hep
