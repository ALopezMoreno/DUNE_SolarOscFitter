
#=
solarModel.jl

Solar model data loading for neutrino production calculations.
This module loads solar structure data and neutrino flux information
required for calculating neutrino production rates and oscillation
probabilities in the solar matter.

Key Features:
- Loading of solar density and electron density profiles
- Neutrino production region data for 8B and HEP processes
- Calculation of production-weighted average densities
- Solar neutrino flux data loading and normalization
- Integration with oscillation probability calculations

The solar model data is essential for calculating MSW (matter) effects
during neutrino propagation through the Sun's interior.

Author: [Author name]
=#

# Load solar model data (density profiles and production regions)
if isfile(solarModelFile)
    solarModel = jldopen(solarModelFile, "r") do file
        # Load solar structure datasets
        radii = file["radii"]                    # Solar radii (fraction of solar radius)
        prodFractionBoron = file["prodFractionBoron"]  # 8B production fraction vs radius
        prodFractionHep = file["prodFractionHep"]      # HEP production fraction vs radius
        n_e = file["n_e"]                        # Electron density vs radius
        
        # Calculate production-weighted average electron densities
        # These are used for fast oscillation probability calculations
        avgNeBoron = sum(prodFractionBoron .* n_e) / sum(prodFractionBoron)
        avgNeHep = sum(prodFractionHep .* n_e) / sum(prodFractionHep)

        # Return solar model as named tuple
        return (radii=radii, 
                prodFractionBoron=prodFractionBoron, 
                prodFractionHep=prodFractionHep, 
                n_e=n_e, 
                avgNeBoron=avgNeBoron, 
                avgNeHep=avgNeHep)
    end
else
    error("Solar model file not found: $solarModelFile")
end

# Load solar neutrino flux data
if isfile(flux_file_path)
    energies, flux8B, fluxHep = jldopen(flux_file_path, "r") do file
        # Load energy-dependent flux spectra
        energies = file["energies"]              # Neutrino energies (GeV)
        flux8B = file["flux8B"]                  # 8B neutrino flux spectrum
        fluxHep = file["fluxHep"]                # HEP neutrino flux spectrum
        
        # Load total flux normalizations
        # Factor 0.514 accounts for updated 8B flux measurements
        global total_flux_8B = file["total8B"] * 0.514
        global total_flux_hep = file["totalHep"]
        
        return energies, flux8B, fluxHep, total_flux_8B, total_flux_hep
    end
else
    error("Solar flux file not found: $flux_file_path")
end