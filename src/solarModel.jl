
# Check if the solar model file exists and read the datasets
if isfile(solarModelFile)
    solarModel = jldopen(solarModelFile, "r") do file
        # Load the necessary datasets from the file
        radii = file["radii"]
        prodFractionBoron = file["prodFractionBoron"]
        prodFractionHep = file["prodFractionHep"]
        n_e = file["n_e"]
        # Calculate weighted averages
        avgNeBoron = sum(prodFractionBoron .* n_e) / sum(prodFractionBoron)
        avgNeHep = sum(prodFractionHep .* n_e) / sum(prodFractionHep)

        # Return a named tuple with both prodFractionBoron and prodFractionHep
        return (radii=radii, prodFractionBoron=prodFractionBoron, prodFractionHep=prodFractionHep, n_e=n_e, avgNeBoron=avgNeBoron, avgNeHep=avgNeHep)
    end

else
    error("File not found: $solarModelFile")
end

#  Check if the solar flux file exists and read the datasets
if isfile(flux_file_path)
    energies, flux8B, fluxHep = jldopen(flux_file_path, "r") do file
        energies = file["energies"]
        flux8B = file["flux8B"]
        fluxHep = file["fluxHep"]
        global total_flux_8B = file["total8B"] * 0.514
        global total_flux_hep = file["totalHep"]
        return energies, flux8B, fluxHep, total_flux_8B, total_flux_hep
    end
else
    error("File not found: $flux_file_path")
end