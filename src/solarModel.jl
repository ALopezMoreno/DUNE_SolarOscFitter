
include(joinpath(@__DIR__, "modelInputs.jl"))   # load_solar_model

# Solar model: density profile and 8B/hep production regions, validated (SPEC 8.2)
solarModel, _solar_model_flux_hep = load_solar_model(solarModelFile)

# The Asimov hep flux is set in the config, not taken from the file: warn if it differs from the
# model's own prediction (SF-III files store it), e.g. after switching solar_model_file.
@noinline function _warn_hep_flux_mismatch(file_flux, true_flux, path)
    if file_flux !== nothing && abs(true_flux / file_flux - 1) > 0.01
        @warn "true_integrated_HEP_flux = $true_flux differs from the hep flux of the solar model, " *
              "$file_flux ($path). Intended?"
    end
end
_warn_hep_flux_mismatch(_solar_model_flux_hep, integrated_HEP_flux_true, solarModelFile)

# Load solar neutrino flux data
if isfile(flux_file_path)
    energies, flux8B, fluxHep = jldopen(flux_file_path, "r") do file
        # Load energy-dependent flux spectra
        energies = file["energies"]              # Neutrino energies (GeV)
        flux8B = file["flux8B"]                  # 8B neutrino flux spectrum
        fluxHep = file["fluxHep"]                # HEP neutrino flux spectrum

        return energies, flux8B, fluxHep
    end
else
    error("Solar flux file not found: $flux_file_path")
end