using JLD2
include("../src/objects.jl")

# Load solar model
SolarModelFile = "inputs/AGSS09_high_z.jld2"

# Check if the solar model file exists and read the datasets
if isfile(SolarModelFile)
    solarModel = jldopen(SolarModelFile, "r") do file
        # Load the necessary datasets from the file
        radii = file["radii"]
        prodFractionBoron = file["prodFractionBoron"]
        prodFractionHep = file["prodFractionHep"]
        n_e = file["n_e"]
        
        # Return a named tuple with both prodFractionBoron and prodFractionHep
        return (radii=radii, prodFractionBoron=prodFractionBoron, prodFractionHep=prodFractionHep, n_e=n_e)
    end
else
    error("File not found: $SolarModelFile")
end


# Load test fake data
file_path = "inputs/testEvents.jld2"
energies8B, energiesHep, events8B_es, events8B_cc, eventsHep_es, eventsHep_cc = jldopen(file_path, "r") do file
    energies8B = file["energies8B"]
    energiesHep = file["energiesHep"]

    events8B_es = file["events8B_es"]
    events8B_cc = file["events8B_cc"]

    eventsHep_es = file["eventsHep_es"]
    eventsHep_cc = file["eventsHep_cc"]

    return energies8B, energiesHep, events8B_es, events8B_cc, eventsHep_es, eventsHep_cc
end

# Initialise (fake) data and mc samples:
data = NuSpectrum(
    energies8B,  # ETrue8B
    events8B_es,  # events8B_es
    events8B_cc,  # events8B_cc
    energiesHep,  # ETrueHep
    eventsHep_es,  # eventsHep_es
    eventsHep_cc   # eventsHep_cc
)
monteCarlo = NuSpectrum(
    energies8B,  # ETrue8B
    events8B_es,  # events8B_es
    events8B_cc,  # events8B_cc
    energiesHep,  # ETrueHep
    eventsHep_es,  # eventsHep_es
    eventsHep_cc   # eventsHep_cc
)

# Initialise parameters and set oservations to Asimov parameter values (PDG)
params = OscillationParameters()
oscReweight!(data, params)

true_par_values = (params.oscpars[1], params.oscpars[2], params.oscpars[3])


