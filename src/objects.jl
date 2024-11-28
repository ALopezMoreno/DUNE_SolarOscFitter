using StaticArrays

# Fermi constant
const G_f = 5.4489e-5
# For now, we keep the remaining mass splitting constant and in NO
const m32 = 2.43e-3
# Electron mass in GeV/c^2
m_e = 5.11e-4
# Weinberg angle
sin2nuW = 0.231

# Cross section Constants
g1_nue = 1/2 + sin2nuW
g2_nue = sin2nuW

g1_nuother = -1/2 + sin2nuW
g2_nuother = sin2nuW
sigma_0 = 1.939e-13 * 1.97e-16^2 # in cm^-2

# Define a mutable struct to hold the oscillation parameters with default values
mutable struct OscillationParameters
    oscpars::SVector{3,Float64}
    function OscillationParameters()
        new(SVector{3,Float64}(0.307, 0.022, 7.53e-5))  # Default values
    end
end

# Define a mutable struct to hold the measured and expected binned spectra
mutable struct NuSpectrum
    ETrue8B::Vector{Float64}
    events8B_es::Vector{Float64}
    events8B_cc::Vector{Float64}

    ETrueHep::Vector{Float64}
    eventsHep_es::Vector{Float64}
    eventsHep_cc::Vector{Float64}

    oscweights8B::Union{Vector{Float64},Nothing}
    oscweightsHep::Union{Vector{Float64},Nothing}

    events_es_oscillated_nue::Union{Vector{Float64},Nothing}
    events_es_oscillated_other::Union{Vector{Float64},Nothing}
    events_cc_oscillated::Union{Vector{Float64},Nothing}

    function NuSpectrum(ETrue8B, events8B_es, events8B_cc, ETrueHep, eventsHep_es, eventsHep_cc)
        new(ETrue8B, events8B_es, events8B_cc, ETrueHep, eventsHep_es, eventsHep_cc, nothing, nothing, nothing, nothing, nothing)
    end
end