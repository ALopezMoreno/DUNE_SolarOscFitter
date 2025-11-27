#=
objects.jl

Defines fundamental physics constants, cross-section parameters, and data structures
used throughout the Solar Oscillation Fitter.

Key components:
- Physical constants (Fermi constant, particle masses, mixing angles)
- Neutrino-electron scattering cross-section parameters
- NuSpectrum struct for storing neutrino event spectra and oscillation weights

Author: [Author name]
=#

using StaticArrays

#####################################################################
#### PHYSICS CONSTANTS FOR OSCILLATION CALCULATIONS ################
#####################################################################
                                                                    #
# Fermi constant in natural units (GeV^-2)                         #
const G_f = 5.4489e-5                                               #
# Atmospheric mass-squared difference (eV^2) - Normal Ordering     #
const m32 = 2.43e-3                                                 #
# Electron mass in GeV/c^2                                          #
m_e = 5.11e-4                                                       #
# Weinberg angle (sin^2(θ_W))                                      #
sin2nuW = 0.231                                                     #
                                                                    #
#####################################################################

# Neutrino-electron scattering cross-section constants
# These determine the interaction strength for different neutrino flavors

# Coupling constants for electron neutrinos (ν_e + e^- scattering)
g1_nue = 1/2 + sin2nuW      # Vector coupling for ν_e
g2_nue = sin2nuW            # Axial coupling for ν_e

# Coupling constants for other neutrino flavors (ν_μ,τ + e^- scattering)  
g1_nuother = -1/2 + sin2nuW # Vector coupling for ν_μ,τ
g2_nuother = sin2nuW        # Axial coupling for ν_μ,τ

# Reference cross-section in cm^2
sigma_0 = 88.06e-46 #1.939e-13 * (1.97e-16)^2 # in cm^-2

#=
NuSpectrum: Mutable struct to store neutrino event spectra and oscillation weights

This structure holds both the unoscillated Monte Carlo truth information and
the oscillated event rates for different detection channels and time periods.

Fields:
- ETrue8B, ETrueHep: True neutrino energies for 8B and HEP processes
- events8B_*, eventsHep_*: Unoscillated event rates by process and channel
- oscweights*: Oscillation probability weights for day/night
- events_*_oscillated_*: Final oscillated event rates by channel and time

The struct is initialized with only the basic MC truth information,
and oscillation weights/rates are computed and stored later.
=#
mutable struct NuSpectrum
    # True neutrino energies for different solar processes
    ETrue8B::Vector{Float64}   # 8B neutrino energies (GeV)
    events8B_es::Vector{Float64}   # 8B events in elastic scattering channel
    events8B_cc::Vector{Float64}   # 8B events in charged current channel

    ETrueHep::Vector{Float64}      # HEP neutrino energies (GeV)  
    eventsHep_es::Vector{Float64}  # HEP events in elastic scattering channel
    eventsHep_cc::Vector{Float64}  # HEP events in charged current channel

    # Oscillation probability weights (computed later)
    oscweights8B_day::Union{Vector{Float64},Nothing}    # 8B day weights
    oscweightsHep_day::Union{Vector{Float64},Nothing}   # HEP day weights

    oscweights8B_night::Union{Matrix{Float64},Nothing}  # 8B night weights (zenith-dependent)
    oscweightsHep_night::Union{Matrix{Float64},Nothing} # HEP night weights (zenith-dependent)

    # Final oscillated event rates by channel and time period
    events_es_oscillated_nue_day::Union{Vector{Float64},Nothing}      # ES ν_e day events
    events_es_oscillated_other_day::Union{Vector{Float64},Nothing}    # ES ν_μ,τ day events  
    events_cc_oscillated_day::Union{Vector{Float64},Nothing}          # CC day events

    events_es_oscillated_nue_night::Union{Matrix{Float64},Nothing}    # ES ν_e night events
    events_es_oscillated_other_night::Union{Matrix{Float64},Nothing}  # ES ν_μ,τ night events
    events_cc_oscillated_night::Union{Matrix{Float64},Nothing}        # CC night events

    # Constructor: Initialize with MC truth, oscillation fields set to nothing
    function NuSpectrum(ETrue8B, events8B_es, events8B_cc, ETrueHep, eventsHep_es, eventsHep_cc)
        new(ETrue8B, events8B_es, events8B_cc, ETrueHep, eventsHep_es, eventsHep_cc, 
            nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end

# Structure of inputs for the likelihood function called in BAT
struct LikelihoodInputs
    nObserved      # should have ES_day, ES_night, CC_day, CC_night
    energies
    Mreco
    SSM
    MC_no_osc
    BG
    f              # propagation function
    ES_mode::Bool
    CC_mode::Bool
    index_ES::Int
    index_CC::Int
end