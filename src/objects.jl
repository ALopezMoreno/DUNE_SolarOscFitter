"""
This module defines constants and data structures for modeling neutrino oscillation and interaction 
parameters, specifically focusing on electron neutrinos and other neutrino flavors. It provides a 
framework for storing and manipulating oscillation parameters and neutrino spectra data.

Constants:
- `G_f`: Fermi constant, a fundamental constant in weak interactions.
- `m32`: Mass splitting between neutrino mass states, assumed constant and in Normal Ordering (NO).
- `m_e`: Electron mass in GeV/c².
- `sin2nuW`: Weinberg angle, a parameter in the electroweak theory.
- `g1_nue`, `g2_nue`, `g1_nuother`, `g2_nuother`: Cross-section constants for electron neutrinos and 
  other neutrino flavors.
- `sigma_0`: Base cross-section value in cm⁻², used for calculating interaction probabilities.

Data Structures:
- `NuSpectrum`: A mutable struct that stores measured and expected binned spectra for neutrino events 
  from different sources (e.g., 8B and Hep neutrinos). It includes fields for true energy, elastic 
  scattering events, charged current events, and oscillation weights.

Usage:
- The `OscillationParameters` struct is initialized with default oscillation parameters, which can be 
  modified as needed for simulations or analyses.
- The `NuSpectrum` struct is designed to hold data for neutrino event spectra, allowing for the 
  incorporation of oscillation effects and comparison between measured and expected values.

Note:
- The module assumes that the constants and default values are appropriate for the specific neutrino 
  physics context being modeled.
- The use of `Union{Vector{Float64},Nothing}` for some fields in `NuSpectrum` allows for optional 
  inclusion of oscillation weights and oscillated event counts.
"""


using StaticArrays

#####################################################################
#### THESE MAY OR MAY NOT BE USED BY THE OSCILLATIONS CALCULATOR ####
#####################################################################
                                                                    #
# Fermi constant                                                    #
const G_f = 5.4489e-5                                               #
# For now, we keep the remaining mass splitting constant and in NO  #
const m32 = 2.43e-3                                                 #
# Electron mass in GeV/c^2                                          #
m_e = 5.11e-4                                                       #
# Weinberg angle                                                    #
sin2nuW = 0.231                                                     #
                                                                    #
#####################################################################

# Cross section Constants
g1_nue = 1/2 + sin2nuW
g2_nue = sin2nuW

g1_nuother = -1/2 + sin2nuW
g2_nuother = sin2nuW

sigma_0 = 88.06e-46 #1.939e-13 * (1.97e-16)^2 # in cm^-2

# Define a mutable struct to hold the measured and expected binned spectra
mutable struct NuSpectrum
    ETrue8B::Vector{Float64}
    events8B_es::Vector{Float64}
    events8B_cc::Vector{Float64}

    ETrueHep::Vector{Float64}
    eventsHep_es::Vector{Float64}
    eventsHep_cc::Vector{Float64}

    oscweights8B_day::Union{Vector{Float64},Nothing}
    oscweightsHep_day::Union{Vector{Float64},Nothing}

    oscweights8B_night::Union{Matrix{Float64},Nothing}
    oscweightsHep_night::Union{Matrix{Float64},Nothing}

    events_es_oscillated_nue_day::Union{Vector{Float64},Nothing}
    events_es_oscillated_other_day::Union{Vector{Float64},Nothing}
    events_cc_oscillated_day::Union{Vector{Float64},Nothing}

    events_es_oscillated_nue_night::Union{Matrix{Float64},Nothing}
    events_es_oscillated_other_night::Union{Matrix{Float64},Nothing}
    events_cc_oscillated_night::Union{Matrix{Float64},Nothing}

    function NuSpectrum(ETrue8B, events8B_es, events8B_cc, ETrueHep, eventsHep_es, eventsHep_cc)
        new(ETrue8B, events8B_es, events8B_cc, ETrueHep, eventsHep_es, eventsHep_cc, nothing, nothing, nothing, nothing, nothing,
                                                                                     nothing, nothing, nothing, nothing, nothing)
    end
end