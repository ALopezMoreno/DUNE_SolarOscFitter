using StaticArrays

# Type-stable extraction of all NamedTuple fields whose names start with `pfx`.
# Returns a Tuple, allowing ForwardDiff Dual numbers to flow through without
# type inference breaking on Symbol("name_", i) runtime construction.
@generated function _params_by_prefix(params, ::Val{pfx}) where {pfx}
    prefix = string(pfx)
    fnames = sort(
        [f for f in fieldnames(params) if startswith(string(f), prefix)],
        by = string,
    )
    isempty(fnames) && return :(())
    exprs = [:(getfield(params, $(QuoteNode(f)))) for f in fnames]
    return :(tuple($(exprs...)))
end

#####################################################################
#### PHYSICS CONSTANTS FOR OSCILLATION CALCULATIONS ################
#####################################################################
                                                                    #
# Fermi constant in natural units (GeV^-2)                         #
const G_f = 5.4489e-5                                               #
# Atmospheric mass-squared difference (eV^2) - Normal Ordering     #
const m32 = 2.43e-3                                                 #
# Electron mass in GeV/c^2                                          #
const m_e = 5.11e-4                                                 #
# Weinberg angle (sin^2(θ_W))                                      #
const sin2nuW = 0.231                                               #
                                                                    #
#####################################################################

# Neutrino-electron scattering cross-section constants
# These determine the interaction strength for different neutrino flavors

# EW radiative corrections from Bahcall, Kamionkowski & Sirlin (1995), PRD 51, 6146.
# rho_nc: NC current renormalization (~0.8% from W/Z/top/Higgs loops).
# kappa: effective low-q² enhancement of sin²θ_W relative to the MS-bar value at M_Z
#        (~3.3%, dominant effect is running from M_Z to q²≈0 for solar neutrinos).
# Higgs-loop contribution to kappa is ~0.03% (negligible individually).
# Values are for m_top = 173 GeV, m_H = 125 GeV; see BKS95 Table 1 and
# PDG Electroweak Model Review for updated sin²θ_W(q²=0).
const rho_nc   = 1.0081
const kappa_ew = 1.0329   # = sin²θ_W(q²→0, MS-bar) / sin²θ_W(M_Z, MS-bar) ≈ 0.23867 / 0.231

# Radiatively corrected coupling constants for electron neutrinos (ν_e + e^- scattering)
const g1_nue = sqrt(rho_nc) * (1/2 + kappa_ew * sin2nuW)
const g2_nue = sqrt(rho_nc) * kappa_ew * sin2nuW

# Radiatively corrected coupling constants for other neutrino flavors (ν_μ,τ + e^- scattering)
const g1_nuother = sqrt(rho_nc) * (-1/2 + kappa_ew * sin2nuW)
const g2_nuother = sqrt(rho_nc) * kappa_ew * sin2nuW

# Reference cross-section in cm^2
const sigma_0 = 88.06e-46 #1.939e-13 * (1.97e-16)^2 # in cm^-2

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