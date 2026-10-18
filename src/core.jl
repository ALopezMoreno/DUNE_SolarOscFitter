using StaticArrays

# Type-stable extraction of all NamedTuple fields whose names start with `pfx`.
# Returns a Tuple, allowing ForwardDiff Dual numbers to flow through without
# type inference breaking on Symbol("name_", i) runtime construction.
# Indexed names sort numerically (`_2` before `_10`); non-numeric suffixes sort after, by text.
function _prefix_sort_key(f::Symbol, prefix::String)
    suffix = string(f)[ncodeunits(prefix)+1:end]
    n = tryparse(Int, suffix)
    return n === nothing ? (typemax(Int), suffix) : (n, suffix)
end

@generated function _params_by_prefix(params, ::Val{pfx}) where {pfx}
    prefix = string(pfx)
    fnames = sort(
        [f for f in fieldnames(params) if startswith(string(f), prefix)],
        by = f -> _prefix_sort_key(f, prefix),
    )
    isempty(fnames) && return :(())
    exprs = [:(getfield(params, $(QuoteNode(f)))) for f in fnames]
    return :(tuple($(exprs...)))
end

#####################################################################
#### PHYSICS CONSTANTS FOR OSCILLATION CALCULATIONS ################
#####################################################################
                                                                    #
# A/E (eV^2/GeV) per (mol/cm^3) of electron density                #
const MATTER_POTENTIAL_CONST = 1.5264932436e-4                      #
# Atmospheric mass-squared difference (eV^2) - Normal Ordering     #
const m32 = 2.43e-3                                                 #
# Electron mass in GeV/c^2                                          #
const m_e = 5.11e-4                                                 #
# MS-bar weak mixing angle at M_Z, sin^2(θ̂_W(M_Z)) (PDG 2024)      #
const sin2nuW = 0.23129                                             #
                                                                    #
#####################################################################

# Targets per kt of liquid argon (molar mass 39.948 g/mol, Z = 18, ⁴⁰Ar abundance 99.604 %).
const AR_ATOMS_PER_KT    = 1e9 / 39.948 * 6.02214076e23   # 1.5075e31
const N_ELECTRONS_PER_KT = 18 * AR_ATOMS_PER_KT            # 2.7135e32 (was 2.7e32)
const N_AR40_PER_KT      = 0.99604 * AR_ATOMS_PER_KT       # 1.5015e31 (was 1.45e31, 3.5 % low)

# Default CC ν_e–⁴⁰Ar total cross-section table (config key `cc_xsec_file`): MARLEY 2.0.0,
# HF-CRPA continuum + Bhattacharya 2009 discrete; provenance in the file's header.
const DEFAULT_CC_XSEC_FILE = "inputs/CC_nue_40Ar_total_marley_v2.0.0.csv"

# Neutrino-electron scattering cross-section constants
# These determine the interaction strength for different neutrino flavors

# EW radiative corrections from Bahcall, Kamionkowski & Sirlin (1995), PRD 51, 6146,
# Appendix A. rho_nc renormalises the neutral-current amplitude only (Eq. A2);
# the energy-dependent kappa(T) multiplying sin2nuW is in xsec.jl (Eqs. A4-A7).
# The QED corrections of their Appendix B are not included.
const rho_nc = 1.0126

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
# First unused "batch_N" group name in an output file (anything with `keys`), N ≥ 1: numbers
# continue after the highest existing one, so a resumed run never reuses a group (saveBatch).
function next_batch_name(f)
    nums = [parse(Int, k[7:end]) for k in keys(f) if startswith(k, "batch_") && tryparse(Int, k[7:end]) !== nothing]
    return "batch_$(isempty(nums) ? 1 : maximum(nums) + 1)"
end
