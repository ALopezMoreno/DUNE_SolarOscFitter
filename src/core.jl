using StaticArrays

# Fields of a NamedTuple whose names start with a prefix, as a Tuple (type-stable for dual
# numbers). Indexed names sort numerically (`_2` before `_10`), other suffixes after, by text.
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

const MATTER_POTENTIAL_CONST = 1.5264932436e-4   # A/E (eV²/GeV) per mol/cm³ of electrons
const m32 = 2.43e-3                              # atmospheric Δm² (eV²), normal ordering
const m_e = 5.11e-4                              # electron mass (GeV)
const sin2nuW = 0.23129                          # MS-bar sin²θ_W(M_Z) (PDG 2024)

# Targets per kt of liquid argon (molar mass 39.948 g/mol, Z = 18, ⁴⁰Ar abundance 99.604 %).
const AR_ATOMS_PER_KT    = 1e9 / 39.948 * 6.02214076e23   # 1.5075e31
const N_ELECTRONS_PER_KT = 18 * AR_ATOMS_PER_KT            # 2.7135e32
const N_AR40_PER_KT      = 0.99604 * AR_ATOMS_PER_KT       # 1.5015e31

# Default `cc_xsec_file`: MARLEY 2.0.0 (provenance in the file header)
const DEFAULT_CC_XSEC_FILE = "inputs/CC_nue_40Ar_total_marley_v2.0.0.csv"

# ν–e scattering: EW radiative corrections of Bahcall, Kamionkowski & Sirlin (1995), PRD 51, 6146,
# App. A. rho_nc scales the NC amplitude only (Eq. A2); κ(T) is in xsec.jl. App. B (QED) not included.
const rho_nc = 1.0126

const sigma_0 = 88.06e-46   # reference cross section (cm²)

# Unoscillated ⁸B and hep spectra per channel; oscillation weights and rates are filled in later.
mutable struct NuSpectrum
    ETrue8B::Vector{Float64}   # 8B neutrino energies (GeV)
    events8B_es::Vector{Float64}   # 8B events in elastic scattering channel
    events8B_cc::Vector{Float64}   # 8B events in charged current channel

    ETrueHep::Vector{Float64}      # HEP neutrino energies (GeV)  
    eventsHep_es::Vector{Float64}  # HEP events in elastic scattering channel
    eventsHep_cc::Vector{Float64}  # HEP events in charged current channel

    oscweights8B_day::Union{Vector{Float64},Nothing}    # 8B day weights
    oscweightsHep_day::Union{Vector{Float64},Nothing}   # HEP day weights

    oscweights8B_night::Union{Matrix{Float64},Nothing}  # 8B night weights (zenith-dependent)
    oscweightsHep_night::Union{Matrix{Float64},Nothing} # HEP night weights (zenith-dependent)

    events_es_oscillated_nue_day::Union{Vector{Float64},Nothing}      # ES ν_e day events
    events_es_oscillated_other_day::Union{Vector{Float64},Nothing}    # ES ν_μ,τ day events  
    events_cc_oscillated_day::Union{Vector{Float64},Nothing}          # CC day events

    events_es_oscillated_nue_night::Union{Matrix{Float64},Nothing}    # ES ν_e night events
    events_es_oscillated_other_night::Union{Matrix{Float64},Nothing}  # ES ν_μ,τ night events
    events_cc_oscillated_night::Union{Matrix{Float64},Nothing}        # CC night events

    function NuSpectrum(ETrue8B, events8B_es, events8B_cc, ETrueHep, eventsHep_es, eventsHep_cc)
        new(ETrue8B, events8B_es, events8B_cc, ETrueHep, eventsHep_es, eventsHep_cc, 
            nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end
# First "batch_N" after the highest existing one, so a resumed run never reuses a group.
function next_batch_name(f)
    nums = [parse(Int, k[7:end]) for k in keys(f) if startswith(k, "batch_") && tryparse(Int, k[7:end]) !== nothing]
    return "batch_$(isempty(nums) ? 1 : maximum(nums) + 1)"
end
