using CSV, DataFrames # not used here any more; kept so scripts that include this file still get them
using QuadGK: gauss   # Gauss-Legendre nodes for the ES recoil integral

# Load physics constants and detector parameters
include(joinpath(@__DIR__, "core.jl"))
include(joinpath(@__DIR__, "tabulated.jl"))   # TabulatedInput, read_numeric_table

# ---------------------------------------------------------------------------------------------
# CC ν_e + ⁴⁰Ar total cross section, tabulated (config key `cc_xsec_file`, default
# DEFAULT_CC_XSEC_FILE in core.jl: MARLEY 2.0.0, HF-CRPA + Bhattacharya 2009).
#
# File format: '#' comment lines (provenance), then one header row, then rows
# `E_nu [MeV], sigma [1e-42 cm^2 per 40Ar]`. The table is used as given: no point is added,
# dropped, sorted or deduplicated, and it is never extrapolated (evaluating CC_xsec outside the
# tabulated energies is an error). Only build_unoscillated_sample() evaluates it, at setup.
# ---------------------------------------------------------------------------------------------

"""
    resolve_cc_xsec_path(path) -> String

An absolute path, or a relative one that exists from the working directory, is used as is
(like the other input-file keys); otherwise a relative path is taken from the repository root.
"""
function resolve_cc_xsec_path(path::AbstractString)
    (isabspath(path) || isfile(path)) && return String(path)
    return normpath(joinpath(@__DIR__, "..", path))
end

"""
    read_cc_xsec_table(path) -> (E_MeV::Vector{Float64}, sigma_1e42::Vector{Float64})

Read a CC cross-section table (read_numeric_table, tabulated.jl): skip '#' comment and blank
lines, require a header row (a first row of numbers is an error, so a headerless file cannot lose
its first point), then two numeric columns. Errors unless every value is finite, sigma ≥ 0 and the
energies strictly increase (validate_table).
"""
function read_cc_xsec_table(path::AbstractString)
    isfile(path) || error("CC cross-section table not found: $path (config key cc_xsec_file)")
    E, σ = read_numeric_table(path; name = "CC cross section", ncols = 2, header = true)
    validate_table(E, σ; name = "CC cross section", source = path, nonnegative = true)
    return E, σ
end

CC_XSEC_PATH = resolve_cc_xsec_path(@isdefined(cc_xsec_file) ? cc_xsec_file : DEFAULT_CC_XSEC_FILE)
CC_XSEC_E_MEV, CC_XSEC_SIGMA_1E42 = read_cc_xsec_table(CC_XSEC_PATH)

# σ_CC(E_ν) [cm² per ⁴⁰Ar], E_ν in GeV: linear interpolation, a TabulatedDomainError outside the
# table (never extrapolated). CC_xsec_scale is a fixed overall factor (1 = the table as given).
CC_xsec = TabulatedInput(CC_XSEC_E_MEV .* 1e-3, CC_XSEC_SIGMA_1E42 .* 1e-42 .* CC_xsec_scale;
                         name = "CC cross section", source = CC_XSEC_PATH, units = "GeV",
                         extrapolation = :error, nonnegative = true)
# Tabulated domain in GeV; build_unoscillated_sample() checks the Etrue range against it.
CC_XSEC_DOMAIN = (CC_XSEC_E_MEV[1] * 1e-3, CC_XSEC_E_MEV[end] * 1e-3)

"""
    check_cc_xsec_covers(bin_edges)

The CC table is never extrapolated: the whole Etrue range must lie inside it. The tolerance
absorbs the rounding of `min + n*width` in calculate_bins (QuadGK never evaluates the endpoints).
"""
function check_cc_xsec_covers(bin_edges; xsec = CC_xsec)
    check_covers(xsec, bin_edges[1], bin_edges[end]; what = "range_Etrue (GeV)")
end

# Elastic scattering cross-section calculations
# Neutrino-electron scattering with the BKS95 electroweak radiative corrections
# (constants and references in core.jl). kappa depends on the recoil energy T,
# so the recoil integral is done by Gauss-Legendre quadrature.

const ES_T_NODES, ES_T_WEIGHTS = gauss(16)

function bks_I(T)
    """
    BKS95 Eq. (A5): I(T) = (1/6)[1/3 + (3 - x^2)(x artanh(1/x) - 1)], x = sqrt(1 + 2 m_e/T).
    """
    u = T / (T + 2 * m_e)  # 1/x^2
    # x artanh(1/x) - 1 = sum_k u^k / (2k + 1); the closed form cancels badly for T << m_e.
    s = u < 0.05 ? sum(u^k / (2k + 1) for k in 1:7) : atanh(sqrt(u)) / sqrt(u) - 1
    return (1 / 3 + (3 - 1 / u) * s) / 6
end

# BKS95 kappa(T): Eq. (A4) for ν_e, Eq. (A7) for ν_μ,τ
kappa_nue(T) = 0.9791 + 0.0097 * bks_I(T)
kappa_nuother(T) = 0.9970 - 0.00037 * bks_I(T)

function es_couplings(T, kappa, cc)
    """
    Radiatively corrected chiral couplings (g1, g2) at recoil energy T, BKS95 Eq. (A2)/(A6)
    with the overall sign flipped (the cross section is unchanged). cc = 1 for ν_e, 0 for ν_μ,τ.
    """
    s2 = kappa(T) * sin2nuW
    return rho_nc * (s2 - 1 / 2) + cc, rho_nc * s2
end

function ES_xsec(enu, kappa, cc)
    """
    Elastic scattering cross-section for ν + e- → ν + e-, integrated over the
    accepted recoil range [tpc_threshold, T_max]:

        dσ/dT = (σ_0 / m_e) [g1^2 + g2^2 (1 - T/E)^2 - g1 g2 m_e T / E^2],
        g1 = ρ_NC (κ(T) s^2 - 1/2) + cc,   g2 = ρ_NC κ(T) s^2,

    with cc = 1 for ν_e (charged-current contribution) and 0 for ν_μ,τ.
    """
    Emin = tpc_threshold  # true TPC detection threshold (GeV); set in config, must match MC

    # Maximum kinematically allowed electron recoil energy
    T_max = 2 * enu^2 / (m_e + 2 * enu)

    # No recoil above threshold
    if T_max <= Emin
        return 0.0
    end

    half = (T_max - Emin) / 2
    mid = (T_max + Emin) / 2
    acc = 0.0
    for (x, w) in zip(ES_T_NODES, ES_T_WEIGHTS)
        T = mid + half * x
        g1, g2 = es_couplings(T, kappa, cc)
        acc += w * (g1^2 + g2^2 * (1 - T / enu)^2 - g1 * g2 * m_e * T / enu^2)
    end
    return sigma_0 / m_e * half * acc
end

# Electron neutrinos: NC + CC
ES_xsec_nue(enu) = ES_xsec(enu, kappa_nue, 1.0)

# Other neutrino flavors (μ, τ): NC only
ES_xsec_nuother(enu) = ES_xsec(enu, kappa_nuother, 0.0)
