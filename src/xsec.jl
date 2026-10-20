using CSV, DataFrames # for scripts that include this file
using QuadGK: gauss

include(joinpath(@__DIR__, "core.jl"))
include(joinpath(@__DIR__, "tabulated.jl"))   # TabulatedInput, read_numeric_table

# CC ν_e + ⁴⁰Ar total cross section from `cc_xsec_file` (default MARLEY 2.0.0): '#' lines, a
# header row, then `E_nu [MeV], sigma [1e-42 cm^2 per 40Ar]`. Used as given, never extrapolated.

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

# σ_CC(E_ν) [cm² per ⁴⁰Ar], E_ν in GeV, times the fixed factor CC_xsec_scale.
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

# ν–e elastic scattering with the BKS95 radiative corrections (constants in core.jl). κ depends on
# the recoil energy T, so the recoil integral is done by Gauss–Legendre quadrature.

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
    Emin = tpc_threshold

    T_max = 2 * enu^2 / (m_e + 2 * enu)

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

# CC cross-section shape nuisances (SPEC 3.3): σ(E) = σ_nominal(E) · norm · exp(tilt·p₁ + curv·p₂),
# with p₁ = L − ⟨L⟩ and p₂ = L² − a·L − b (L = ln E) orthogonal to 1 and to each other under the
# unoscillated CC ⁸B rate, so tilt and curv do not move the total rate to first order.

"""
    cc_xsec_shape_basis(energies_GeV, w) -> (; p1, p2, E_pivot, mean_L, a, b)

Orthogonal shape basis for the CC cross-section nuisances, built once at setup from the E_true bin
centres and the weight `w` (unoscillated CC ⁸B rate per bin). `E_pivot` is the `w`-weighted mean
energy; it only fixes the origin of L (p₁, p₂ do not depend on it). Returns zeros if `w` has no
weight (no CC channel).
"""
function cc_xsec_shape_basis(energies_GeV::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    length(energies_GeV) == length(w) || throw(DimensionMismatch("energies and weights differ in length"))
    all(isfinite, w) || throw(ArgumentError("CC xsec shape weights must be finite"))
    all(>=(0), w) || throw(ArgumentError("CC xsec shape weights must be non-negative"))
    n = length(w)
    W = sum(w)
    if !(W > 0)
        z = zeros(n)
        return (; p1 = z, p2 = copy(z), E_pivot = NaN, mean_L = 0.0, a = 0.0, b = 0.0)
    end
    wm(f) = sum(w .* f) / W
    E_pivot = wm(energies_GeV)
    L = log.(energies_GeV ./ E_pivot)
    mean_L = wm(L)
    q = L .- mean_L
    q = q .- wm(q)                              # re-centre (round-off)
    m2 = wm(q .^ 2)
    m2 > 0 || throw(ArgumentError("CC xsec shape basis needs weight in at least two E_true bins"))
    c = wm(q .^ 3) / m2
    p2 = q .^ 2 .- c .* q .- m2                 # ⟨p₂⟩ = ⟨p₂ q⟩ = 0 analytically
    p2 = p2 .- wm(p2)                           # one re-orthogonalisation pass (round-off)
    p2 = p2 .- (wm(p2 .* q) / m2) .* q
    # p₂ = L² − a L − b in terms of L = q + ⟨L⟩ (reported in the setup log)
    a = 2mean_L + c
    b = m2 - mean_L^2 - c * mean_L
    wm(p2 .^ 2) > 1e-12 * m2^2 ||
        @warn "CC xsec shape basis: the CC ⁸B rate sits in only two E_true bins, so p₂ vanishes on it and " *
              "curv is not constrained by the rate (only by its prior); use more E_true bins"
    return (; p1 = q, p2 = p2, E_pivot = E_pivot, mean_L = mean_L, a = a, b = b)
end

"""
    cc_xsec_shape(norm, tilt, curv, p1, p2)

Per-E_true-bin multiplicative CC cross-section factor `norm · exp(tilt·p1 + curv·p2)`; `p1`, `p2`
come from `cc_xsec_shape_basis` (constant data). Equals 1 at norm = 1, tilt = curv = 0.
"""
cc_xsec_shape(norm, tilt, curv, p1, p2) = exp.(tilt .* p1 .+ curv .* p2) .* norm
