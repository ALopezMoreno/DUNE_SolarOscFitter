using CSV             # For cross-section data loading
using DataFrames      # For data manipulation
using Interpolations  # For cross-section interpolation
using QuadGK: gauss   # Gauss-Legendre nodes for the ES recoil integral

# Load physics constants and detector parameters
include(joinpath(@__DIR__, "core.jl"))

# Load charged current cross-section data
# Alternative: Gardiner 2020 cross-sections (commented out)
# df = CSV.File("inputs/CC_nue_40Ar_total_Gardiner2020.csv") |> DataFrame

# Load CC cross-section from MARLEY Monte Carlo calculations
df = CSV.File(joinpath(@__DIR__, "..", "inputs", "CC_nue_40Ar_total_marley.csv")) |> DataFrame   # repo-relative, not cwd-relative

# Extract energy and cross-section data with unit conversions
CC_xsec_energy_raw = vcat(0.0, df[:, 1] * 1e-3)  # Convert MeV to GeV, add zero point
CC_xsec_raw = vcat(0.0, df[:, 2] * 1e-42) .* CC_xsec_scale  # Convert to cm^2, add zero point

# Data cleaning and preparation for interpolation
# Ensure energy values are sorted
sorted_indices = sortperm(CC_xsec_energy_raw)
CC_xsec_energy_sorted = CC_xsec_energy_raw[sorted_indices]
CC_xsec_sorted = CC_xsec_raw[sorted_indices]

# Remove duplicate energy points
CC_xsec_energy_unique = unique(CC_xsec_energy_sorted)
unique_indices = [findfirst(==(val), CC_xsec_energy_sorted) for val in CC_xsec_energy_unique]
CC_xsec_unique = CC_xsec_sorted[unique_indices]

# Create interpolated CC cross-section function
# Uses flat extrapolation beyond data range
CC_xsec = linear_interpolation(CC_xsec_energy_unique, CC_xsec_unique, extrapolation_bc=Flat())

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
