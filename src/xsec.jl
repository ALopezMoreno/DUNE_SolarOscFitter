#=
xsec.jl

Neutrino interaction cross-section calculations for the Solar Oscillation Fitter.
This module loads and processes neutrino-nucleus interaction cross-sections for
both charged current (CC) and elastic scattering (ES) processes.

Key Features:
- CC cross-section loading from MARLEY calculations for ν_e + 40Ar
- ES cross-section analytical calculations for ν + e- scattering
- Energy-dependent cross-section interpolation
- Separate treatment for electron and other neutrino flavors
- Threshold energy handling for detector sensitivity

The cross-sections are combined with neutrino fluxes to calculate
unoscillated interaction rates in the detector.

Author: [Author name]
=#

using CSV             # For cross-section data loading
using DataFrames      # For data manipulation
using Interpolations  # For cross-section interpolation

# Load physics constants and detector parameters
include("../src/objects.jl")

# Load charged current cross-section data
# Alternative: Gardiner 2020 cross-sections (commented out)
# df = CSV.File("inputs/CC_nue_40Ar_total_Gardiner2020.csv") |> DataFrame

# Load CC cross-section from MARLEY Monte Carlo calculations
df = CSV.File("inputs/CC_nue_40Ar_total_marley.csv") |> DataFrame

# Extract energy and cross-section data with unit conversions
CC_xsec_energy_raw = vcat(0.0, df[:, 1] * 1e-3)  # Convert MeV to GeV, add zero point
CC_xsec_raw = vcat(0.0, df[:, 2] * 1e-42)        # Convert to cm^2, add zero point

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
CC_xsec = LinearInterpolation(CC_xsec_energy_unique, CC_xsec_unique, extrapolation_bc=Flat())

# Elastic scattering cross-section calculations
# These use analytical formulas for neutrino-electron scattering

function ES_xsec_nue(enu)
    """
    Elastic scattering cross-section for electron neutrinos with electrons.
    
    Uses the standard formula for ν_e + e- → ν_e + e- scattering
    integrated over the kinematically allowed recoil energy range.
    """
    Emin = E_threshold.ES  # Minimum detectable electron recoil energy

    # Maximum kinematically allowed electron recoil energy
    T_max = 2 * enu^2 / (m_e + 2 * enu)

    # Integrated cross-section over recoil energy range [Emin, T_max]
    xsec = sigma_0 / m_e *
        (
        (g1_nue^2 + g2_nue^2) * (T_max - Emin)
        -
        (g2_nue^2 + g1_nue * g2_nue * m_e / (2 * enu)) * (T_max^2 - Emin^2) / enu
        +
        1 / 3 * g2_nue^2 * ((T_max^3 - Emin^3) / enu^2)
        )
    
    # Ensure non-negative cross-section
    if xsec < 0
        xsec = 0
    end

    return xsec
end

function ES_xsec_nuother(enu)
    """
    Elastic scattering cross-section for other neutrino flavors (μ, τ) with electrons.
    
    Uses the standard formula for ν_μ,τ + e- → ν_μ,τ + e- scattering
    with different coupling constants than electron neutrinos.
    """
    Emin = E_threshold.ES  # Minimum detectable electron recoil energy
    
    # Return zero cross-section below threshold
    if enu <= Emin
        return 0
    else
        # Maximum kinematically allowed electron recoil energy
        T_max = 2 * enu^2 / (m_e + 2 * enu)

        # Integrated cross-section over recoil energy range [Emin, T_max]
        xsec = sigma_0 / m_e *
            (
            (g1_nuother^2 + g2_nuother^2) * (T_max - Emin)
            -
            (g2_nuother^2 + g1_nuother * g2_nuother * m_e / (2 * enu)) * (T_max^2 - Emin^2) / enu
            +
            1 / 3 * g2_nuother^2 * ((T_max^3 - Emin^3) / enu^2)
        )
        return xsec
    end
end