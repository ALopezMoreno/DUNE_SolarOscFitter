using CSV
using DataFrames
using Interpolations

# Calculate binned cross sections for convolving
include("../src/objects.jl")

# Load CC cross section from Gardiner 2020
# df = CSV.File("inputs/CC_nue_40Ar_total_Gardiner2020.csv") |> DataFrame

# Load CC cross section from Marley
df = CSV.File("inputs/CC_nue_40Ar_total_marley.csv") |> DataFrame

# Extract the first column as energy and the second as cross section
CC_xsec_energy_raw = vcat(0.0, df[:, 1] * 1e-3)
CC_xsec_raw = vcat(0.0, df[:, 2] * 1e-42)  # total enu CC cross section with 40Ar in 10^-42 cm^2

# Ensure CC_xsec_energy is sorted
sorted_indices = sortperm(CC_xsec_energy_raw)
CC_xsec_energy_sorted = CC_xsec_energy_raw[sorted_indices]
CC_xsec_sorted = CC_xsec_raw[sorted_indices]

# Remove duplicates and get unique indices
CC_xsec_energy_unique = unique(CC_xsec_energy_sorted)
unique_indices = [findfirst(==(val), CC_xsec_energy_sorted) for val in CC_xsec_energy_unique]
CC_xsec_unique = CC_xsec_sorted[unique_indices]

# Interpolate
# CC_xsec = LinearInterpolation(CC_xsec_energy_unique, CC_xsec_unique)
CC_xsec = LinearInterpolation(CC_xsec_energy_unique, CC_xsec_unique, extrapolation_bc=Flat())

# Calculate ES cross section from formulae

function ES_xsec_nue(enu)
    Emin = E_threshold.ES

    T_max = 2 * enu^2 / (m_e + 2 * enu)

    xsec = sigma_0 / m_e *
        (
        (g1_nue^2 + g2_nue^2) * (T_max - Emin)
        -
        (g2_nue^2 + g1_nue * g2_nue * m_e / (2 * enu)) * (T_max^2 - Emin^2) / enu
        +
        1 / 3 * g2_nue^2 * ((T_max^3 - Emin^3) / enu^2)
        )
    
        if xsec < 0
            xsec = 0
        end

    return xsec
end


function ES_xsec_nuother(enu)
    Emin = E_threshold.ES
    if enu <= Emin
        return 0
    else
        T_max = 2 * enu^2 / (m_e + 2 * enu)

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