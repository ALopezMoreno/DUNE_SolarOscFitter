#=
unoscillatedSample.jl

Unoscillated neutrino event rate calculations for the Solar Oscillation Fitter.
This module computes the expected neutrino interaction rates at the detector
before oscillations, serving as the baseline for oscillation probability calculations.

Key Features:
- Integration of solar neutrino fluxes with interaction cross-sections
- Separate calculations for 8B and HEP neutrino sources
- Support for both ES (elastic scattering) and CC (charged current) channels
- Energy binning and numerical integration over detector response
- Detector mass and exposure time scaling

The unoscillated samples provide the reference event rates that are then
modified by oscillation probabilities to predict observed event rates.

Author: [Author name]
=#

# Numerical integration for flux calculations
using QuadGK

# Load solar flux and cross-section data
include("../src/solarFlux.jl")
include("../src/xsec.jl")

# Detector specifications
global detector_ne = 10 * 2.7e32      # Number of electrons (10 kT LAr, 2.7e32 e-/kT)
global detector_nAr40 = 10 * 1.45e31  # Number of Ar40 nuclei (10 kT LAr, 1.45e31 nuclei/kT)
global detection_time = 3600 * 24 * 365  # Detection time in seconds (1 year)

# Unoscillated interaction rate functions
# These combine solar neutrino fluxes with interaction cross-sections

# 8B neutrinos: Elastic scattering with electron neutrinos
function unoscillatedRate_ES_nue_8B(enu)
    return flux_8B(enu) * ES_xsec_nue(enu)
end

# HEP neutrinos: Elastic scattering with electron neutrinos
function unoscillatedRate_ES_nue_hep(enu)
    return flux_hep(enu) * ES_xsec_nue(enu)
end

# 8B neutrinos: Elastic scattering with other neutrino flavors (μ, τ)
function unoscillatedRate_ES_nuother_8B(enu)
    return flux_8B(enu) * ES_xsec_nuother(enu)
end

# HEP neutrinos: Elastic scattering with other neutrino flavors (μ, τ)
function unoscillatedRate_ES_nuother_hep(enu)
    return flux_hep(enu) * ES_xsec_nuother(enu)
end

# 8B neutrinos: Charged current interactions on Argon-40
function unoscillatedRate_CC_8B(enu)
    return flux_8B(enu) * CC_xsec(enu)
end

# HEP neutrinos: Charged current interactions on Argon-40
function unoscillatedRate_CC_hep(enu)
    return flux_hep(enu) * CC_xsec(enu)
end

# Energy binning utilities

function calculate_bins(bins)
    """Calculate bin edges and centers from binning specification"""
    # Calculate the width of each bin
    bin_width = (bins.max - bins.min) / bins.bin_number
    
    # Generate the bin edges
    bin_edges = [bins.min + i * bin_width for i in 0:bins.bin_number]
    # Calculate the bin centers
    bin_centers = [bins.min + (i + 0.5) * bin_width for i in 0:(bins.bin_number - 1)]
    
    return bin_edges, bin_centers # Both in GeV
end

function integrate_over_bins(f, bin_edges)
    """Integrate function f over each energy bin using adaptive quadrature"""
    nBins = length(bin_edges) - 1
    averages = Vector{Float64}(undef, nBins)
    
    for i in 1:nBins
        a = bin_edges[i]
        b = bin_edges[i + 1]
        # Numerical integration over the bin
        integral, _ = quadgk(f, a, b)
        averages[i] = integral
    end
    
    return averages
end

# Calculate unoscillated event rates by integrating over energy bins

# Set up energy binning
bin_edges, energies_GeV = calculate_bins(Etrue_bins)
# Higher resolution binning for calculations (doubled bin number)
global bin_edges_calc, energies_calc = calculate_bins((max=Etrue_bins.max, min=Etrue_bins.min, bin_number=Etrue_bins.bin_number*2))

# Calculate unoscillated event rates for 8B neutrinos
# ES channel: electron neutrinos
unoscillated_ES_nue_sample_8B = integrate_over_bins(unoscillatedRate_ES_nue_8B, bin_edges) * detector_ne * detection_time * ES_normalisation
# ES channel: other neutrino flavors (after oscillation)
unoscillated_ES_nuother_sample_8B = integrate_over_bins(unoscillatedRate_ES_nuother_8B, bin_edges) * detector_ne * detection_time * ES_normalisation
# CC channel: charged current interactions
unoscillated_CC_sample_8B = integrate_over_bins(unoscillatedRate_CC_8B, bin_edges) * detector_nAr40 * detection_time * CC_normalisation

# Calculate unoscillated event rates for HEP neutrinos
# ES channel: electron neutrinos
unoscillated_ES_nue_sample_hep = integrate_over_bins(unoscillatedRate_ES_nue_hep, bin_edges) * detector_ne * detection_time * ES_normalisation
# ES channel: other neutrino flavors (after oscillation)
unoscillated_ES_nuother_sample_hep = integrate_over_bins(unoscillatedRate_ES_nuother_hep, bin_edges) * detector_ne * detection_time * ES_normalisation
# CC channel: charged current interactions
unoscillated_CC_sample_hep = integrate_over_bins(unoscillatedRate_CC_hep, bin_edges) * detector_nAr40 * detection_time * CC_normalisation

# Package all unoscillated samples into a named tuple for easy access
unoscillatedSample = (ES_nue_8B = unoscillated_ES_nue_sample_8B,
                      ES_nuother_8B = unoscillated_ES_nuother_sample_8B,
                      CC_8B = unoscillated_CC_sample_8B,
                      ES_nue_hep = unoscillated_ES_nue_sample_hep,
                      ES_nuother_hep = unoscillated_ES_nuother_sample_hep,
                      CC_hep = unoscillated_CC_sample_hep)

# Optional: Save unoscillated samples to file for later use
# @save "outputs/unoscillatedSamples.jld2" unoscillated_ES_nue_sample_8B unoscillated_ES_nuother_sample_8B unoscillated_CC_sample_8B unoscillated_ES_nue_sample_hep unoscillated_ES_nuother_sample_hep unoscillated_CC_sample_hep energies_GeV
