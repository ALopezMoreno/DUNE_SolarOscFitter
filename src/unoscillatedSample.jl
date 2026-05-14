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
include(joinpath(@__DIR__, "solarFlux.jl"))
include(joinpath(@__DIR__, "xsec.jl"))

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

"""
    build_unoscillated_sample(det) -> (unoscillatedSample, bin_edges, energies_GeV)

Compute unoscillated event rates for detector `det` (a named tuple from `detector_configs`).
Uses shared global `Etrue_bins`. Returns:
- `unoscillatedSample`: named tuple with per-channel, per-process event rate vectors
- `bin_edges`: Etrue bin edges (GeV)
- `energies_GeV`: Etrue bin centres (GeV)
"""
function build_unoscillated_sample(det)
    global E_threshold = det.E_threshold   # read by ES_xsec_nue / ES_xsec_nuother in xsec.jl
    ES_normalisation = det.ES_normalisation
    CC_normalisation = det.CC_normalisation

    bin_edges, energies_GeV = calculate_bins(Etrue_bins)

    unosc_ES_nue_8B     = integrate_over_bins(unoscillatedRate_ES_nue_8B,     bin_edges) .* (detector_ne   * detection_time * ES_normalisation)
    unosc_ES_nuother_8B = integrate_over_bins(unoscillatedRate_ES_nuother_8B, bin_edges) .* (detector_ne   * detection_time * ES_normalisation)
    unosc_CC_8B         = integrate_over_bins(unoscillatedRate_CC_8B,         bin_edges) .* (detector_nAr40 * detection_time * CC_normalisation)
    unosc_ES_nue_hep     = integrate_over_bins(unoscillatedRate_ES_nue_hep,     bin_edges) .* (detector_ne   * detection_time * ES_normalisation)
    unosc_ES_nuother_hep = integrate_over_bins(unoscillatedRate_ES_nuother_hep, bin_edges) .* (detector_ne   * detection_time * ES_normalisation)
    unosc_CC_hep         = integrate_over_bins(unoscillatedRate_CC_hep,         bin_edges) .* (detector_nAr40 * detection_time * CC_normalisation)

    unoscillatedSample = (
        ES_nue_8B    = unosc_ES_nue_8B,
        ES_nuother_8B = unosc_ES_nuother_8B,
        CC_8B        = unosc_CC_8B,
        ES_nue_hep   = unosc_ES_nue_hep,
        ES_nuother_hep = unosc_ES_nuother_hep,
        CC_hep       = unosc_CC_hep,
    )

    return unoscillatedSample, bin_edges, energies_GeV
end
