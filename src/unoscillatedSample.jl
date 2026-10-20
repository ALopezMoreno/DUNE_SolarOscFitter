using QuadGK

include(joinpath(@__DIR__, "solarFlux.jl"))
include(joinpath(@__DIR__, "xsec.jl"))

global detection_time = 3600 * 24 * 365  # s

# Unoscillated rates: flux × cross section, per channel and source
function unoscillatedRate_ES_nue_8B(enu)
    return flux_8B(enu) * ES_xsec_nue(enu)
end

function unoscillatedRate_ES_nue_hep(enu)
    return flux_hep(enu) * ES_xsec_nue(enu)
end

function unoscillatedRate_ES_nuother_8B(enu)
    return flux_8B(enu) * ES_xsec_nuother(enu)
end

function unoscillatedRate_ES_nuother_hep(enu)
    return flux_hep(enu) * ES_xsec_nuother(enu)
end

function unoscillatedRate_CC_8B(enu)
    return flux_8B(enu) * CC_xsec(enu)
end

function unoscillatedRate_CC_hep(enu)
    return flux_hep(enu) * CC_xsec(enu)
end

# Uniform bin edges and centres (GeV) of `(bin_number, min, max)`
function calculate_bins(bins)
    bin_width = (bins.max - bins.min) / bins.bin_number
    bin_edges = [bins.min + i * bin_width for i in 0:bins.bin_number]
    bin_centers = [bins.min + (i + 0.5) * bin_width for i in 0:(bins.bin_number - 1)]
    return bin_edges, bin_centers
end

# Integral of f over each bin (adaptive quadrature)
function integrate_over_bins(f, bin_edges)
    nBins = length(bin_edges) - 1
    averages = Vector{Float64}(undef, nBins)
    
    for i in 1:nBins
        a = bin_edges[i]
        b = bin_edges[i + 1]
        integral, _ = quadgk(f, a, b)
        averages[i] = integral
    end
    
    return averages
end

"""
    build_unoscillated_sample(det) -> (unoscillatedSample, bin_edges, energies_GeV)

Unoscillated event rates per channel and source for detector `det`, on the Etrue bins (GeV).
"""
function build_unoscillated_sample(det)
    ES_normalisation = det.ES_normalisation
    CC_normalisation = det.CC_normalisation
    detector_ne      = N_ELECTRONS_PER_KT * det.module_mass_kt
    detector_nAr40   = N_AR40_PER_KT      * det.module_mass_kt

    bin_edges, energies_GeV = calculate_bins(Etrue_bins)

    check_cc_xsec_covers(bin_edges)

    unosc_ES_nue_8B     = integrate_over_bins(unoscillatedRate_ES_nue_8B,     bin_edges) .* (detector_ne   * detection_time * ES_normalisation)
    unosc_ES_nuother_8B = integrate_over_bins(unoscillatedRate_ES_nuother_8B, bin_edges) .* (detector_ne   * detection_time * ES_normalisation)
    unosc_CC_8B         = integrate_over_bins(unoscillatedRate_CC_8B,         bin_edges) .* (detector_nAr40 * detection_time * CC_normalisation)

    # CC xsec shape basis, orthogonal under the unoscillated CC ⁸B rate (SPEC 3.3)
    xsec_basis = cc_xsec_shape_basis(energies_GeV, unosc_CC_8B)
    if det.CC_mode && !(sum(unosc_CC_8B) > 0)
        @warn "CC channel on but its unoscillated ⁸B rate is zero (CC_exposure 0?): tilt/curv have no effect"
    elseif det.CC_mode
        @logmsg Setup @sprintf("  CC xsec shape basis: E_pivot = %.4f MeV, L = ln(E/E_pivot), p1 = L - (%.6g), p2 = L^2 - (%.6g) L - (%.6g)",
                               xsec_basis.E_pivot * 1e3, xsec_basis.mean_L, xsec_basis.a, xsec_basis.b)
    end

    unosc_ES_nue_hep     = integrate_over_bins(unoscillatedRate_ES_nue_hep,     bin_edges) .* (detector_ne   * detection_time * ES_normalisation)
    unosc_ES_nuother_hep = integrate_over_bins(unoscillatedRate_ES_nuother_hep, bin_edges) .* (detector_ne   * detection_time * ES_normalisation)
    unosc_CC_hep         = integrate_over_bins(unoscillatedRate_CC_hep,         bin_edges) .* (detector_nAr40 * detection_time * CC_normalisation)

    unoscillatedSample = (
        ES_nue_8B      = unosc_ES_nue_8B,
        ES_nuother_8B  = unosc_ES_nuother_8B,
        CC_8B          = unosc_CC_8B,
        ES_nue_hep     = unosc_ES_nue_hep,
        ES_nuother_hep = unosc_ES_nuother_hep,
        CC_hep         = unosc_CC_hep,
        xsec_p1        = xsec_basis.p1,
        xsec_p2        = xsec_basis.p2,
    )

    return unoscillatedSample, bin_edges, energies_GeV
end
