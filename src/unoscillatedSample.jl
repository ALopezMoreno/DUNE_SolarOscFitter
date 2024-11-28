# Create the unoscillated flux at the detector
using QuadGK

# Bring in flux and cross section data
include("../src/solarFlux.jl")
include("../src/xsec.jl")


function unoscillatedRate_ES_nue_8B(enu)
    return flux_8B(enu) * ES_xsec_nue(enu) * 1e38 # TEMPORAL FIX UNTIL I FIGURE OUT XSEC
end

# Function for unoscillated rate with hep flux and ES cross-section for nue
function unoscillatedRate_ES_nue_hep(enu)
    return flux_hep(enu) * ES_xsec_nue(enu) * 1e38 # TEMPORAL FIX UNTIL I FIGURE OUT XSEC
end

# Function for unoscillated rate with 8B flux and ES cross-section for nuother
function unoscillatedRate_ES_nuother_8B(enu)
    return flux_8B(enu) * ES_xsec_nuother(enu) * 1e38 # TEMPORAL FIX UNTIL I FIGURE OUT XSEC
end

# Function for unoscillated rate with hep flux and ES cross-section for nuother
function unoscillatedRate_ES_nuother_hep(enu)
    return flux_hep(enu) * ES_xsec_nuother(enu) * 1e38 # TEMPORAL FIX UNTIL I FIGURE OUT XSEC
end

# Function for unoscillated rate with 8B flux and CC cross-section
function unoscillatedRate_CC_8B(enu)
    return flux_8B(enu) * CC_xsec(enu) * 1e38 # TEMPORAL FIX UNTIL I FIGURE OUT XSEC
end

# Function for unoscillated rate with hep flux and CC cross-section
function unoscillatedRate_CC_hep(enu)
    return flux_hep(enu) * CC_xsec(enu) * 1e38 # TEMPORAL FIX UNTIL I FIGURE OUT XSEC
end


function calculate_bins(bins)
    # Calculate the width of each bin
    bin_width = (bins.max - bins.min) / bins.bin_number
    
    # Generate the bin edges
    bin_edges = [bins.min + i * bin_width for i in 0:bins.bin_number]
    # Calculate the bin centers
    bin_centers = [bins.min + (i + 0.5) * bin_width for i in 0:(bins.bin_number - 1)]
    
    return bin_edges, bin_centers # Both in GeV
end

function average_over_bins(f, bin_edges)
    nBins = length(bin_edges) - 1
    averages = Vector{Float64}(undef, nBins)
    
    for i in 1:nBins
        a = bin_edges[i]
        b = bin_edges[i + 1]
        integral, _ = quadgk(f, a, b)
        averages[i] = integral / (b - a)
    end
    
    return averages
end

# Integrate over energy bins to get binned event rates
bin_edges, energies_GeV = calculate_bins(bins)

unoscillated_ES_nue_sample_8B = average_over_bins(unoscillatedRate_ES_nue_8B, bin_edges) * ES_normalisation
unoscillated_ES_nuother_sample_8B = average_over_bins(unoscillatedRate_ES_nuother_8B, bin_edges) * ES_normalisation
unoscillated_CC_sample_8B = average_over_bins(unoscillatedRate_CC_8B, bin_edges) * CC_normalisation

unoscillated_ES_nue_sample_hep = average_over_bins(unoscillatedRate_ES_nue_hep, bin_edges) * ES_normalisation
unoscillated_ES_nuother_sample_hep = average_over_bins(unoscillatedRate_ES_nuother_hep, bin_edges) * ES_normalisation
unoscillated_CC_sample_hep = average_over_bins(unoscillatedRate_CC_hep, bin_edges) * CC_normalisation

const unoscillatedSample = (ES_nue_8B = unoscillated_ES_nue_sample_8B,
                            ES_nuother_8B = unoscillated_ES_nuother_sample_8B,
                            CC_8B = unoscillated_CC_sample_8B,
                            ES_nue_hep = unoscillated_ES_nue_sample_hep,
                            ES_nuother_hep = unoscillated_ES_nuother_sample_hep,
                            CC_hep = unoscillated_CC_sample_hep)