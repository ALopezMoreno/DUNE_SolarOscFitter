"""
This script calculates unoscillated neutrino event rates at a detector for different interaction channels 
and solar neutrino sources. It integrates flux and cross-section data over specified energy bins to produce 
binned event rates, which are then saved for further analysis.

Dependencies:
- Uses the `QuadGK` package for numerical integration.
- Includes external modules `solarFlux.jl` and `xsec.jl` for solar neutrino flux and cross-section data.
- Assumes the existence of constants `ES_normalisation` and `CC_normalisation` for normalizing event rates.

Functions:
- `unoscillatedRate_ES_nue_8B`, `unoscillatedRate_ES_nue_hep`: Calculate unoscillated event rates for 
  electron neutrinos (nue) from 8B and hep solar processes using elastic scattering (ES) cross-sections.
- `unoscillatedRate_ES_nuother_8B`, `unoscillatedRate_ES_nuother_hep`: Calculate unoscillated event rates 
  for other neutrino flavors (nuother) from 8B and hep processes using ES cross-sections.
- `unoscillatedRate_CC_8B`, `unoscillatedRate_CC_hep`: Calculate unoscillated event rates for 8B and hep 
  processes using charged current (CC) cross-sections.

- `calculate_bins`: Computes bin edges and centers for a given range and number of bins, facilitating 
  energy binning for integration.
- `integrate_over_bins`: Integrates a given function over specified energy bins and calculates the average 
  value in each bin.

Process:
1. Defines functions to compute unoscillated event rates for different neutrino flavors and interaction 
   channels using flux and cross-section data.
2. Calculates bin edges and centers for energy integration using `calculate_bins`.
3. Integrates the unoscillated rate functions over energy bins using `integrate_over_bins` to obtain binned 
   event rates.
4. Normalizes the binned event rates using predefined normalization constants.
5. Stores the results in a constant `unoscillatedSample` for easy access and saves the data to a JLD2 file 
   for further analysis.

Output:
- A JLD2 file named `unoscillatedSamples.jld2` containing binned event rates for different neutrino flavors 
  and interaction channels, along with the energy bin centers.

Note:
- The script uses a temporary scaling factor (`1e38`) in the rate calculations, which should be adjusted 
  once the cross-section units are finalized.
- Ensure that the included modules and constants are correctly defined and accessible in the working 
  environment.
"""


# Create the unoscillated flux at the detector
using QuadGK

# Bring in flux and cross section data
include("../src/solarFlux.jl")
include("../src/xsec.jl")

detector_ne = 2 * 2.7e33 #2 * 2.7e34 # 2 x number of electrons in 10KTon of LAr (2 modules)
detection_time = 3600 * 24 * 365 * 10  #10 * 3.1536e7 # 10 years of runtime x number of seconds in a year

function unoscillatedRate_ES_nue_8B(enu)
    return flux_8B(enu) * ES_xsec_nue(enu)
end

# Function for unoscillated rate with hep flux and ES cross-section for nue
function unoscillatedRate_ES_nue_hep(enu)
    return flux_hep(enu) * ES_xsec_nue(enu)
end

# Function for unoscillated rate with 8B flux and ES cross-section for nuother
function unoscillatedRate_ES_nuother_8B(enu)
    return flux_8B(enu) * ES_xsec_nuother(enu)
end

# Function for unoscillated rate with hep flux and ES cross-section for nuother
function unoscillatedRate_ES_nuother_hep(enu)
    return flux_hep(enu) * ES_xsec_nuother(enu)
end

# Function for unoscillated rate with 8B flux and CC cross-section
function unoscillatedRate_CC_8B(enu)
    return flux_8B(enu) * CC_xsec(enu)
end

# Function for unoscillated rate with hep flux and CC cross-section
function unoscillatedRate_CC_hep(enu)
    return flux_hep(enu) * CC_xsec(enu)
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

# Integrate over energy bins to get binned event rates
bin_edges, energies_GeV = calculate_bins(Etrue_bins)
global bin_edges_calc, energies_calc = calculate_bins((max=Etrue_bins.max, min=Etrue_bins.min, bin_number=Etrue_bins.bin_number*3))

unoscillated_ES_nue_sample_8B = integrate_over_bins(unoscillatedRate_ES_nue_8B, bin_edges) * detector_ne * detection_time * ES_normalisation
unoscillated_ES_nuother_sample_8B = integrate_over_bins(unoscillatedRate_ES_nuother_8B, bin_edges) * detector_ne * detection_time * ES_normalisation
unoscillated_CC_sample_8B = integrate_over_bins(unoscillatedRate_CC_8B, bin_edges) * detector_ne * detection_time * CC_normalisation

unoscillated_ES_nue_sample_hep = integrate_over_bins(unoscillatedRate_ES_nue_hep, bin_edges) * detector_ne * detection_time * ES_normalisation
unoscillated_ES_nuother_sample_hep = integrate_over_bins(unoscillatedRate_ES_nuother_hep, bin_edges) * detector_ne * detection_time * ES_normalisation
unoscillated_CC_sample_hep = integrate_over_bins(unoscillatedRate_CC_hep, bin_edges) * detector_ne * detection_time * CC_normalisation

unoscillatedSample = (ES_nue_8B = unoscillated_ES_nue_sample_8B,
                      ES_nuother_8B = unoscillated_ES_nuother_sample_8B,
                      CC_8B = unoscillated_CC_sample_8B,
                      ES_nue_hep = unoscillated_ES_nue_sample_hep,
                      ES_nuother_hep = unoscillated_ES_nuother_sample_hep,
                      CC_hep = unoscillated_CC_sample_hep)

@save "outputs/unoscillatedSamples.jld2" unoscillated_ES_nue_sample_8B unoscillated_ES_nuother_sample_8B unoscillated_CC_sample_8B unoscillated_ES_nue_sample_hep unoscillated_ES_nuother_sample_hep unoscillated_CC_sample_hep energies_GeV
