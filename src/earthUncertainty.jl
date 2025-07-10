#=
earthUncertainty.jl

Earth matter density uncertainty processing for systematic error analysis.
This module loads and processes uncertainties in the Earth's density profile
(PREM model uncertainties) and propagates them through the neutrino oscillation
calculations to estimate systematic uncertainties in predicted event rates.

Key Features:
- Loading of PREM (Preliminary Reference Earth Model) uncertainty data
- Interpolation and rebinning of uncertainty matrices
- Propagation of Earth model uncertainties through full analysis chain
- Calculation of covariance matrices for systematic uncertainties
- Statistical analysis of uncertainty distributions

The Earth matter uncertainties affect neutrino oscillation probabilities
during nighttime propagation and contribute to systematic errors in the analysis.

Author: [Author name]
=#

using JLD2           # For loading uncertainty data files
using Interpolations # For uncertainty matrix interpolation
using Plots          # For diagnostic plotting (optional)
using Statistics     # For statistical calculations

include("../src/propagateSample.jl")

# Load PREM uncertainty data
# This contains oscillation probability uncertainties due to Earth density variations
earth_uncertainty = JLD2.load("./inputs/prem_uncertainties_4.500e-5.jld2", "probabilities")

# Define coordinate extents for interpolation
extent = [(-1, 0), (3, 20)]  # Original data extent: [cos(zenith), energy(MeV)]
extent2 = [(cosz_bins.min, cosz_bins.max), (Etrue_bins.min * 1e3, Etrue_bins.max * 1e3)]  # Analysis extent

# Uncertainty matrix processing functions

function interpolate_matrices(earth_uncertainty, extent)
    """
    Interpolate each uncertainty matrix in the 3D array to enable rebinning.
    
    Creates B-spline interpolation objects for each uncertainty realization
    to allow evaluation at arbitrary coordinate points.
    
    Arguments:
    - earth_uncertainty: 3D array of uncertainty matrices
    - extent: Coordinate ranges for interpolation
    
    Returns:
    - interpolated_matrices: Array of interpolation objects
    """
    interpolated_matrices = []
    
    # Build coordinate ranges from the extent
    x_range = range(extent[1][1], extent[1][2], length=size(earth_uncertainty, 1))
    y_range = range(extent[2][1], extent[2][2], length=size(earth_uncertainty, 2))

    # Process each uncertainty realization
    for i in 1:size(earth_uncertainty, 1)
        matrix = reverse(earth_uncertainty[:, :, i], dims=1)

        # Create B-spline interpolation with coordinate scaling
        itp = interpolate(matrix, BSpline(Linear()))
        sitp = Interpolations.scale(itp, x_range, y_range)
        push!(interpolated_matrices, sitp)
    end
    return interpolated_matrices
end

function average_over_bins(interpolated_matrices, nbins_z, nbins_e, extent)
    """
    Average interpolated uncertainty matrices over analysis bins.
    
    Evaluates each interpolated matrix on a fine grid within each analysis bin
    and computes the average value for that bin.
    
    Arguments:
    - interpolated_matrices: Array of interpolation objects
    - nbins_z: Number of zenith angle bins
    - nbins_e: Number of energy bins
    - extent: Coordinate ranges for binning
    
    Returns:
    - averaged_matrices: Array of rebinned uncertainty matrices
    """
    averaged_matrices = []
    x_bins = range(extent[1][1], extent[1][2], length=nbins_z + 1)
    y_bins = range(extent[2][1], extent[2][2], length=nbins_e + 1)
    
    for myfunct in interpolated_matrices
        new_matrix = zeros(nbins_z, nbins_e)
        for i in 1:nbins_z
            for j in 1:nbins_e
                x_start, x_end = x_bins[i], x_bins[i + 1]
                y_start, y_end = y_bins[j], y_bins[j + 1]
                
                # Generate fine grid within current bin for accurate averaging
                x_points = range(x_start, x_end, length=30)
                y_points = range(y_start, y_end, length=30)
                
                # Evaluate interpolated function at grid points
                values = [myfunct(x, y) for x in x_points, y in y_points]
                
                # Store average value for this bin
                new_matrix[i, j] = mean(values)
            end
        end
        # Apply coordinate transformations for proper orientation
        push!(averaged_matrices, reverse(reverse(new_matrix', dims=1), dims=2))
    end
    return averaged_matrices
end

# Interpolate each matrix
interpolated_earth_uncertainty = interpolate_matrices(earth_uncertainty, extent)

nbins_z = cosz_bins.bin_number
nbins_e = Etrue_bins.bin_number

averaged_earth_uncertainty = average_over_bins(interpolated_earth_uncertainty, nbins_e, nbins_z, extent2)

# p2 = heatmap(averaged_earth_uncertainty[1],
#              title="Covariance matrix",
#              xlabel="bins",
#              ylabel="bins",
#              aspect_ratio=:equal,  # Ensures the plot is square
#              size=(500,500))       # Increase the size for clarity

# display(p2)
# sleep(10)

uncertainty_ES_nue_day_list = []
uncertainty_ES_nuother_day_list = []
uncertainty_CC_day_list = []
uncertainty_ES_nue_night_list = []
uncertainty_ES_nuother_night_list = []
uncertainty_CC_night_list = []

@logmsg Setup ("########################")
@logmsg Setup ("propagating PREM samples")
@logmsg Setup ("########################")
println(" ")

# Iterate over each matrix in averaged_earth_uncertainty
for uncertaintyProbs in averaged_earth_uncertainty
    # Call propagateSamplesUncertainty for each matrix
    uncertainty_ES_nue_day, uncertainty_ES_nuother_day, uncertainty_CC_day, uncertainty_ES_nue_night, uncertainty_ES_nuother_night, uncertainty_CC_night = propagateSamplesUncertainty(unoscillatedSample, responseMatrices, true_params, solarModel, bin_edges, CC_bg, uncertaintyProbs)
    
    # Append each result to the corresponding list
    push!(uncertainty_ES_nue_day_list, uncertainty_ES_nue_day)
    push!(uncertainty_ES_nuother_day_list, uncertainty_ES_nuother_day)
    push!(uncertainty_CC_day_list, uncertainty_CC_day)
    push!(uncertainty_ES_nue_night_list, uncertainty_ES_nue_night)
    push!(uncertainty_ES_nuother_night_list, uncertainty_ES_nuother_night)
    push!(uncertainty_CC_night_list, uncertainty_CC_night)
end


function compute_elementwise_std(matrix_list)
    # Get the dimensions of the matrices
    num_matrices = length(matrix_list)
    num_rows, num_cols = size(matrix_list[1])

    # Initialize an array to store the standard deviation
    std_matrix = zeros(num_rows, num_cols)

    # Compute standard deviation for each element
    for i in 1:num_rows
        for j in 1:num_cols
            elements = [matrix[i, j] for matrix in matrix_list]
            std_matrix[i, j] = std(elements)
        end
    end

    return std_matrix
end

function compute_elementwise_mean(matrix_list)
    # Get the dimensions of the matrices
    num_matrices = length(matrix_list)
    num_rows, num_cols = size(matrix_list[1])

    # Initialize an array to store the standard deviation
    std_matrix = zeros(num_rows, num_cols)

    # Compute standard deviation for each element
    for i in 1:num_rows
        for j in 1:num_cols
            elements = [matrix[i, j] for matrix in matrix_list]
            std_matrix[i, j] = mean(elements)
        end
    end

    return std_matrix
end



function compute_covariance_matrix(matrix_list)
    # Flatten each matrix into a vector
    flattened_matrices = [vec(matrix') for matrix in matrix_list]

    # Stack vectors into a 2D array
    data_matrix = vcat(flattened_matrices'...)
    
    # Compute the covariance matrix
    covariance_matrix = cov(data_matrix)
    
    return covariance_matrix
end

function safe_elementwise_division(numerator_matrix, denominator_matrix)
    # Ensure the matrices have the same dimensions
    num_rows, num_cols = size(numerator_matrix)
    result_matrix = zeros(num_rows, num_cols)

    # Perform element-wise division with zero-check
    for i in 1:num_rows
        for j in 1:num_cols
            if denominator_matrix[i, j] == 0
                result_matrix[i, j] = numerator_matrix[i, j] / 0.5
            else
                result_matrix[i, j] = numerator_matrix[i, j] / denominator_matrix[i, j]
            end
        end
    end

    return result_matrix
end

# Example usage for one of the lists
covariance_matrix_ES_nue_night = compute_covariance_matrix(uncertainty_ES_nue_night_list)
covariance_matrix_ES_nuother_night = compute_covariance_matrix(uncertainty_ES_nuother_night_list)
covariance_matrix_CC_night = compute_covariance_matrix(uncertainty_CC_night_list)

std_matrix_ES_nue_night = compute_elementwise_std(uncertainty_ES_nue_night_list)
std_matrix_ES_nuother_night = compute_elementwise_std(uncertainty_ES_nuother_night_list)
std_matrix_CC_night = compute_elementwise_std(uncertainty_CC_night_list)

mean_matrix_ES_nue_night = compute_elementwise_mean(uncertainty_ES_nue_night_list)
mean_matrix_ES_nuother_night = compute_elementwise_mean(uncertainty_ES_nuother_night_list)
mean_matrix_CC_night = compute_elementwise_mean(uncertainty_CC_night_list)

global uncertainty_ratio_matrix_ES_nue_night = safe_elementwise_division(std_matrix_ES_nue_night, mean_matrix_ES_nue_night)
global uncertainty_ratio_matrix_ES_nuother_night = safe_elementwise_division(std_matrix_ES_nuother_night, mean_matrix_ES_nuother_night)
global uncertainty_ratio_matrix_CC_night = safe_elementwise_division(std_matrix_CC_night, mean_matrix_CC_night)

# Create individual heatmaps with square aspect ratio and larger size
#p1 = heatmap(uncertainty_ratio_matrix_CC_night,
#             title="ratio matrix",
#             xlabel="E",
#             ylabel="cosz",
#             aspect_ratio=:equal,
#             size=(1500,1500))

#p2 = heatmap(measuredRate_CC_night,
#             title="measured matrix",
#             xlabel="E",
#             ylabel="cosz",
#             aspect_ratio=:equal,
#             size=(1500,1500))

# Display the plots side by side
#plot(p1, p2, layout=(1, 2))
#display(current())
#sleep(2)