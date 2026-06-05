using Interpolations  # For exposure interpolation
using QuadGK          # For numerical integration
using CSV             # For reading exposure data
using DataFrames      # For data manipulation

# Load solar exposure data from CSV file
# Format: cos(zenith), relative_exposure
exposure = CSV.File(solarExposureFile, header=false) |> DataFrame

# Extract zenith angle and exposure data
cosz_x = exposure[:, 1]  # cos(zenith) values
cosz_y = exposure[:, 2]  # Relative exposure values

# Create interpolated exposure function
# Uses flat extrapolation beyond data range
exposure_intp = linear_interpolation(cosz_x, cosz_y, extrapolation_bc=Flat())

# Calculate normalization by integrating over the full zenith range
lower_limit = cosz_bins.min  # Minimum cos(zenith) = -1 (upward-going)
upper_limit = cosz_bins.max  # Maximum cos(zenith) = 0 (horizontal)

# Integrate exposure function over full range for normalization
exposure_intp_int, _ = quadgk(exposure_intp, lower_limit, upper_limit)

# Alternative: Create normalized interpolation function (currently commented)
#exposure_intp_norm = linear_interpolation(cosz_x, cosz_y / exposure_intp_int, extrapolation_bc=Flat())

# Calculate exposure weights for each zenith angle bin
bin_edges = COARSE_COSZ_EDGES

# Define integration function for each bin
compute_integral(lower, upper) = quadgk(exposure_intp, lower, upper)[1]

# Calculate normalized exposure weights for each bin
# These weights represent the fraction of total exposure in each zenith bin
global exposure_weights = reshape(
    compute_integral.(bin_edges[1:end-1], bin_edges[2:end]) ./ exposure_intp_int,
    :, 1
)
