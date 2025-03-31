using Interpolations
using QuadGK
using CSV
using DataFrames

exposure = CSV.File(solarExposureFile, header=false) |> DataFrame

cosz_x = exposure[:, 1]
cosz_y = exposure[:, 2]

# create interpolated exposure
exposure_intp = LinearInterpolation(cosz_x, cosz_y, extrapolation_bc=Flat())

# Normalise
lower_limit = cosz_bins.min
upper_limit = cosz_bins.max

# Integrate the functions using QuadGK
exposure_intp_int, _ = quadgk(exposure_intp, lower_limit, upper_limit)

# Remake interpolation
#exposure_intp_norm = LinearInterpolation(cosz_x, cosz_y / exposure_intp_int, extrapolation_bc=Flat())

# Get weights across cosz bins
bin_edges = range(cosz_bins.min, cosz_bins.max, length=cosz_bins.bin_number+1)

# Define a function to compute the integral for a given range
compute_integral(lower, upper) = quadgk(exposure_intp, lower, upper)[1]

# Use broadcasting to compute integrals for each bin
global exposure_weights = compute_integral.(bin_edges[1:end-1], bin_edges[2:end]) ./ exposure_intp_int 
