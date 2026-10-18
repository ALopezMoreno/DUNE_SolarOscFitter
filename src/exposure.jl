using QuadGK          # For numerical integration
include(joinpath(@__DIR__, "modelInputs.jl"))   # load_exposure

# Solar exposure vs cos z (headerless CSV: cos(zenith), relative_exposure; every row is a point).
# Only evaluated inside the table: the integrals below start at COARSE_COSZ_EDGES[1], the lowest
# night cos z of this file, and end at 0, and NightOsc's BinnedP1e evaluates it inside the cos z
# grid. (It used Flat() extrapolation before; NightOsc clamps its zenith nodes to the bins, which
# a 1-ulp round trip can otherwise leave just below the table, see night/binning.jl.)
exposure_intp = load_exposure(solarExposureFile)

# Normalise over the actual bin range, not the full [-1,0] range.
# The coarse grid may start above cosz_bins.min (e.g. clipped to the exposure
# compact support). Integrating from cosz_bins.min would leave the table (an error now; with
# the old flat extrapolation it put phantom exposure in the denominator, making
# Σexp_weights < 1 and breaking the day/night 50/50 split for the background).
lower_limit = COARSE_COSZ_EDGES[1]
upper_limit = cosz_bins.max  # Maximum cos(zenith) = 0 (horizontal)

# Integrate exposure function over full range for normalization
exposure_intp_int, _ = quadgk(exposure_intp, lower_limit, upper_limit)

# Calculate exposure weights for each zenith angle bin
bin_edges = COARSE_COSZ_EDGES

# Define integration function for each bin
compute_integral(lower, upper) = quadgk(exposure_intp, lower, upper)[1]

# Calculate normalized exposure weights for each bin
# These weights represent the fraction of total exposure in each zenith bin
const exposure_weights = reshape(
    compute_integral.(bin_edges[1:end-1], bin_edges[2:end]) ./ exposure_intp_int,
    :, 1
)
