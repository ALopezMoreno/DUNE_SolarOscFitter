using QuadGK          # For numerical integration
using Printf
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

# Day/night split of the exposure (SPEC 4.6), from the whole table: f_night = ∫_{cos z<0} / ∫_all,
# f_day = 1 − f_night. Every day term (signal and background) carries DAY_FRACTION and every night
# term NIGHT_FRACTION (the night rows are further split by exposure_weights, which sum to 1).
# Assumption: the exposure file is weighted by time (fraction of time the Sun spends at each cos z,
# no 1/r² seasonal flux weighting), so its night fraction is also the live-time night fraction that
# the backgrounds use. Setup-time constants (const globals, like exposure_weights): the likelihood
# reads them as constants, which keeps it type-stable for Enzyme.
const DAY_FRACTION, NIGHT_FRACTION = let f = exposure_day_night_fractions(exposure_intp)
    f.day, f.night
end
@logmsg Setup @sprintf("Exposure day/night split from %s: f_day = %.6f, f_night = %.6f",
                       solarExposureFile, DAY_FRACTION, NIGHT_FRACTION)
