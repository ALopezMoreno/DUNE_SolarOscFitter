using QuadGK          # For numerical integration
using Printf
include(joinpath(@__DIR__, "modelInputs.jl"))   # load_exposure, exposure_day_night_fractions

# Solar exposure vs cos z (SPEC 4.6): two weightings on the same cos z nodes (load_exposure).
#   signal       flux-weighted, (1 AU / r)²: the solar-neutrino rate follows the flux. By Kepler's
#                second law dt ∝ r² dλ, so this is uniform in solar longitude (f_night = 0.5).
#   backgrounds  time-weighted: their rates are fixed in time, so they follow the live time.
# Each has its own day/night split and its own night cos z weights; there is no shared set, so
# nothing can silently use the wrong one. Only evaluated inside the table: the integrals below start
# at COARSE_COSZ_EDGES[1], the lowest night cos z of this file, and end at 0, and NightOsc's BinnedP1e
# evaluates the signal column inside the cos z grid. (It used Flat() extrapolation before; NightOsc
# clamps its zenith nodes to the bins, which a 1-ulp round trip can otherwise leave just below the
# table, see night/binning.jl.)
const _exposure_tables = load_exposure(solarExposureFile)
const exposure_intp_signal = _exposure_tables.flux
const exposure_intp_bg     = _exposure_tables.time

# Night cos z weights per coarse bin, normalised over the bin range [COARSE_COSZ_EDGES[1], 0] (Σ = 1).
# The coarse grid may start above cosz_bins.min (clipped to the exposure compact support).
# Integrating from cosz_bins.min would leave the table (an error now; with the old flat
# extrapolation it put phantom exposure in the denominator, making Σ weights < 1).
function night_cosz_weights(expo, edges, upper)
    # Normalised by the sum of the bin integrals, not by one more quadgk over the range: the two agree
    # only to quadgk's tolerance (~1e-8 on the kinked interpolant), and a background 1e6 × the signal
    # turns that into a fake night excess. This way Σ = 1 to rounding.
    edges[end] == upper || error("night cos z edges end at $(edges[end]), not at $upper")
    w = [quadgk(expo, edges[i], edges[i+1])[1] for i in 1:length(edges)-1]
    return reshape(w ./ sum(w), :, 1)
end
const exposure_weights_signal = night_cosz_weights(exposure_intp_signal, COARSE_COSZ_EDGES, cosz_bins.max)
const exposure_weights_bg     = night_cosz_weights(exposure_intp_bg,     COARSE_COSZ_EDGES, cosz_bins.max)

# Day/night splits over the whole table: f_night = ∫_{cos z<0} / ∫_all, f_day = 1 − f_night. Every
# day signal term carries SIGNAL_DAY_FRACTION and every night signal term SIGNAL_NIGHT_FRACTION ×
# exposure_weights_signal; backgrounds carry BG_DAY_FRACTION and BG_NIGHT_FRACTION × exposure_weights_bg.
# Setup-time constants (const globals): the likelihood reads them as constants, which keeps it
# type-stable for Enzyme.
const SIGNAL_DAY_FRACTION, SIGNAL_NIGHT_FRACTION = let f = exposure_day_night_fractions(exposure_intp_signal)
    f.day, f.night
end
const BG_DAY_FRACTION, BG_NIGHT_FRACTION = let f = exposure_day_night_fractions(exposure_intp_bg)
    f.day, f.night
end
@logmsg Setup @sprintf("Exposure day/night split from %s: signal (flux-weighted) f_day = %.6f, f_night = %.6f; backgrounds (time-weighted) f_day = %.6f, f_night = %.6f",
                       solarExposureFile, SIGNAL_DAY_FRACTION, SIGNAL_NIGHT_FRACTION, BG_DAY_FRACTION, BG_NIGHT_FRACTION)
const SIGNAL_FRACTIONS = (day = SIGNAL_DAY_FRACTION, night = SIGNAL_NIGHT_FRACTION)
const BG_FRACTIONS     = (day = BG_DAY_FRACTION,     night = BG_NIGHT_FRACTION)
# What the predictions actually carry of a background spectrum BG, by day and over all night bins:
# f_day^B·BG and f_night^B·Σ_k w^B_k·BG. Σ_k w^B_k = 1 to rounding (night_cosz_weights), but backgrounds
# can be 1e6 × the signal, so subtracting the background for the signal asymmetry uses exactly the
# factors the predictions carry.
const BG_TOTAL_FACTORS = (day = BG_DAY_FRACTION, night = BG_NIGHT_FRACTION * sum(exposure_weights_bg))
