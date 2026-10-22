using QuadGK
using Printf
include(joinpath(@__DIR__, "modelInputs.jl"))

# Solar exposure vs cos z (SPEC 4.6): the signal is flux-weighted (uniform in solar longitude by
# Kepler's second law, f_night = 0.5), the backgrounds time-weighted. Each has its own day/night
# split and night cos z weights; there is no shared set to use by mistake.
const _exposure_tables = load_exposure(solarExposureFile)
const exposure_intp_signal = _exposure_tables.flux
const exposure_intp_bg     = _exposure_tables.time

# Night cos z weights per coarse bin over [COARSE_COSZ_EDGES[1], 0]. Normalised by the sum of the
# bin integrals, so Σ = 1 to rounding: a background 1e6 × the signal turns a 1e-8 excess into a fake
# night signal.
function night_cosz_weights(expo, edges, upper)
    edges[end] == upper || error("night cos z edges end at $(edges[end]), not at $upper")
    w = [quadgk(expo, edges[i], edges[i+1])[1] for i in 1:length(edges)-1]
    return reshape(w ./ sum(w), :, 1)
end
const exposure_weights_signal = night_cosz_weights(exposure_intp_signal, COARSE_COSZ_EDGES, cosz_bins.max)
const exposure_weights_bg     = night_cosz_weights(exposure_intp_bg,     COARSE_COSZ_EDGES, cosz_bins.max)

# Day/night splits (const, for type stability): day terms carry *_DAY_FRACTION, night terms
# *_NIGHT_FRACTION × the night weights.
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
# The day and night-total factors a background spectrum carries in the predictions, used exactly when
# subtracting it for the signal asymmetry.
const BG_TOTAL_FACTORS = (day = BG_DAY_FRACTION, night = BG_NIGHT_FRACTION * sum(exposure_weights_bg))
