#=
osc_setup.jl

Selects the oscillation calculation from the config, once. The choice is stored in the
type of `OscSetup`, so the likelihood dispatches on it and stays type-stable.

Night backends:
- NightOscNight:    NightOsc bin averages (nightOsc: true, default)
- NumOscNight{F}:   3-flavour per-path NumOsc (nightOsc: false; F = fastFit)
=#

abstract type NightBackend end

struct NightOscNight <: NightBackend
    engine::NightOsc.BinnedP1e
end

struct NumOscNight{Fast,P,L} <: NightBackend
    paths::P        # makePaths paths at cosz_calc
    lookup::L       # average potential per layer
end
NumOscNight{Fast}(paths::P, lookup::L) where {Fast,P,L} = NumOscNight{Fast,P,L}(paths, lookup)

# EarthUnc: earth_norm_i parameters are sampled. FastSolar: mean production density (fastFit).
struct OscSetup{N<:NightBackend,EarthUnc,FastSolar}
    night::N
end

earth_uncertainty(::OscSetup{N,EU,FS}) where {N,EU,FS} = EU
fast_solar(::OscSetup{N,EU,FS}) where {N,EU,FS} = FS
uses_night_osc(osc::OscSetup) = osc.night isa NightOscNight

"""
    build_osc_setup(; night_osc, fast, earth_uncertainty, ...)

Build the `OscSetup` from the config switches and check it against the analysis grids,
the exposure and the Earth prior. `legacy_path(cz)` returns a makePaths path, used to check
that both Earth parsers number the layers the same way.
"""
function build_osc_setup(; night_osc::Bool, fast::Bool, earth_uncertainty::Bool,
                         earth_model_file, depth_km, cosz_edges, energy_edges, exposure,
                         exposure_weights, n_E_calc, n_earth_norms,
                         legacy_paths, legacy_lookup, legacy_path)
    n_cosz, n_E = length(cosz_edges) - 1, length(energy_edges) - 1
    length(exposure_weights) == n_cosz ||
        error("osc setup: $(length(exposure_weights)) exposure weights for $n_cosz cos z bins")
    n_E_calc == 2 * n_E ||
        error("osc setup: $n_E_calc E_calc points for $n_E E_true bins (expected two per bin)")
    if earth_uncertainty && n_earth_norms === nothing
        error("osc setup: earth_potential_uncertainties is on but no Earth normalisation prior was loaded")
    end

    night = if night_osc
        earth = NightOsc.EarthModel(earth_model_file; potential_const=MATTER_POTENTIAL_CONST)
        if earth_uncertainty && n_earth_norms != NightOsc.n_layers(earth)
            error("nightOsc: Earth model $earth_model_file has $(NightOsc.n_layers(earth)) layers, " *
                  "but the Earth-uncertainty prior has $n_earth_norms normalisations")
        end
        check_layer_agreement(earth, legacy_path, depth_km)
        engine = NightOsc.BinnedP1e(cosz_edges, energy_edges, exposure, earth;
                                    depth_km=depth_km, phase_const=Osc.PHASE_CONST)
        engine.shape == (n_cosz, n_E) ||
            error("nightOsc: engine grid $(engine.shape) differs from the analysis grid ($n_cosz, $n_E)")
        check_exposure_normalisation(engine, cosz_edges, exposure, exposure_weights)
        NightOscNight(engine)
    else
        NumOscNight{fast}(legacy_paths, legacy_lookup)
    end
    return OscSetup{typeof(night),earth_uncertainty,fast}(night)
end

# makePaths and NightOsc must cross the same layers in the same order (so earth_norm_i
# means the same layer in both), with lengths within tol_km
function check_layer_agreement(earth::NightOsc.EarthModel, legacy_path, depth_km;
                               cosz=(-0.05, -0.3, -0.6, -0.85, -0.95, -0.99, -1.0), tol_km=5.0)
    for cz in cosz
        legacy = legacy_path(cz).segments
        night  = NightOsc.build_earth_path(cz, earth; depth_km=depth_km)
        li = [Int(s.index) for s in legacy]
        ni = [s.layer for s in night]
        li == ni || error("Earth layers differ at cos z = $cz: makePaths crosses $li, NightOsc crosses $ni. " *
                          "earth_norm_i would scale different layers in the two calculations.")
        dl = maximum(abs(a.length - b.length) for (a, b) in zip(legacy, night))
        dl <= tol_km || error("Earth layer crossings differ by $(round(dl; digits=2)) km at cos z = $cz " *
                              "between makePaths and NightOsc")
    end
    return nothing
end

# The engine's per-bin exposure must equal exposure_weights up to one common factor.
# rtol = 1e-2 (SPEC 4.6). The bin totals the likelihood uses are exposure_weights_signal, exact cell
# integrals of the table; NightOsc's sampled weights only set the weighting inside a bin, and it
# normalises its bin average by them. The flux-weighted table has a narrow integrable peak at the
# declination turning point, cos z = −cos(φ + ε) ≈ −0.377, which NightOsc's zenith pieces sample
# coarsely: on the regression grid the two bins next to it are off by −0.33 % and +0.16 % (all others
# agree to 1e-6). Its effect on the bin-averaged P1e, against pieces 4× and 16× finer: ≤ 3.7e-4
# relative (2.4e-4 absolute) at Δm²₂₁ = 2e-5 and ≤ 4.2e-5 at 6e-5 in those two bins, against ≤ 5.3e-5
# elsewhere (NightOsc's own accuracy, SPEC 1.3). A wrong column (the time weighting given to the
# signal) still trips the guard: its per-bin ratio to the flux weighting spreads by 3.9 %.
function check_exposure_normalisation(engine::NightOsc.BinnedP1e, cosz_edges, exposure, exposure_weights; rtol=1e-2)
    den = NightOsc._den(engine)
    w = vec(exposure_weights)
    exposed = findall(>(0), w)
    zero_bins = findall(i -> (w[i] > 0) != (den[i] > 0), eachindex(w))
    isempty(zero_bins) ||
        error("nightOsc: cos z bins $zero_bins are exposed in one of the engine and exposure_weights but not the other")
    ratio = den[exposed] ./ w[exposed]
    spread = (maximum(ratio) - minimum(ratio)) / maximum(ratio)
    spread <= rtol ||
        error("nightOsc: per-bin exposure differs from exposure_weights beyond a common normalisation " *
              "(spread $(round(spread; sigdigits=3))); the engine and the likelihood use different exposures or grids")
    return nothing
end
