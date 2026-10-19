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

# The engine's per-bin exposure must equal exposure_weights up to one common factor (SPEC 4.6).
# The bin totals the likelihood uses are exposure_weights_signal, exact cell integrals of the table;
# NightOsc's sampled weights only set the weighting inside a bin, and it normalises its bin average by
# them. The flux-weighted table has a narrow integrable peak at the declination turning point,
# cos z = −cos(φ + ε) ≈ −0.377, which NightOsc's zenith pieces sample coarsely: how far off the bins
# next to it are depends on where an edge falls (0.33 % on the regression grid, 0.58 % on the
# production one, 0.96 % with 60 + 20 bins). Elsewhere the two agree to ≤ 8e-5 on all those grids.
# So the check is per bin: within peak_halfwidth of the table's night-side maximum, rtol_peak; elsewhere
# rtol, against the median ratio of those bins. A wrong column (the time weighting given to the
# signal) is off by ~3 % in the deep bins and still trips it. Effect of the peak bins on the bin-averaged
# P1e, against pieces 16× finer: ≤ 1.2e-4 on the production grid, the same as with the time column
# (NightOsc's own accuracy, SPEC 1.3).
function check_exposure_normalisation(engine::NightOsc.BinnedP1e, cosz_edges, exposure, exposure_weights;
                                      rtol=1e-3, rtol_peak=5e-2, peak_halfwidth=0.03)
    den = NightOsc._den(engine)
    w = vec(exposure_weights)
    exposed = findall(>(0), w)
    zero_bins = findall(i -> (w[i] > 0) != (den[i] > 0), eachindex(w))
    isempty(zero_bins) ||
        error("nightOsc: cos z bins $zero_bins are exposed in one of the engine and exposure_weights but not the other")
    grid = range(first(cosz_edges), last(cosz_edges); length=2001)
    c_peak = grid[argmax(exposure.(grid))]
    near(i) = cosz_edges[i] < c_peak + peak_halfwidth && cosz_edges[i+1] > c_peak - peak_halfwidth
    far = filter(!near, exposed)
    ratio = den ./ w
    ref = sort(ratio[isempty(far) ? exposed : far])[cld(length(isempty(far) ? exposed : far), 2)]
    for i in exposed
        dev, tol = abs(ratio[i] / ref - 1), near(i) ? rtol_peak : rtol
        dev <= tol ||
            error("nightOsc: per-bin exposure of cos z bin $i [$(cosz_edges[i]), $(cosz_edges[i+1])] differs from " *
                  "exposure_weights by $(round(dev; sigdigits=3)) beyond a common normalisation (tolerance $tol); " *
                  "the engine and the likelihood use different exposures or grids")
    end
    return nothing
end
