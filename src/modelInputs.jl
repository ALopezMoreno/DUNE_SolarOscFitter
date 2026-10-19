#=
modelInputs.jl — loaders for the tabulated model inputs, built on TabulatedInput (tabulated.jl).
No top-level computation: solarFlux.jl, exposure.jl and earthProfile.jl call these with the
configured files, and tests/test_tabulated.jl calls them directly. The CC cross section is read
in xsec.jl (read_cc_xsec_table). The NightOsc Earth model keeps its own reader and interpolation
(src/oscillations/night/earth_model.jl), validated to the same contract.
=#
using QuadGK
include(joinpath(@__DIR__, "core.jl"))         # MATTER_POTENTIAL_CONST
include(joinpath(@__DIR__, "tabulated.jl"))

if !@isdefined(EARTH_RADIUS_KM)
    const EARTH_RADIUS_KM = 6371.0  # Earth radius in kilometers
end

"""
    solar_flux_spectra(energies, flux8B, fluxHep, E_max; source) -> NamedTuple

The ⁸B and hep spectra (energies in GeV, from flux_model_file), normalised to unit integral.
Two modelling points complete the table; they are part of it, not an extrapolation: zero flux
at E = 0 and, when the analysis range ends above the last tabulated energy (E_max >
energies[end]), zero flux at E_max (a linear ramp from the last tabulated value down to 0).
Evaluating outside [0, max(energies[end], E_max)] is an error.
"""
function solar_flux_spectra(energies, flux8B, fluxHep, E_max; source::AbstractString = "")
    E = [0.0; energies]
    f8B = [0.0; flux8B]
    fhep = [0.0; fluxHep]
    if E_max > energies[end]
        E = [0.0; energies; E_max]
        f8B = [0.0; flux8B; 0.0]
        fhep = [0.0; fluxHep; 0.0]
    end
    raw_8B = TabulatedInput(E, f8B; name = "8B spectrum (raw)", source, units = "GeV", nonnegative = true, register = false)
    raw_hep = TabulatedInput(E, fhep; name = "hep spectrum (raw)", source, units = "GeV", nonnegative = true, register = false)
    integral_8B, _ = quadgk(raw_8B, E[1], E[end])
    integral_hep, _ = quadgk(raw_hep, E[1], E[end])
    return (
        energies = E,
        raw_8B = raw_8B, raw_hep = raw_hep,
        integral_8B = integral_8B, integral_hep = integral_hep,
        flux_8B = TabulatedInput(E, f8B / integral_8B; name = "8B spectrum", source, units = "GeV", nonnegative = true),
        flux_hep = TabulatedInput(E, fhep / integral_hep; name = "hep spectrum", source, units = "GeV", nonnegative = true),
    )
end

"""
    load_exposure(path) -> (time = TabulatedInput, flux = TabulatedInput)

Solar exposure vs cos z (SPEC 4.6): a CSV with the header row `cosz,time_weight,flux_weight` ('#'
comment lines allowed), made by `utils/make_solar_exposure.py`; every row is a point, cos z < 0 is
night (Sun below the horizon).
- `time_weight`: fraction of time the Sun spends per unit cos z. Backgrounds (rates fixed in time).
- `flux_weight`: the same weighted by the solar flux, (1 AU / r)². The signal.

The two columns are read as two TabulatedInputs on the same nodes. Evaluating outside the table is
an error. (It used Flat() extrapolation before.) The integrals start at the lowest night cos z of
this file (exposure.jl), and NightOsc clamps its zenith nodes to the bin range, which they can miss by
1 ulp (night/binning.jl). The table must reach both sides of cos z = 0: the day/night splits come from
it (exposure_day_night_fractions). The legacy single-weight file (headerless `cos z, exposure`) is an
error: one column for both would give the signal the wrong weighting.
"""
function load_exposure(path::AbstractString)
    isfile(path) || error("solar exposure: table not found: $path")
    lines = strip.(readlines(path))
    first_row = findfirst(l -> !isempty(l) && !startswith(l, "#"), lines)
    hdr = first_row === nothing ? "" : lines[first_row]
    want = ["cosz", "time_weight", "flux_weight"]
    if strip.(split(hdr, ',')) != want
        legacy = all(f -> tryparse(Float64, strip(f)) !== nothing, split(hdr, ',')) && length(split(hdr, ',')) == 2
        error("solar exposure $path: expected the header row \"$(join(want, ','))\" " *
              (legacy ? "(this is a legacy two-column file: one weight for signal and backgrounds alike) " :
                        "(found \"$hdr\") ") *
              "; make the table with `python3 utils/make_solar_exposure.py -o <file>`")
    end
    cosz, wt, wf = read_numeric_table(path; name = "solar exposure", ncols = 3, header = true)
    time = TabulatedInput(cosz, wt; name = "solar exposure (time)", source = path, units = "cos z",
                          extrapolation = :error, nonnegative = true)
    flux = TabulatedInput(cosz, wf; name = "solar exposure (flux)", source = path, units = "cos z",
                          extrapolation = :error, nonnegative = true)
    return (time = time, flux = flux)
end

"""
    exposure_day_night_fractions(expo::TabulatedInput) -> (day = f_day, night = f_night)

Day/night split of one exposure column (SPEC 4.6): f_night = ∫ exposure over cos z < 0 divided by ∫
exposure over the whole table, f_day = 1 − f_night (night = Sun below the horizon, cos z < 0). Applied
to the flux-weighted column it gives the signal split, to the time-weighted one the background split.

The integrals are exact for the linear interpolant: trapezoids on the nodes plus cos z = 0, each side
summed outward from cos z = 0, so a table mirror-symmetric in cos z gives exactly 0.5. Errors if the
table does not reach both sides of the horizon, or has no exposure on one side (a night-only file
cannot fix the split).
"""
function exposure_day_night_fractions(expo::TabulatedInput)
    x, y = expo.x, expo.y
    w = _where(expo.name, expo.source)
    (x[1] < 0 && x[end] > 0) ||
        error("$w: the day/night exposure split needs a table on both sides of cos z = 0 " *
              "(it covers [$(x[1]), $(x[end])])")
    y0 = Float64(expo.itp(0.0))
    k = searchsortedlast(x, 0.0)            # x[k] ≤ 0 < x[k+1]
    night = 0.0                             # outward from 0: [x[k], 0], then [x[k-1], x[k]], …
    xa, ya = 0.0, y0
    for i in k:-1:1
        night += (xa - x[i]) * (ya + y[i]) / 2
        xa, ya = x[i], y[i]
    end
    day = 0.0
    xa, ya = 0.0, y0
    for i in k+1:length(x)
        day += (x[i] - xa) * (ya + y[i]) / 2
        xa, ya = x[i], y[i]
    end
    (night > 0 && day > 0) ||
        error("$w: the day/night exposure split needs exposure on both sides of cos z = 0 " *
              "(night $night, day $day)")
    total = night + day
    f_night = night / total
    return (day = 1 - f_night, night = f_night)
end

"""
    load_earth_model(path) -> Dict(:radius, :potential, :source)

Earth model table: whitespace-delimited (r/R⊕, density [g/cm³], electron fraction), no header
row. Radii are in km, stretched by 1e-10 so that points on the surface (r = 6371 km up to
rounding, e.g. the end of a makePaths track) stay inside the table; potential in eV²/GeV.
"""
function load_earth_model(path::AbstractString)
    r_frac, density, e_fraction = read_numeric_table(path; name = "Earth model", ncols = 3, header = false, delim = nothing)
    validate_table(r_frac, density; name = "Earth density", source = path, nonnegative = true)
    validate_table(r_frac, e_fraction; name = "Earth electron fraction", source = path, nonnegative = true)
    radius = r_frac .* EARTH_RADIUS_KM .* 1.0000000001
    potential = density .* e_fraction .* MATTER_POTENTIAL_CONST
    return Dict{Symbol,Any}(:radius => radius, :potential => potential, :source => String(path))
end

"""
    create_interpolated_model(earth_model) -> TabulatedInput

Earth matter potential vs radius [km]: linear between the table points, an error outside it.
"""
function create_interpolated_model(earth_model::Dict)
    return TabulatedInput(earth_model[:radius], earth_model[:potential]; name = "Earth potential",
                          source = get(earth_model, :source, ""), units = "km",
                          extrapolation = :error, nonnegative = true)
end
