#=
modelInputs.jl — loaders for the tabulated model inputs (solar model and spectra, exposure, Earth),
built on TabulatedInput. Function definitions only. The CC cross section is read in xsec.jl; the
NightOsc Earth model has its own reader (night/earth_model.jl).
=#
using QuadGK
using JLD2
include(joinpath(@__DIR__, "core.jl"))
include(joinpath(@__DIR__, "tabulated.jl"))

if !@isdefined(EARTH_RADIUS_KM)
    const EARTH_RADIUS_KM = 6371.0
end

"""
    solar_flux_spectra(energies, flux8B, fluxHep, E_max; source) -> NamedTuple

The ⁸B and hep spectra (energies in GeV), normalised to unit integral. The table is completed by
zero flux at E = 0 and, if E_max > energies[end], at E_max. Evaluating outside it is an error.
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
    validate_solar_model(radii, n_e, prod_8B, prod_hep; source) -> nothing

Solar-model input contract (SPEC 8.2): `radii` [R/R_sun] finite, strictly increasing, in (0, 1];
`n_e` [N_A/cm^3] finite and > 0; the production weights finite, >= 0, not all zero, one per radius.
Weights are normalised by the fitter, so dP/dr or per-zone fractions both work. Errors name the file.
"""
function validate_solar_model(radii, n_e, prod_8B, prod_hep; source::AbstractString = "")
    w = "solar model $(isempty(source) ? "" : "($source)")"
    n = length(radii)
    n >= 2 || error("$w: fewer than two radii")
    for (name, v) in (("n_e", n_e), ("prodFractionBoron", prod_8B), ("prodFractionHep", prod_hep))
        length(v) == n || error("$w: $name has $(length(v)) entries for $n radii")
        all(isfinite, v) || error("$w: $name has non-finite values")
    end
    all(isfinite, radii) && all(>(0), diff(radii)) || error("$w: radii must be finite and strictly increasing")
    (radii[1] > 0 && radii[end] <= 1) || error("$w: radii must lie in (0, 1] (fractions of the solar radius)")
    all(>(0), n_e) || error("$w: n_e must be > 0")
    for (name, v) in (("prodFractionBoron", prod_8B), ("prodFractionHep", prod_hep))
        all(>=(0), v) || error("$w: $name has negative weights")
        sum(v) > 0 || error("$w: $name is all zero")
    end
    return nothing
end

"""
    load_solar_model(path) -> (model::NamedTuple, flux_hep)

Read and validate a solar-model JLD2 file and add the production-weighted mean densities (fastFit).
`flux_hep` is the file's hep flux (SF-III files), else `nothing`; returned separately so the model
tuple keeps its type.
"""
function load_solar_model(path::AbstractString)
    isfile(path) || error("Solar model file not found: $path")
    jldopen(path, "r") do file
        radii, n_e = file["radii"], file["n_e"]
        pB, pH = file["prodFractionBoron"], file["prodFractionHep"]
        validate_solar_model(radii, n_e, pB, pH; source = path)
        model = (radii = radii, prodFractionBoron = pB, prodFractionHep = pH, n_e = n_e,
                 avgNeBoron = sum(pB .* n_e) / sum(pB), avgNeHep = sum(pH .* n_e) / sum(pH))
        return model, (haskey(file, "flux_hep") ? Float64(file["flux_hep"]) : nothing)
    end
end

"""
    load_exposure(path) -> (time = TabulatedInput, flux = TabulatedInput)

Solar exposure vs cos z (SPEC 4.6) from a CSV with header `cosz,time_weight,flux_weight`
(`utils/make_solar_exposure.py`; cos z < 0 is night): `time_weight` for the backgrounds,
`flux_weight` (× (1 AU / r)²) for the signal. Evaluating outside the table is an error. A legacy
single-weight file is rejected.
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

Day/night split of one exposure column (SPEC 4.6): f_night = ∫_{cos z<0} / ∫, f_day = 1 − f_night.
Exact for the linear interpolant, summed outward from cos z = 0 so a symmetric table gives exactly
0.5. Errors unless the table has exposure on both sides of the horizon.
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

Earth model table (r/R⊕, density [g/cm³], electron fraction; no header). Radii in km, stretched by
1e-10 so surface points stay inside the table; potential in eV²/GeV.
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
