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
    load_exposure(path) -> TabulatedInput

Solar exposure vs cos z: a headerless CSV of (cos z, relative exposure); every row is a point.
Evaluating outside the table is an error. (It used Flat() extrapolation before.) The integrals
start at the lowest night cos z of this file (exposure.jl), and NightOsc clamps its zenith nodes to
the bin range, which they can miss by 1 ulp (night/binning.jl). The table must cover
[lowest night cos z, 0]: a night-only file has to include cos z = 0.
"""
function load_exposure(path::AbstractString)
    cosz, w = read_numeric_table(path; name = "solar exposure", ncols = 2, header = false)
    return TabulatedInput(cosz, w; name = "solar exposure", source = path, units = "cos z",
                          extrapolation = :error, nonnegative = true)
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
