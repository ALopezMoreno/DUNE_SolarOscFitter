#!/usr/bin/env julia
#=
make_solar_model_inputs.jl — build the fitter's solar-model files from the SF-III (B23) package.

Source: Herrera & Serenelli, "Standard Solar Models B23 / SF-III", Zenodo record 10822316, v1.2,
SolarModels_ICE2023_v1.2.zip (the tables parchita uses). Download and unzip it, then run

    julia --project utils/make_solar_model_inputs.jl <unzipped dir>/SolarModels

It writes, for a high-Z and a low-Z composition,

    inputs/SF3_GS98.jld2       GS98 (Grevesse & Sauval 1998), high-Z
    inputs/SF3_AGSS09met.jld2  AGSS09met (Asplund et al. 2009), low-Z

with the keys solarModel.jl reads (radii, n_e, prodFractionBoron, prodFractionHep) and provenance:
the model's 8B and hep fluxes at Earth with fractional errors, the source files and their SHA-256.

Conventions (same as the B16 files AGSS09_high_z.jld2 / GS98_low_z.jld2 it replaces):
- radii: R/R_sun on the model's 5e-4 grid, (0, 0.5]. Beyond 0.5 R_sun the model has < 1e-8 of the
  hep and < 1e-24 of the 8B production.
- n_e: free electron density [N_A/cm^3], 10^log_ne.
- prodFraction*: dP/d(R/R_sun) from nu_8B / nu_hep. The fitter normalises them itself.
  Spline-noise negatives near r = 0 (|x| < 1e-10) are clipped to 0, as in parchita.
=#
using JLD2, SHA, Printf

const MODELS = [("GS98", "SF3_GS98", "GS98 (Grevesse & Sauval 1998), high-Z"),
                ("AGSS09", "SF3_AGSS09met", "AGSS09met (Asplund et al. 2009), low-Z")]
const R_MAX = 0.5

function read_struct(path)
    lines = readlines(path)
    header = split(lstrip(lines[1], ['#', ' ']))
    rows = [parse.(Float64, split(l)) for l in lines[2:end] if !isempty(strip(l))]
    data = reduce(vcat, permutedims.(rows))
    col(name) = data[:, findfirst(==(name), header)]
    return col
end

# Flux row and fractional-error row of fluxes_SF3_<model>.dat (sources in its header line).
function read_fluxes(path)
    lines = readlines(path)
    i = findfirst(l -> endswith(strip(l), "Flux"), lines)
    sources = split(lines[i-1])
    flux = parse.(Float64, split(lines[i])[1:end-1])
    err = parse.(Float64, split(lines[i+1])[1:end-1])
    return Dict(s => (flux[k], err[k]) for (k, s) in enumerate(sources))
end

function build(dir, tag, outname, label)
    spath = joinpath(dir, "struct+nu_SF3_$tag.dat")
    fpath = joinpath(dir, "fluxes_SF3_$tag.dat")
    col = read_struct(spath)
    r = col("R_sun")
    keep = (r .> 0) .& (r .<= R_MAX + 1e-12)
    clip(x) = (minimum(x) > -1e-10 || error("$spath: production negative beyond spline noise"); max.(x, 0.0))
    radii = r[keep]
    n_e = 10 .^ col("log_ne")[keep]
    p8B = clip(col("nu_8B")[keep])
    phep = clip(col("nu_hep")[keep])
    fl = read_fluxes(fpath)
    out = joinpath(@__DIR__, "..", "inputs", outname * ".jld2")
    isfile(out) && rm(out)
    jldsave(out;
        radii, n_e, prodFractionBoron = p8B, prodFractionHep = phep,
        model = label,
        source = "Herrera & Serenelli, SF-III / B23, Zenodo 10822316 v1.2 (SolarModels_ICE2023_v1.2.zip)",
        source_files = [basename(spath), basename(fpath)],
        source_sha256 = [bytes2hex(open(sha256, spath)), bytes2hex(open(sha256, fpath))],
        flux_8B = fl["8B"][1], flux_8B_frac_err = fl["8B"][2],
        flux_hep = fl["hep"][1], flux_hep_frac_err = fl["hep"][2],
        flux_units = "cm^-2 s^-1 at Earth")
    @printf("%-20s %4d radii  8B %.3e (±%.1f%%)  hep %.3e (±%.1f%%)  -> %s\n", outname, length(radii),
            fl["8B"][1], 100fl["8B"][2], fl["hep"][1], 100fl["hep"][2], normpath(out))
end

length(ARGS) == 1 || error("usage: julia --project utils/make_solar_model_inputs.jl <SF-III SolarModels dir>")
for (tag, outname, label) in MODELS
    build(ARGS[1], tag, outname, label)
end
