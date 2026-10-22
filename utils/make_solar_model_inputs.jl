#!/usr/bin/env julia
#=
make_solar_model_inputs.jl — the fitter's solar-model files from the SF-III (B23) package
(Herrera & Serenelli, Zenodo 10822316 v1.2, SolarModels_ICE2023_v1.2.zip):

    julia --project utils/make_solar_model_inputs.jl <unzipped dir>/SolarModels

writes inputs/SF3_GS98.jld2 (high-Z) and inputs/SF3_AGSS09met.jld2 (low-Z): radii (R/R_sun, (0, 0.5],
beyond which < 1e-8 of the production lies), n_e = 10^log_ne [N_A/cm³], prodFraction* = dP/d(R/R_sun),
plus the model's ⁸B and hep fluxes and the source files' SHA-256.
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
