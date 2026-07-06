#!/usr/bin/env julia
using JLD2, Plots, Printf

const SCRIPT_DIR = @__DIR__
const REF_FILE   = joinpath(SCRIPT_DIR, "osc_convergence_reference.jld2")

isfile(REF_FILE) || error("Reference not found: $REF_FILE\nRun checkOscGridConvergence.jl first.")

P_night, cosz_ref, E_ref = jldopen(REF_FILE, "r") do f
    f["P_night"], f["cosz_ref"], f["E_ref"]
end

E_MeV = E_ref .* 1e3
const CORE_MANTLE_COSZ = -sqrt(1.0 - (3480.0/6371.0)^2)
const INNER_CORE_COSZ  = -sqrt(1.0 - (1221.0/6371.0)^2)

hm = heatmap(E_MeV, cosz_ref, P_night;
             xlabel = "E_true (MeV)",
             ylabel = "cos(z)",
             title  = "P_night  (sin²θ₁₂ = 0.4, Δm²₂₁ = 15×10⁻⁵ eV²)  [1000×1000]",
             colorbar_title = "P(νe survives)",
             c = :viridis,
             size = (900, 600))
hline!(hm, [CORE_MANTLE_COSZ]; color=:red,    lw=1.5, ls=:dash,
       label=@sprintf("core-mantle  cos(z) = %.4f", CORE_MANTLE_COSZ))
hline!(hm, [INNER_CORE_COSZ];  color=:orange, lw=1.5, ls=:dash,
       label=@sprintf("inner-core   cos(z) = %.4f", INNER_CORE_COSZ))

out = joinpath(SCRIPT_DIR, "osc_reference_heatmap.pdf")
savefig(hm, out)
println("Saved → $out")
