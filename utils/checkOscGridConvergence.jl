#!/usr/bin/env julia
#=
checkOscGridConvergence.jl

Empirical convergence test for block-averaging P_night to the coarse analysis grid.

Both the 40-bin coarse grid and the 1000-bin reference grid are piecewise-uniform,
with bin edges placed exactly at the inner-core (≈−0.982) and core-mantle (≈−0.838)
cos(z) boundaries so no bin straddles a density discontinuity.

Computes a 1000×1000 reference at the worst-case parameter point
(sin²θ₁₂ = 0.4, Δm²₂₁ = 15×10⁻⁵ eV²), saves to utils/ for reuse, then measures
how the block-average to the 40-bin coarse grid converges as N_sub increases.

Usage:
  julia utils/checkOscGridConvergence.jl [path/to/config.yaml]
=#

using JLD2, Plots, Printf, Statistics, LinearAlgebra, YAML, DelimitedFiles

# ── Config ───────────────────────────────────────────────────────────────────

const SCRIPT_DIR = @__DIR__
const REPO_ROOT  = dirname(SCRIPT_DIR)
config_path = length(ARGS) > 0 ? ARGS[1] : joinpath(REPO_ROOT, "config.yaml")
config = YAML.load_file(config_path)

const solarModelFile = joinpath(REPO_ROOT, config["solar_model_file"])
const flux_file_path = joinpath(REPO_ROOT, config["flux_model_file"])
const earthModelFile = joinpath(REPO_ROOT, config["earth_model_file"])

# Analysis binning
const N_COARSE  = 40
const COSZ_MIN  = -1.0
const COSZ_MAX  =  0.0
const E_MIN_GEV = config["range_Etrue"][1] * 1e-3
const E_MAX_GEV = config["range_Etrue"][2] * 1e-3

# Worst-case parameters
const SIN2_TH12 = 0.4
const DM2_21    = 20e-5   # eV²
const SIN2_TH13 = 0.022

const N_REF      = 1000
const N_SUB_TEST = [1, 2, 3, 5, 8, 10, 15, 20, 33]
const REF_FILE   = joinpath(SCRIPT_DIR, "osc_convergence_reference.jld2")

# ── Physical boundaries & piecewise grid construction ────────────────────────

# Scan the PREM data for the two largest density discontinuities.
# Returns cos(z) values sorted ascending (most-negative / deepest first).
function find_density_boundaries(file_path)
    data     = readdlm(file_path)
    r_frac   = data[:, 1]          # fractional radius [0, 1]
    density  = data[:, 2]          # g/cm³
    jumps    = abs.(diff(density))
    order    = sortperm(jumps, rev=true)
    # Take the two rows with the largest density jumps (CMB ≈ 4.5 g/cm³, ICB ≈ 2.3 g/cm³).
    # The boundary radius is the mean of the two adjacent rows.
    top2_i   = sort(order[1:2])    # sort by index so we get them in radial order
    r_bdry   = [(r_frac[i] + r_frac[i+1]) / 2.0 for i in top2_i]
    cosz_vals = sort([-sqrt(max(0.0, 1.0 - r^2)) for r in r_bdry])  # ascending (deepest first)
    return cosz_vals[1], cosz_vals[2]   # (inner-core, core-mantle)
end

INNER_CORE_COSZ, CORE_MANTLE_COSZ = find_density_boundaries(earthModelFile)

const SEG_BREAKS  = Float64[COSZ_MIN, INNER_CORE_COSZ, CORE_MANTLE_COSZ, COSZ_MAX]
const SEG_LENGTHS = diff(SEG_BREAKS)

# Largest-remainder allocation of n_total bins across segments.
function alloc_bins(n_total, seg_lengths)
    fracs  = seg_lengths ./ sum(seg_lengths)
    counts = max.(1, floor.(Int, fracs .* n_total))
    rem    = fracs .* n_total .- counts
    for _ in 1:(n_total - sum(counts))
        idx = argmax(rem);  counts[idx] += 1;  rem[idx] -= 1.0
    end
    counts
end

const N_COARSE_SEGS = alloc_bins(N_COARSE, SEG_LENGTHS)
const N_REF_SEGS    = alloc_bins(N_REF,    SEG_LENGTHS)

# Piecewise-uniform edges (boundaries fall exactly on edges).
function piecewise_edges(seg_breaks, n_per_seg)
    edges = [seg_breaks[1]]
    for i in eachindex(n_per_seg)
        append!(edges, collect(range(seg_breaks[i], seg_breaks[i+1]; length=n_per_seg[i]+1))[2:end])
    end
    edges
end

# Piecewise-uniform midpoints (representative sample positions).
function piecewise_midpoints(seg_breaks, n_per_seg)
    pts = Float64[]
    for i in eachindex(n_per_seg)
        lo, hi = seg_breaks[i], seg_breaks[i+1]
        step = (hi - lo) / n_per_seg[i]
        append!(pts, [lo + (k - 0.5) * step for k in 1:n_per_seg[i]])
    end
    pts
end

const coarse_edges = piecewise_edges(SEG_BREAKS, N_COARSE_SEGS)

# ── Load physics ──────────────────────────────────────────────────────────────

const cosz_bins = (bin_number = N_COARSE, min = COSZ_MIN, max = COSZ_MAX)
earth_normalisation_true = nothing

include(joinpath(REPO_ROOT, "src", "core.jl"))
include(joinpath(REPO_ROOT, "src", "earthProfile.jl"))
include(joinpath(REPO_ROOT, "src", "oscillations", "makePaths.jl"))
include(joinpath(REPO_ROOT, "src", "solarModel.jl"))
include(joinpath(REPO_ROOT, "src", "oscillations", "osc.jl"))
import .Osc: oscPars, osc_prob_both_fast
import .Osc.NumOsc.Fast: osc_prob_earth

# ── Compute or load reference ─────────────────────────────────────────────────

function compute_reference()
    println("Computing $(N_REF)×$(N_REF) reference (sin²θ₁₂=$SIN2_TH12, Δm²₂₁=$DM2_21 eV²)")
    cosz_ref = piecewise_midpoints(SEG_BREAKS, N_REF_SEGS)
    E_ref    = collect(range(E_MIN_GEV, E_MAX_GEV, length=N_REF))
    mixPars  = oscPars(DM2_21, asin(sqrt(SIN2_TH12)), asin(sqrt(SIN2_TH13)))

    print("  Earth paths ($N_REF)... "); flush(stdout)
    t = @elapsed paths = [make_potential_for_integrand(z, earth) for z in cosz_ref]
    @printf("%.1f s\n", t)

    lookup = get_avg_densities(paths)

    print("  osc_prob_earth... "); flush(stdout)
    t = @elapsed p1e = osc_prob_earth(E_ref, mixPars, lookup, paths)
    @printf("%.1f s\n", t)

    print("  P_night... "); flush(stdout)
    t = @elapsed _, P_night = osc_prob_both_fast(E_ref, p1e, mixPars, solarModel; process="8B")
    @printf("%.1f s\n", t)

    jldsave(REF_FILE; P_night, cosz_ref, E_ref)
    println("  Saved → $REF_FILE")
    return P_night, cosz_ref, E_ref
end

P_ref, cosz_ref, E_ref = if isfile(REF_FILE)
    println("Loading reference from $REF_FILE")
    jldopen(REF_FILE, "r") do f; f["P_night"], f["cosz_ref"], f["E_ref"]; end
else
    compute_reference()
end

# ── Flux weighting ────────────────────────────────────────────────────────────
# `energies` and `flux8B` are loaded by solarModel.jl (in GeV, arbitrary units).
# Linear interpolation onto the reference E grid.
function interp_flux(E_pts, E_grid, F_grid)
    n = length(E_grid)
    [begin
        j = clamp(searchsortedfirst(E_grid, e) - 1, 1, n-1)
        t = (e - E_grid[j]) / (E_grid[j+1] - E_grid[j])
        F_grid[j] * (1-t) + F_grid[j+1] * t
    end for e in E_pts]
end

flux_ref = interp_flux(E_ref, energies, flux8B)
FP_ref   = P_ref .* flux_ref'   # (N_REF cosz) × (N_REF E), flux-weighted P_night

# ── Binning helpers ───────────────────────────────────────────────────────────

# Map each coarse cosz bin → the row range in the reference grid.
# Within each segment, reference rows are divided uniformly among coarse bins.
coarse_row_ranges = let
    row = 0
    ranges = Vector{UnitRange{Int}}(undef, N_COARSE)
    coarse_bin = 0
    for seg in eachindex(N_COARSE_SEGS)
        n_c = N_COARSE_SEGS[seg]
        n_r = N_REF_SEGS[seg]
        for k in 1:n_c
            coarse_bin += 1
            r_lo = row + round(Int, (k-1) * n_r / n_c) + 1
            r_hi = row + round(Int,  k    * n_r / n_c)
            ranges[coarse_bin] = r_lo:r_hi
        end
        row += n_r
    end
    ranges
end

const FPC_E = N_REF / N_COARSE   # fine E-points per coarse E-bin (uniform)

# "Truth": average ALL reference fine points into each coarse bin.
function truth_coarse(P)
    [mean(P[coarse_row_ranges[i],
            round(Int,(j-1)*FPC_E)+1 : round(Int,j*FPC_E)])
     for i in 1:N_COARSE, j in 1:N_COARSE]
end

# "N_sub approximation": N_sub uniformly-spaced sample centres per coarse bin,
# snapped to nearest reference point, averaged.
function nsub_approx(P, n_sub)
    result = zeros(N_COARSE, N_COARSE)
    for i in 1:N_COARSE
        lo, hi = coarse_edges[i], coarse_edges[i+1]
        w = hi - lo
        sample_cz = [lo + (k - 0.5) * w / n_sub for k in 1:n_sub]
        rows = [argmin(abs.(cosz_ref .- c)) for c in sample_cz]
        for j in 1:N_COARSE
            cols = round(Int,(j-1)*FPC_E)+1 : round(Int,j*FPC_E)
            result[i,j] = mean(P[rows, cols])
        end
    end
    result
end

# Asymmetric approximation: n_sub_mantle above the core-mantle boundary,
# n_sub_core below it.  The CM boundary is a bin edge, so the split is clean.
function nsub_approx_asym(P, n_sub_mantle, n_sub_core)
    result = zeros(N_COARSE, N_COARSE)
    for i in 1:N_COARSE
        lo, hi = coarse_edges[i], coarse_edges[i+1]
        w  = hi - lo
        ns = hi <= CORE_MANTLE_COSZ ? n_sub_core : n_sub_mantle
        sample_cz = [lo + (k - 0.5) * w / ns for k in 1:ns]
        rows = [argmin(abs.(cosz_ref .- c)) for c in sample_cz]
        for j in 1:N_COARSE
            cols = round(Int,(j-1)*FPC_E)+1 : round(Int,j*FPC_E)
            result[i,j] = mean(P[rows, cols])
        end
    end
    result
end

# ── Report grid layout ────────────────────────────────────────────────────────

@printf("\nBoundaries from earth model:\n")
@printf("  Inner-core  cos(z) = %.6f\n", INNER_CORE_COSZ)
@printf("  Core-mantle cos(z) = %.6f\n", CORE_MANTLE_COSZ)
@printf("\nPiecewise coarse grid (%d bins, boundaries on edges):\n", N_COARSE)
@printf("  Seg 1 [inner core]:   %2d bin(s)  [%.4f, %.4f)  bin width %.4f\n",
        N_COARSE_SEGS[1], COSZ_MIN, INNER_CORE_COSZ, SEG_LENGTHS[1]/N_COARSE_SEGS[1])
@printf("  Seg 2 [outer core]:   %2d bin(s)  [%.4f, %.4f)  bin width %.4f\n",
        N_COARSE_SEGS[2], INNER_CORE_COSZ, CORE_MANTLE_COSZ, SEG_LENGTHS[2]/N_COARSE_SEGS[2])
@printf("  Seg 3 [mantle/crust]: %2d bin(s)  [%.4f,  %.4f)  bin width %.4f\n",
        N_COARSE_SEGS[3], CORE_MANTLE_COSZ, COSZ_MAX, SEG_LENGTHS[3]/N_COARSE_SEGS[3])

# Adjacent bins for boundary diagnostics
const IC_BDY_BINS = (N_COARSE_SEGS[1],   N_COARSE_SEGS[1]+1)
const CM_BDY_BINS = (sum(N_COARSE_SEGS[1:2]), sum(N_COARSE_SEGS[1:2])+1)

# ── Convergence table ─────────────────────────────────────────────────────────
# Errors are measured in the flux-weighted night probability φ(E)·P_night(cosz,E),
# integrated over E for each coarse cosz bin — this is the quantity that enters
# the event rate, so it is what actually needs to converge.

println("\nComputing truth (full average of $(N_REF)×$(N_REF), flux-weighted)...")
FP_truth = truth_coarse(FP_ref)                 # (N_COARSE cosz) × (N_COARSE E)
R_truth  = vec(sum(FP_truth, dims=2))           # E-integrated rate per cosz bin

# Rate error for a given approximation matrix (cosz × E, coarse)
function rate_err(FP_approx)
    R_approx = vec(sum(FP_approx, dims=2))
    abs.(R_approx .- R_truth) ./ max.(R_truth, 1e-30)   # (N_COARSE,)
end

println("\n  N_sub │ uniform max   │ asym(N,2) max │ asym(N,1) max │ near IC bdry  │ near CM bdry  │ rest max")
println("────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼──────────────")

max_errs   = Float64[]
asym2_errs = Float64[]
asym1_errs = Float64[]
ic_errs    = Float64[]
cm_errs    = Float64[]
rest_errs  = Float64[]

for n_sub in N_SUB_TEST
    err       = rate_err(nsub_approx(FP_ref, n_sub))
    err_asym2 = rate_err(nsub_approx_asym(FP_ref, n_sub, 2))
    err_asym1 = rate_err(nsub_approx_asym(FP_ref, n_sub, 1))
    push!(max_errs,   maximum(err))
    push!(asym2_errs, maximum(err_asym2))
    push!(asym1_errs, maximum(err_asym1))
    push!(ic_errs,    maximum(err[[IC_BDY_BINS...]]))
    push!(cm_errs,    maximum(err[[CM_BDY_BINS...]]))
    mask = trues(N_COARSE); mask[[IC_BDY_BINS..., CM_BDY_BINS...]] .= false
    push!(rest_errs,  maximum(err[mask]))
    @printf("    %3d │   %7.4f%%    │   %7.4f%%    │   %7.4f%%    │   %7.4f%%    │   %7.4f%%    │  %7.4f%%\n",
            n_sub, max_errs[end]*100, asym2_errs[end]*100, asym1_errs[end]*100,
            ic_errs[end]*100, cm_errs[end]*100, rest_errs[end]*100)
end

# ── 50/50 grid test ──────────────────────────────────────────────────────────
# Allocate N_COARSE/2 bins to each side of the CM boundary.
# Within the core half, maintain the IC boundary on an edge.

n_core_5050   = N_COARSE ÷ 2
n_mantle_5050 = N_COARSE - n_core_5050
core_segs_5050 = alloc_bins(n_core_5050, SEG_LENGTHS[1:2])
N_SEGS_5050    = [core_segs_5050; n_mantle_5050]

coarse_edges_5050 = piecewise_edges(SEG_BREAKS, N_SEGS_5050)

# For each 50/50 coarse bin find the reference rows that fall inside it.
row_ranges_5050 = let
    n_c = length(coarse_edges_5050) - 1
    ranges = Vector{UnitRange{Int}}(undef, n_c)
    for i in 1:n_c
        lo, hi = coarse_edges_5050[i], coarse_edges_5050[i+1]
        idx_lo = searchsortedfirst(cosz_ref, lo - 1e-12)
        idx_hi = i < n_c ? searchsortedlast(cosz_ref, hi - 1e-12) :
                           searchsortedlast(cosz_ref, hi)
        ranges[i] = idx_lo:idx_hi
    end
    ranges
end

function truth_coarse_5050(P)
    n_c = length(row_ranges_5050)
    [mean(P[row_ranges_5050[i], round(Int,(j-1)*FPC_E)+1 : round(Int,j*FPC_E)])
     for i in 1:n_c, j in 1:N_COARSE]
end

function nsub_approx_5050(P, n_sub)
    n_c = length(row_ranges_5050)
    result = zeros(n_c, N_COARSE)
    for i in 1:n_c
        lo, hi = coarse_edges_5050[i], coarse_edges_5050[i+1]
        w = hi - lo
        sample_cz = [lo + (k - 0.5) * w / n_sub for k in 1:n_sub]
        rows = [argmin(abs.(cosz_ref .- c)) for c in sample_cz]
        for j in 1:N_COARSE
            cols = round(Int,(j-1)*FPC_E)+1 : round(Int,j*FPC_E)
            result[i,j] = mean(P[rows, cols])
        end
    end
    result
end

FP_truth_5050 = truth_coarse_5050(FP_ref)
R_truth_5050  = vec(sum(FP_truth_5050, dims=2))

function rate_err_5050(FP_approx)
    abs.(vec(sum(FP_approx, dims=2)) .- R_truth_5050) ./ max.(R_truth_5050, 1e-30)
end

IC_BDY_BINS_5050 = (N_SEGS_5050[1],   N_SEGS_5050[1]+1)
CM_BDY_BINS_5050 = (sum(N_SEGS_5050[1:2]), sum(N_SEGS_5050[1:2])+1)

@printf("\n50/50 coarse grid (%d core + %d mantle = %d bins):\n",
        n_core_5050, n_mantle_5050, N_COARSE)
@printf("  Core  [%.4f, %.4f): %d bins, width %.4f each\n",
        COSZ_MIN, CORE_MANTLE_COSZ,
        n_core_5050, (CORE_MANTLE_COSZ - COSZ_MIN) / n_core_5050)
@printf("  Mantle[%.4f,  %.4f): %d bins, width %.4f each\n",
        CORE_MANTLE_COSZ, COSZ_MAX,
        n_mantle_5050, (COSZ_MAX - CORE_MANTLE_COSZ) / n_mantle_5050)

println("\n  N_sub │ uniform max   │ asym(N,2) max │ asym(N,1) max │ near IC bdry  │ near CM bdry  │ rest max")
println("────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼──────────────")

for n_sub in N_SUB_TEST
    err       = rate_err_5050(nsub_approx_5050(FP_ref, n_sub))
    err_asym2 = rate_err_5050(nsub_approx_5050(FP_ref, n_sub))   # no asymmetry needed on 5050
    err_asym1 = rate_err_5050(nsub_approx_5050(FP_ref, n_sub))   # same
    ic_e = maximum(err[[IC_BDY_BINS_5050...]])
    cm_e = maximum(err[[CM_BDY_BINS_5050...]])
    mask = trues(N_COARSE); mask[[IC_BDY_BINS_5050..., CM_BDY_BINS_5050...]] .= false
    @printf("    %3d │   %7.4f%%    │      —        │      —        │   %7.4f%%    │   %7.4f%%    │  %7.4f%%\n",
            n_sub, maximum(err)*100, ic_e*100, cm_e*100, maximum(err[mask])*100)
end

# ── Plots ─────────────────────────────────────────────────────────────────────

# 1. Convergence curves (flux-weighted E-integrated rate)
pl = plot(N_SUB_TEST, max_errs  .* 100, label="Uniform N_sub (max)",       marker=:circle,    lw=2,
          xscale=:log10, yscale=:log10,
          xlabel="Sub-bins per coarse bin (mantle)",
          ylabel="Max fractional rate error (%)",
          title="φ(E)·P_night rate convergence  (sin²θ₁₂=$SIN2_TH12, Δm²₂₁=$DM2_21 eV²)")
plot!(pl, N_SUB_TEST, asym2_errs .* 100, label="Asym: N_sub mantle, 2 core", marker=:star5,   lw=2, ls=:dashdot)
plot!(pl, N_SUB_TEST, ic_errs   .* 100, label="Near inner-core bdry",       marker=:utriangle, lw=1, ls=:dash)
plot!(pl, N_SUB_TEST, cm_errs   .* 100, label="Near core-mantle bdry",      marker=:square,    lw=1, ls=:dash)
plot!(pl, N_SUB_TEST, rest_errs .* 100, label="Remaining bins (max)",       marker=:diamond,   lw=1, ls=:dot)
hline!(pl, [0.1], color=:gray, lw=1, ls=:dot, label="0.1% target")

# 2. Flux-weighted error heatmap at current code settings (N_sub = 3)
n_now = 2
FP_now = nsub_approx(FP_ref, n_now)
# Normalise each (cosz,E) cell relative to its cosz bin's total E-integrated rate,
# so that high-E cells with tiny flux don't inflate the colour scale.
R_truth_mat = repeat(R_truth, 1, N_COARSE)   # (N_COARSE, N_COARSE) broadcast-ready
rel_now = abs.(FP_now .- FP_truth) ./ max.(R_truth_mat, 1e-30) .* 100

E_c  = collect(range(E_MIN_GEV * 1e3, E_MAX_GEV * 1e3, length=N_COARSE))
cz_c = [(coarse_edges[i]+coarse_edges[i+1])/2 for i in 1:N_COARSE]
hm = heatmap(E_c, cz_c, rel_now,
             xlabel="E_true (MeV)", ylabel="cos(z)",
             title="φ(E)·P_night error (% of total rate) at N_sub = $n_now  vs truth",
             colorbar_title="% of total rate", c=:heat)
hline!(hm, [CORE_MANTLE_COSZ], color=:cyan,   lw=1.5, ls=:dash, label="core-mantle boundary")
hline!(hm, [INNER_CORE_COSZ],  color=:yellow, lw=1.5, ls=:dash, label="inner-core boundary")

savefig(pl, joinpath(SCRIPT_DIR, "osc_convergence.pdf"))
savefig(hm, joinpath(SCRIPT_DIR, "osc_error_heatmap_N$(n_now).pdf"))
println("\nSaved: utils/osc_convergence.pdf, utils/osc_error_heatmap_N$(n_now).pdf")

# ── Parameter sensitivity ─────────────────────────────────────────────────────
# Translate the rate error into equivalent parameter bias at the nominal best-fit.
# Earth paths depend only on cos(z), so we compute them once for the coarse
# analysis grid and reuse across all parameter evaluations.

const NOM_SIN2_TH12 = 0.307
const NOM_DM2_21    = 7.53e-5
const NOM_SIN2_TH13 = 0.022

print("\nComputing Earth paths for sensitivity grid ($(N_COARSE) cosz)... "); flush(stdout)
cosz_sens = [(coarse_edges[i]+coarse_edges[i+1])/2 for i in 1:N_COARSE]
E_sens    = collect(range(E_MIN_GEV, E_MAX_GEV, length=N_COARSE))
flux_sens = interp_flux(E_sens, energies, flux8B)
t = @elapsed begin
    paths_sens  = [make_potential_for_integrand(z, earth) for z in cosz_sens]
    lookup_sens = get_avg_densities(paths_sens)
end
@printf("%.1f s\n", t)

# Total flux-weighted night rate (scalar) at a given parameter point.
function night_rate_total(s12, dm21)
    mp  = oscPars(dm21, asin(sqrt(clamp(s12, 0.0, 1.0))), asin(sqrt(NOM_SIN2_TH13)))
    p1e = osc_prob_earth(E_sens, mp, lookup_sens, paths_sens)
    _, P = osc_prob_both_fast(E_sens, p1e, mp, solarModel; process="8B")
    sum(P .* flux_sens')   # scalar: Σ_{cosz,E} φ(E)·P_night
end

R_nom = night_rate_total(NOM_SIN2_TH12, NOM_DM2_21)

δs = 0.005
δm = 0.2e-5
dR_ds12 = (night_rate_total(NOM_SIN2_TH12 + δs, NOM_DM2_21) -
           night_rate_total(NOM_SIN2_TH12 - δs, NOM_DM2_21)) / (2δs)
dR_dm21 = (night_rate_total(NOM_SIN2_TH12, NOM_DM2_21 + δm) -
           night_rate_total(NOM_SIN2_TH12, NOM_DM2_21 - δm)) / (2δm)

# Parameter shift Δθ such that |ΔR/R| = target
target = 0.001   # 0.1%
bias_s12 = abs(target * R_nom / dR_ds12)
bias_dm2  = abs(target * R_nom / dR_dm21)

println("\nParameter bias for 0.1% total-rate error (at nominal best-fit):")
@printf("  Δ(sin²θ₁₂) = %.5f\n", bias_s12)
@printf("  Δ(Δm²₂₁)   = %.3e eV²\n", bias_dm2)
@printf("  (current PDG 1σ: sin²θ₁₂ ≈ 0.013,  Δm²₂₁ ≈ 0.18×10⁻⁵ eV²)\n")

# ── Error maps: parameter shift vs sub2 grid error ────────────────────────────
# Compute flux-weighted P_night on the coarse grid at a given parameter point.
function fp_coarse_grid(s12, dm21)
    mp  = oscPars(dm21, asin(sqrt(clamp(s12, 0.0, 1.0))), asin(sqrt(NOM_SIN2_TH13)))
    p1e = osc_prob_earth(E_sens, mp, lookup_sens, paths_sens)
    _, P = osc_prob_both_fast(E_sens, p1e, mp, solarModel; process="8B")
    P .* flux_sens'   # (N_COARSE cosz) × (N_COARSE E)
end

FP_nom    = fp_coarse_grid(NOM_SIN2_TH12, NOM_DM2_21)
R_nom_vec = vec(sum(FP_nom, dims=2))          # total rate per cosz bin at nominal
R_nom_mat = repeat(R_nom_vec, 1, N_COARSE)   # broadcast shape

# Error from a 1σ shift in sin²θ₁₂
FP_s12p = fp_coarse_grid(NOM_SIN2_TH12 + 0.013, NOM_DM2_21)
err_s12  = abs.(FP_s12p .- FP_nom) ./ max.(R_nom_mat, 1e-30) .* 100

# Error from a 1σ shift in Δm²₂₁
FP_dm2p = fp_coarse_grid(NOM_SIN2_TH12, NOM_DM2_21 + 0.18e-5)
err_dm2  = abs.(FP_dm2p .- FP_nom) ./ max.(R_nom_mat, 1e-30) .* 100

# sub2 grid approximation error at worst-case parameters
FP_sub2  = nsub_approx(FP_ref, 2)
err_sub2 = abs.(FP_sub2 .- FP_truth) ./ max.(R_truth_mat, 1e-30) .* 100

@printf("\nPeak error maps (percent of total cosz-bin rate):\n")
@printf("  1σ sin²θ₁₂ shift:  max = %.4f%%\n", maximum(err_s12))
@printf("  1σ Δm²₂₁ shift:    max = %.4f%%\n", maximum(err_dm2))
@printf("  sub2 grid error:   max = %.4f%%\n", maximum(err_sub2))

# Shared colour axis: the 1σ parameter shifts set the physical scale
clim_top = max(maximum(err_s12), maximum(err_dm2))

hm_s12 = heatmap(E_c, cz_c, err_s12;
    xlabel="E_true (MeV)", ylabel="cos(z)",
    title="1σ sin²θ₁₂ shift (+0.013)",
    colorbar_title="% of rate", c=:viridis,
    clims=(0, clim_top))
hline!(hm_s12, [CORE_MANTLE_COSZ]; color=:red,    lw=1.5, ls=:dash, label="")
hline!(hm_s12, [INNER_CORE_COSZ];  color=:orange, lw=1.5, ls=:dash, label="")

hm_dm2 = heatmap(E_c, cz_c, err_dm2;
    xlabel="E_true (MeV)", ylabel="cos(z)",
    title="1σ Δm²₂₁ shift (+0.18×10⁻⁵ eV²)",
    colorbar_title="% of rate", c=:viridis,
    clims=(0, clim_top))
hline!(hm_dm2, [CORE_MANTLE_COSZ]; color=:red,    lw=1.5, ls=:dash, label="")
hline!(hm_dm2, [INNER_CORE_COSZ];  color=:orange, lw=1.5, ls=:dash, label="")

hm_sub2 = heatmap(E_c, cz_c, err_sub2;
    xlabel="E_true (MeV)", ylabel="cos(z)",
    title="sub2 grid error (N_sub=2, worst-case params)",
    colorbar_title="% of rate", c=:viridis,
    clims=(0, clim_top))
hline!(hm_sub2, [CORE_MANTLE_COSZ]; color=:red,    lw=1.5, ls=:dash, label="")
hline!(hm_sub2, [INNER_CORE_COSZ];  color=:orange, lw=1.5, ls=:dash, label="")

comparison = plot(hm_s12, hm_dm2, hm_sub2; layout=(1,3), size=(1500, 500))

savefig(comparison, joinpath(SCRIPT_DIR, "osc_error_comparison.pdf"))
println("Saved: utils/osc_error_comparison.pdf")

# ── Parameter bias from sub-bin approximation ────────────────────────────────
# For each N_sub, compute the per-bin rate offset at nominal parameters
# and project it onto the parameter gradient to get the first-order best-fit
# bias.  This is the correct approach: the per-bin errors have mostly alternating
# signs that cancel in the projection, so the actual best-fit shift is much
# smaller than the raw per-bin max error.

# Build uniformly-spaced sub-bin cosz centres and their Earth paths.
function build_subbin_infra(n_sub)
    n_sub == 1 && return (cosz_sens, paths_sens, lookup_sens)
    cz = Float64[]
    for i in 1:N_COARSE
        lo, hi = coarse_edges[i], coarse_edges[i+1]
        w = hi - lo
        append!(cz, [lo + (k-0.5)*w/n_sub for k in 1:n_sub])
    end
    ps  = [make_potential_for_integrand(z, earth) for z in cz]
    lk  = get_avg_densities(ps)
    return (cz, ps, lk)
end

print("\nBuilding sub-bin paths for bias estimate (N_sub=2,3,8,20)... "); flush(stdout)
t = @elapsed begin
    (_, ps2,  lk2)  = build_subbin_infra(2)
    (_, ps3,  lk3)  = build_subbin_infra(3)
    (_, ps8,  lk8)  = build_subbin_infra(8)
    (_, ps20, lk20) = build_subbin_infra(20)
end
@printf("%.1f s\n", t)

infra_map = Dict(1  => (paths_sens, lookup_sens, 1),
                 2  => (ps2,  lk2,  2),
                 3  => (ps3,  lk3,  3),
                 8  => (ps8,  lk8,  8),
                 20 => (ps20, lk20, 20))

# Flux-weighted night rate per coarse cosz bin.
# P_night shape: (n_sub*N_COARSE, N_E); reshape → (n_sub, N_COARSE, N_E).
function night_rate_vec(s12, dm21, ps, lk, n_sub)
    mp  = oscPars(dm21, asin(sqrt(clamp(s12, 0.0, 1.0))), asin(sqrt(NOM_SIN2_TH13)))
    p1e = osc_prob_earth(E_sens, mp, lk, ps)
    _, P = osc_prob_both_fast(E_sens, p1e, mp, solarModel; process="8B")
    P_c = dropdims(mean(reshape(P, n_sub, N_COARSE, N_COARSE), dims=1), dims=1)
    vec(sum(P_c .* flux_sens', dims=2))
end

# Converged reference = N_sub=20.
R_ref20 = night_rate_vec(NOM_SIN2_TH12, NOM_DM2_21, ps20, lk20, 20)

# Per-parameter gradient vectors at nominal (using N_sub=20 paths).
gS = (night_rate_vec(NOM_SIN2_TH12 + δs, NOM_DM2_21,     ps20, lk20, 20) .-
      night_rate_vec(NOM_SIN2_TH12 - δs, NOM_DM2_21,     ps20, lk20, 20)) ./ (2δs)
gM = (night_rate_vec(NOM_SIN2_TH12,     NOM_DM2_21 + δm, ps20, lk20, 20) .-
      night_rate_vec(NOM_SIN2_TH12,     NOM_DM2_21 - δm, ps20, lk20, 20)) ./ (2δm)

println("\nSub-bin approximation bias at nominal best-fit:")
println("  gradient projection of per-bin δR onto ∂R/∂θ — the correct best-fit shift estimate")
println()
@printf("  N_sub │  mean(δR/R)  │  rms(δR/R)  │  max|δR/R|  │  Δsin²θ₁₂     │  Δ(Δm²₂₁)     │  frac 1σ\n")
@printf("  ──────┼─────────────┼─────────────┼─────────────┼────────────────┼────────────────┼─────────\n")
for n_sub in [1, 2, 3, 8]
    ps_n, lk_n, ns = infra_map[n_sub]
    R_n = night_rate_vec(NOM_SIN2_TH12, NOM_DM2_21, ps_n, lk_n, ns)
    δR  = R_n .- R_ref20
    bs  = dot(gS, δR) / dot(gS, gS)
    bm  = dot(gM, δR) / dot(gM, gM)
    @printf("  %4d  │  %+7.4f%%   │  %7.4f%%   │  %7.4f%%   │  %+10.5f    │  %+10.3e   │  %5.1f%%\n",
            n_sub,
            mean(δR ./ R_ref20) * 100,
            sqrt(mean((δR ./ R_ref20).^2)) * 100,
            maximum(abs.(δR ./ R_ref20)) * 100,
            bs, bm, abs(bs) / 0.013 * 100)
end
println()
@printf("  PDG 1σ: Δsin²θ₁₂ = 0.013,  Δ(Δm²₂₁) = 1.8×10⁻⁶ eV²\n")
println("  (Note: max|δR/R| has alternating signs across bins; it does not drive the best-fit bias)")

# ── Bias vs N_sub plot ────────────────────────────────────────────────────────
n_sub_curve = [1, 2, 3, 5, 8, 10, 15, 20]
biases_s12 = Float64[]
biases_dm2 = Float64[]

print("\nComputing bias curve vs N_sub... "); flush(stdout)
infra_extra = Dict{Int, Tuple}()
for n_sub in n_sub_curve
    if haskey(infra_map, n_sub)
        ps_n, lk_n, ns = infra_map[n_sub]
    else
        (_, ps_n, lk_n) = build_subbin_infra(n_sub)
        ns = n_sub
    end
    R_n = night_rate_vec(NOM_SIN2_TH12, NOM_DM2_21, ps_n, lk_n, ns)
    δR  = R_n .- R_ref20
    push!(biases_s12, dot(gS, δR) / dot(gS, gS))
    push!(biases_dm2, dot(gM, δR) / dot(gM, gM))
end
println("done")

bias_plt = plot(n_sub_curve, abs.(biases_s12),
    marker=:circle, lw=2, xscale=:log10, yscale=:log10,
    xlabel="N_sub (sub-bins per coarse cosz bin)",
    ylabel="|best-fit bias|",
    title="Best-fit bias from sub-bin approximation  (nominal params, gradient projection)",
    label="|Δsin²θ₁₂|",
    legend=:topright)
plot!(bias_plt, n_sub_curve, abs.(biases_dm2) ./ 1.8e-6 .* 0.013,
    marker=:square, lw=2, ls=:dash,
    label="|Δ(Δm²₂₁)| rescaled to sin²θ₁₂ units")
hline!(bias_plt, [0.013],      color=:red,       lw=1, ls=:dot, label="PDG 1σ (sin²θ₁₂ = 0.013)")
hline!(bias_plt, [0.013/10],   color=:orange,    lw=1, ls=:dot, label="10% of 1σ")
hline!(bias_plt, [0.013/100],  color=:lightgray, lw=1, ls=:dot, label="1% of 1σ")

savefig(bias_plt, joinpath(SCRIPT_DIR, "osc_contour_bias.pdf"))
println("Saved: utils/osc_contour_bias.pdf  (best-fit bias vs N_sub)")
