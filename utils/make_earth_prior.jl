#=
make_earth_prior.jl — Earth prior of the fast mode from the parchita draws (utils/make_earth_draws.py).

Each draw's night P₁ₑ per (cos z, E) bin is computed by NightOsc on its own radial profile. The fast-mode
parameters (8 PREM layer scales, and the CMB shift) are fitted to each draw's deviation from PREM by
exposure-weighted least squares, linearised at PREM and over several Δm²₂₁ at once; their covariance
over the draws is the prior. Other Δm²₂₁ values check how well the fit transfers.

    julia -t auto --project utils/make_earth_prior.jl [draws.h5] [out.jld2]
=#
using HDF5, JLD2, LinearAlgebra, Statistics, Printf
const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "modelInputs.jl"))
include(joinpath(ROOT, "src", "oscillations", "osc.jl"))
include(joinpath(ROOT, "src", "oscillations", "osc_night.jl"))
using .NightOsc

const DRAWS_FILE = get(ARGS, 1, joinpath(ROOT, "inputs", "earth_draws_parchita.h5"))
const OUT_FILE   = get(ARGS, 2, joinpath(ROOT, "inputs", "earth_prior_parchita.jld2"))
const DM2_FIT    = [5e-5, 7.5e-5, 1e-4]
const DM2_CHECK  = [4e-5, 6e-5, 8.5e-5, 1.3e-4]
const TH12, TH13 = asin(sqrt(0.307)), asin(sqrt(0.022))
const CMB_STEP_KM = 2.0

# Grid: night cos z over the exposure's support, E_true 3–20 MeV
const EXPOSURE = load_exposure(joinpath(ROOT, "inputs", "solarExposure_v2.csv")).flux
const CZ_EDGES = collect(range(EXPOSURE.x[1], 0.0; length=41))
const E_EDGES  = collect(3e-3:0.5e-3:20e-3)
const N_CZ, N_E = length(CZ_EDGES) - 1, length(E_EDGES) - 1
const W_BIN = let w = [sum(EXPOSURE(c) for c in range(CZ_EDGES[i], CZ_EDGES[i+1]; length=9)) for i in 1:N_CZ]
    vec(repeat(w ./ sum(w), 1, N_E))
end

engine(earth) = BinnedP1e(CZ_EDGES, E_EDGES, c -> EXPOSURE(c), earth; phase_const=Osc.PHASE_CONST)
p1e(b, dm2s, norms=nothing) = reduce(vcat, [vec(b(dm2, TH12, TH13, norms)) for dm2 in dm2s])

# PREM with the CMB moved by s km; the outer core and lower mantle are stretched rigidly (as EarthModels)
function prem_cmb_shifted(s)
    r, rho, ye = prem_table()
    b = copy(PREM_BOUNDARIES_KM); b[2] += s
    lo, cmb, hi = PREM_BOUNDARIES_KM[1], PREM_BOUNDARIES_KM[2], PREM_BOUNDARIES_KM[3]
    r_new = map(r) do ri
        ri <= lo || ri >= hi ? ri : ri < cmb ? lo + (ri - lo) * (cmb + s - lo) / (cmb - lo) :
                                               hi - (hi - ri) * (hi - cmb - s) / (hi - cmb)
    end
    return NightOsc.EarthModel(r_new, rho .* ye .* MATTER_POTENTIAL_CONST, b)
end

function draw_model(rad, n_e, bounds)
    o = sortperm(rad)
    return NightOsc.EarthModel(rad[o], n_e[o] .* MATTER_POTENTIAL_CONST, sort(bounds))
end

# Response of the stacked P₁ₑ to the 8 layer scales (exact derivative) and the CMB shift (central difference)
function response(dm2s)
    b0 = engine(prem_cmb_shifted(0.0))
    J_layers = reduce(vcat, map(dm2s) do dm2
        x = vcat(dm2, TH12, TH13, ones(8))
        _, J = NightOsc._value_jacobian(b0, x, true, Val(3 + length(b0.used_layers)))
        J[:, 4:end]
    end)
    J_cmb = (p1e(engine(prem_cmb_shifted(CMB_STEP_KM)), dm2s, ones(8)) .-
             p1e(engine(prem_cmb_shifted(-CMB_STEP_KM)), dm2s, ones(8))) ./ (2CMB_STEP_KM)
    return p1e(b0, dm2s, ones(8)), hcat(J_layers, J_cmb)
end

# Exposure-weighted least squares of each column of Y on X; returns (θ, residuals)
function fit(X, Y, w)
    sw = sqrt.(w)
    θ = (sw .* X) \ (sw .* Y)
    return θ, Y .- X * θ
end

wrms(A, w) = sqrt(sum(w .* mean(A .^ 2; dims=2)) / sum(w))

function main()
    bnd, rad, ne = h5open(DRAWS_FILE) do f
        read(f, "boundaries_km"), read(f, "radius_km"), read(f, "n_e")
    end
    # h5py arrays arrive transposed: (points, draws)
    n_draws = size(rad, 2)
    @info "Earth prior from $n_draws draws of $DRAWS_FILE"
    dm2_all = vcat(DM2_FIT, DM2_CHECK)
    nb = N_CZ * N_E
    P0, J = response(dm2_all)
    D = zeros(length(P0), n_draws)
    Threads.@threads for d in 1:n_draws
        D[:, d] = p1e(engine(draw_model(rad[:, d], ne[:, d], bnd[:, d])), dm2_all) .- P0
    end
    rows(dm2s) = reduce(vcat, [(findfirst(==(m), dm2_all) - 1) * nb .+ (1:nb) for m in dm2s])
    w_fit = repeat(W_BIN, length(DM2_FIT))
    out = Dict{String,Any}()
    for (name, cols) in (("layers", 1:8), ("layers+cmb", 1:9))
        θ, _ = fit(J[rows(DM2_FIT), cols], D[rows(DM2_FIT), :], w_fit)
        @printf("%-11s", name)
        for dm2 in dm2_all
            r = rows([dm2])
            res = D[r, :] .- J[r, cols] * θ
            @printf("  %.1e: %4.1f%%", dm2, 100 * wrms(res, W_BIN) / wrms(D[r, :], W_BIN))
        end
        println("   (weighted rms residual / rms draw deviation; first $(length(DM2_FIT)) fitted)")
        out[name] = θ
    end
    θ = out["layers+cmb"]
    μ, Σ = vec(mean(θ; dims=2)), cov(θ; dims=2)
    names = vcat(["earth_norm_$i" for i in 1:8], "cmb_shift_km")
    println("\nparameter      mean        sd    (norms as deviations from 1)")
    for i in 1:9
        @printf("%-13s %+.2e  %.2e\n", names[i], μ[i], sqrt(Σ[i, i]))
    end
    C = Σ ./ sqrt.(diag(Σ) * diag(Σ)')
    println("\ncorrelations:"); for i in 1:9; println(join([@sprintf("%+.2f", C[i, j]) for j in 1:9], " ")); end

    # Per-bin spread: draws vs the Gaussian prior propagated linearly
    println("\nper-bin spread, fast mode / draws (exposure-weighted over bins):")
    for (name, cols) in (("layers", 1:8), ("layers+cmb", 1:9))
        Σc = cov(out[name]; dims=2)
        @printf("%-11s", name)
        for dm2 in dm2_all
            r = rows([dm2])
            s_fast = sqrt.(max.(diag(J[r, cols] * Σc * J[r, cols]'), 0))
            s_draw = vec(std(D[r, :]; dims=2))
            @printf("  %.1e: %.3f", dm2, sum(W_BIN .* s_fast) / sum(W_BIN .* s_draw))
        end
        println()
    end

    Σ8 = cov(out["layers"]; dims=2)
    jldsave(OUT_FILE; covariance=Σ8, var"true"=ones(8), mean=1 .+ vec(mean(out["layers"]; dims=2)),
            covariance_with_cmb=Σ, mean_with_cmb=μ, names_with_cmb=names,
            source="utils/make_earth_prior.jl from $(basename(DRAWS_FILE)), $n_draws draws",
            dm2_fit=DM2_FIT, cz_edges=CZ_EDGES, e_edges=E_EDGES)
    println("\nwrote $OUT_FILE (covariance: 8 layer scales; covariance_with_cmb: + CMB shift)")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
