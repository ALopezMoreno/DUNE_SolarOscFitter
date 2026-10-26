#=
make_earth_prior.jl — Earth prior of the fast mode (layer scales) from parchita draws (SPEC 2.9).

Each draw's night P₁ₑ per (cos z, E) bin is computed by NightOsc on its own radial profile; the
covariance of the layer scales is built from them by `earth_prior_covariance` (src/earthPrior.jl),
fitted on several Δm²₂₁ at once and checked on others. Bins are weighted by expected CC events
(exposure × ⁸B flux × σ_CC). Layers no path crosses get no parameter. A bootstrap over the draws gives
the statistical uncertainty of each width. The prior is centred at PREM; spreads are about PREM, and
the covariance is scaled to cover the draws' per-bin spread with a small margin (COVERAGE) at every
Δm²₂₁ evaluated.

    python3 utils/make_earth_draws.py --draws 1000 --out inputs/earth_draws_parchita.h5
    julia -t auto --project utils/make_earth_prior.jl [draws.h5] [out.jld2]
=#
using HDF5, JLD2, LinearAlgebra, Statistics, Printf, Random, SHA, DelimitedFiles, Dates
const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "modelInputs.jl"))
include(joinpath(ROOT, "src", "oscillations", "osc.jl"))
include(joinpath(ROOT, "src", "oscillations", "osc_night.jl"))
include(joinpath(ROOT, "src", "earthPrior.jl"))
using .NightOsc

const DRAWS_FILE = get(ARGS, 1, joinpath(ROOT, "inputs", "earth_draws_parchita.h5"))
const OUT_FILE   = get(ARGS, 2, joinpath(ROOT, "inputs", "earth_prior_parchita.jld2"))
const DM2_FIT    = [5e-5, 7.5e-5, 1e-4]
const DM2_CHECK  = [4e-5, 6e-5, 8.5e-5, 1.3e-4]
const TH12, TH13 = asin(sqrt(0.307)), asin(sqrt(0.022))
const N_BOOT     = 200
const COVERAGE   = 1.1     # prior per-bin variance / draws', at the worst Δm²₂₁: slightly over
const FLUX_FILE  = joinpath(ROOT, "inputs", "fluxes.jld2")
const XSEC_FILE  = joinpath(ROOT, "inputs", "CC_nue_40Ar_total_marley_v2.0.0.csv")
const EXPOSURE_FILE = joinpath(ROOT, "inputs", "solarExposure_v2.csv")

# Grid: night cos z over the exposure's support, E_true 3–20 MeV
const EXPOSURE = load_exposure(EXPOSURE_FILE).flux
const CZ_EDGES = collect(range(EXPOSURE.x[1], 0.0; length=41))
const E_EDGES  = collect(3e-3:0.5e-3:20e-3)
const N_CZ, N_E = length(CZ_EDGES) - 1, length(E_EDGES) - 1

engine(earth) = BinnedP1e(CZ_EDGES, E_EDGES, c -> EXPOSURE(c), earth; phase_const=Osc.PHASE_CONST)
p1e(b, dm2s, norms=nothing) = reduce(vcat, [vec(b(dm2, TH12, TH13, norms)) for dm2 in dm2s])

function draw_model(rad, n_e, bounds)
    o = sortperm(rad)
    return NightOsc.EarthModel(rad[o], n_e[o] .* MATTER_POTENTIAL_CONST, sort(bounds))
end

# Expected CC events per (cos z, E) bin, normalised: exposure × ∫ ⁸B flux × σ over the E bin
function event_weights()
    interp(x, X, Y) = (i = clamp(searchsortedlast(X, x), 1, length(X) - 1);
                       Y[i] + (x - X[i]) * (Y[i+1] - Y[i]) / (X[i+1] - X[i]))
    fl = JLD2.load(FLUX_FILE)
    xs = readdlm(XSEC_FILE, ','; comments=true, skipstart=1)
    w_cz = [sum(EXPOSURE(c) for c in range(CZ_EDGES[i], CZ_EDGES[i+1]; length=9)) for i in 1:N_CZ]
    w_e = [sum(interp(E, fl["energies"], fl["flux8B"]) * interp(1e3E, xs[:, 1], xs[:, 2])
               for E in range(E_EDGES[j], E_EDGES[j+1]; length=11)) for j in 1:N_E]
    w = vec(w_cz * w_e')
    return w ./ sum(w)
end

# Response of the stacked P₁ₑ to the scales of the crossed layers, at PREM
function response(b0, dm2s)
    J = reduce(vcat, map(dm2s) do dm2
        _, J = NightOsc._value_jacobian(b0, vcat(dm2, TH12, TH13, ones(b0.n_layers)), true,
                                        Val(3 + length(b0.used_layers)))
        J[:, 3 .+ b0.used_layers]
    end)
    return p1e(b0, dm2s, ones(b0.n_layers)), J
end

sha256_file(f) = bytes2hex(open(sha256, f))

function main()
    bnd, rad, ne, attrs = h5open(DRAWS_FILE) do f
        read(f, "boundaries_km"), read(f, "radius_km"), read(f, "n_e"), Dict(k => read_attribute(f, k) for k in keys(attributes(f)))
    end
    n_draws = size(rad, 2)                      # h5py arrays arrive transposed: (points, draws)
    @info "Earth prior from $n_draws draws of $DRAWS_FILE"
    m = load_earth_model("PREM")
    b0 = engine(NightOsc.EarthModel(m[:radius], m[:potential], m[:boundaries]))
    layers = b0.used_layers
    dm2_all = vcat(DM2_FIT, DM2_CHECK)
    P0, J = response(b0, dm2_all)
    D = zeros(length(P0), n_draws)
    Threads.@threads for d in 1:n_draws
        D[:, d] = p1e(engine(draw_model(rad[:, d], ne[:, d], bnd[:, d])), dm2_all) .- P0
    end
    nb = N_CZ * N_E
    rows(m) = (findfirst(==(m), dm2_all) - 1) * nb .+ (1:nb)
    w_ev = event_weights()
    rf = reduce(vcat, rows.(DM2_FIT))
    blocks = rows.(dm2_all)
    function build(idx)
        p = earth_prior_covariance(J[rf, :], D[rf, idx], repeat(w_ev, length(DM2_FIT)))
        k, Σ = cover(J, D[:, idx], w_ev, p.cov, blocks, COVERAGE)
        return p, k, Σ
    end
    prior, k, Σ = build(1:n_draws)

    rng = Xoshiro(1)
    boot = reduce(hcat, [sqrt.(diag(build(rand(rng, 1:n_draws, n_draws))[3])) for _ in 1:N_BOOT])
    sd, sd_err = sqrt.(diag(Σ)), vec(std(boot; dims=2))

    println("\nlayer   sd       ± boot   (projection sd, extra sd, before ×$(round(sqrt(k), digits=3)))   mean offset")
    for (i, l) in enumerate(layers)
        @printf("%3d   %.4f   %.4f   (%.4f, %.4f)   %+.4f\n", l, sd[i], sd_err[i],
                sqrt(prior.cov_projection[i, i]), sqrt(prior.extra_variance[i]), prior.mean[i])
    end
    Cr = Σ ./ (sd * sd')
    println("\ncorrelations:"); foreach(i -> println(join([@sprintf("%+.2f", Cr[i, j]) for j in eachindex(layers)], " ")), eachindex(layers))
    cov_by_dm2 = Dict(name => [coverage(J[r, :], D[r, :], w_ev, S) for r in blocks]
                      for (name, S) in (("projection", prior.cov_projection), ("matched", prior.cov), ("prior", Σ)))
    println("\nper-bin variance / draws' mean square about PREM, event-weighted (fitted Δm²: $DM2_FIT):")
    for name in ("projection", "matched", "prior")
        @printf("  %-10s %s\n", name, join([@sprintf("%.1e: %.2f", m, c) for (m, c) in zip(dm2_all, cov_by_dm2[name])], "  "))
    end
    isposdef(Σ) || error("prior covariance is not positive definite")
    k > 2 && @warn "the matched covariance had to be scaled by $(round(k, digits=2)) to reach the coverage target"

    git(args...) = try readchomp(`git -C $ROOT $args`) catch; "unknown" end
    jldsave(OUT_FILE; covariance=Σ, var"true"=ones(length(layers)), layers=collect(layers),
            mean_draws=1 .+ prior.mean, covariance_projection=prior.cov_projection, extra_variance=prior.extra_variance,
            coverage_target=COVERAGE, coverage_scale=k, coverage=cov_by_dm2["prior"], sd_bootstrap=sd_err,
            n_draws, draws_file=basename(DRAWS_FILE), draws_sha256=sha256_file(DRAWS_FILE), draws_attrs=attrs,
            dm2_fit=DM2_FIT, dm2_check=DM2_CHECK, cz_edges=CZ_EDGES, e_edges=E_EDGES,
            weights="CC events: $(basename(EXPOSURE_FILE)) flux weight x $(basename(FLUX_FILE)) 8B x $(basename(XSEC_FILE))",
            weights_sha256=Dict(basename(f) => sha256_file(f) for f in (EXPOSURE_FILE, FLUX_FILE, XSEC_FILE)),
            fitter_commit=git("rev-parse", "--short", "HEAD") * (isempty(git("status", "--porcelain", "--untracked-files=no", "--", "src", "utils")) ? "" : "-dirty"),
            created=string(now()))
    println("\nwrote $OUT_FILE ($(length(layers)) layer scales: $(collect(layers)))")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
