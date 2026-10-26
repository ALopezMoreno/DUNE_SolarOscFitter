using Test, Random, Statistics, LinearAlgebra
using BAT, Distributions, DensityInterface, AdvancedHMC, JLD2

# Warm start from a previous run (src/warmStart.jl): the mass matrix is the covariance of the
# previous samples in BAT's pretransformed space, and it must stay fixed through the burn-in.
include(joinpath(@__DIR__, "..", "src", "logger.jl"))
include(joinpath(@__DIR__, "..", "src", "warmStart.jl"))

@testset "warm start from a previous run" begin
    rng = Xoshiro(3)
    prior = distprod(a = Normal(0, 1), b = Uniform(0, 10), c = truncated(Normal(1, 2), 0, Inf))
    ll(p) = -0.5 * ((p.a - 0.3)^2 / 1e-4 + (p.b - 4)^2 / 0.04 + (p.c - 2)^2 / 0.25)
    post = PosteriorMeasure(logfuncdensity(ll), prior)
    _, fpt = BAT.transform_and_unshape(PriorToNormal(), post, BATContext())
    finv = BAT.inverse(fpt)

    # "previous run": z ~ N(μ, Σ) in the pretransformed space, saved in parameter space in the
    # layout saveBatch writes (two batches, integer weights, an extra column not in the prior)
    μ = [0.3, -0.25, 0.13]
    Σ = [1e-4 0 0; 0 4e-3 3e-3; 0 3e-3 0.09]
    Z = rand(rng, MvNormal(μ, Σ), 3000)
    X = [finv(Z[:, i]) for i in 1:size(Z, 2)]
    w = rand(rng, 1:3, length(X))
    dir = mktempdir(); prefix = joinpath(dir, "prev")
    jldopen(prefix * ".jld2", "w") do f
        for (b, r) in (("batch_1", 1:1000), ("batch_2", 1001:3000))
            g = JLD2.Group(f, b)
            for k in (:a, :b, :c)
                g[string(k)] = [getfield(X[i], k) for i in r]
            end
            g["unrelated"] = zeros(length(r))
            g["weights"] = w[r]; g["stepno"] = collect(r); g["chainid"] = ones(Int, length(r))
        end
    end

    ws = build_warm_start(prefix, post, prior, PriorToNormal())
    A, b = ws.transform.f.A, ws.transform.f.b
    # weights are multiplicities: compare with the plain mean / covariance of the repeated draws
    Zrep = reduce(hcat, [repeat(Z[:, i], 1, w[i]) for i in eachindex(w)])
    μw = vec(mean(Zrep, dims = 2)); Σw = cov(Zrep, dims = 2)
    @test b ≈ μw rtol = 1e-8
    @test A * A' ≈ Σw rtol = 1e-8
    @test istril(A)
    @test ws.scales ≈ sqrt.(diag(Σw)) rtol = 1e-8
    @test ws.nsamples == 3000
    # and they estimate the true ones
    @test b ≈ μ atol = 0.02
    @test A * A' ≈ Σ rtol = 0.15

    # thinning keeps an even spread of the run
    @test build_warm_start(prefix, post, prior, PriorToNormal(); max_samples = 500).nsamples == 500

    # a previous run without one of the fitted parameters is refused
    prior4 = distprod(a = Normal(0, 1), b = Uniform(0, 10), c = truncated(Normal(1, 2), 0, Inf), d = Normal())
    post4 = PosteriorMeasure(logfuncdensity(p -> ll(p) - p.d^2 / 2), prior4)
    @test_throws ErrorException build_warm_start(prefix, post4, prior4, PriorToNormal())
    @test_throws ErrorException build_warm_start(joinpath(dir, "absent"), post, prior, PriorToNormal())

    # boundary samples: PriorToNormal clamps them to |z| ≈ 8; they are dropped, not kept as outliers
    jldopen(joinpath(dir, "edge.jld2"), "w") do f
        g = JLD2.Group(f, "batch_1")
        g["a"] = [x.a for x in X]; g["c"] = [x.c for x in X]
        g["b"] = [i <= 10 ? 10.0 : X[i].b for i in eachindex(X)]       # 10 draws on the Uniform's edge
        g["weights"] = ones(Int, length(X))
    end
    we = build_warm_start(joinpath(dir, "edge"), post, prior, PriorToNormal())
    @test we.nsamples == length(X) - 10
    @test we.scales[2] < 2 * sqrt(Σ[2, 2])

    # the fitter's own wiring: _build_samplingalg with the global warm_start, init + burn-in, then a
    # checkpoint → rebuild_live_state resume (which builds a default sampler) and more steps.
    # The transform must not change anywhere along the way.
    include(joinpath(@__DIR__, "..", "src", "mcmcHelpers.jl"))
    global hmc_context = BATContext(ad = BAT.ADSelector(:ForwardDiff), rng = BAT.Random123.Philox4x((UInt64(7), UInt64(0))))
    global posterior = post
    global proposal_algorithm = HamiltonianMC()
    global mcmcChains, batchSteps, convergence = 2, 50, AssumeConvergence()
    global burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 50, max_ncycles = 1, nsteps_final = 5)
    init0 = MCMCChainPoolInit(init_tries_per_chain = BAT.ClosedInterval(4, 20), nsteps_init = 10,
                              initval_alg = InitFromTarget())
    global warm_start = build_warm_start(prefix, post, prior, PriorToNormal(); init = init0)
    global init = warm_start.init
    @test init.nsteps_init == 10 && init.init_tries_per_chain == init0.init_tries_per_chain
    salg = _build_samplingalg()
    @test salg.adaptive_transform === warm_start.transform
    @test salg.transform_tuning isa BAT.NoMCMCTransformTuning
    tm, fpt2 = BAT.transform_and_unshape(salg.pretransform, posterior, hmc_context)
    states, _ = BAT.mcmc_init!(salg, tm, BAT.apply_trafo_to_init(fpt2, salg.init), BAT.nop_func, hmc_context)
    states = BAT.mcmc_burnin!(nothing, states, salg, BAT.nop_func)
    for s in states
        @test s.chain_state.f_transform.A == A
        @test s.chain_state.f_transform.b == b
        x = BAT.inverse(fpt2)(s.chain_state.current.x.v[1])           # started inside the previous posterior
        @test abs(x.a - 0.3) < 0.1 && abs(x.b - 4) < 1 && abs(x.c - 2) < 2
    end
    file = joinpath(dir, "state.bin")
    save_runstate(RunState(states, fpt2, 0, [:a, :b, :c]), file)
    global warm_start = nothing                        # a resumed run does not rebuild the warm start
    # a resume with a different parameter layout is refused (positions and transform are by index)
    @test_throws ErrorException load_runstate(file, distprod(b = Uniform(0, 10), a = Normal(0, 1), c = Normal()))
    rs = rebuild_live_state(load_runstate(file, prior))
    new_states, _ = _sample_one_batch(rs.mcmc_states, rs.f_pretransform, 50)
    @test all(s.chain_state.f_transform.A == A && s.chain_state.f_transform.b == b for s in new_states)
    @test all(s.trafo_tuner_state isa BAT.NoMCMCTransformTuningState for s in new_states)
end

# A vector-valued parameter (the Earth layer scales, SPEC 2.6), stored in the chain file as e_1…e_K
@testset "warm start with a vector-valued parameter" begin
    rng = Xoshiro(5)
    prior = distprod(a = Normal(0, 1), e = MvNormal([1.0, 1.0], [0.01 0.006; 0.006 0.02]))
    post = PosteriorMeasure(logfuncdensity(p -> -0.5 * ((p.a - 0.3)^2 / 1e-2 + sum(abs2, p.e .- 1) / 1e-3)), prior)
    _, fpt = BAT.transform_and_unshape(PriorToNormal(), post, BATContext())
    finv = BAT.inverse(fpt)
    μ, Σ = [0.3, 0.1, -0.2], [0.01 0 0; 0 0.02 0.01; 0 0.01 0.03]
    Z = rand(rng, MvNormal(μ, Σ), 2000)
    X = [finv(Z[:, i]) for i in 1:size(Z, 2)]
    prefix = joinpath(mktempdir(), "vec")
    jldopen(prefix * ".jld2", "w") do f
        g = JLD2.Group(f, "batch_1")
        g["a"] = [x.a for x in X]; g["e_1"] = [x.e[1] for x in X]; g["e_2"] = [x.e[2] for x in X]
        g["weights"] = ones(Int, length(X)); g["stepno"] = collect(1:length(X)); g["chainid"] = ones(Int, length(X))
    end
    flat(p) = (a = p.a, e_1 = p.e[1], e_2 = p.e[2])
    unflat(q) = (e = [q.e_1, q.e_2], a = q.a)          # deliberately not in the prior's order
    ws = build_warm_start(prefix, post, prior, PriorToNormal(); flat, unflat)
    @test ws.names == [:a, :e_1, :e_2]
    @test ws.transform.f.b ≈ vec(mean(Z, dims = 2)) rtol = 1e-8
    @test ws.transform.f.A * ws.transform.f.A' ≈ cov(Z, dims = 2) rtol = 1e-8
end
