using Test, Random, Statistics, LinearAlgebra
using BAT, Distributions, DensityInterface, AdvancedHMC, JLD2

# Warm start from a previous run (src/warmStart.jl): the mass matrix is the covariance of the
# previous samples in BAT's pretransformed space, and it must stay fixed through the burn-in.
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
    # the weighted mean / covariance of the same draws, computed directly in z
    Wt = sum(w); μw = Z * w / Wt
    Σw = sum(w[i] * (Z[:, i] - μw) * (Z[:, i] - μw)' for i in eachindex(w)) / (Wt - sum(abs2, w) / Wt)
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

    # the warm transform survives init + burn-in unchanged (BAT's RAM tuning would rescale it),
    # and the chains start inside the previous posterior
    ctx = BATContext(ad = BAT.ADSelector(:ForwardDiff), rng = BAT.Random123.Philox4x((UInt64(7), UInt64(0))))
    tm, fpt2 = BAT.transform_and_unshape(PriorToNormal(), post, ctx)
    init = MCMCChainPoolInit(init_tries_per_chain = BAT.ClosedInterval(4, 20), nsteps_init = 10,
                             initval_alg = InitFromTarget())
    ws2 = build_warm_start(prefix, post, prior, PriorToNormal(); init = init)
    @test ws2.init.nsteps_init == 10 && ws2.init.init_tries_per_chain == init.init_tries_per_chain
    salg = TransformedMCMC(proposal = HamiltonianMC(), adaptive_transform = ws2.transform,
                           transform_tuning = BAT.NoMCMCTransformTuning(), nchains = 2,
                           init = ws2.init, nsteps = 50,
                           burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 50, max_ncycles = 1, nsteps_final = 5))
    states, _ = BAT.mcmc_init!(salg, tm, BAT.apply_trafo_to_init(fpt2, salg.init), BAT.nop_func, ctx)
    states = BAT.mcmc_burnin!(nothing, states, salg, BAT.nop_func)
    for s in states
        @test s.chain_state.f_transform.A == A
        @test s.chain_state.f_transform.b == b
        x = BAT.inverse(fpt2)(s.chain_state.current.x.v[1])
        @test abs(x.a - 0.3) < 0.1 && abs(x.b - 4) < 1 && abs(x.c - 2) < 2
    end
end
