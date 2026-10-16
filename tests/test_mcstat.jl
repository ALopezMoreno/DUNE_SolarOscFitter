using Test, Random, Statistics
using Distributions: Poisson

# MC statistical uncertainty in the likelihood (src/likelihoods/mcstat.jl). Each test compares
# with an independent computation: brute-force minimisation over β, brute-force resampling of the
# MC, MaCh3's formulas (transcribed from SampleHandlerInterface::GetTestStatLLH), or limits.
include(joinpath(@__DIR__, "..", "src", "logger.jl"))
include(joinpath(@__DIR__, "..", "src", "histHelpers.jl"))
include(joinpath(@__DIR__, "..", "src", "likelihoods", "mcstat.jl"))

const DA, BB = DembinskiAbdelmotteleb(), BarlowBeeston()
bc(n, μ) = n > 0 ? μ - n + n * log(n / μ) : μ                      # Baker–Cousins, independent copy
penalty(::DembinskiAbdelmotteleb, β, a) = β * a - a - a * log(β)
penalty(::BarlowBeeston, β, a) = a * (β - 1)^2 / 2
# brute force: −log L of a group profiled over β on a fine grid + golden-section refinement
function brute(method, μs, ns, a)
    f(β) = sum(bc(n, β * μ) for (μ, n) in zip(μs, ns)) + penalty(method, β, a)
    lo, hi = -60.0, 20.0                                # golden section in log β (f is unimodal; Conway can reach β = 0)
    for _ in 1:300
        m1, m2 = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
        f(exp(m1)) < f(exp(m2)) ? (hi = m2) : (lo = m1)
    end
    return f(exp((lo + hi) / 2))
end

@testset "MC-statistics likelihood" begin
    rng = Xoshiro(1)

    @testset "closed-form β is the profile minimum (single bin and shared groups)" begin
        for method in (DA, BB), _ in 1:200
            k = rand(rng, 1:6)                                  # bins sharing one β
            μs = 10 .^ (3 .* rand(rng, k) .- 1)                 # 0.1 … 100
            ns = Float64.(rand(rng, 0:150, k)) .* (rand(rng) < 0.2 ? 0 : 1)
            a  = 10^(4 * rand(rng) - 1)                         # effective MC counts 0.1 … 1000
            @test _group_negllh(method, μs, ns, a) ≈ brute(method, μs, ns, a) rtol=1e-7 atol=1e-7
        end
    end

    @testset "matches MaCh3's per-bin formulas" begin
        # GetTestStatLLH(data, mc, w2) with w2 = mc²/a; it returns −log L (same normalisation)
        function mach3_da(data, mc, w2)
            k = mc^2 / w2; β = (data + k) / (mc + k); newmc = mc * β
            pen = k * β - k + k * log(k / (k * β))
            return (data > 0 ? newmc - data + data * log(data / newmc) : newmc) + pen
        end
        function mach3_bb(data, mc, w2)
            f2 = w2 / mc^2; temp = mc * f2 - 1; β = (-temp + sqrt(temp^2 + 4 * data * f2)) / 2
            newmc = mc * β
            stat = data > 0 ? newmc - data + data * log(data / newmc) : mc * β
            return stat + (β - 1)^2 / (2 * f2)
        end
        for _ in 1:200
            mc = 10^(3 * rand(rng) - 0.5); data = Float64(rand(rng, 0:200)); a = 10^(3 * rand(rng))
            @test _group_negllh(DA, [mc], [data], a) ≈ mach3_da(data, mc, mc^2 / a) rtol=1e-9 atol=1e-11
            @test _group_negllh(BB, [mc], [data], a) ≈ mach3_bb(data, mc, mc^2 / a) rtol=1e-9 atol=1e-11
        end
    end

    @testset "limits" begin
        μs, ns = [5.0, 20.0, 1.0], [7.0, 18.0, 0.0]
        pois = sum(bc(n, _mu_floor(μ)) for (μ, n) in zip(μs, ns))
        for method in (DA, BB)
            @test _group_negllh(method, μs, ns, Inf) ≈ pois rtol=1e-14         # no MC uncertainty
            @test _group_negllh(method, μs, ns, 1e12) ≈ pois rtol=1e-8         # infinite MC
            @test _group_negllh(method, μs, μs, 3.0) ≈ 0 atol=1e-6             # Asimov: β = 1, nothing to pay
            # profiling β can only raise the likelihood (β = 1 has zero penalty)
            for a in (0.5, 5.0, 50.0)
                @test _group_negllh(method, μs, ns, a) <= pois + 1e-12
            end
        end
        # a stronger MC constraint (larger a) gives a likelihood closer to Poisson
        @test _group_negllh(DA, μs, ns, 2.0) < _group_negllh(DA, μs, ns, 20.0) < pois
    end

    @testset "channel log L: shared vs independent, shapes, threshold" begin
        nE, nz, idx = 6, 4, 2
        day, night = 50 .* rand(rng, nE) .+ 1, 10 .* rand(rng, nz, nE) .+ 0.5
        od, on = Float64.(rand(rng, 0:60, nE)), Float64.(rand(rng, 0:15, nz, nE))
        a = [Inf, 3.0, 40.0, Inf, 1e3, 7.0]
        for method in (DA, BB)
            sh = mcstat_loglikelihood(method, true, day, night, od, on, a, idx)
            @test sh ≈ -sum(_group_negllh(method, vcat(day[j], night[:, j]), vcat(od[j], on[:, j]), a[j]) for j in idx:nE)
            ind = mcstat_loglikelihood(method, false, day, night, od, on, a, idx)
            @test ind ≈ -sum(_group_negllh(method, [m], [n], a[j]) for j in idx:nE
                             for (m, n) in zip(vcat(day[j], night[:, j]), vcat(od[j], on[:, j])))
            # per-bin pieces add up to the channel log L (all fitted bins: n_fit = nE)
            pd, pn = mcstat_perbin(method, true, day, night, od, on, a, idx, nE)
            @test sum(pd) + sum(pn) ≈ sh
            @test all(iszero, pd[1:idx-1]) && all(iszero, pn[:, 1:idx-1])
        end
        # angular shapes: day (n_cos, n_E), night (n_cos, n_E, n_cosz)
        dA, nA = rand(rng, 3, nE) .+ 1, rand(rng, 3, nE, nz) .+ 0.2
        odA, onA = Float64.(rand(rng, 0:3, 3, nE)), Float64.(rand(rng, 0:2, 3, nE, nz))
        shA = mcstat_loglikelihood(DA, true, dA, nA, odA, onA, a, idx)
        @test shA ≈ -sum(_group_negllh(DA, vcat(dA[:, j], vec(nA[:, j, :])), vcat(odA[:, j], vec(onA[:, j, :])), a[j]) for j in idx:nE)
        pd, pn = mcstat_perbin(DA, true, dA, nA, odA, onA, a, idx, nE)
        @test sum(pd) + sum(pn) ≈ shA
    end

    @testset "signal MC variance agrees with resampling the MC" begin
        # 4 E_true rows × 5 E_reco bins; N_i MC events per row, multinomial over reco bins.
        P = [0.6 0.3 0.1 0 0; 0.1 0.5 0.3 0.1 0; 0 0.2 0.4 0.3 0.1; 0 0 0.2 0.5 0.3]
        Nrow = [400, 300, 200, 100]
        T = [1000.0, 800.0, 500.0, 300.0]                       # true-energy rates
        eff, g = [0.9, 0.8, 0.7, 0.9, 0.5], [1.0, 0.5, 1.0, 1.0, 0.8]
        draw() = vcat([(c = zeros(5); for _ in 1:Nrow[i]; c[findfirst(>=(rand(rng)), cumsum(P[i, :]))] += 1; end; c')
                       for i in 1:4]...)
        reps = [signal_column(T, draw(), eff, zeros(5), g)[1] for _ in 1:4000]
        var_mc = vec(var(reduce(hcat, reps), dims=2))
        # the analytic variance from ONE MC sample, averaged over samples to remove its own noise
        var_an = mean(signal_column(T, draw(), eff, zeros(5), g)[2] for _ in 1:400)
        @test var_an ≈ var_mc rtol=0.08

        # with the efficiency term on: eff_j = sel_j / tot_j from the same toy, sel ~ Binomial(tot, ε)
        tot0, ε0 = [300.0, 250.0, 200.0, 150.0, 100.0], [0.9, 0.8, 0.7, 0.9, 0.5]
        draw_eff() = [count(_ -> rand(rng) < ε0[j], 1:Int(tot0[j])) / tot0[j] for j in 1:5]
        reps2 = [signal_column(T, draw(), draw_eff(), tot0, g)[1] for _ in 1:4000]
        var_mc2 = vec(var(reduce(hcat, reps2), dims=2))
        var_an2 = mean(signal_column(T, draw(), draw_eff(), tot0, g)[2] for _ in 1:400)
        @test var_an2 ≈ var_mc2 rtol=0.08
    end

    @testset "guards" begin
        using ForwardDiff
        # Conway at N = 0, M = a exactly: finite gradient (sqrt'(0) used to give NaN)
        @test all(isfinite, ForwardDiff.gradient(x -> _group_negllh(BB, x, [0.0, 0.0], 1.0), [0.6, 0.4]))
        # a significantly negative prediction is an error, as in poissonLogLikelihood
        @test_throws ArgumentError _group_negllh(DA, [-5.0, 1.0], [1.0, 1.0], 10.0)
        @test_throws ArgumentError mcstat_loglikelihood(DA, true, [1.0, -5.0], ones(2, 2), ones(2), ones(2, 2), [10.0, 10.0], 1)
        @test_throws ArgumentError mcstat_loglikelihood(DA, true, [-5.0, 1.0], ones(2, 2), ones(2), ones(2, 2), [Inf, Inf], 1)
        @test_throws ArgumentError mcstat_perbin(DA, true, [-5.0, 1.0], ones(2, 2), ones(2), ones(2, 2), [10.0, 10.0], 1, 2)
        @test isfinite(_group_negllh(DA, [-1e-6, 1.0], [0.0, 1.0], 10.0))        # numerical noise is floored
    end

    @testset "background relative variance Σw²/(Σw)²" begin
        bins = (bin_number = 2, min = 0.0, max = 2.0, bins = [-Inf, 1.0, Inf])
        E = [0.5, 0.7, 1.5, 1.6, 1.7]; w = [1.0, 3.0, 2.0, 2.0, 0.0]
        @test mc_relative_variance(E, w, bins) ≈ [(1 + 9) / 16, 8 / 16]
        @test mc_relative_variance([0.5], [1.0], bins) == [1.0, 0.0]          # empty bin → 0
        # compound Poisson: var(Σ w) = Σ w² for a Poisson number of weighted events
        λ, ws = 50, [0.5, 1.0, 4.0]
        sums = [sum(ws[rand(rng, 1:3)] for _ in 1:rand(rng, Poisson(λ))) for _ in 1:20000]
        @test var(sums) / mean(sums)^2 ≈ mean(ws .^ 2) / (λ * mean(ws)^2) rtol=0.05
    end
end
