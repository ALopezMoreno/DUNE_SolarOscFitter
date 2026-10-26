using Test
using LinearAlgebra, Random, Statistics

include(joinpath(@__DIR__, "..", "src", "earthPrior.jl"))

# ‖x .* g‖ and min(g) relative to the scale of Aᵀb, for the NNLS optimality conditions
function kkt_violation(A, b, x)
    g = A' * (A * x .- b) ./ norm(A' * b, Inf)
    return max(maximum(abs.(x .* g)) / max(maximum(x), eps()), -minimum(g))
end

# SPEC 2.9 on synthetic draws: a random linear response and draws with known structure
@testset "Earth layer-scale prior" begin
    rng = Xoshiro(7)
    nbin, nl, ndraw = 300, 6, 400
    J = randn(rng, nbin, nl) .* 1e-2
    w = rand(rng, nbin) .^ 4                         # far from uniform
    A = randn(rng, nl, nl) .* 0.02

    @testset "exact layer-scale draws" begin
        θ = 0.01 .+ A * randn(rng, nl, ndraw)          # offset from the nominal counts as spread
        p = earth_prior_covariance(J, J * θ, w)
        M2 = θ * θ' ./ ndraw
        @test maximum(abs.(p.cov_projection .- M2)) < 1e-10 * maximum(abs.(M2))
        @test maximum(p.extra_variance) < 1e-10 * maximum(diag(M2))
        @test p.mean ≈ vec(mean(θ; dims=2)) atol = 1e-12
    end

    # heteroscedastic noise the scales cannot represent
    D = J * (0.01 .+ A * randn(rng, nl, ndraw)) .+ 3e-4 .* randn(rng, nbin, ndraw) .* (0.2 .+ 3 .* rand(rng, nbin))
    p = earth_prior_covariance(J, D, w)

    @testset "weighted projection" begin
        θw = (J' * (w .* J)) \ (J' * (w .* D))
        @test p.cov_projection ≈ θw * θw' ./ ndraw rtol = 1e-10
        @test p.mean ≈ vec(mean(θw; dims=2)) rtol = 1e-10
        θu = J \ D                                     # unweighted: must differ
        @test !isapprox(p.cov_projection, θu * θu' ./ ndraw; rtol=1e-3)
    end

    @testset "added variances match the per-bin variance" begin
        sw = sqrt.(w)
        Aw = sw .* J .^ 2
        bw = sw .* (vec(mean(D .^ 2; dims=2)) .- bin_variance(J, p.cov_projection))
        @test all(>=(0), p.extra_variance) && any(>(0), p.extra_variance)
        @test kkt_violation(Aw, bw, p.extra_variance) < 1e-8
        @test isposdef(p.cov)
        @test p.cov ≈ p.cov_projection + Diagonal(p.extra_variance)
    end

    @testset "a layer with no extra variance" begin
        # per-bin noise variance J² s with one s < 0: the unconstrained match goes negative there
        σ2 = max.((J .^ 2) * [1.0, 1.0, 1.0, 1.0, 1.0, -0.8] .* 1e-2, 1e-12)
        Dn = J * (A * randn(rng, nl, ndraw)) .+ sqrt.(σ2) .* randn(rng, nbin, ndraw)
        q = earth_prior_covariance(J, Dn, w)
        sw = sqrt.(w)
        Aw = sw .* J .^ 2
        bw = sw .* (vec(mean(Dn .^ 2; dims=2)) .- bin_variance(J, q.cov_projection))
        @test any(<(0), Aw \ bw)                      # the constraint is active
        @test any(iszero, q.extra_variance)
        @test kkt_violation(Aw, bw, q.extra_variance) < 1e-8
    end

    @testset "coverage scaling" begin
        blocks = [1:100, 101:200, 201:300]
        k, Σ = cover(J, D, w[1:100], p.cov, blocks, 1.1)
        c = [sum(w[1:100] .* bin_variance(J[r, :], Σ)) / sum(w[1:100] .* vec(mean(D[r, :] .^ 2; dims=2))) for r in blocks]
        @test minimum(c) ≈ 1.1 rtol = 1e-12
        @test Σ ≈ k .* p.cov
    end

    @testset "nnls optimality" begin
        for scale in (1e-10, 1.0, 1e6), _ in 1:10
            M, b = scale .* randn(rng, 50, 7), randn(rng, 50)
            x = nnls(M, b)
            @test all(>=(0), x)
            @test kkt_violation(M, b, x) < 1e-8
        end
        M = rand(rng, 30, 4); x0 = [0.5, 0.0, 2.0, 1.0]
        @test nnls(M, M * x0) ≈ x0 atol = 1e-10
        @test nnls(M, zeros(30)) == zeros(4)
    end
end
