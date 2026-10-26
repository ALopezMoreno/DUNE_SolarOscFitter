using Test
using LinearAlgebra, Random, Statistics

include(joinpath(@__DIR__, "..", "src", "earthPrior.jl"))

# SPEC 2.9 on synthetic draws: a random linear response and draws with known structure
@testset "Earth layer-scale prior" begin
    rng = Xoshiro(7)
    nbin, nl, ndraw = 300, 6, 400
    J = randn(rng, nbin, nl) .* 1e-2
    w = rand(rng, nbin)
    A = randn(rng, nl, nl) .* 0.02

    @testset "exact layer-scale draws" begin
        θ = A * randn(rng, nl, ndraw)
        p = earth_prior_covariance(J, J * θ, w)
        @test maximum(abs.(p.cov_projection .- cov(θ; dims=2))) < 1e-10 * maximum(abs.(cov(θ; dims=2)))
        @test maximum(p.extra_variance) < 1e-10 * maximum(diag(cov(θ; dims=2)))
        @test p.mean ≈ vec(mean(θ; dims=2)) atol = 1e-12
    end

    @testset "draws with structure the scales cannot represent" begin
        D = J * (A * randn(rng, nl, ndraw)) .+ 3e-4 .* randn(rng, nbin, ndraw) .* (1 .+ rand(rng, nbin))
        p = earth_prior_covariance(J, D, w)
        @test isposdef(p.cov)
        @test all(>=(0), p.extra_variance) && any(>(0), p.extra_variance)
        @test p.cov ≈ p.cov_projection + Diagonal(p.extra_variance)
        @test sum(w .* bin_variance(J, p.cov)) > sum(w .* bin_variance(J, p.cov_projection))
    end

    @testset "nnls optimality" begin
        for _ in 1:20
            M, b = randn(rng, 50, 7), randn(rng, 50)
            x = nnls(M, b)
            g = M' * (M * x .- b)                       # gradient of ½‖Mx − b‖²
            @test all(>=(0), x)
            @test all(>=(-1e-8), g)
            @test maximum(abs.(x .* g)) < 1e-8
        end
        M = rand(rng, 30, 4); x0 = [0.5, 0.0, 2.0, 1.0]
        @test nnls(M, M * x0) ≈ x0 atol = 1e-10
        @test nnls(M, zeros(30)) == zeros(4)
    end
end
