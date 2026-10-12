using Test

global tpc_threshold = 0.2e-3
global CC_xsec_scale = 1.0
include(joinpath(@__DIR__, "..", "src", "xsec.jl"))

@testset "ES cross section – BKS95 radiative corrections" begin
    # ν_μ e couplings at T → 0 against PDG 2024 Table 10.3: g_V = g1 + g2 = -0.0395,
    # g_A = g1 - g2 = -0.5063.
    g1, g2 = es_couplings(1e-12, kappa_nuother, 0.0)
    @test isapprox(g1 + g2, -0.0395; atol=5e-4)
    @test isapprox(g1 - g2, -0.5063; atol=5e-4)

    # Below threshold (T_max <= tpc_threshold) the cross section is exactly 0, never negative.
    for E in (0.1e-3, 0.25e-3, 0.3e-3)
        @test ES_xsec_nue(E) == 0.0
        @test ES_xsec_nuother(E) == 0.0
    end

    # Reference values [cm^2] at 10 MeV, cross-checked against SnuPpy's independent implementation.
    @test isapprox(ES_xsec_nue(10e-3), 8.8323e-44; rtol=1e-4)
    @test isapprox(ES_xsec_nuother(10e-3), 1.5464e-44; rtol=1e-4)
end
