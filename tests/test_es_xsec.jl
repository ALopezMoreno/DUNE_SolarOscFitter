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

# Independent checks: each compares the code with a closed form evaluated here, not with its
# own output.
@testset "ES cross section – independent checks" begin
    # BKS95 Eq. (A5), I(T) = (1/6)[1/3 + (3 - x²)(x artanh(1/x) - 1)], x = √(1 + 2 m_e/T), in
    # BigFloat (the closed form cancels badly in Float64 at small T, where the code uses a series).
    I_ref(T) = (x = sqrt(1 + 2 * big(m_e) / big(T)); Float64((1 / big(3) + (3 - x^2) * (x * atanh(1 / x) - 1)) / 6))
    for T_MeV in (0.01, 0.05, 0.2, 0.3, 1.0, 5.0, 15.0)
        T = T_MeV * 1e-3
        @test isapprox(bks_I(T), I_ref(T); rtol=1e-9, atol=1e-12)
    end
    @test bks_I(1e-9) ≈ 0 atol=1e-6          # κ(T→0) → the T=0 values 0.9791 / 0.9970

    # σ₀ = 2 G_F² m_e² (ħc)² / π  (G_F and ħc from PDG 2024).
    G_F, ħc = 1.1663788e-5, 1.973269804e-14  # GeV⁻², GeV·cm
    @test isapprox(sigma_0, 2 * G_F^2 * m_e^2 * ħc^2 / π; rtol=1e-3)

    # The 16-point recoil quadrature against the analytic integral of dσ/dT for constant κ:
    # ∫ [g1² + g2² (1 - T/E)² - g1 g2 m_e T/E²] dT from T0 to T_max.
    for (enu, cc) in ((5e-3, 1.0), (10e-3, 1.0), (15e-3, 0.0))
        g1, g2 = es_couplings(0.0, _ -> 1.0, cc)
        T0, Tm = tpc_threshold, 2 * enu^2 / (m_e + 2 * enu)
        prim(T) = g1^2 * T - g2^2 * enu / 3 * (1 - T / enu)^3 - g1 * g2 * m_e * T^2 / (2 * enu^2)
        @test isapprox(ES_xsec(enu, _ -> 1.0, cc), sigma_0 / m_e * (prim(Tm) - prim(T0)); rtol=1e-12)
    end
end
