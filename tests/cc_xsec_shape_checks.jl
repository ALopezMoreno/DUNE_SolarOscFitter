#=
Shared by test_cc_xsec.jl (synthetic ⁸B-like weights) and derived_mode.jl (real setup weights).
Needs cc_xsec_shape_basis / cc_xsec_shape (src/xsec.jl).
=#
using Test, ForwardDiff

# SPEC 3.3: shape nuisances norm · exp(tilt·p₁ + curv·p₂). At the nominal point the shape is 1;
# with N = Σ w·shape (w = unoscillated CC ⁸B rate per E_true bin), ∂ln N/∂tilt = ∂ln N/∂curv = 0
# at tilt = curv = 0 and Σ w p₁ p₂ = 0 (first order only); tilt tilts and curv curves σ in ln E.
# Here a ⁸B-like spectrum folded with the table; derived_mode.jl repeats the check on the real
# setup weights of the fixture detectors.
"""Check SPEC 3.3 for E_true bin centres `E` (GeV) and weights `w`; returns the basis."""
function check_cc_xsec_shape_spec(E, w; rtol = 1e-10)
    B = cc_xsec_shape_basis(E, w)
    p1, p2 = B.p1, B.p2
    W = sum(w)
    s1, s2 = sqrt(sum(w .* p1 .^ 2) / W), sqrt(sum(w .* p2 .^ 2) / W)   # natural scales
    @test s1 > 0 && s2 > 0

    # nominal point: the cross section is the table; norm is purely multiplicative
    @test all(cc_xsec_shape(1.0, 0.0, 0.0, p1, p2) .== 1.0)
    @test cc_xsec_shape(1.3, 0.0, 0.0, p1, p2) ≈ fill(1.3, length(E)) rtol=1e-15

    # first-order orthogonality of the unoscillated rate N: AD and finite differences
    lnN(θ) = log(sum(w .* cc_xsec_shape(1.0, θ[1], θ[2], p1, p2)))
    g = ForwardDiff.gradient(lnN, [0.0, 0.0])
    @test abs(g[1]) <= rtol * s1
    @test abs(g[2]) <= rtol * s2
    h = 1e-4
    @test abs((lnN([h, 0.0]) - lnN([-h, 0.0])) / 2h) <= 1e-7 * s1
    @test abs((lnN([0.0, h]) - lnN([0.0, -h])) / 2h) <= 1e-7 * s2
    # Fisher-type cross term
    @test abs(sum(w .* p1 .* p2) / W) <= rtol * s1 * s2
    # second order is not zero: ∂²ln N = ⟨p p'⟩ (diagonal ⟨p₁²⟩, ⟨p₂²⟩, off-diagonal 0)
    H = ForwardDiff.hessian(lnN, [0.0, 0.0])
    @test H[1, 1] ≈ s1^2 rtol=1e-8
    @test H[2, 2] ≈ s2^2 rtol=1e-8
    @test abs(H[1, 2]) <= 1e-8 * s1 * s2

    # tilt: ln(σ/σ_nom) linear in ln E with slope tilt (monotone in E);
    # curv: ln(σ/σ_nom) quadratic in ln E with second derivative 2·curv
    lnE = log.(E)
    r_t = log.(cc_xsec_shape(1.0, 0.2, 0.0, p1, p2))
    @test all(diff(r_t) .> 0)
    @test all(isapprox.(diff(r_t) ./ diff(lnE), 0.2; rtol=1e-9))
    @test all(diff(log.(cc_xsec_shape(1.0, -0.2, 0.0, p1, p2))) .< 0)
    r_c = log.(cc_xsec_shape(1.0, 0.0, 0.3, p1, p2))
    d2(r) = [2 * ((r[i+1] - r[i]) / (lnE[i+1] - lnE[i]) - (r[i] - r[i-1]) / (lnE[i] - lnE[i-1])) /
             (lnE[i+1] - lnE[i-1]) for i in 2:length(r)-1]
    @test all(isapprox.(d2(r_c), 2 * 0.3; rtol=1e-6))

    # the coefficients logged at setup: p₁ = L − ⟨L⟩, p₂ = L² − aL − b, L = ln(E/E_pivot)
    L = log.(E ./ B.E_pivot)
    @test B.E_pivot ≈ sum(w .* E) / W rtol=1e-14
    @test p1 ≈ L .- B.mean_L atol=1e-12
    @test p2 ≈ L .^ 2 .- B.a .* L .- B.b atol=1e-10
    return B
end
