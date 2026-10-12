using Test
using ForwardDiff
using Enzyme
using Random
using DelimitedFiles, Interpolations, Statistics, QuadGK

include(joinpath(@__DIR__, "..", "src", "oscillations", "osc.jl"))
include(joinpath(@__DIR__, "..", "src", "oscillations", "osc_night.jl"))
using .NightOsc

const _BN_PHASE = 2.5338653580781976
const _BN_FILE  = joinpath(@__DIR__, "..", "inputs", "EARTH_MODEL_PREM_DENSE.dat")

# Legacy Earth paths (makePaths.jl) for the layer-agreement check; it reads these globals.
earth_normalisation_true = nothing
earth = let data = readdlm(_BN_FILE)
    linear_interpolation(data[:, 1] .* 6371.0 .* 1.0000000001, data[:, 2] .* data[:, 3] .* MATTER_POTENTIAL_CONST;
                         extrapolation_bc=Throw())
end
include(joinpath(@__DIR__, "..", "src", "oscillations", "makePaths.jl"))
include(joinpath(@__DIR__, "..", "src", "propagation", "osc_setup.jl"))

# Builds and uses an engine that becomes unreachable on return; returns the number of stores.
@noinline function _use_temporary_engine(expo, earth_model)
    tmp = BinnedP1e([-0.5, -0.1], [5e-3, 6e-3], expo, earth_model; phase_const=_BN_PHASE)
    tmp(6e-5, 0.6, 0.149)
    return length(task_local_storage(:night_osc_store))
end

@testset "NightOsc" begin
    earth_model = EarthModel(_BN_FILE; potential_const=MATTER_POTENTIAL_CONST)
    expo(c) = 1.0 + 0.3c          # smooth stand-in for the solar exposure per unit cos z

    @testset "Earth model and chords" begin
        @test NightOsc.n_layers(earth_model) == 6
        @test issorted(earth_model.layer_boundaries)
        for cz in (-0.05, -0.4, -0.9, -1.0), step in (nothing, 200.0)
            path = build_earth_path(cz, earth_model; depth_km=0.001, max_step_km=step)
            @test sum(s.length for s in path) ≈ chord_length(6371.0 - 0.001, cz)
            step === nothing || @test maximum(s.length for s in path) <= step + 1e-9
            # outermost layer at both ends, deepest layer in the middle
            @test first(path).layer == 1 && last(path).layer == 1
        end
    end

    # Small grid with bins that straddle the core-mantle (cz ≈ -0.84) and 670 km (cz ≈ -0.43) tangencies
    cz_edges = [-0.95, -0.9, -0.84, -0.7, -0.45, -0.35, -0.1, -0.03, 0.0]
    E_edges  = collect(4e-3:1e-3:10e-3)
    b = BinnedP1e(cz_edges, E_edges, expo, earth_model; phase_const=_BN_PHASE)

    # Dense reference: 64 cos z midpoints per bin (exposure-weighted), energies resolving 0.3 rad
    function dense(dm2, th12, th13, norms)
        out = zeros(length(cz_edges) - 1, length(E_edges) - 1)
        for i in axes(out, 1)
            lo, hi = cz_edges[i], cz_edges[i+1]
            cs = [lo + (k - 0.5) * (hi - lo) / 64 for k in 1:64]
            w = expo.(cs); w ./= sum(w)
            paths = [build_earth_path(c, earth_model; max_step_km=300.0) for c in cs]
            for j in axes(out, 2)
                a, bb = E_edges[j], E_edges[j+1]
                n = max(16, ceil(Int, _BN_PHASE * 2 * 6371 * dm2 * 1.6 * (1 / a - 1 / bb) / 0.3))
                Es = [a + (m - 0.5) * (bb - a) / n for m in 1:n]
                out[i, j] = sum(w .* vec(sum(p1e_paths(Es, dm2, th12, th13, paths; layer_norms=norms,
                                                       phase_const=_BN_PHASE), dims=2) ./ n))
            end
        end
        out
    end

    @testset "bin averages match dense sampling" begin
        for (dm2, s2, norms) in ((2e-5, 0.307, ones(6)), (1e-4, 0.307, ones(6)),
                                 (4e-5, 0.35, [1.15, 0.85, 1.1, 0.9, 1.2, 1.0]))
            th12, th13 = asin(sqrt(s2)), asin(sqrt(0.022))
            @test maximum(abs.(b(dm2, th12, th13, norms) .- dense(dm2, th12, th13, norms))) < 1e-3
        end
    end

    Random.seed!(7)
    W = randn(b.shape)
    f(x) = sum(W .* b(x[1] * 1e-5, x[2], x[3], x[4:9]))

    @testset "ForwardDiff gradient matches finite differences" begin
        x0 = [4.0, 0.60, 0.149, 0.9, 1.1, 1.15, 0.85, 1.05, 1.0]
        g = ForwardDiff.gradient(f, x0)
        fd = map(eachindex(x0)) do i
            h = 1e-5 * max(abs(x0[i]), 1.0)
            xp = copy(x0); xm = copy(x0); xp[i] += h; xm[i] -= h
            (f(xp) - f(xm)) / (2h)
        end
        @test maximum(abs.(g .- fd) ./ max.(abs.(fd), 1e-6)) < 1e-5
        @test g[9] == 0          # inner core: not crossed by this grid
    end

    @testset "cached Jacobian rule equals direct duals under chunking" begin
        θ0 = [4.0, 0.30, 0.022, 0.9, 1.1, 1.15, 0.85, 1.05, 1.0, 1.3]
        h(θ, direct) = begin
            args = (θ[1] * 1e-5, asin(sqrt(θ[2])), asin(sqrt(θ[3])), Tuple(θ[4:9]))
            P = direct ? b(args...) : binned_p1e(b, args...)
            sum(W .* P) * θ[10]
        end
        hc(θ) = h(θ, false)
        hd(θ) = h(θ, true)
        gc = ForwardDiff.gradient(hc, θ0, ForwardDiff.GradientConfig(hc, θ0, ForwardDiff.Chunk{3}()))
        gd = ForwardDiff.gradient(hd, θ0, ForwardDiff.GradientConfig(hd, θ0, ForwardDiff.Chunk{3}()))
        @test maximum(abs.(gc .- gd)) < 1e-9 * maximum(abs.(gd))
        @test hc(θ0) == hd(θ0)
        # no Earth-density inputs
        g3c = ForwardDiff.gradient(θ -> sum(W .* binned_p1e(b, θ[1] * 1e-5, θ[2], θ[3])), [4.0, 0.6, 0.149])
        g3d = ForwardDiff.gradient(θ -> sum(W .* b(θ[1] * 1e-5, θ[2], θ[3])), [4.0, 0.6, 0.149])
        @test maximum(abs.(g3c .- g3d)) < 1e-9 * maximum(abs.(g3d))
    end

    @testset "Enzyme reverse rule equals ForwardDiff" begin
        he(θ) = sum(W .* binned_p1e(b, θ[1] * 1e-5, θ[2], θ[3], (θ[4], θ[5], θ[6], θ[7], θ[8], θ[9])))
        θ0 = [4.0, 0.60, 0.149, 0.9, 1.1, 1.15, 0.85, 1.05, 1.0]
        ge = Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(he), θ0)[1]
        gf = ForwardDiff.gradient(he, θ0)
        @test maximum(abs.(ge .- gf)) < 1e-9 * maximum(abs.(gf))
        @test he(θ0) == sum(W .* b(θ0[1] * 1e-5, θ0[2], θ0[3], θ0[4:9]))   # primal unchanged by the rule
    end

    @testset "step propagator matches the 3-flavour NumOsc solver" begin
        # Independent check of the physics: NumOsc (full 3ν, constant potential per 10 km step)
        # against p1e_paths at the fitter's 300 km linear-slope steps. Measured ≤ 5e-4; the
        # two-flavour reduction alone is ≤ 2e-4.
        as_numosc(path) = (segments = [(avgRho = s.avg_potential, length = s.length) for s in path],)
        Es = collect(range(3e-3, 15e-3; length=13))
        for (dm2, s2) in ((6e-5, 0.307), (2e-5, 0.307), (1.5e-4, 0.35))
            th12, th13 = asin(sqrt(s2)), asin(sqrt(0.022))
            pars = Osc.oscPars(dm2, th12, th13)
            for cz in (-0.1, -0.6, -0.85, -0.99)
                ref = Osc.NumOsc.Slow.osc_prob_earth(Es, pars, nothing,
                                                     [as_numosc(build_earth_path(cz, earth_model; max_step_km=10.0))])
                p = p1e_paths(Es, dm2, th12, th13, [build_earth_path(cz, earth_model; max_step_km=300.0)];
                              phase_const=_BN_PHASE)
                @test maximum(abs.(p .- ref)) < 1e-3
            end
        end
    end

    @testset "Earth layers agree with makePaths" begin
        @test check_layer_agreement(earth_model, cz -> make_potential_for_integrand(cz, earth), depth) === nothing
        # a model with an extra layer must be caught
        shifted = NightOsc.EarthModel(earth_model.radius, earth_model.potential_grid,
                                      sort(vcat(earth_model.layer_boundaries, 4500.0)))
        @test_throws ErrorException check_layer_agreement(shifted, cz -> make_potential_for_integrand(cz, earth), depth)
    end

    @testset "exposure normalisation check" begin
        w = [quadgk(expo, cz_edges[i], cz_edges[i+1])[1] for i in 1:length(cz_edges)-1]
        @test check_exposure_normalisation(b, cz_edges, expo, 3.7 .* w) === nothing   # any common factor
        w_bad = copy(w); w_bad[2] *= 1.2
        @test_throws ErrorException check_exposure_normalisation(b, cz_edges, expo, w_bad)
    end

    @testset "day-night factorisation error is small" begin
        # The fitter forms P_N = P̄_D + c₁₃² ⟨cos2θ₁₂ᵐ⟩ (P̄₁ₑ − c₁₂²c₁₃²) from separate bin
        # averages (P̄_D, ⟨cos2θ₁₂ᵐ⟩ from the two E_calc points per bin). The exact bin average of
        # P_N also has the within-bin covariance of cos2θ₁₂ᵐ and P₁ₑ. Measured ≤ 3e-4 at the
        # prior edge (Δm²₂₁ = 2e-5), ≤ 1e-4 at 6e-5.
        n_e = 90.0
        for dm2 in (2e-5, 6e-5)
            pars = Osc.oscPars(dm2, asin(sqrt(0.307)), asin(sqrt(0.022)))
            c13sq = cos(pars.θ₁₃)^2; P0 = cos(pars.θ₁₂)^2 * c13sq
            worst = 0.0
            for (lo, hi) in ((-0.03, -0.0225), (-0.40, -0.35)), (a, bb) in ((1e-3, 1.5e-3), (4e-3, 4.5e-3), (10e-3, 10.5e-3))
                cs = [lo + (k - 0.5) * (hi - lo) / 24 for k in 1:24]
                paths = [build_earth_path(c, earth_model; max_step_km=100.0) for c in cs]
                Es = [a + (m - 0.5) * (bb - a) / 200 for m in 1:200]
                P1e = vec(sum(p1e_paths(Es, dm2, pars.θ₁₂, pars.θ₁₃, paths; phase_const=_BN_PHASE), dims=1)) ./ 24
                Pd, c2m = Osc.mswProb(Es, pars, n_e)
                exact = sum(Pd .+ c13sq .* c2m .* (P1e .- P0)) / 200
                Pd2, c2m2 = Osc.mswProb([a + (bb - a) / 4, a + 3(bb - a) / 4], pars, n_e)
                fact = sum(Pd2) / 2 + c13sq * sum(c2m2) / 2 * (sum(P1e) / 200 - P0)
                worst = max(worst, abs(fact - exact))
            end
            @test worst < 5e-4
        end
    end

    @testset "unsafe Float64-word paths equal the generic loops" begin
        # _interp_nodes! and _contract_last! reinterpret arrays of Float64, ComplexF64 and duals
        # as Float64 words. Compare with the generic loops on the same data.
        D = ForwardDiff.Dual{Nothing,Float64,3}
        rnd(::Type{Float64}, d...) = randn(d...)
        rnd(::Type{ComplexF64}, d...) = randn(ComplexF64, d...)
        rnd(::Type{D}, d...) = map(_ -> D(randn(), ForwardDiff.Partials(Tuple(randn(3)))), zeros(d...))
        rnd(::Type{Complex{D}}, d...) = complex.(rnd(D, d...), rnd(D, d...))
        nq, nn = 7, 10
        lag_first = rand(1:nn-NightOsc.E_ORDER+1, nq)
        lag_w = randn(nq, NightOsc.E_ORDER)
        B = zeros(5, 6); for t in 1:5, p in t:min(t+2, 6); B[t, p] = randn(); end
        for E in (Float64, ComplexF64, D, Complex{D})
            @test NightOsc._f64(E)
            inp = rnd(E, 4, nn, 2)
            fast = NightOsc._interp_nodes!(Array{E}(undef, 4, nq, 2), inp, lag_first, lag_w)
            slow = NightOsc._interp_loop!(Array{E}(undef, 4, nq, 2), inp, lag_first, lag_w)
            @test fast == slow
            inp2 = rnd(E, 3, 4, 6)
            fast2 = NightOsc._contract_last!(Array{E}(undef, 3, 4, 5), inp2, B)
            slow2 = reshape(NightOsc._contract_loop!(Array{E}(undef, 12, 5), reshape(inp2, 12, 6), B), 3, 4, 5)
            @test fast2 == slow2
        end
        @test !NightOsc._f64(Float32)
    end

    @testset "per-task stores are released with the engine" begin
        # The engine is built and used inside a function, so nothing in the test keeps it alive.
        n0 = length(get!(NightOsc._StoreDict, task_local_storage(), :night_osc_store))
        @test _use_temporary_engine(expo, earth_model) == n0 + 1
        GC.gc(true); GC.gc(true)
        @test length(task_local_storage(:night_osc_store)) == n0
    end
end
