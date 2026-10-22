using Test
using DelimitedFiles, Random

include(joinpath(@__DIR__, "..", "src", "modelInputs.jl"))   # load_earth_model, prem.jl

# Analytic PREM with continental crust (src/prem.jl) against parchita, and as the Earth of both
# night calculations. Reference: tests/fixtures/prem_parchita.csv (make_prem_fixture.py).
@testset "PREM Earth model" begin
    @testset "profile against parchita" begin
        ref = readdlm(joinpath(@__DIR__, "fixtures", "prem_parchita.csv"), ','; skipstart=1)
        n_e = prem_electron_density.(ref[:, 1])
        @test maximum(abs.(n_e ./ ref[:, 2] .- 1)) < 1e-12
    end

    @testset "jumps and layers" begin
        @test length(PREM_BOUNDARIES_KM) == 7 && issorted(PREM_BOUNDARIES_KM)
        for (k, b) in enumerate(PREM_BOUNDARIES_KM)
            @test prem_region(b - 1e-6) == k && prem_region(b) == k + 1
            @test abs(prem_density(b, k + 1) - prem_density(b, k)) > 0.05
        end
        # no jump inside the transition zone, where the polynomial changes
        @test prem_density(PREM_R_TZ - 1e-9) ≈ prem_density(PREM_R_TZ) atol = 1e-4
        @test prem_density(PREM_RADIUS_KM - 1) == 2.6     # upper crust, not ocean

        m = load_earth_model("PREM")
        @test all(diff(m[:radius]) .> 0)
        @test m[:boundaries] == PREM_BOUNDARIES_KM
        earth = night_earth_model("PREM")
        @test NightOsc.n_layers(earth) == 8
        @test earth.layer_boundaries == PREM_BOUNDARIES_KM
        r = rand(MersenneTwister(1), 500) .* PREM_RADIUS_KM
        v = [NightOsc.potential(earth, ri) for ri in r]
        @test maximum(abs.(v ./ (MATTER_POTENTIAL_CONST .* prem_electron_density.(r)) .- 1)) < 1e-6
    end

    @testset "makePaths and NightOsc agree on PREM" begin
        tab = create_interpolated_model(load_earth_model("PREM"))
        @test check_layer_agreement(night_earth_model("PREM"), cz -> make_potential_for_integrand(cz, tab), depth) === nothing
        path = make_potential_for_integrand(-0.95, tab)
        @test [Int(s.index) for s in path.segments] == [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]
    end
end
