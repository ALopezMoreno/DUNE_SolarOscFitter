using Test

@testset "poissonLogLikelihood – correctness & edge cases" begin
    # simple case with all-positive entries where the formula is easy
    nExpected = [10.0, 20.0, 30.0, 15.0]
    nMeasured = [12.0, 18.0, 29.0, 14.0]

    # reference (your “deviance” without any of the 0.5 hacks needed for zeros)
    ref = sum((e - m) + m * log(m / e) for (e, m) in zip(nExpected, nMeasured))

    result = poissonLogLikelihood(nExpected, nMeasured)

    @test result ≈ -ref  atol=1e-10

    # m == e => deviance=0 => loglike should be 0
    n = [1.0, 5.0, 10.0]
    @test poissonLogLikelihood(n, n) ≈ 0.0 atol=1e-12

    # length mismatch
    @test_throws ArgumentError poissonLogLikelihood([1.0, 2.0], [1.0])

    # negative entries
    @test_throws ArgumentError poissonLogLikelihood([-1.0, 2.0], [1.0, 2.0])
    @test_throws ArgumentError poissonLogLikelihood([1.0, 2.0], [1.0, -2.0])

    # zeros allowed, but result must be finite
    nExpected2 = [0.0, 1.0, 2.0]
    nMeasured2 = [0.0, 0.0, 3.0]
    @test isfinite(poissonLogLikelihood(nExpected2, nMeasured2))
end

@testset "barlowBeestonLogLikelihood – consistency with Poisson" begin
    nExpected = [10.0, 20.0, 30.0]
    nMeasured = [9.0, 19.0, 31.0]
    sigmaVar  = zeros(length(nExpected))

    res_sys = barlowBeestonLogLikelihood(nExpected, nMeasured, sigmaVar)
    res_poi = poissonLogLikelihood(nExpected, nMeasured)

    @test res_sys ≈ res_poi atol=1e-10

    # basic sanity with non-zero sigma
    sigmaVar2 = [0.1, 0.2, 0.3]
    res_sys2 = barlowBeestonLogLikelihood(nExpected, nMeasured, sigmaVar2)
    @test isfinite(res_sys2)

    # length mismatch
    @test_throws ArgumentError barlowBeestonLogLikelihood(
        [1.0, 2.0], [1.0, 2.0], [0.1],
    )
end

@testset "check_earth_norm_bounds – behaviour" begin
    # If you have check_earth_norm_bounds in likelihood_debug.jl
    params_ok = Dict(
        :earth_norm_1 => 0.5,
        :earth_norm_2 => 1.9,
        :other        => 3.0,
    )
    @test check_earth_norm_bounds(params_ok)

    params_low = Dict(:earth_norm_1 => -0.01)
    params_high = Dict(:earth_norm_1 => 2.01)

    @test !check_earth_norm_bounds(params_low)
    @test !check_earth_norm_bounds(params_high)
end

# --- helpers for fake inputs -----------------------------------------------

"""
Build a tiny LikelihoodInputs with a fake propagate function and fake observed data.
This keeps tests fast and independent from the real MC.
"""
function fake_likelihood_inputs()
    # simple observed data
    nObserved = (
        ES_day   = [1.0, 2.0],
        ES_night = [1.0 2.0; 3.0 4.0],
        CC_day   = [3.0, 4.0],
        CC_night = [1.0 1.0; 2.0 2.0],
    )

    # fake propagate function: ignore parameters, just return fixed arrays
    fake_f = function (MC_no_osc, Mreco, parameters, SSM, energies, BG)
        ES_day   = [1.0, 2.0]
        CC_day   = [3.0, 4.0]
        ES_night = [1.0 2.0; 3.0 4.0]
        CC_night = [1.0 1.0; 2.0 2.0]
        BG_ES    = [0.1, 0.2]
        BG_CC    = [0.3, 0.4]
        return ES_day, CC_day, ES_night, CC_night, BG_ES, BG_CC
    end

    inputs = LikelihoodInputs(
        nObserved,     # ereco_data (you use it as nObserved)
        nothing,       # bin_edges
        nothing,       # responseMatrices
        nothing,       # solarModel
        nothing,       # unoscillatedSample
        nothing,       # backgrounds
        fake_f,        # propagateSamples
        true,          # ES_mode
        true,          # CC_mode
        1,             # index_ES
        1,             # index_CC
    )

    return inputs
end

@testset "expected_rates – wiring & shapes" begin
    inputs = fake_likelihood_inputs()
    params = Dict(:theta => 1.0)  # no earth_norm, so always allowed

    rates = expected_rates(inputs, params)

    # We know exactly what fake_f returns, so check equality
    @test rates.ES_day   == [1.0, 2.0]
    @test rates.CC_day   == [3.0, 4.0]
    @test rates.ES_night == [1.0 2.0; 3.0 4.0]
    @test rates.CC_night == [1.0 1.0; 2.0 2.0]
    @test rates.BG_ES_tot == [0.1, 0.2]
    @test rates.BG_CC_tot == [0.3, 0.4]
end

@testset "make_likelihood – ES/CC composition" begin
    inputs = fake_likelihood_inputs()
    params = Dict(:theta => 1.0)

    # full ES + CC likelihood
    total_llh = make_likelihood(inputs;
        use_ES = true,
        use_CC = true,
        ES_llh = llh_ES_poisson,
        CC_llh = llh_CC_poisson,
    )

    rates = expected_rates(inputs, params)
    ref_ES = llh_ES_poisson(inputs, params, rates)
    ref_CC = llh_CC_poisson(inputs, params, rates)

    @test total_llh(params) ≈ (ref_ES + ref_CC)

    # ES-only
    llh_ES_only = make_likelihood(inputs;
        use_ES = true,
        use_CC = false,
        ES_llh = llh_ES_poisson,
        CC_llh = llh_CC_poisson,
    )
    @test llh_ES_only(params) ≈ ref_ES

    # CC-only
    llh_CC_only = make_likelihood(inputs;
        use_ES = false,
        use_CC = true,
        ES_llh = llh_ES_poisson,
        CC_llh = llh_CC_poisson,
    )
    @test llh_CC_only(params) ≈ ref_CC
end

@testset "make_likelihood – earth_norm bounds via total llh" begin
    inputs = fake_likelihood_inputs()

    # build likelihood with defaults
    total_llh = make_likelihood(inputs;
        use_ES = true,
        use_CC = true,
        ES_llh = llh_ES_poisson,
        CC_llh = llh_CC_poisson,
    )

    params_ok   = Dict(:earth_norm_1 => 1.0)
    params_bad1 = Dict(:earth_norm_1 => -0.1)
    params_bad2 = Dict(:earth_norm_1 => 2.1)

    @test total_llh(params_ok)   > -Inf
    @test total_llh(params_bad1) == -Inf
    @test total_llh(params_bad2) == -Inf
end
