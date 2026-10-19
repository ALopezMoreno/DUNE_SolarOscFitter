using Test
include(joinpath(@__DIR__, "test_tiers.jl"))   # TEST_TIER / RUN_SLOW (JULIA_TESTS=fast|slow)
@info "Test tier: $TEST_TIER" * (RUN_SLOW ? "" : " (JULIA_TESTS=slow adds the Enzyme unit tests, warm start and derived mode)") *
      (get(ENV, "JULIA_REGRESSION", "0") == "1" ? "; with the likelihood regression" : "; JULIA_REGRESSION=1 adds the likelihood regression")

# Bring in likelihood implementation
include(joinpath(@__DIR__, "..", "src", "core.jl"))   # _params_by_prefix + physics constants
include(joinpath(@__DIR__, "..", "src", "likelihoods", "likelihood_core.jl"))
include(joinpath(@__DIR__, "..", "src", "likelihoods", "likelihood_debug.jl"))
include(joinpath(@__DIR__, "..", "src", "likelihoods", "likelihood_statistical.jl"))
include(joinpath(@__DIR__, "..", "src", "likelihoods", "likelihood_builder.jl"))

# Unit tests for correctness
include(joinpath(@__DIR__, "test_likelihood_correctness.jl"))
include(joinpath(@__DIR__, "test_config_and_backgrounds.jl"))
include(joinpath(@__DIR__, "test_run_record.jl"))
include(joinpath(@__DIR__, "test_es_xsec.jl"))
include(joinpath(@__DIR__, "test_cc_xsec.jl"))
include(joinpath(@__DIR__, "test_osc_night.jl"))
include(joinpath(@__DIR__, "test_tabulated.jl"))
include(joinpath(@__DIR__, "test_response_and_backgrounds.jl"))
include(joinpath(@__DIR__, "test_input_checks.jl"))
include(joinpath(@__DIR__, "test_mcstat.jl"))
# test_warm_start.jl sets fitter globals in Main (posterior, init, warm_start, …): keep it last.
# Slow tier: it builds BAT samplers and runs short chains (~30 s).
RUN_SLOW && include(joinpath(@__DIR__, "test_warm_start.jl"))

# End-to-end test of derived mode on the two-detector fixtures (own process: it runs the full
# setup; ~1.5 min). Slow tier; JULIA_SKIP_DERIVED=1 skips it there too.
if RUN_SLOW && get(ENV, "JULIA_SKIP_DERIVED", "0") != "1"
    @testset "derived mode (end to end)" begin
        cmd = `$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) $(joinpath(@__DIR__, "derived_mode.jl"))`
        @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
    end
elseif RUN_SLOW
    @info "Skipping derived-mode test (JULIA_SKIP_DERIVED=1)"
end

# End-to-end likelihood regression (full setup, one process per variant; a few minutes)
if get(ENV, "JULIA_REGRESSION", "0") == "1"
    @testset "likelihood regression" begin
        for variant in ("night", "legacy", "night_da", "night_bb", "night_da_indep")
            cmd = `$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) $(joinpath(@__DIR__, "regression_likelihood.jl")) $variant`
            @test success(pipeline(cmd; stdout=stdout, stderr=stderr))
        end
    end
else
    @info "Skipping likelihood regression (set JULIA_REGRESSION=1 to enable)"
end

# Benchmarks / speed tests (optional; see file for ENV guard)
include(joinpath(@__DIR__, "benchmark_likelihoods.jl"))

# Profiling / speed tests (optional; see file for ENV guard)
include(joinpath(@__DIR__, "test_profile.jl"))