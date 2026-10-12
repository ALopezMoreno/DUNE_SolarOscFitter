using Test

# Bring in likelihood implementation
include(joinpath(@__DIR__, "..", "src", "core.jl"))   # _params_by_prefix + physics constants
include(joinpath(@__DIR__, "..", "src", "likelihoods", "likelihood_core.jl"))
include(joinpath(@__DIR__, "..", "src", "likelihoods", "likelihood_debug.jl"))
include(joinpath(@__DIR__, "..", "src", "likelihoods", "likelihood_statistical.jl"))
include(joinpath(@__DIR__, "..", "src", "likelihoods", "likelihood_builder.jl"))

# Unit tests for correctness
include(joinpath(@__DIR__, "test_likelihood_correctness.jl"))
include(joinpath(@__DIR__, "test_es_xsec.jl"))
include(joinpath(@__DIR__, "test_osc_night.jl"))

# End-to-end likelihood regression (full setup, one process per variant; a few minutes)
if get(ENV, "JULIA_REGRESSION", "0") == "1"
    @testset "likelihood regression" begin
        for variant in ("night", "legacy")
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
