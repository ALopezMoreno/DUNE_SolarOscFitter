using Test

# Bring in likelihood implementation
include(joinpath(@__DIR__, "..", "src", "likelihoods", "likelihood_core.jl"))
include(joinpath(@__DIR__, "..", "src", "likelihoods", "likelihood_debug.jl"))
include(joinpath(@__DIR__, "..", "src", "likelihoods", "likelihood_statistical.jl"))
include(joinpath(@__DIR__, "..", "src", "likelihoods", "likelihood_builder.jl"))

# Unit tests for correctness
include(joinpath(@__DIR__, "test_likelihood_correctness.jl"))

# Benchmarks / speed tests (optional; see file for ENV guard)
include(joinpath(@__DIR__, "benchmark_likelihoods.jl"))

# Profiling / speed tests (optional; see file for ENV guard)
include(joinpath(@__DIR__, "test_profile.jl"))