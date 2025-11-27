using Test

if get(ENV, "JULIA_PROFILE", "0") == "1"
    @info "Running test_profile (JULIA_PROFILE=1)"

    # Emulate: julia readConfig.jl profilingConfig.yaml
    empty!(ARGS)
    push!(ARGS, "configs/profilingConfig.yaml")

    include(joinpath(@__DIR__, "..", "src", "readConfig.jl"))
else
    @info "Skipping test_profile (set JULIA_PROFILE=1 to enable)"
end