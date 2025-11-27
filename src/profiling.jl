using BenchmarkTools
using Profile
using PProf

using LinearAlgebra, Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets

include(joinpath(@__DIR__, "..", "src", "setup.jl"))
include(joinpath(@__DIR__, "..", "src", "likelihoods", "likelihood_main.jl"))

@info "Running likelihood benchmark/profiling (RunMode = BENCH)"

# Parameters: use the NamedTuple, not the Dict
params_prop = true_params   # for both total_llh and propagateSamples

# ---------------- Warmup (avoid measuring compilation time) ----------------

total_llh(params_prop)
propagateSamples(unoscillatedSample, responseMatrices,
                 params_prop, solarModel, bin_edges, backgrounds)


# ---------------- CPU profiling → PProf format ----------------

# Profile total_llh
@info "Profiling total_llh(parameters) → profile_total_llh.pb.gz"
Profile.clear()
@profile for i in 1:3000
    total_llh(params_prop)
end
PProf.pprof(web = false, out = "profile_total_llh.pb.gz")
@info "Wrote profile_total_llh.pb.gz (use `pprof` CLI to inspect)"

# Profile propagateSamples
@info "Profiling propagateSamples(...) → profile_propagateSamples.pb.gz"
Profile.clear()
@profile for i in 1:3000
    propagateSamples(unoscillatedSample,
                     responseMatrices,
                     params_prop,
                     solarModel,
                     bin_edges,
                     backgrounds)
end
PProf.pprof(web = false, out = "profile_propagateSamples.pb.gz")
@info "Wrote profile_propagateSamples.pb.gz (use `pprof` CLI to inspect)"

@info "Benchmark/profiling complete. Exiting."