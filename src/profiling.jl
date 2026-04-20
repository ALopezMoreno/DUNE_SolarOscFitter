using BenchmarkTools
using Profile
using PProf
using ProfileSVG

using LinearAlgebra, Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets

include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "likelihoods", "likelihood_main.jl"))

@info "Running likelihood profiling (RunMode = PROFILE)"

params_prof = true_params

# --- Build partial likelihoods for ES-only and CC-only targets ----------------
es_only_llh = make_likelihood(likelihood_inputs; use_ES = true,  use_CC = false)
cc_only_llh = make_likelihood(likelihood_inputs; use_ES = false, use_CC = true)

# --- Prepare inputs for osc-only target ---------------------------------------
mixingPars_prof = get_mixing_parameters(true_params)

# --- Helper: warmup → clear → profile → save PProf + ProfileSVG --------------
function profile_target(f, label; n_warmup = 3, n_profile = 500)
    @info "Warming up: $label"
    for _ in 1:n_warmup
        f()
    end

    pb_path  = label * ".pb.gz"
    svg_path = label * ".svg"

    t_ms = @belapsed($f()) * 1e3
    @info "$label  wall time: $(round(t_ms, digits=2)) ms/call"

    @info "Profiling: $label  →  $pb_path  /  $svg_path"
    Profile.clear()
    @profile for _ in 1:n_profile
        f()
    end
    PProf.pprof(web = false, out = pb_path)
    Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do
        ProfileSVG.save(svg_path; maxdepth = 200, width = 12000)
    end
    Profile.clear()

    @info "Done: $label"
end

# --- Profile targets ----------------------------------------------------------

profile_target(
    () -> total_llh(params_prof),
    "profile_total_llh";
    n_profile = 500,
)

profile_target(
    () -> es_only_llh(params_prof),
    "profile_es_only_llh";
    n_profile = 500,
)

profile_target(
    () -> cc_only_llh(params_prof),
    "profile_cc_only_llh";
    n_profile = 500,
)

profile_target(
    () -> osc_prob_earth(E_calc, mixingPars_prof, earth_lookup, earth_paths),
    "profile_osc_earth";
    n_profile = 1000,
)

@info "Profiling complete. Outputs written to the working directory."
