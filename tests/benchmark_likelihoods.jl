# Run with:  JULIA_LIKELIHOOD_BENCH=1 julia --project -e 'using Pkg; Pkg.test()'
using BenchmarkTools

# Only run benchmarks when explicitly asked, to keep normal CI fast
if get(ENV, "JULIA_LIKELIHOOD_BENCH", "0") == "1"
    @info "Running likelihood benchmarks (JULIA_LIKELIHOOD_BENCH=1)"

    # --- simple benchmark for poissonLogLikelihood ------------------------

    @info "Benchmark: poissonLogLikelihood on length 10_000 vectors"
    let
        n = 10_000
        nExpected = rand(10.0:50.0, n) .|> float
        nMeasured = rand(10.0:50.0, n) .|> float

        b = @benchmark poissonLogLikelihood($nExpected, $nMeasured)
        @info "poissonLogLikelihood median time (ms):" median(b).time / 1e6
        @info "poissonLogLikelihood memory (Kb):" median(b).memory / 1e3
    end

    # --- benchmark ES + CC likelihood with “realistic-ish” sizes ---------

    @info "Benchmark: total likelihood (ES+CC) with fake (200x200) sizes"
    let
        # fake but larger arrays to simulate real analysis sizes
        n_bins_ES = 200
        n_bins_CC = 200
        n_zenith  = 200

        nObserved = (
            ES_day   = rand(10.0:50.0, n_bins_ES),
            ES_night = rand(10.0:50.0, n_zenith, n_bins_ES),
            CC_day   = rand(10.0:50.0, n_bins_CC),
            CC_night = rand(10.0:50.0, n_zenith, n_bins_CC),
        )

        # fake propagate that returns something proportional to a parameter
        fake_f = function (MC_no_osc, Mreco, parameters, SSM, energies, BG)
            scale = get(parameters, :theta, 1.0)
            ES_day   = fill(scale, n_bins_ES)
            CC_day   = fill(scale, n_bins_CC)
            ES_night = fill(scale, n_zenith, n_bins_ES)
            CC_night = fill(scale, n_zenith, n_bins_CC)
            BG_ES    = zeros(n_bins_ES)
            BG_CC    = zeros(n_bins_CC)
            return ES_day, CC_day, ES_night, CC_night, BG_ES, BG_CC
        end

        inputs = LikelihoodInputs(
            nObserved,
            nothing,   # bin_edges
            nothing,   # responseMatrices
            nothing,   # solarModel
            nothing,   # unoscillatedSample
            nothing,   # backgrounds
            fake_f,
            true,      # ES_mode
            true,      # CC_mode
            1,         # index_ES
            1,         # index_CC
        )

        total_llh = make_likelihood(inputs;
            use_ES = true,
            use_CC = true,
            ES_llh = llh_ES_poisson,
            CC_llh = llh_CC_poisson,
        )

        params = Dict(:theta => 1.0)

        b = @benchmark $total_llh($params)
        @info "total_llh median time (ms):"  median(b).time / 1e6
        @info "total_llh memory (Kb):"    median(b).memory /1e3
    end
else
    @info "Skipping likelihood benchmarks (set JULIA_LIKELIHOOD_BENCH=1 to enable)"
end