#=
derived_mode.jl — end-to-end test of RunMode "derived" (src/derive_variables_from_chain.jl).

Writes a tiny synthetic chain (three weighted steps in two batches) for the two-detector
fixture config (tests/fixtures/regression_config.yaml), runs derived mode on it, and checks the
output against values computed here directly from `expected_rates` and the per-bin likelihood:
per-step day-night asymmetries, weighted posterior-predictive means and variances, per-bin
log-L means, the shared best-fit oscillation map and the file layout. If python3 with h5py is
available it also checks that utils/posteriorHelpers.py reads each detector back.

    julia --project tests/derived_mode.jl        (~1–2 min; run by tests/runtests.jl)
=#
using Test, YAML, JLD2

const ROOT = normpath(joinpath(@__DIR__, ".."))
tmp = mktempdir()
chain = joinpath(tmp, "chain")

# --- the synthetic chain: truth, a displaced point, truth again; BAT-style integer weights ---
cfg = YAML.load_file(joinpath(ROOT, "tests", "fixtures", "regression_config.yaml"))
p_truth = Dict("sin2_th12" => cfg["true_sin2_th12"], "dm2_21" => cfg["true_dm2_21"],
               "FXA_CC_bg_norm_1" => 2.2e-6)
p_moved = Dict("sin2_th12" => cfg["true_sin2_th12"] + 0.02, "dm2_21" => 1.2 * cfg["true_dm2_21"],
               "FXA_CC_bg_norm_1" => 1.05 * 2.2e-6)
steps   = [p_truth, p_moved, p_truth]
weights = [2, 1, 3]
jldopen(chain * ".jld2", "w") do f
    for (b, idx) in (("batch_1", 1:2), ("batch_2", 3:3))
        g = JLD2.Group(f, b)
        for k in keys(p_truth)
            g[k] = [steps[i][k] for i in idx]
        end
        g["weights"] = weights[idx]
        g["stepno"]  = collect(idx)
        g["chainid"] = fill(1, length(idx))
    end
end

merge!(cfg, Dict{String,Any}("RunMode" => "derived", "prevFile" => chain, "thinning" => 1,
                             "outFile" => joinpath(tmp, "derived"), "earth_potential_uncertainties" => true))
cfg_file = joinpath(tmp, "derived.yaml")
YAML.write_file(cfg_file, cfg)

cd(ROOT)
empty!(ARGS); push!(ARGS, cfg_file)
include(joinpath(ROOT, "src", "readConfig.jl"))
main()

# --- reference values, computed independently of the derived-mode aggregation ---
pars = [Base.invokelatest(() -> merge(true_params, NamedTuple{Tuple(Symbol.(keys(s)))}(Tuple(values(s)))))
        for s in steps]
w = Float64.(weights)
wmean(xs) = sum(w .* xs) / sum(w)
wvar(xs)  = max.(wmean([x .^ 2 for x in xs]) .- wmean(xs) .^ 2, 0.0)

function asym(day, night, bg, idx)
    b = sum(bg[idx:end])
    D = sum(day[(ndims(day) == 1 ? (idx:length(day),) : (:, idx:size(day, 2)))...]) - 0.5b
    N = sum(night[:, idx:end, (ndims(night) == 3 ? (:,) : ())...]) - 0.5b
    return 2 * (D - N) / (D + N)
end
above(x, idx) = (y = copy(x); ndims(y) == 1 ? (y[1:idx-1] .= 0) : (y[:, 1:idx-1, (ndims(y) == 3 ? (:,) : ())...] .= 0); y)

saved = jldopen(f -> Dict(k => f[k] for k in keys(f) if startswith(k, "derived_")), chain * ".jld2")

@testset "derived mode, two detectors" begin
    dets = Base.invokelatest(() -> collect(keys(detector_outputs)))
    @test saved["derived_detectors"] == join(dets, ",") == "FXA,FXB"

    for dname in dets
        out = Base.invokelatest(() -> detector_outputs[dname])
        li  = out.likelihood_inputs
        cfgd = Base.invokelatest(() -> detector_configs[dname])
        rates  = [Base.invokelatest(expected_rates, li, p) for p in pars]
        perbin = [Base.invokelatest(detector_perbin_fns[dname], p) for p in pars]
        key(k) = "derived_$(dname)_$k"

        @testset "$dname" begin
            # per-step asymmetries (every step, unweighted)
            @test saved[key("ES_asymmetry")] ≈ [asym(r.ES_day, r.ES_night, r.BG_ES_tot, li.index_ES) for r in rates] rtol=1e-10
            @test saved[key("CC_asymmetry")] ≈ [asym(r.CC_day, r.CC_night, r.BG_CC_tot, li.index_CC) for r in rates] rtol=1e-10

            # weighted posterior-predictive moments, zero below threshold
            for (ch, idx) in (("ES", li.index_ES), ("CC", li.index_CC)), t in ("day", "night")
                xs = [above(getfield(r, Symbol(ch, "_", t)), idx) for r in rates]
                @test saved[key("$(ch)$(t)_pp_mean")] ≈ wmean(xs) rtol=1e-10
                @test saved[key("$(ch)$(t)_pp_var")]  ≈ wvar(xs)  rtol=1e-6 atol=1e-6 * maximum(abs, wmean(xs))^2
            end
            @test saved[key("BGCCday_pp_mean")] ≈ wmean([0.5 .* r.BG_CC_tot for r in rates]) rtol=1e-10

            # per-bin log-L means (absolute scale) and the total
            for (ch, f) in (("ESday", :ES_day), ("ESnight", :ES_night), ("CCday", :CC_day), ("CCnight", :CC_night))
                m = wmean([getfield(pb, f) for pb in perbin])
                @test saved[key("$(ch)_mean_llh")] ≈ m rtol=1e-8 atol=1e-8
                @test only(saved[key("$(ch)_tot_mean_llh")]) ≈ sum(m) rtol=1e-8 atol=1e-8
            end

            # truth-level and layout entries
            @test saved[key("index_ES")] == li.index_ES && saved[key("index_CC")] == li.index_CC
            @test saved[key("data_CCday")] == li.nObserved.CC_day
            @test length(saved[key("Ereco_edges_ES")]) == cfgd.Ereco_bins_ES.bin_number + 1
            @test saved[key("inclusive_mode")] == cfgd.inclusive_analysis
        end
    end

    # shared: best-fit oscillation map at the weighted posterior mean of the sampled parameters
    pmean = Dict(k => wmean([s[k] for s in steps]) for k in keys(p_truth))
    best  = Base.invokelatest(() -> merge(true_params, NamedTuple{Tuple(Symbol.(keys(pmean)))}(Tuple(values(pmean)))))
    osc   = Base.invokelatest(compute_shared_osc_probs, best, solarModel)
    @test saved["derived_oscmap_8B_day"] ≈ collect(osc.nue_8B_day) rtol=1e-10
    @test !haskey(saved, "derived_ES_asymmetry")          # per-detector keys only carry a prefix

    # the Python loaders read each detector back (skipped if h5py is not available)
    if success(pipeline(`python3 -c "import h5py, numpy"`; stderr=devnull))
        py = """
        import sys; sys.path.insert(0, sys.argv[1] + '/utils')
        import numpy as np, posteriorHelpers as ph
        for det in ('FXA', 'FXB'):
            d = ph.load_bin_diagnostics([sys.argv[2]], detector=det, verbose=False)
            assert d['detector'] == det
            p = ph.load_posterior([sys.argv[2]], ['sin2_th12'], burnin=0, exclude_chains=[],
                                  test=[('derived_CC_asymmetry', 'A')], detector=det)
            assert p['derived_CC_asymmetry'].shape == (3,), p['derived_CC_asymmetry'].shape
        assert ph.load_bin_diagnostics([sys.argv[2]], verbose=False)['detector'] == 'FXA'
        print('python loaders ok')
        """
        @test success(pipeline(`python3 -c $py $ROOT $chain`; stdout=stdout, stderr=stderr))
    else
        @warn "python3 with h5py not available: skipping the posteriorHelpers check"
    end
end
