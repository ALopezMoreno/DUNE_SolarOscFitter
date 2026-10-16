#=
derived_mode.jl — end-to-end test of RunMode "derived" (src/derive_variables_from_chain.jl).

Writes a tiny synthetic chain (three weighted steps in two batches) for the two-detector
fixture config (tests/fixtures/regression_config.yaml), runs derived mode on it, and checks the
output against values computed here directly from `expected_rates` and the per-bin likelihood:
per-step day-night asymmetries, weighted posterior-predictive means, variances and quantile
bands, background and signal-only arrays, per-bin log-L means, variances and correlations, the
CC background components, the shared best-fit oscillation map and the file layout, plus the
per-bin breakdown CSVs written at setup (src/inputChecks.jl). A third,
purely inclusive detector (FXC, no CC channel) is added to this test's copy of the config. If python3 with h5py is
available it also checks that utils/posteriorHelpers.py reads each detector's values back and
rejects an unknown detector.

    julia --project tests/derived_mode.jl        (~1–2 min; run by tests/runtests.jl)
=#
using Test, YAML, JLD2, CSV, DataFrames

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
# A third, purely inclusive detector (no CC channel), only in this test's copy of the config.
cfg["detectors"]["FXC"] = merge(deepcopy(cfg["detectors"]["FXB"]),
                                Dict("inclusiveMode" => true, "semiInclusiveMode" => false))
cfg["active_detectors"] = ["FXA", "FXB", "FXC"]
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
# Weighted correlation of per-bin values xs with a parameter θs (xs centred first, for precision).
function wcorr(xs, θs)
    c  = [x .- xs[1] for x in xs]
    cov = wmean([ci .* θ for (ci, θ) in zip(c, θs)]) .- wmean(c) .* wmean(θs)
    return cov ./ sqrt.(max.(wvar(c), 1e-30) .* max(wvar(θs), 1e-30))
end
# Smallest value whose cumulative weight fraction reaches q.
function wquantile(vals, q)
    o = sortperm(vals); c = cumsum(w[o]) ./ sum(w)
    return vals[o[clamp(searchsortedfirst(c, q), 1, length(vals))]]
end

chains_mc(has_CC) = has_CC ? (("ES", 0), ("CC", 0)) : (("ES", 0),)

function asym(day, night, bg, idx)
    b = sum(bg[idx:end])
    D = sum(day[(ndims(day) == 1 ? (idx:length(day),) : (:, idx:size(day, 2)))...]) - 0.5b
    N = sum(night[:, idx:end, (ndims(night) == 3 ? (:,) : ())...]) - 0.5b
    return 2 * (D - N) / (D + N)
end
above(x, idx) = (y = copy(x); ndims(y) == 1 ? (y[1:idx-1] .= 0) : (y[:, 1:idx-1, (ndims(y) == 3 ? (:,) : ())...] .= 0); y)

saved = jldopen(f -> Dict(k => f[k] for k in keys(f) if startswith(k, "derived_")), chain * ".jld2")

@testset "derived mode, three detectors" begin
    dets = Base.invokelatest(() -> collect(keys(detector_outputs)))
    @test saved["derived_detectors"] == join(dets, ",") == "FXA,FXB,FXC"
    θ1s = [p.sin2_th12 for p in pars]; θ2s = [p.dm2_21 for p in pars]
    truth = Base.invokelatest(() -> true_params)

    for dname in dets
        out = Base.invokelatest(() -> detector_outputs[dname])
        li  = out.likelihood_inputs
        cfgd = Base.invokelatest(() -> detector_configs[dname])
        rates  = [Base.invokelatest(expected_rates, li, p) for p in pars]
        perbin = [Base.invokelatest(detector_perbin_fns[dname], p) for p in pars]
        key(k) = "derived_$(dname)_$k"

        has_CC = !cfgd.inclusive_analysis
        chans  = has_CC ? (("ES", li.index_ES), ("CC", li.index_CC)) : (("ES", li.index_ES),)
        r_true = Base.invokelatest(expected_rates, li, truth)

        @testset "$dname" begin
            # per-step asymmetries (every step, unweighted)
            @test saved[key("ES_asymmetry")] ≈ [asym(r.ES_day, r.ES_night, r.BG_ES_tot, li.index_ES) for r in rates] rtol=1e-10
            if has_CC
                @test saved[key("CC_asymmetry")] ≈ [asym(r.CC_day, r.CC_night, r.BG_CC_tot, li.index_CC) for r in rates] rtol=1e-10
            else
                @test !any(k -> startswith(k, key("CC")), keys(saved))     # no CC channel, no CC keys
            end

            for (ch, idx) in chans
                # weighted posterior-predictive moments, zero below threshold
                for t in ("day", "night")
                    xs = [above(getfield(r, Symbol(ch, "_", t)), idx) for r in rates]
                    @test saved[key("$(ch)$(t)_pp_mean")] ≈ wmean(xs) rtol=1e-10
                    @test saved[key("$(ch)$(t)_pp_var")]  ≈ wvar(xs)  rtol=1e-6 atol=1e-6 * maximum(abs, wmean(xs))^2
                end
                BG = Symbol("BG_", ch, "_tot")
                bg_mean = wmean([0.5 .* getfield(r, BG) for r in rates])
                @test saved[key("BG$(ch)day_pp_mean")] ≈ bg_mean rtol=1e-10
                @test saved[key("BG$(ch)night_pp_mean")] ≈ exposure_weights .* bg_mean' rtol=1e-10

                # 1/2/3σ bands of (total day rate − posterior-mean background), per bin above threshold
                angular = ch == "ES" && cfgd.angular_reco
                tot(r) = angular ? vec(sum(r.ES_day, dims=1)) : getfield(r, Symbol(ch, "_day"))
                for (k, q) in (("lo", 0.1587), ("hi", 0.8413), ("lo3", 0.0013), ("hi3", 0.9987))
                    band = saved[key("$(ch)day_pp_$k")]
                    @test all(iszero, band[1:idx-1])
                    @test band[idx:end] ≈ [wquantile([tot(r)[j] for r in rates] .- bg_mean[j], q) for j in idx:length(band)] rtol=1e-10
                end

                # signal-only rates at the truth (angular ES keeps the background by design)
                if !angular
                    @test saved[key("signal_$(ch)day")] ≈ getfield(r_true, Symbol(ch, "_day")) .- 0.5 .* getfield(r_true, BG) rtol=1e-10
                end
            end

            # per-bin log-L: means (absolute scale), totals, variances, correlations with θ
            for (ch, f) in (has_CC ? (("ESday", :ES_day), ("ESnight", :ES_night), ("CCday", :CC_day), ("CCnight", :CC_night)) :
                                     (("ESday", :ES_day), ("ESnight", :ES_night)))
                xs = [getfield(pb, f) for pb in perbin]
                m = wmean(xs)
                @test saved[key("$(ch)_mean_llh")] ≈ m rtol=1e-8 atol=1e-8
                @test only(saved[key("$(ch)_tot_mean_llh")]) ≈ sum(m) rtol=1e-8 atol=1e-8
                @test saved[key("$(ch)_var_llh")] ≈ wvar([x .- xs[1] for x in xs]) rtol=1e-6 atol=1e-9
                @test saved[key("$(ch)_corr_llh_sin2_th12")] ≈ wcorr(xs, θ1s) rtol=1e-5 atol=1e-6
                @test saved[key("$(ch)_corr_llh_dm2_21")]    ≈ wcorr(xs, θ2s) rtol=1e-5 atol=1e-6
            end

            if has_CC
                # the CC background components add up to the total CC background (day, truth)
                comps = saved[key("CCbg_comp_day")]
                @test vec(sum(comps, dims=2)) ≈ 0.5 .* r_true.BG_CC_tot rtol=1e-10
                @test length(split(saved[key("CCbg_comp_names")], ",")) == size(comps, 2)
                @test saved[key("data_CCday")] == li.nObserved.CC_day
            end

            # setup's per-bin breakdown CSVs (src/inputChecks.jl): components add up to the total
            # background at the truth (day + all night bins; CC × backward fraction if semi-inclusive)
            nf = 0.5 * (1 + sum(exposure_weights))
            for (ch, _) in chans
                bd = CSV.read(joinpath(tmp, "derived_$(dname)_$(ch)_breakdown.csv"), DataFrame)
                bgcols = filter(c -> startswith(c, "bg_"), names(bd))
                f = ch == "CC" && cfgd.semi_inclusive_analysis ? nf * li.Mreco.CC_split.below : nf
                @test sum(bd[!, c] for c in bgcols) ≈ f .* getfield(r_true, Symbol("BG_", ch, "_tot")) rtol=1e-10
                # data = the Asimov data, summed over day and night (and over cos θ for angular ES)
                obs_d, obs_n = getfield(li.nObserved, Symbol(ch, "_day")), getfield(li.nObserved, Symbol(ch, "_night"))
                ang = ch == "ES" && cfgd.angular_reco
                @test bd.data ≈ (ang ? vec(sum(obs_d, dims=1)) .+ vec(sum(obs_n, dims=(1, 3))) :
                                       obs_d .+ vec(sum(obs_n, dims=1))) rtol=1e-12
                @test bd.in_fit == ((1:nrow(bd)) .>= getfield(li, Symbol("index_", ch)))
            end

            # MC statistics (src/likelihoods/mcstat.jl): the prediction rebuilt from its signal and
            # background components reproduces the Asimov data in every E_reco bin (day + night;
            # angular summed over cos θ).
            for (ch, _) in chains_mc(has_CC)
                mcc = out.mc_counts[Symbol(ch)]
                obs_d, obs_n = getfield(li.nObserved, Symbol(ch, "_day")), getfield(li.nObserved, Symbol(ch, "_night"))
                ang = ch == "ES" && cfgd.angular_reco
                data_col = ang ? vec(sum(obs_d, dims=1)) .+ vec(sum(obs_n, dims=(1, 3))) : obs_d .+ vec(sum(obs_n, dims=1))
                @test mcc.M ≈ data_col rtol=1e-8
                @test mcc.a ≈ mcc.M .^ 2 ./ mcc.V                                   # definition
            end

            # the fit starts at bin 2, whose lower edge is the threshold (Ereco_min_*)
            @test li.index_ES == 2 && li.index_CC == 2
            @test saved[key("Ereco_edges_ES")][2] ≈ cfg["detectors"][dname]["Ereco_min_ES"]
            has_CC && @test saved[key("Ereco_edges_CC")][2] ≈ cfg["detectors"][dname]["Ereco_min_CC"]
            @test saved[key("Ereco_edges_ES")][1] >= 0                     # underflow drawn from ≥ 0

            # layout entries
            @test saved[key("index_ES")] == li.index_ES && saved[key("index_CC")] == li.index_CC
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
        import sys, h5py, numpy as np
        sys.path.insert(0, sys.argv[1] + '/utils')
        import posteriorHelpers as ph
        chain = sys.argv[2]
        f = h5py.File(chain + '.jld2', 'r')
        vals = {}
        for det in ('FXA', 'FXB', 'FXC'):
            d = ph.load_bin_diagnostics([chain], detector=det, verbose=False)
            assert d['detector'] == det
            for k in ('derived_ESday_pp_mean', 'derived_ESnight_mean_llh'):
                assert np.allclose(np.squeeze(d[k]), np.squeeze(f['derived_' + det + '_' + k[8:]][()])), (det, k)
            vals[det] = np.squeeze(d['derived_ESday_pp_mean'])
            assert d['inclusive_mode'] == (det == 'FXC')
        assert not np.isclose(vals['FXA'].sum(), vals['FXB'].sum())       # detectors really differ
        for det in ('FXA', 'FXB'):
            p = ph.load_posterior([chain], ['sin2_th12'], burnin=0, exclude_chains=[],
                                  test=[('derived_CC_asymmetry', 'A')], detector=det)
            assert np.allclose(p['derived_CC_asymmetry'], f['derived_' + det + '_CC_asymmetry'][()])
        assert ph.load_bin_diagnostics([chain], verbose=False)['detector'] == 'FXA'
        try:
            ph.load_bin_diagnostics([chain], detector='NOPE', verbose=False); raise SystemExit('no error')
        except ValueError:
            pass
        print('python loaders ok')
        """
        @test success(pipeline(`python3 -c $py $ROOT $chain`; stdout=stdout, stderr=stderr))
    else
        @warn "python3 with h5py not available: skipping the posteriorHelpers check"
    end
end
