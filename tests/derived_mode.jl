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

# SPEC 4.6: an exposure table whose two weightings differ strongly, so that a signal/background mix-up
# shows: the time column of inputs/solarExposure_v2.csv, and a flux column tilted by (1 + 0.6 cos z).
expo_rows = [parse.(Float64, split(l, ',')) for l in eachline(joinpath(ROOT, "inputs", "solarExposure_v2.csv"))
             if !isempty(strip(l)) && !startswith(strip(l), "#") && tryparse(Float64, split(l, ',')[1]) !== nothing]
expo_c  = getindex.(expo_rows, 1)
expo_wt = getindex.(expo_rows, 2)
expo_wf = expo_wt .* (1 .+ 0.6 .* expo_c)
expo_file = joinpath(tmp, "exposure_tilted.csv")
open(expo_file, "w") do io
    println(io, "cosz,time_weight,flux_weight")
    for i in eachindex(expo_c)
        println(io, repr(expo_c[i]), ",", repr(expo_wt[i]), ",", repr(expo_wf[i]))
    end
end
merge!(cfg, Dict{String,Any}("RunMode" => "derived", "prevFile" => chain, "thinning" => 1,
                             "outFile" => joinpath(tmp, "derived"), "earth_potential_uncertainties" => true,
                             "solar_exposure_file" => expo_file))
# A third, purely inclusive detector (no CC channel), only in this test's copy of the config.
cfg["detectors"]["FXC"] = merge(deepcopy(cfg["detectors"]["FXB"]),
                                Dict("inclusiveMode" => true, "semiInclusiveMode" => false))
cfg["active_detectors"] = ["FXA", "FXB", "FXC"]
# Left to their (identical) defaults, so the run record has to fill them in.
delete!(cfg, "TPC_threshold"); delete!(cfg, "prior_cc_xsec_norm")
cfg_file = joinpath(tmp, "derived.yaml")
YAML.write_file(cfg_file, cfg)

cd(ROOT)
include(joinpath(ROOT, "src", "readConfig.jl"))

# A new MCMC run onto an existing chain file must stop before it touches that chain's run record.
@testset "aborted run keeps the existing record" begin
    d = mktempdir()
    mcfg = merge(deepcopy(cfg), Dict{String,Any}("RunMode" => "MCMC", "outFile" => joinpath(d, "run")))
    delete!(mcfg, "prevFile")
    YAML.write_file(joinpath(d, "mcmc.yaml"), mcfg)
    write(joinpath(d, "run.jld2"), "existing chain")
    write(joinpath(d, "run_config.yaml"), "existing record")
    empty!(ARGS); push!(ARGS, joinpath(d, "mcmc.yaml"))
    @test_throws ErrorException main()
    @test read(joinpath(d, "run_config.yaml"), String) == "existing record"
    @test !isfile(joinpath(d, "run_config.old1.yaml"))

    # derived mode without a usable chain stops before setup, leaving the record alone
    for pf in (nothing, joinpath(d, "nochain"), joinpath(d, "run.jld2"))
        dcfg = merge(deepcopy(mcfg), Dict{String,Any}("RunMode" => "derived"))
        pf === nothing || (dcfg["prevFile"] = pf)
        YAML.write_file(joinpath(d, "derived.yaml"), dcfg)
        empty!(ARGS); push!(ARGS, joinpath(d, "derived.yaml"))
        @test_throws ErrorException main()
    end
    @test read(joinpath(d, "run_config.yaml"), String) == "existing record"
    @test !any(f -> occursin("old", f) || occursin("derived_config", f), readdir(d))
end

empty!(ARGS); push!(ARGS, cfg_file)
main()

# --- SPEC 3.3 on the real setup weights (unoscillated CC ⁸B rate per E_true bin) ---
include(joinpath(@__DIR__, "cc_xsec_shape_checks.jl"))
@testset "CC xsec shape basis on the setup weights (SPEC 3.3)" begin
    E = Base.invokelatest(() -> calculate_bins(Etrue_bins)[2])
    ncc = 0
    for dname in Base.invokelatest(() -> collect(keys(detector_outputs)))
        li = Base.invokelatest(() -> detector_outputs[dname].likelihood_inputs)
        li.CC_mode || continue
        ncc += 1
        U = li.MC_no_osc
        B = Base.invokelatest(check_cc_xsec_shape_spec, E, U.CC_8B)
        @test U.xsec_p1 == B.p1 && U.xsec_p2 == B.p2           # the basis the likelihood uses
        # and the propagation applies it to CC ⁸B and hep, day and night (unit oscillation probs)
        nz = size(Base.invokelatest(() -> exposure_weights_signal), 1)
        o1, oN = ones(length(E)), ones(nz, length(E))
        probs = (nue_8B_day = o1, nue_hep_day = o1, nue_8B_night = oN, nue_hep_night = oN)
        par(t, c) = Base.invokelatest(() -> merge(true_params, (cc_xsec_norm = 1.0, cc_xsec_tilt = t, cc_xsec_curv = c)))
        cc(t, c) = Base.invokelatest(compute_oscillated_samples, U, par(t, c), probs; es_mode=false, cc_mode=true).CC
        c0, c1 = cc(0.0, 0.0), cc(0.15, -0.2)
        shape = Base.invokelatest(cc_xsec_shape, 1.0, 0.15, -0.2, B.p1, B.p2)
        nzb = c0.day .> 0
        @test c1.day[nzb] ≈ (c0.day .* shape)[nzb] rtol=1e-12
        @test c1.night ≈ c0.night .* shape' rtol=1e-12
    end
    @test ncc > 0
end

# --- the run record (src/runRecord.jl): the config used, defaults filled in, and provenance ---
@testset "run record of the derived run" begin
    # derived mode writes into prevFile.jld2, so its record sits next to the chain
    @test !isfile(joinpath(tmp, "derived_config.yaml"))
    rec = YAML.load_file(chain * "_derived_config.yaml")
    @test rec["provenance"]["config_file"] == abspath(cfg_file)
    @test rec["provenance"]["run_mode"] == "derived"
    # every input file is pinned by its hash
    sha = rec["provenance"]["input_sha256"]
    for f in (cfg["solar_model_file"], cfg["flux_model_file"], cfg["earth_model_file"])
        @test sha[f] == bytes2hex(open(SHA.sha256, f))
    end
    res, dflt = rec["config"], rec["defaults_applied"]
    for (k, v) in cfg                                     # the file's own values round-trip
        k == "detectors" && continue
        @test res[k] == v
    end
    for (n, b) in cfg["detectors"], (k, v) in b
        @test res["detectors"][n][k] == v
    end
    @test isempty(intersect(keys(dflt), setdiff(keys(cfg), ["detectors"])))
    # the defaults recorded are the values the run used
    g(s) = Base.invokelatest(getglobal, Main, s)
    @test dflt["TPC_threshold"] * 1e-3 ≈ g(:tpc_threshold) rtol=1e-14
    @test Base.invokelatest(string_to_distribution, dflt["prior_cc_xsec_norm"]) == g(:prior_cc_xsec_norm)
    @test dflt["hmc_max_depth"] == g(:hmc_max_depth)
    @test dflt["CC_xsec_scale"] == g(:CC_xsec_scale)
    @test dflt["cc_xsec_file"] == g(:cc_xsec_file) == g(:DEFAULT_CC_XSEC_FILE)
    @test dflt["mcmc_seed"] === g(:mcmc_seed) === nothing
    @test haskey(dflt["detectors"]["FXA"], "module_mass_kt")
    # re-parsing the resolved detector blocks gives the detectors the run used
    for n in cfg["active_detectors"]
        @test Base.invokelatest(parse_detector_block, res["detectors"][n], n; defaults = res) == g(:detector_configs)[n]
    end
end

# --- SPEC 4.6: signal (flux-weighted) and background (time-weighted) splits, and the totals built ---
S  = Base.invokelatest(() -> SIGNAL_FRACTIONS)
Bf = Base.invokelatest(() -> BG_FRACTIONS)
wS = Base.invokelatest(() -> exposure_weights_signal)
wB = Base.invokelatest(() -> exposure_weights_bg)
# night fraction of a column, independently of the fitter: trapezoids plus the linear value at 0
function night_fraction_ref(c, e)
    trap(x, y) = sum((x[2:end] .- x[1:end-1]) .* (y[2:end] .+ y[1:end-1]) ./ 2)
    k = findlast(<(0), c)
    e0 = e[k] + (e[k+1] - e[k]) * (0 - c[k]) / (c[k+1] - c[k])
    return trap([c[1:k]; 0.0], [e[1:k]; e0]) / trap(c, e)
end
@testset "day/night exposure splits (SPEC 4.6)" begin
    @test S.night ≈ night_fraction_ref(expo_c, expo_wf) rtol=1e-12
    @test Bf.night ≈ night_fraction_ref(expo_c, expo_wt) rtol=1e-12
    @test Bf.night ≈ 0.4978 atol=1e-4
    @test S.night < Bf.night - 0.1                     # the tilted flux column: a strongly different split
    @test S.day + S.night == 1 && Bf.day + Bf.night == 1
    @test sum(wS) ≈ 1 rtol=1e-7   # quadgk tolerance on the kinked interpolant
    @test sum(wB) ≈ 1 rtol=1e-7
    @test maximum(abs.(wS ./ wB .- 1)) > 0.1           # and different night cos z weights
    li = Base.invokelatest(() -> detector_outputs["FXA"].likelihood_inputs)
    truth = Base.invokelatest(() -> true_params)
    r = Base.invokelatest(expected_rates, li, truth)
    probs = Base.invokelatest(compute_shared_osc_probs, truth, solarModel)
    osc = Base.invokelatest(compute_oscillated_samples, li.MC_no_osc, truth, probs)
    M = li.Mreco
    # rates per unit (full) exposure, folded with scale 1 (the night rows carry the signal weights)
    fold_d(x, R, e) = Base.invokelatest(apply_day_response, x, R, e; scale=1.0)
    fold_n(x, R, e) = Base.invokelatest(apply_night_response, x, R, e; scale=1.0)
    R_CC_day, R_CC_night = fold_d(osc.CC.day, M.CC, M.eff.CC), fold_n(osc.CC.night, M.CC, M.eff.CC)
    R_ES_day = fold_d(osc.ES.nue_day, M.ES.nue, M.eff.ES_nue) .+ fold_d(osc.ES.nuother_day, M.ES.nuother, M.eff.ES_nuother)
    R_ES_night = fold_n(osc.ES.nue_night, M.ES.nue, M.eff.ES_nue) .+ fold_n(osc.ES.nuother_night, M.ES.nuother, M.eff.ES_nuother)
    # with no Earth effect the night signal rows are the signal weights ⊗ the day rate
    U = li.MC_no_osc
    nE = length(U.CC_8B)
    flat = (nue_8B_day = ones(nE), nue_hep_day = ones(nE), nue_8B_night = ones(length(wS), nE), nue_hep_night = ones(length(wS), nE))
    c0 = Base.invokelatest(compute_oscillated_samples, U, truth, flat; es_mode=false, cc_mode=true).CC
    @test c0.night ≈ wS .* c0.day' rtol=1e-12
    for (day, night, Rd, Rn, BG) in ((r.CC_day, r.CC_night, R_CC_day, R_CC_night, r.BG_CC_tot),
                                     (r.ES_day, r.ES_night, R_ES_day, R_ES_night, r.BG_ES_tot))
        @test day ≈ S.day .* Rd .+ Bf.day .* BG rtol=1e-12
        @test night ≈ S.night .* Rn .+ Bf.night .* (BG' .* wB) rtol=1e-12
        @test day .+ vec(sum(night, dims=1)) ≈ S.day .* Rd .+ S.night .* vec(sum(Rn, dims=1)) .+ BG .* (Bf.day + Bf.night * sum(wB)) rtol=1e-12
        # the swapped weighting is far off
        @test !isapprox(night, S.night .* Rn .+ S.night .* (BG' .* wS); rtol=1e-3)
    end
end

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

# Signal A_DN per unit (signal) exposure, background removed with its own split (SPEC 4.6)
function asym(day, night, bg, idx)
    b = sum(bg[idx:end])
    D = sum(day[(ndims(day) == 1 ? (idx:length(day),) : (:, idx:size(day, 2)))...]) - Bf.day * b
    N = sum(night[:, idx:end, (ndims(night) == 3 ? (:,) : ())...]) - Bf.night * sum(wB) * b
    return 2 * (D / S.day - N / S.night) / (D / S.day + N / S.night)
end
above(x, idx) = (y = copy(x); ndims(y) == 1 ? (y[1:idx-1] .= 0) : (y[:, 1:idx-1, (ndims(y) == 3 ? (:,) : ())...] .= 0); y)

saved = jldopen(f -> Dict(k => f[k] for k in keys(f) if startswith(k, "derived_")), chain * ".jld2")
@testset "day/night fractions saved (SPEC 4.6)" begin
    @test saved["derived_signal_day_fraction"] == S.day && saved["derived_signal_night_fraction"] == S.night
    @test saved["derived_bg_day_fraction"] == Bf.day && saved["derived_bg_night_fraction"] == Bf.night
    @test saved["derived_exposure_weights_signal"] == vec(wS)
    @test saved["derived_exposure_weights_bg"] == vec(wB)
    # the single, ambiguous set of older files is no longer written
    @test !any(haskey.(Ref(saved), ["derived_exposure_weights", "derived_day_fraction", "derived_night_fraction"]))
end

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
                bg_mean = wmean([Bf.day .* getfield(r, BG) for r in rates])
                @test saved[key("BG$(ch)day_pp_mean")] ≈ bg_mean rtol=1e-10
                @test saved[key("BG$(ch)night_pp_mean")] ≈ (Bf.night / Bf.day) .* wB .* bg_mean' rtol=1e-10

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
                    @test saved[key("signal_$(ch)day")] ≈ getfield(r_true, Symbol(ch, "_day")) .- Bf.day .* getfield(r_true, BG) rtol=1e-10
                    @test saved[key("signal_$(ch)night")] ≈ getfield(r_true, Symbol(ch, "_night")) .- Bf.night .* (getfield(r_true, BG)' .* wB) rtol=1e-10
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
                @test vec(sum(comps, dims=2)) ≈ Bf.day .* r_true.BG_CC_tot rtol=1e-10
                @test length(split(saved[key("CCbg_comp_names")], ",")) == size(comps, 2)
                @test saved[key("data_CCday")] == li.nObserved.CC_day
            end

            # setup's per-bin breakdown CSVs (src/inputChecks.jl): components add up to the total
            # background at the truth (day + all night bins; CC × backward fraction if semi-inclusive)
            nf = Bf.day + Bf.night * sum(wB)
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
        d = ph.load_bin_diagnostics([chain], verbose=False)
        assert d['detector'] == 'FXA'
        for k in ('signal_day_fraction', 'signal_night_fraction', 'bg_day_fraction', 'bg_night_fraction'):
            assert np.isclose(float(np.mean(d['derived_' + k])), f['derived_' + k][()]), k
        for k in ('exposure_weights_signal', 'exposure_weights_bg'):
            assert np.allclose(np.squeeze(d['derived_' + k]), np.squeeze(f['derived_' + k][()])), k
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
