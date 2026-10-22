using Test
using YAML
using CSV
using DataFrames
using StatsBase
using Distributions

include(joinpath(@__DIR__, "..", "src", "readConfig.jl"))
include(joinpath(@__DIR__, "..", "src", "backgrounds.jl"))

# ── warn_if_bg_arrays_mismatched: shorter norms/sys arrays (allowed) must show in the log ──
@testset "warn_if_bg_arrays_mismatched" begin
    @test_logs (:warn,) warn_if_bg_arrays_mismatched(
        "ES", ["a", "b", "c"], [1.0, 2.0], [0.1, 0.2],
    )
    @test_logs warn_if_bg_arrays_mismatched(
        "ES", ["a", "b"], [1.0, 2.0], [0.1, 0.2],
    )
end

# ── every maintained config parses (catches typos before a long run) ──
function parses_ok(config)
    if haskey(config, "detectors")
        active = get(config, "active_detectors", collect(keys(config["detectors"])))
        for name in active
            parse_detector_block(config["detectors"][name], name)
        end
    else
        parse_detector_block(config, "default")
    end
    return true
end

# Old flat example configs without `reconstruction_sample_ES_angle`: @test_broken until fixed or removed
const KNOWN_STALE_CONFIGS = Set([
    "config.yaml",   # configs/config.yaml (flat legacy example, not the root one)
    "config1.yaml", "config2.yaml", "config3.yaml", "config4.yaml",
])

@testset "committed configs parse without throwing" begin
    repo_root = joinpath(@__DIR__, "..")
    root_config = joinpath(repo_root, "config.yaml")
    examples_dir = joinpath(repo_root, "configs")
    example_configs = filter(f -> endswith(f, ".yaml"), readdir(examples_dir, join=true))
    @test !isempty(example_configs)

    @testset "config.yaml (root)" begin
        config = load_parameters(root_config)
        @test config !== nothing
        @test config !== nothing && parses_ok(config)
    end

    for path in example_configs
        name = basename(path)
        @testset "configs/$name" begin
            config = load_parameters(path)
            @test config !== nothing
            if config !== nothing
                if name in KNOWN_STALE_CONFIGS
                    @test_broken parses_ok(config)
                else
                    @test parses_ok(config)
                end
            end
        end
    end
end

# ── unknown config keys ──────────────────────────────────────────────────────
# A misspelled key stops the run, naming the key, its section and the closest known key; obsolete
# keys only warn.
@testset "validate_config_keys" begin
    repo_root = joinpath(@__DIR__, "..")
    root = load_parameters(joinpath(repo_root, "config.yaml"))
    errmsg(c) = try validate_config_keys(c); "" catch e; sprint(showerror, e) end

    @testset "shipped and test configs pass" begin
        @test validate_config_keys(root) === nothing
        # Committed configs only: untracked work-in-progress configs in configs/ are the user's.
        tracked = readlines(`git -C $repo_root ls-files -- 'configs/*.yaml'`)
        @test !isempty(tracked)
        for path in vcat(joinpath(repo_root, "tests", "fixtures", "regression_config.yaml"),
                         joinpath.(repo_root, tracked))
            @test validate_config_keys(load_parameters(path)) === nothing
        end
    end

    @testset "typo at the top level" begin
        c = deepcopy(root); c["prior_cc_xsec_nrom"] = "Normal(1.0, 0.2)"
        m = errmsg(c)
        @test occursin("'prior_cc_xsec_nrom'", m) && occursin("top level", m) &&
              occursin("did you mean 'prior_cc_xsec_norm'", m)
    end

    @testset "typo in a detector entry" begin
        c = deepcopy(root); c["detectors"]["HD"]["Ereco_min_ESS"] = 5.0
        m = errmsg(c)
        @test occursin("'Ereco_min_ESS'", m) && occursin("detectors.HD", m) &&
              occursin("did you mean 'Ereco_min_ES'", m)
    end

    @testset "key in the wrong section" begin
        c = deepcopy(root); c["CC_exposure"] = 10.0         # per-detector key at the top level
        @test occursin("per-detector key", errmsg(c))
        c = deepcopy(root); c["detectors"]["HD"]["nSteps"] = 10
        @test occursin("top-level key", errmsg(c))
        # flat config (no detectors:): detector keys belong at the top level
        flat = load_parameters(joinpath(repo_root, "configs", "hmcTinyTest.yaml"))
        @test validate_config_keys(flat) === nothing
        flat["ES_exposre"] = 1.0
        @test occursin("did you mean 'ES_exposure'", errmsg(flat))
    end

    @testset "active_detectors must exist" begin
        c = deepcopy(root); c["active_detectors"] = ["HDD"]
        @test occursin("did you mean 'HD'", errmsg(c))
        c = deepcopy(root); c["active_detectors"] = "HD"
        @test occursin("must be a list", errmsg(c))
        c = deepcopy(root); c["active_detectors"] = String[]
        @test occursin("is empty", errmsg(c))
        c = deepcopy(root); c["active_detectors"] = ["HD", "HD"]
        @test occursin("more than once", errmsg(c))
        c = deepcopy(root); c["detectors"] = Dict{Any,Any}()  # nothing to suggest from
        @test occursin("active_detectors names 'HD'", errmsg(c))
    end

    @testset "misspelled per-detector key at the top level" begin
        c = deepcopy(root); c["ES_exposre"] = 1.0
        m = errmsg(c)
        @test occursin("did you mean 'ES_exposure'", m) && occursin("per-detector key", m)
        # valid at both levels: no "move it" hint
        c = deepcopy(root); c["mc_stat_likelihod"] = "Poisson"
        m = errmsg(c)
        @test occursin("did you mean 'mc_stat_likelihood'", m) && !occursin("per-detector key", m)
        # flat config: detector keys belong at the top level, so no "move it" hint either
        flat = load_parameters(joinpath(repo_root, "configs", "hmcTinyTest.yaml"))
        flat["ES_exposre"] = 1.0
        @test !occursin("per-detector key", errmsg(flat))
    end

    # Every key src/ reads from the YAML must be accepted; otherwise a valid config fails the check.
    # Catches config[...], block[...], defaults[...], get(...) and haskey(...) with a literal key.
    # Not caught: keys built by interpolation (block["Ereco_min_$ch"] in threshold_binning) or read
    # through another variable name; list those by hand.
    @testset "key lists cover every key read in src/" begin
        reads_top, reads_det = Set{String}(), Set{String}()
        for (dir, _, files) in walkdir(joinpath(repo_root, "src")), f in files
            endswith(f, ".jl") || continue
            src = read(joinpath(dir, f), String)
            for m in eachmatch(r"(?:\b(config|block|defaults)\[\s*|\b(?:get|haskey)\(\s*(config|block|defaults)\s*,\s*)\"([^\"$]+)\"", src)
                var = something(m.captures[1], m.captures[2])
                push!(var == "block" ? reads_det : reads_top, m.captures[3])
            end
        end
        @test !isempty(reads_top) && !isempty(reads_det)
        @test setdiff(reads_top, union(TOP_LEVEL_CONFIG_KEYS, OBSOLETE_CONFIG_KEYS)) == Set()
        @test setdiff(reads_det, DETECTOR_CONFIG_KEYS) == Set()
    end

    @testset "obsolete keys warn, not error" begin
        c = deepcopy(root); c["proposal_matrix"] = "old_cov.jld2"
        @test validate_config_keys(c) === nothing
        @test_logs (:warn, r"proposal_matrix") load_proposal_matrix(c)
        c = deepcopy(root); c["nuFast"] = false
        @test_logs (:warn, r"nuFast") validate_config_keys(c)
        c = deepcopy(root); c["binnedNight"] = true
        @test validate_config_keys(c) === nothing      # warns where it is read, in main()
    end

    @testset "Asimov: false is not implemented" begin
        c = deepcopy(root); c["Asimov"] = false
        @test occursin("not implemented", errmsg(c))
        c = deepcopy(root); c["Asimov"] = "true"           # a string, not a Bool
        @test occursin("must be true or false", errmsg(c))
        c = deepcopy(root); delete!(c, "Asimov")           # optional, defaults to true
        @test validate_config_keys(c) === nothing
    end
end

# ── config inheritance (`base:`, src/configFiles.jl) ─────────────────────────
# A config built from bases must load as the flat file holding the same settings: mappings merged
# at every depth, the later file winning, lists and scalars replaced whole.
@testset "config inheritance (base:)" begin
    repo_root = joinpath(@__DIR__, "..")
    d = mktempdir()
    w(name, text) = (p = joinpath(d, name); mkpath(dirname(p)); write(p, text); p)
    errmsg(f) = try f(); "" catch e; sprint(showerror, e) end

    @testset "an override resolves to the equivalent flat file" begin
        w("parts/a.yaml", """
            nChains: 4
            range_Etrue: [1, 21]
            detectors:
              VD: {CC_exposure: 10, ES_exposure: 10, range_Ereco_ES: [2, 20]}
            """)
        w("parts/b.yaml", "nChains: 8\nnSteps: 100\n")
        top = w("run.yaml", """
            base: [parts/a.yaml, parts/b.yaml]
            nSteps: 200
            range_Etrue: [2, 20]
            detectors:
              VD: {ES_exposure: 5, range_Ereco_ES: [3, 15]}
            """)
        flat = w("flat.yaml", """
            nChains: 8
            nSteps: 200
            range_Etrue: [2, 20]
            detectors:
              VD: {CC_exposure: 10, ES_exposure: 5, range_Ereco_ES: [3, 15]}
            """)
        @test resolve_config(top) == YAML.load_file(flat)
        @test load_parameters(top) == YAML.load_file(flat)
        @test config_files(top) == [joinpath(d, "parts", "a.yaml"), joinpath(d, "parts", "b.yaml"), top]
    end

    @testset "a config without base: loads unchanged" begin
        path = joinpath(repo_root, "config.yaml")
        @test resolve_config(path) == YAML.load_file(path)
        @test config_files(path) == [abspath(path)]
    end

    @testset "bases chain, each relative to the file naming it" begin
        w("chain/deep/c.yaml", "nTuning: 7\n")
        w("chain/b.yaml", "base: deep/c.yaml\nnTuning: 8\nllh_bins: 3\n")
        top = w("chain/top.yaml", "base: b.yaml\nllh_bins: 4\n")
        @test resolve_config(top) == Dict("nTuning" => 8, "llh_bins" => 4)
        @test length(config_files(top)) == 3
    end

    @testset "a shared base is applied once and cannot undo a later override" begin
        w("diamond/d.yaml", "nSteps: 1\nnChains: 1\n")
        w("diamond/b.yaml", "base: d.yaml\nnSteps: 2\n")
        w("diamond/c.yaml", "base: d.yaml\nnChains: 3\n")
        top = w("diamond/a.yaml", "base: [b.yaml, c.yaml]\n")
        @test resolve_config(top) == Dict("nSteps" => 2, "nChains" => 3)
        @test basename.(config_files(top)) == ["d.yaml", "b.yaml", "c.yaml", "a.yaml"]
    end

    @testset "errors" begin
        w("cyc/x.yaml", "base: y.yaml\n"); w("cyc/y.yaml", "base: x.yaml\n")
        @test occursin("cycle", errmsg(() -> resolve_config(joinpath(d, "cyc", "x.yaml"))))
        miss = w("miss.yaml", "base: nowhere.yaml\n")
        m = errmsg(() -> resolve_config(miss))
        @test occursin("not found", m) && occursin("named in", m)
        @test occursin("base must be", errmsg(() -> resolve_config(w("bad.yaml", "base: {a: 1}\n"))))
        @test occursin("mapping", errmsg(() -> resolve_config(w("list.yaml", "- 1\n- 2\n"))))
    end
end

# ── tutorial_example/ ────────────────────────────────────────────────────────
# The run configs load, pass the key check and parse; the full tutorial config sets every key of
# the root config.yaml; a nested override changes only the key it names.
@testset "tutorial_example configs" begin
    repo_root = joinpath(@__DIR__, "..")
    tut = joinpath(repo_root, "tutorial_example")
    runs = filter(f -> endswith(f, ".yaml"), readdir(tut, join = true))
    @test length(runs) >= 6
    for path in vcat(joinpath(tut, "configs", "tutorial.yaml"), runs)
        @testset "$(relpath(path, repo_root))" begin
            c = load_parameters(path)
            @test c !== nothing && validate_config_keys(c) === nothing && parses_ok(c)
        end
    end

    root = load_parameters(joinpath(repo_root, "config.yaml"))
    full = load_parameters(joinpath(tut, "configs", "tutorial.yaml"))
    @test isempty(setdiff(keys(root), keys(full)))
    @test isempty(setdiff(keys(root["detectors"]["VD"]), keys(full["detectors"]["VD"])))

    mcmc    = load_parameters(joinpath(tut, "02_mcmc.yaml"))
    es_only = load_parameters(joinpath(tut, "03_mcmc_ES_only.yaml"))
    expected = deepcopy(mcmc)
    expected["outFile"] = es_only["outFile"]
    expected["detectors"]["VD"]["singleChannel"] = "ES"
    @test es_only == expected
end

# ── build_backgrounds: par_counts alignment & normalisation arithmetic ──────
# Guards the actual downstream consumer of the arrays above: a mismatch
# (intentional or not) must not silently misalign `ES_par_counts`/
# `CC_par_counts` against `backgrounds.ES`/`backgrounds.CC`, and the plain
# (sys == 0) rescale must match the documented convention exactly.
function write_bg_csv(path; Ereco, mask)
    open(path, "w") do io
        println(io, "Ereco,mask")
        for (e, m) in zip(Ereco, mask)
            println(io, "$e,$m")
        end
    end
end

function fake_det(; ES_filepaths_BG, CC_filepaths_BG, ES_bg_norms, CC_bg_norms,
                     ES_bg_sys, CC_bg_sys)
    (
        ES_normalisation        = 1.0,
        CC_normalisation        = 1.0,
        inclusive_analysis      = false,
        semi_inclusive_analysis = false,
        ES_mode                 = true,
        CC_mode                 = true,
        ES_filepaths_BG         = ES_filepaths_BG,
        CC_filepaths_BG         = CC_filepaths_BG,
        ES_bg_norms             = ES_bg_norms,
        CC_bg_norms             = CC_bg_norms,
        ES_bg_sys               = ES_bg_sys,
        CC_bg_sys               = CC_bg_sys,
    )
end

@testset "build_backgrounds – alignment & arithmetic" begin
    global detection_time = 1.0
    bin_info = (bin_number=1, min=0.0, max=10.0, bins=[0.0, 10.0])

    mktempdir() do dir
        es_files = [joinpath(dir, "es$i.csv") for i in 1:3]
        write_bg_csv(es_files[1]; Ereco=[1.0, 2.0, 3.0, 4.0], mask=[true, true, false, false])
        write_bg_csv(es_files[2]; Ereco=[1.0, 2.0, 3.0, 4.0], mask=[true, true, false, false])
        write_bg_csv(es_files[3]; Ereco=[1.0, 2.0, 3.0, 4.0], mask=[true, true, false, false])

        cc_file = joinpath(dir, "cc1.csv")
        write_bg_csv(cc_file; Ereco=[1.0, 2.0, 3.0], mask=[true, false, true])

        @testset "matched arrays: par_counts tracks every background" begin
            det = fake_det(
                ES_filepaths_BG = es_files,
                CC_filepaths_BG = [cc_file],
                ES_bg_norms     = [2.0, 2.0, 2.0],
                CC_bg_norms     = [2.2e-6],
                ES_bg_sys       = [0.0, 0.0, 0.0],
                CC_bg_sys       = [0.0],
            )
            bg, _, _, _, _ = build_backgrounds(det, bin_info, bin_info)

            @test length(bg.ES) == length(bg.ES_par_counts) == 3
            @test length(bg.CC) == length(bg.CC_par_counts) == 1

            # ES: normalise=true histogram is 1.0 in the single bin; eff = 2/4;
            # attenuation = 4/50e6; then flux norm 2.0 [cm⁻² s⁻¹] × exposed area (no `side`
            # column → all faces: 2·12·64 + 2·12·12 m² = 1.824e7 cm²), as in the sys≠0 path.
            # Each background is itself a 1-element histogram (single-bin binning).
            area_all    = (2 * 12 * 64 + 2 * 12 * 12) * 1e4
            expected_es = 1.0 * detection_time * 0.5 * det.ES_normalisation * (4 / 50e6) * 2.0 * area_all
            @test only(bg.ES[1]) ≈ expected_es atol=1e-12
            @test only(bg.ES[2]) ≈ expected_es atol=1e-12
            @test only(bg.ES[3]) ≈ expected_es atol=1e-12

            # CC: 2 of 3 events pass the mask, x10 kt scaling, x1 CC_normalisation,
            # then rescaled by norm/2.2e-6 == 1.0 (sys==0 branch).
            expected_cc = 2 * 10 * det.CC_normalisation * (2.2e-6 / 2.2e-6)
            @test only(bg.CC[1]) ≈ expected_cc atol=1e-12
        end

        @testset "mismatched arrays: truncation is silent on the data, loud in the logs" begin
            det = fake_det(
                ES_filepaths_BG = es_files,       # 3 files
                CC_filepaths_BG = [cc_file],
                ES_bg_norms     = [2.0, 2.0],      # only 2 norms/sys
                CC_bg_norms     = [2.2e-6],
                ES_bg_sys       = [0.0, 0.0],
                CC_bg_sys       = [0.0],
            )
            bg = @test_logs (:warn,) match_mode=:any build_backgrounds(det, bin_info, bin_info)[1]

            # Only the first 2 files are used, and the data, par_counts and conversion
            # factors stay aligned (the 3rd file is not loaded at all).
            @test length(bg.ES) == 2
            @test length(bg.ES_par_counts) == 2
            @test length(bg.ES_conv) == 2
        end
    end
end

@testset "threshold_binning – fit starts exactly at Ereco_min" begin
    for (thr, hi, n) in ((5, 25, 30), (3, 15, 30), (4, 20, 15), (0.1, 25, 50))
        b = threshold_binning(Dict("Ereco_min_ES" => thr, "range_Ereco_ES" => [1, hi], "nBins_Ereco_ES" => n), "ES", "T")
        edges = collect(range(b.min, b.max, length = b.bin_number + 1))
        @test b.bin_number == n + 1                          # n fitted bins + 1 underflow
        @test edges[2] ≈ thr * 1e-3                          # threshold is the lower edge of bin 2
        @test edges[end] ≈ hi * 1e-3
        @test all(diff(edges) .≈ (hi - thr) * 1e-3 / n)      # uniform width above (and for) the underflow
    end
    @test_throws ErrorException threshold_binning(Dict("Ereco_min_CC" => 30, "range_Ereco_CC" => [1, 25], "nBins_Ereco_CC" => 10), "CC", "T")

    # The threshold is exactly the second edge (no rounding), so the first fitted bin is bin 2
    # and an event at the threshold is in it. Includes Ereco_min = 0, where a relative tolerance
    # would vanish, and grids where range() lands one ulp off.
    for thr in (0, 0.1, 3, 5), (hi, n) in ((20, 29), (25, 3), (25, 7), (25, 40), (15, 30))
        b = threshold_binning(Dict("Ereco_min_ES" => thr, "range_Ereco_ES" => [1, hi], "nBins_Ereco_ES" => n), "ES", "T")
        e = ereco_edges(b)
        @test e[2] == thr * 1e-3
        e[1] = -Inf; e[end] = Inf
        @test findfirst(x -> x >= thr * 1e-3, e) == 2
        h, _ = create_histogram([thr * 1e-3, thr * 1e-3 - 1e-9], (bin_number = b.bin_number, min = b.min, max = b.max, bins = e), normalise = false)
        @test h[1] == 1 && h[2] == 1                    # just below → underflow, at threshold → first fitted bin
    end
end

# ── inputs read by the plotting code are in git ──────────────────────────────
# inputs/ is gitignored; a fresh clone has only the files added on purpose (git add -f).
@testset "plotting inputs are tracked" begin
    repo_root = joinpath(@__DIR__, "..")
    tracked = Set(readlines(`git -C $repo_root ls-files -- inputs`))
    py = [joinpath(d, f) for (d, _, fs) in walkdir(joinpath(repo_root, "utils", "plotting")) for f in fs if endswith(f, ".py")]
    append!(py, joinpath.(repo_root, "utils", ["plotOutput.py", "compareOutput.py", "plotLLH.py", "plotInputs.py"]))
    for f in py, m in eachmatch(r"[\"'](inputs/[A-Za-z0-9_./-]+\.\w+)[\"']", read(f, String))
        @test m.captures[1] in tracked
    end
end
