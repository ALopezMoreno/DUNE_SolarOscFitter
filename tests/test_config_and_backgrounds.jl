using Test
using YAML
using CSV
using DataFrames
using StatsBase
using Distributions

include(joinpath(@__DIR__, "..", "src", "readConfig.jl"))
include(joinpath(@__DIR__, "..", "src", "backgrounds.jl"))

# ── warn_if_bg_arrays_mismatched ─────────────────────────────────────────────
# build_backgrounds zips the background file list against its normalisation/
# systematic arrays, so a shorter norms/sys array is a *supported* way to only
# use the first N background files (confirmed by the maintainer). The risk is
# that this silent truncation looks identical to an accidental typo in a
# hand-edited config — a mismatch must now be visible in the logs.
@testset "warn_if_bg_arrays_mismatched" begin
    @test_logs (:warn,) warn_if_bg_arrays_mismatched(
        "ES", ["a", "b", "c"], [1.0, 2.0], [0.1, 0.2],
    )
    @test_logs warn_if_bg_arrays_mismatched(
        "ES", ["a", "b"], [1.0, 2.0], [0.1, 0.2],
    )
end

# ── config schema smoke test ─────────────────────────────────────────────────
# These YAML files are hand-edited and copy-pasted (the array-length mismatch
# above was found this way); a cheap parse-only pass over every maintained
# config catches structural typos (missing keys, wrong types) before a
# multi-hour MCMC run fails or silently misbehaves.
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

# Pre-angular-reco flat-format example configs — these predate the
# `reconstruction_sample_ES_angle` key and are already known to fail; kept as
# @test_broken so a future fix (or removal) is noticed instead of re-silenced.
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
# A misspelled key used to be ignored silently (its default was used instead). It must stop the
# run, naming the key, its section and the closest known key; obsolete keys only warn.
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
        c = deepcopy(root); c["detectors"] = Dict{Any,Any}()  # nothing to suggest from
        @test occursin("active_detectors names 'HD'", errmsg(c))
    end

    @testset "misspelled per-detector key at the top level" begin
        c = deepcopy(root); c["ES_exposre"] = 1.0
        m = errmsg(c)
        @test occursin("did you mean 'ES_exposure'", m) && occursin("per-detector key", m)
    end

    # Every key src/ reads from the YAML must be accepted; otherwise a valid config fails the check.
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
