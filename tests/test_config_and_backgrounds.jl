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
            # attenuation = 4/50e6; then rescaled by norm=2.0 (sys==0 branch).
            # Each background is itself a 1-element histogram (single-bin binning).
            expected_es = 1.0 * detection_time * 0.5 * det.ES_normalisation * (4 / 50e6) * 2.0
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

            # All 3 histograms are still loaded, but only the first 2 are
            # scaled/counted — this is the exact invariant break any code
            # consuming ES_par_counts positionally against ES must guard against.
            @test length(bg.ES) == 3
            @test length(bg.ES_par_counts) == 2
        end
    end
end
