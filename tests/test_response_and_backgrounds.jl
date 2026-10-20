using Test
using CSV, DataFrames, StatsBase, Distributions, LinearAlgebra

# Response, backgrounds and exposure folding (SPEC 2.7, 4.1–4.3, 4.6, 7.1). Each test states the
# physical requirement, not the current output of the code.
include(joinpath(@__DIR__, "..", "src", "core.jl"))
include(joinpath(@__DIR__, "..", "src", "response.jl"))
include(joinpath(@__DIR__, "..", "src", "backgrounds.jl"))
include(joinpath(@__DIR__, "..", "src", "propagation", "propagation_core.jl"))   # DetFlags
include(joinpath(@__DIR__, "..", "src", "propagation", "propagation_bg.jl"))

function write_csv(path, cols::Pair...)
    CSV.write(path, DataFrame(cols...))
    return path
end

# Extended binning as build_response_matrices makes it: outer edges at ±Inf.
function extended_bins(lo, hi, n)
    b = collect(range(lo, hi, length=n + 1)); b[1] = -Inf; b[end] = Inf
    return (bin_number=n, min=lo, max=hi, bins=b)
end

@testset "response matrix keeps out-of-window events (SPEC 4.1)" begin
    etrue = (bin_number=1, min=0.0, max=10.0)
    ereco = extended_bins(1.0, 10.0, 3)                 # edges -Inf, 4, 7, Inf
    # One true-energy bin; reconstructed at 0.5 and 20 (outside [1, 10]), 2 and 3 (bin 1).
    R = create_response_matrix((x=fill(5.0, 4), y=[0.5, 2.0, 3.0, 20.0]), etrue, ereco)
    @test R[1, :] ≈ [3/4, 0.0, 1/4]                     # underflow → bin 1, overflow → bin 3
    @test sum(R[1, :]) ≈ 1

    # Without extended edges, out-of-range events are still dropped (Etrue rows are conditional
    # on Etrue being in range; that is the intended behaviour for the x axis).
    plain = (bin_number=3, min=1.0, max=10.0)
    Rp = create_response_matrix((x=[5.0, 50.0], y=[2.0, 2.0]), etrue, plain)
    @test Rp[1, :] ≈ [1.0, 0.0, 0.0]
end

@testset "next_batch_name never reuses a group (SPEC 7.1)" begin
    @test next_batch_name(Dict{String,Int}()) == "batch_1"
    @test next_batch_name(Dict("batch_1" => 0, "batch_2" => 0, "meta" => 0)) == "batch_3"
    @test next_batch_name(Dict("batch_1" => 0, "batch_10" => 0, "batch_2" => 0)) == "batch_11"
end

@testset "_params_by_prefix orders indices numerically (SPEC 2.7)" begin
    names_ = Tuple(Symbol("earth_norm_$i") for i in (10, 2, 1, 11, 3))
    p = NamedTuple{names_}((10.0, 2.0, 1.0, 11.0, 3.0))
    @test _params_by_prefix(p, Val(:earth_norm_)) == (1.0, 2.0, 3.0, 10.0, 11.0)
end

@testset "background normalisation (SPEC 4.2, 4.3)" begin
    global detection_time = 1.0
    bins = extended_bins(0.0, 10.0, 2)                  # edges -Inf, 5, Inf

    mktempdir() do dir
        es1 = write_csv(joinpath(dir, "es1.csv"), :Ereco => [1.0, 2.0, 6.0, 7.0], :mask => [true, true, true, false], :side => fill(0, 4))
        es2 = write_csv(joinpath(dir, "es2.csv"), :Ereco => [1.0, 6.0, 7.0, 8.0], :mask => [true, true, true, true],  :side => fill(1, 4))
        cc1 = write_csv(joinpath(dir, "cc1.csv"), :Ereco => [1.0, 6.0, 7.0],      :mask => [true, true, false])

        det(; ES_files=[es1, es2], ES_norms=[2.0, 3.0], ES_sys=[0.1, 0.1], CC_norms=[2.2e-6], CC_sys=[0.1], kw...) = (
            ES_normalisation = 1.0, CC_normalisation = 1.0,
            inclusive_analysis = false, semi_inclusive_analysis = false,
            ES_mode = true, CC_mode = true,
            ES_filepaths_BG = ES_files, CC_filepaths_BG = [cc1],
            ES_bg_norms = ES_norms, CC_bg_norms = CC_norms,
            ES_bg_sys = ES_sys, CC_bg_sys = CC_sys,
            kw...)

        area_long = 2 * 12 * 64 * 1e4                    # side 0, cm²
        area_end  = 2 * 12 * 12 * 1e4                    # side 1, cm²

        @testset "each ES background has its own norm" begin
            bg, _ = build_backgrounds(det(), bins, bins)
            @test bg.ES_conv ≈ [area_long, area_end]
            p = (D_ES_bg_norm_1 = 2.0, D_ES_bg_norm_2 = 3.0, D_CC_bg_norm_1 = 2.2e-6)
            BG_ES, _ = normalize_backgrounds(bg, p, "D")
            @test BG_ES ≈ bg.ES[1] .* area_long .* 2.0 .+ bg.ES[2] .* area_end .* 3.0
            # Changing only norm_2 changes only the second background's share.
            BG_ES2, _ = normalize_backgrounds(bg, (D_ES_bg_norm_1 = 2.0, D_ES_bg_norm_2 = 6.0, D_CC_bg_norm_1 = 2.2e-6), "D")
            @test BG_ES2 .- BG_ES ≈ bg.ES[2] .* area_end .* 3.0
        end

        @testset "fixed (sys = 0) and free (sys > 0) backgrounds agree at the nominal norm" begin
            # CC norm ≠ the reference flux, so the CC conversion factor is not 1.
            free, _  = build_backgrounds(det(CC_norms=[1.3e-6]), bins, bins)
            fixed, _ = build_backgrounds(det(ES_sys=[0.0, 0.0], CC_sys=[0.0], CC_norms=[1.3e-6]), bins, bins)
            p = (D_ES_bg_norm_1 = 2.0, D_ES_bg_norm_2 = 3.0, D_CC_bg_norm_1 = 1.3e-6)
            ES_free,  CC_free  = normalize_backgrounds(free,  p, "D")
            ES_fixed, CC_fixed = normalize_backgrounds(fixed, NamedTuple(), "D")
            @test ES_fixed ≈ ES_free
            @test CC_fixed ≈ CC_free
            @test all(>(0), ES_free)
            # And the CC scale is norm / reference flux.
            ref, _ = build_backgrounds(det(CC_sys=[0.0], CC_norms=[2.2e-6]), bins, bins)
            @test CC_fixed ≈ ref.CC[1] .* (1.3e-6 / 2.2e-6)
        end

        @testset "weighted samples: efficiency ≤ 1, uniform weight c scales the rate by c" begin
            esw = write_csv(joinpath(dir, "esw.csv"), :Ereco => [1.0, 2.0, 6.0, 7.0], :mask => [true, true, true, false],
                            :side => fill(0, 4), :weights => fill(5.0, 4))
            unw, _ = build_backgrounds(det(ES_files=[es1], ES_norms=[2.0], ES_sys=[0.0]), bins, bins)
            wtd, _ = build_backgrounds(det(ES_files=[esw], ES_norms=[2.0], ES_sys=[0.0]), bins, bins)
            @test wtd.ES[1] ≈ 5.0 .* unw.ES[1]

            # Non-uniform weights: each bin holds (selected weighted events) / (generated events),
            # times exposure, area and flux norm. Bin 1: 1+2 selected; bin 2: 4 selected, 8 not.
            esv = write_csv(joinpath(dir, "esv.csv"), :Ereco => [1.0, 2.0, 6.0, 7.0], :mask => [true, true, true, false],
                            :side => fill(0, 4), :weights => [1.0, 2.0, 4.0, 8.0])
            v, _ = build_backgrounds(det(ES_files=[esv], ES_norms=[2.0], ES_sys=[0.0]), bins, bins)
            @test v.ES[1] ≈ [3.0, 4.0] ./ 50e6 .* area_long .* 2.0
        end

        @testset "conventions come from the detector block" begin
            base, _ = build_backgrounds(det(), bins, bins)
            alt, _  = build_backgrounds(det(module_mass_kt=20.0, CC_bg_reference_flux=4.4e-6,
                                            ES_bg_generated_events=25e6, ES_bg_face_areas_m2=[1.0, 2.0]), bins, bins)
            @test alt.CC[1] ≈ 2 .* base.CC[1]
            @test alt.CC_conv ≈ base.CC_conv ./ 2
            @test alt.ES[1] ≈ 2 .* base.ES[1]
            @test alt.ES_conv ≈ [1.0e4, 2.0e4]
        end

        @testset "a free background needs a positive nominal norm" begin
            @test_throws ErrorException build_backgrounds(det(ES_norms=[0.0, 3.0]), bins, bins)
        end
    end
end

@testset "semi-inclusive mis-ID uses finite ES bin centres" begin
    global Etrue_bins = (bin_number=4, min=0.0, max=20.0)
    mktempdir() do dir
        # ES events in every reco bin, CC events spread over all true bins.
        E = [1.0, 6.0, 11.0, 16.0, 19.0]
        nue  = write_csv(joinpath(dir, "nue.csv"),  :Etrue => E, :Ereco => E, :mask => trues(5))
        oth  = write_csv(joinpath(dir, "oth.csv"),  :Etrue => E, :Ereco => E, :mask => trues(5))
        ang  = write_csv(joinpath(dir, "ang.csv"),  :Ereco => repeat(E, 2), :cos_scatter => [fill(0.2, 5); fill(0.8, 5)])
        cc   = write_csv(joinpath(dir, "cc.csv"),   :Etrue => E, :Ereco => E, :mask => trues(5))
        d = (nue_filepath=nue, other_filepath=oth, angular_filepath=ang, CC_filepath=cc,
             Ereco_bins_ES=(bin_number=4, min=0.0, max=20.0), Ereco_bins_CC=(bin_number=4, min=0.0, max=20.0),
             cos_scatter_bins=(bin_number=2, min=0.0, max=1.0), angular_cos_cut=0.5,
             inclusive_analysis=false, semi_inclusive_analysis=true)
        R, _, _ = build_response_matrices(d)
        # Every ES reco bin, including the lowest and the highest, maps to the CC response row of
        # the true-energy bin containing its (finite) centre; here that is the identity, so the
        # chain is the backward fraction (1/2 per bin) on the diagonal.
        @test R.CC_misID_nue ≈ 0.5 .* Matrix(1.0I, 4, 4)
    end
end

@testset "target counts per kt of liquid argon" begin
    # 1 kt LAr: 1e9 g / 39.948 g/mol × N_A atoms; Z = 18; ⁴⁰Ar is 99.604 % (atomic abundance).
    @test isapprox(N_ELECTRONS_PER_KT, 2.7135e32; rtol=1e-4)
    @test isapprox(N_AR40_PER_KT, 1.5015e31; rtol=1e-4)
    @test N_ELECTRONS_PER_KT / N_AR40_PER_KT ≈ 18 / 0.99604
end

# SPEC 4.6: the signal is folded with its (flux-weighted) day/night split and night cos z weights, the
# background with its (time-weighted) ones; the signal day-night asymmetry is taken per unit (signal)
# exposure after removing the background with its own split, so it is zero with no Earth effect,
# whatever the two splits. The splits here differ strongly, so mixing them up is visible.
@testset "day/night exposure splits and A_DN per unit exposure (SPEC 4.6)" begin
    R   = [0.7 0.3 0.0; 0.1 0.6 0.3; 0.0 0.2 0.8; 0.0 0.0 1.0]       # (n_true, n_reco)
    eff = [0.9, 0.8, 0.7]
    rate = [5.0, 4.0, 3.0, 2.0]                                         # E_true signal rate, full exposure
    BG   = [30.0, 20.0, 10.0]                                           # background, full exposure
    S = (day = 0.62, night = 0.38); wS = [0.1, 0.3, 0.6]               # signal split, night cos z weights
    B = (day = 0.45, night = 0.55); wB = [0.5, 0.3, 0.2]               # background split and weights
    # no keyword default: the caller must say which side it folds
    @test_throws UndefKeywordError apply_day_response(rate, R, eff)
    full = (R' * rate) .* eff
    @test apply_day_response(rate, R, eff; scale=S.day) ≈ S.day .* full rtol=1e-14
    night_rows = wS .* rate'                                            # no Earth effect: every row ∝ day
    @test vec(sum(apply_night_response(night_rows, R, eff; scale=S.night), dims=1)) ≈ S.night .* full rtol=1e-14

    # predictions as propagateSamples builds them, without any day-night effect
    day   = apply_day_response(rate, R, eff; scale=S.day) .+ B.day .* BG
    night = apply_night_response(night_rows, R, eff; scale=S.night) .+ B.night .* (BG' .* wB)
    @test sum(day) + sum(night) ≈ sum(full) + sum(BG) rtol=1e-13     # each split sums to 1
    for idx in (1, 2)
        @test abs(dn_asymmetry(day, night, BG, idx, S, B)) < 1e-14
        # the wrong split for either part fakes an asymmetry of several %
        @test abs(dn_asymmetry(day, night, BG, idx, B, B)) > 0.05
        @test abs(dn_asymmetry(day, night, BG, idx, S, S)) > 0.05
    end
    # raw counts would show a fake asymmetry of 2(f_day − f_night)/(f_day + f_night)
    D, N = sum(day) - B.day * sum(BG), sum(night) - B.night * sum(BG)
    @test 2 * (D - N) / (D + N) ≈ 2 * (S.day - S.night) rtol=1e-10
    # angular layout: day (cos θ, E), night (cos θ, E, cos z); energy is axis 2
    A = [0.25 0.5 0.4; 0.75 0.5 0.6]
    dayA   = A .* day'
    nightA = reshape(A, 2, 3, 1) .* reshape(night', 1, 3, 3)
    @test abs(dn_asymmetry(dayA, nightA, BG, 1, S, B)) < 1e-14
    # with f = 0.5 it is the earlier 2(D − N)/(D + N) on raw counts
    @test dn_asymmetry(12.0, 10.0, 0.5, 0.5) == 2 * (12.0 - 10.0) / 22.0
    @test dn_asymmetry(0.0, 0.0, S.day, S.night) == 0.0
end
