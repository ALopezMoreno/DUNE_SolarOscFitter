using Test
using CSV, DataFrames, Logging

# src/inputChecks.jl: per-file checks and the per-bin rate breakdown.
include(joinpath(@__DIR__, "..", "src", "logger.jl"))       # Setup log level
include(joinpath(@__DIR__, "..", "src", "inputChecks.jl"))

@testset "input checks" begin
    @testset "looks_prebinned" begin
        @test looks_prebinned(collect(0.0035:0.0002:0.0221))                  # bin centres, 0.2 MeV grid
        @test looks_prebinned([0.0035, 0.0037, 0.0041, 0.0049, 0.0051, 0.0091]) # grid with gaps
        @test !looks_prebinned(1e-3 .* (3 .+ 10 .* rand(500)))                  # event-level sample
        @test !looks_prebinned([0.004, 0.005, 0.006])                           # too few values to tell
        # bin centres that went through Float32 (ROOT TH1F) are still histograms
        @test looks_prebinned(Float64.(Float32.(0.0035:0.0002:0.0221)))
        # event-level samples stored with limited precision are not
        ev = 1e-3 .* (3 .+ 15 .* rand(100_000))
        @test !looks_prebinned(round.(ev, digits=6))                            # 1 keV rounding
        @test !looks_prebinned(round.(ev, digits=8))                            # 10 eV rounding
        @test !looks_prebinned(Float64.(Float32.(ev)))                          # Float32 upcast
        @test !looks_prebinned(round.(ev[1:1000], digits=7))                    # few events, 0.1 keV
    end

    mktempdir() do dir
        w(name, df) = (p = joinpath(dir, name); CSV.write(p, df); p)
        ev  = w("ev.csv",  DataFrame(Ereco = 1e-3 .* (3 .+ 10 .* rand(200)), mask = rand(200) .< 0.5, side = 0))
        all_sel = w("all.csv", DataFrame(Ereco = 1e-3 .* (3 .+ 10 .* rand(200)), mask = trues(200), side = 0))
        binned  = w("binned.csv", DataFrame(Ereco = collect(0.0035:0.0002:0.0101), weights = rand(34), mask = trues(34)))
        copy_ev = joinpath(dir, "ev_copy.csv"); cp(ev, copy_ev)

        @testset "identical_file_groups" begin
            @test identical_file_groups([ev, all_sel, copy_ev, binned]) == [[ev, copy_ev]]
            @test isempty(identical_file_groups([ev, all_sel]))
        end

        det(; ES_bg=[ev], CC_bg=[binned]) = (
            name = "T", ES_mode = true, CC_mode = true, angular_reco = false,
            inclusive_analysis = false, semi_inclusive_analysis = false,
            E_threshold = (ES = 4e-3, CC = 4e-3),
            Ereco_bins_ES = (bin_number = 5, min = 2e-3, max = 20e-3),
            Ereco_bins_CC = (bin_number = 5, min = 2e-3, max = 20e-3),
            nue_filepath = ev, other_filepath = ev, angular_filepath = ev, CC_filepath = ev,
            ES_filepaths_BG = ES_bg, CC_filepaths_BG = CC_bg,
            ES_bg_norms = ones(length(ES_bg)), ES_bg_sys = zeros(length(ES_bg)),
            CC_bg_norms = ones(length(CC_bg)), CC_bg_sys = zeros(length(CC_bg)))

        @testset "warnings" begin
            # pre-binned CC background (the signal files all use the same path: not "identical files")
            @test_logs (:warn, r"looks pre-binned") min_level=Logging.Warn check_input_files(det())
            # ES background with every event selected
            @test_logs (:warn, r"passes the selection") (:warn, r"looks pre-binned") min_level=Logging.Warn check_input_files(det(ES_bg=[all_sel]))
            # a copied file
            @test_logs (:warn, r"looks pre-binned") (:warn, r"identical contents") min_level=Logging.Warn check_input_files(det(ES_bg=[copy_ev]))
            # clean inputs: no warnings
            @test_logs min_level=Logging.Warn check_input_files(det(CC_bg=[all_sel]))
        end

        @testset "per-file summary" begin
            rows = check_input_files(det(CC_bg=[all_sel]))
            r = only(filter(r -> r.role == "CC background", rows))
            df = CSV.read(all_sel, DataFrame)
            @test r.rows == 200 && r.selected == 200
            @test r.in_window ≈ count(df.Ereco .>= 4e-3)                         # threshold to +∞
        end

        @testset "save_rate_breakdown" begin
            edges = [-Inf, 4e-3, 8e-3, Inf]
            raw   = [[1.0, 2.0, 3.0], [10.0, 0.0, 5.0]]
            data  = [100.0, 200.0, 300.0]
            # first background free (conv 2, true norm 3), second fixed; night factor 0.7
            df = save_rate_breakdown(joinpath(dir, "out"), "T", "CC", edges, 2, data, raw,
                                     [2.0, 1.0], [1, 0], [3.0], ["a", "b"], 0.7)
            @test df.bg_a ≈ raw[1] .* 6 .* 0.7
            @test df.bg_b ≈ raw[2] .* 0.7
            @test df.background ≈ df.bg_a .+ df.bg_b
            @test df.signal ≈ data .- df.background
            @test df.in_fit == [false, true, true]
            @test isfile(joinpath(dir, "out_T_CC_breakdown.csv"))
        end
    end
end
