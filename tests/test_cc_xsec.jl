#=
CC ν_e–⁴⁰Ar total cross section (SPEC 3.4): the nominal table is MARLEY 2.0.0 (HF-CRPA continuum
+ Bhattacharya 2009 discrete), used point by point, interpolated linearly, never extrapolated.
Reference values below are copied from the MARLEY 2.0.0 `marley xsec` output (KEmin 0, KEmax 30,
301 steps), not from the CSV the code reads.
=#
using Test, YAML

if !isdefined(Main, :read_cc_xsec_table)
    global tpc_threshold = 0.2e-3
    global CC_xsec_scale = 1.0
    include(joinpath(@__DIR__, "..", "src", "xsec.jl"))
end

@testset "CC cross section – MARLEY 2.0.0 nominal table" begin
    @test basename(CC_XSEC_PATH) == basename(DEFAULT_CC_XSEC_FILE) == "CC_nue_40Ar_total_marley_v2.0.0.csv"
    E, σ = read_cc_xsec_table(CC_XSEC_PATH)

    # Every generated point is present, the first one (0 MeV) included: 0–30 MeV in 0.1 MeV steps.
    @test length(E) == 301
    @test E[1] == 0.0 && σ[1] == 0.0
    @test E[end] == 30.0
    @test all(isapprox.(E, 0.1 .* (0:300); atol=1e-9))
    @test CC_XSEC_DOMAIN == (0.0, 0.03)

    # Reference values [1e-42 cm²] from the MARLEY output (exact on the grid points).
    marley = Dict(3.7 => 0.0, 3.8 => 0.00695103, 5.0 => 0.181581, 6.0 => 0.555616, 8.0 => 2.70547,
                  10.0 => 6.8166, 14.0 => 20.8907, 16.0 => 31.2696, 20.0 => 60.1531, 30.0 => 179.362)
    for (e, s) in marley
        @test CC_xsec(e * 1e-3) ≈ s * 1e-42 * CC_xsec_scale rtol=1e-12 atol=1e-60
    end
    # Linear between grid points
    @test CC_xsec(6.05e-3) ≈ (0.555616 + 0.620577) / 2 * 1e-42 * CC_xsec_scale rtol=1e-12

    # Zero below threshold (the lowest B09 level, E_x ≈ 2.29 MeV, plus Q ≈ 1.50 MeV: 3.7–3.9 MeV),
    # non-negative everywhere, strictly increasing above threshold.
    @test all(>=(0), σ)
    i_th = findfirst(>(0), σ)
    @test 3.7 < E[i_th] <= 3.9
    @test all(iszero, σ[1:i_th-1])
    @test all(>(0), diff(σ[i_th:end]))

    # Never extrapolated: evaluating outside [0, 30] MeV is an error.
    @test_throws BoundsError CC_xsec(30.05e-3)
    @test_throws BoundsError CC_xsec(-1e-4)

    # The true-energy range of every shipped config lies inside the table.
    root = joinpath(@__DIR__, "..")
    cfgs = vcat(joinpath(root, "config.yaml"), joinpath(root, "tests", "fixtures", "regression_config.yaml"),
                filter(endswith(".yaml"), readdir(joinpath(root, "configs"); join=true)))
    for f in cfgs
        c = YAML.load_file(f)
        haskey(c, "range_Etrue") || continue
        lo, hi = c["range_Etrue"]
        @test E[1] <= lo && hi <= E[end]
    end
end

@testset "CC cross section – table format checks" begin
    mktempdir() do dir
        tab(name, lines) = (p = joinpath(dir, name); write(p, join(lines, "\n") * "\n"); p)

        # Comments and a header row are skipped; the first data row is kept.
        E, σ = read_cc_xsec_table(tab("ok.csv", ["# provenance", "E_nu_MeV,sigma", "1.0,2.0", "2.0,4.0", "3.0,8.0"]))
        @test E == [1.0, 2.0, 3.0] && σ == [2.0, 4.0, 8.0]

        # A headerless file would lose its first point: rejected.
        @test_throws ErrorException read_cc_xsec_table(tab("nohdr.csv", ["1.0,2.0", "2.0,4.0"]))
        @test_throws ErrorException read_cc_xsec_table(tab("neg.csv",   ["E,s", "1.0,2.0", "2.0,-4.0"]))
        @test_throws ErrorException read_cc_xsec_table(tab("nan.csv",   ["E,s", "1.0,NaN", "2.0,4.0"]))
        @test_throws ErrorException read_cc_xsec_table(tab("inf.csv",   ["E,s", "1.0,2.0", "Inf,4.0"]))
        @test_throws ErrorException read_cc_xsec_table(tab("unsort.csv", ["E,s", "2.0,2.0", "1.0,4.0"]))
        @test_throws ErrorException read_cc_xsec_table(tab("dup.csv",   ["E,s", "1.0,2.0", "1.0,4.0", "2.0,5.0"]))
        @test_throws ErrorException read_cc_xsec_table(tab("cols.csv",  ["E,s", "1.0,2.0,3.0", "2.0,4.0,5.0"]))
        @test_throws ErrorException read_cc_xsec_table(joinpath(dir, "missing.csv"))
    end
end
