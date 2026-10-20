#=
Tabulated model inputs (SPEC 8.1): validation, extrapolation policy, interpolation, used range and
header handling. Each ported input is compared (===, at nodes and midpoints) with an independent
reference implementation reproduced below.
=#
using Test, Interpolations, QuadGK, DelimitedFiles, CSV, DataFrames, JLD2, Logging

include(joinpath(@__DIR__, "..", "src", "modelInputs.jl"))
if !isdefined(Main, :NightOsc)
    include(joinpath(@__DIR__, "..", "src", "oscillations", "osc_night.jl"))
end

const _TAB_INPUTS = joinpath(@__DIR__, "..", "inputs")
_nodes_and_midpoints(x) = vcat(x, (x[1:end-1] .+ x[2:end]) ./ 2)
_same(f, g, qs) = all(q -> f(q) === g(q), qs)

@testset "TabulatedInput – validation" begin
    ok(x, y; kw...) = TabulatedInput(x, y; name = "test table", source = "unit test", register = false, kw...)
    @test ok([1.0, 2.0, 3.0], [1.0, 4.0, 9.0]) isa TabulatedInput
    @test_throws ErrorException ok([1.0, 3.0, 2.0], [1.0, 2.0, 3.0])      # unsorted
    @test_throws ErrorException ok([1.0, 2.0, 2.0], [1.0, 2.0, 3.0])      # duplicate node
    @test_throws ErrorException ok([1.0, NaN, 3.0], [1.0, 2.0, 3.0])      # NaN node
    @test_throws ErrorException ok([1.0, 2.0, 3.0], [1.0, NaN, 3.0])      # NaN value
    @test_throws ErrorException ok([1.0, 2.0, Inf], [1.0, 2.0, 3.0])      # Inf node
    @test_throws ErrorException ok([1.0, 2.0, 3.0], [1.0, -Inf, 3.0])     # Inf value
    @test_throws ErrorException ok([1.0, 2.0], [1.0, 2.0, 3.0])           # lengths differ
    @test_throws ErrorException ok([1.0], [1.0])                          # one point
    @test_throws ErrorException ok([1.0, 2.0], [1.0, 2.0]; extrapolation = :linear)  # not a policy
    # Negative values: rejected only where the quantity requires y ≥ 0
    @test_throws ErrorException ok([1.0, 2.0, 3.0], [1.0, -2.0, 3.0]; nonnegative = true)
    @test ok([1.0, 2.0, 3.0], [1.0, -2.0, 3.0]) isa TabulatedInput
    @test ok([1.0, 2.0, 3.0], [0.0, 0.0, 3.0]; nonnegative = true) isa TabulatedInput

    # The message names the input, its source and the offending row
    msg = try ok([1.0, 2.0, 2.0], [1.0, 2.0, 3.0]); "" catch e; sprint(showerror, e) end
    @test occursin("test table", msg) && occursin("unit test", msg) && occursin("strictly increasing", msg)
    msg = try ok([1.0, 2.0, 3.0], [1.0, -2.0, 3.0]; nonnegative = true); "" catch e; sprint(showerror, e) end
    @test occursin("test table", msg) && occursin("negative", msg) && occursin("-2.0", msg)
end

@testset "TabulatedInput – interpolation and extrapolation policies" begin
    x = [0.0, 0.5, 2.0, 3.0, 7.5]
    y = [1.0, 3.0, 0.0, 4.0, 2.5]
    t = TabulatedInput(x, y; name = "policy table", source = "unit test", units = "MeV", register = false)
    @test t.extrapolation === :error       # the default
    @test domain(t) == (0.0, 7.5)

    # Exact at the nodes, linear between them
    @test all(t(x[i]) === y[i] for i in eachindex(x))
    for i in 1:length(x)-1, s in (0.25, 0.5, 0.9)
        q = x[i] + s * (x[i+1] - x[i])
        @test t(q) ≈ y[i] + s * (y[i+1] - y[i]) rtol = 1e-14 atol = 1e-15
    end
    @test t([0.0, 0.25, 7.5]) == [t(0.0), t(0.25), t(7.5)]   # vectors are evaluated element-wise

    # :error — a TabulatedDomainError naming the input, the query and the domain
    @test_throws TabulatedDomainError t(-0.1)
    @test_throws TabulatedDomainError t(7.6)
    msg = try t(8.25); "" catch e; sprint(showerror, e) end
    @test occursin("policy table", msg) && occursin("unit test", msg) && occursin("8.25", msg) &&
          occursin("[0.0, 7.5]", msg) && occursin("MeV", msg)

    # :flat — the end values, as Interpolations' Flat()
    tf = TabulatedInput(x, y; name = "flat table", extrapolation = :flat, register = false)
    @test tf(-1.0) === 1.0 && tf(100.0) === 2.5
    @test tf(3.3) === t(3.3)
    # :zero — zero outside
    tz = TabulatedInput(x, y; name = "zero table", extrapolation = :zero, register = false)
    @test tz(-1.0) === 0.0 && tz(100.0) === 0.0
    @test tz(3.3) === t(3.3)

    # Out-of-domain queries are counted under every policy, never silent
    reset_usage!(tf); tf(-1.0); tf(-2.0); tf(9.0); tf(1.0)
    @test n_outside(tf) == (2, 1)
    @test used_range(tf) == (-2.0, 9.0)
    @test any(l -> occursin("flat table", l) && occursin("2 below, 1 above", l), tabulated_summary([tf]))
end

@testset "TabulatedInput – range used and setup log" begin
    t = TabulatedInput([0.0, 1.0, 2.0], [0.0, 1.0, 4.0]; name = "logged table", source = "unit test", units = "GeV")
    @test isnan(used_range(t)[1])
    t(0.5); t(1.75); t(1.0)
    @test used_range(t) == (0.5, 1.75)
    @test n_outside(t) == (0, 0)
    @test TABULATED_REGISTRY["logged table"] === t
    line = only(tabulated_summary([t]))
    @test occursin("logged table", line) && occursin(":error", line) && occursin("domain [0, 2] GeV", line) &&
          occursin("used [0.5, 1.75]", line) && occursin("none outside", line)
    # Logged once, at the Setup level (LogLevel(2), see logger.jl)
    logs = Test.collect_test_logs(; min_level = LogLevel(2)) do
        log_tabulated_inputs([t])
    end[1]
    @test length(logs) == 1 && logs[1].level == LogLevel(2) && occursin("logged table", logs[1].message)
    delete!(TABULATED_REGISTRY, "logged table")

    @test check_covers(t, 0.0, 2.0; what = "range") === nothing
    @test check_covers(t, 0.0, 2.0 + 1e-12; what = "range") === nothing   # rounding of computed edges
    @test_throws ErrorException check_covers(t, -0.1, 2.0; what = "range")
    @test_throws ErrorException check_covers(t, 0.0, 2.1; what = "range")
end

@testset "read_numeric_table – headers and first rows" begin
    mktempdir() do dir
        tab(name, lines) = (p = joinpath(dir, name); write(p, join(lines, "\n") * "\n"); p)
        # Headerless: every row is a point, the first one included
        c = read_numeric_table(tab("nohdr.csv", ["# comment", "1.0,2.0", "", "2.0,4.0"]); name = "t", ncols = 2, header = false)
        @test c == [[1.0, 2.0], [2.0, 4.0]]
        @test_throws ErrorException read_numeric_table(tab("hdr.csv", ["E,s", "1.0,2.0"]); name = "t", ncols = 2, header = false)
        # Header required: a numeric first row is an error, not a dropped point
        @test read_numeric_table(tab("hdr2.csv", ["E,s", "1.0,2.0", "2.0,4.0"]); name = "t", ncols = 2, header = true) ==
              [[1.0, 2.0], [2.0, 4.0]]
        @test_throws ErrorException read_numeric_table(tab("nohdr2.csv", ["1.0,2.0", "2.0,4.0"]); name = "t", ncols = 2, header = true)
        # Whitespace-delimited, wrong column count
        @test read_numeric_table(tab("ws.dat", ["0.0  1.0 0.5", "1.0 2.0  0.5"]); name = "t", ncols = 3, header = false, delim = nothing) ==
              [[0.0, 1.0], [1.0, 2.0], [0.5, 0.5]]
        @test_throws ErrorException read_numeric_table(tab("ws2.dat", ["0.0 1.0", "1.0 2.0"]); name = "t", ncols = 3, header = false, delim = nothing)
        @test_throws ErrorException read_numeric_table(joinpath(dir, "missing.csv"); name = "t", ncols = 2, header = false)
    end
end

@testset "ported inputs equal the pre-port implementation" begin
    # ⁸B / hep spectra (master: solarFlux.jl), with the shipped analysis range (21 MeV)
    energies, flux8B, fluxHep = jldopen(joinpath(_TAB_INPUTS, "fluxes.jld2"), "r") do f
        f["energies"], f["flux8B"], f["fluxHep"]
    end
    for E_max in (0.021, energies[end])      # with and without the added (E_max, 0) point
        old_E = [0.0; energies]; old_8B = [0.0; flux8B]; old_hep = [0.0; fluxHep]
        if E_max > energies[end]
            old_E = [0.0; energies; E_max]; old_8B = [0.0; flux8B; 0.0]; old_hep = [0.0; fluxHep; 0.0]
        end
        i8B, _ = quadgk(linear_interpolation(old_E, old_8B), old_E[1], old_E[end])
        ihep, _ = quadgk(linear_interpolation(old_E, old_hep), old_E[1], old_E[end])
        old_flux_8B = linear_interpolation(old_E, old_8B / i8B)
        old_flux_hep = linear_interpolation(old_E, old_hep / ihep)

        s = solar_flux_spectra(energies, flux8B, fluxHep, E_max; source = "fluxes.jld2")
        @test s.energies == old_E
        @test s.integral_8B === i8B && s.integral_hep === ihep
        @test _same(s.flux_8B, old_flux_8B, _nodes_and_midpoints(old_E))
        @test _same(s.flux_hep, old_flux_hep, _nodes_and_midpoints(old_E))
        @test s.flux_8B.extrapolation === :error
        @test_throws TabulatedDomainError s.flux_8B(old_E[end] + 1e-6)
        @test_throws TabulatedDomainError s.flux_hep(-1e-6)
    end

    # CC cross section (master: xsec.jl since the MARLEY 2.0 change): header skipped, all rows kept
    if isdefined(Main, :CC_xsec)
        rows = [strip(l) for l in eachline(CC_XSEC_PATH)]
        rows = [l for l in rows if !isempty(l) && !startswith(l, "#")][2:end]
        E = [parse(Float64, split(l, ',')[1]) for l in rows]
        σ = [parse(Float64, split(l, ',')[2]) for l in rows]
        old = linear_interpolation(E .* 1e-3, σ .* 1e-42 .* CC_xsec_scale, extrapolation_bc = Throw())
        @test _same(CC_xsec, old, _nodes_and_midpoints(E .* 1e-3))
    end

    # Solar exposure (SPEC 4.6): two weight columns on the same nodes, each the linear interpolant of
    # its CSV column (read here with CSV.jl), never extrapolated. (The legacy single-column file is an
    # error, tested with the fractions below.)
    expo_file = joinpath(_TAB_INPUTS, "solarExposure_v2.csv")
    df = CSV.File(expo_file; comment = "#") |> DataFrame
    @test names(df) == ["cosz", "time_weight", "flux_weight"]
    tabs = load_exposure(expo_file)
    for (t, col) in ((tabs.time, :time_weight), (tabs.flux, :flux_weight))
        ref = linear_interpolation(df.cosz, df[!, col], extrapolation_bc = Throw())
        @test t.x == df.cosz && length(t.x) == nrow(df)
        @test _same(t, ref, _nodes_and_midpoints(df.cosz))
        @test t.extrapolation === :error
        @test_throws TabulatedDomainError t(t.x[1] - 1e-6)
    end

    # Earth potential (master: earthProfile.jl, readdlm, radii × 6371 × (1 + 1e-10), Throw())
    earth_file = joinpath(_TAB_INPUTS, "EARTH_MODEL_PREM_DENSE.dat")
    data = readdlm(earth_file)
    old_r = data[:, 1] .* 6371.0 .* 1.0000000001
    old_earth = linear_interpolation(old_r, data[:, 2] .* data[:, 3] .* MATTER_POTENTIAL_CONST; extrapolation_bc = Throw())
    em = load_earth_model(earth_file)
    @test em[:radius] == old_r
    new_earth = create_interpolated_model(em)
    @test _same(new_earth, old_earth, _nodes_and_midpoints(old_r))
    @test new_earth(6371.0) === old_earth(6371.0)       # the surface stays inside the table
    @test_throws TabulatedDomainError new_earth(old_r[end] * (1 + 1e-12))

    # makePaths path potential (master: linear_interpolation(distances, potential; Flat())),
    # on a chord of the real Earth model, at the Gauss-Legendre nodes used and at the midpoints
    d = collect(range(0.0, 12000.0; length = 12000))
    r = sqrt.((6371.0 - 0.001)^2 .+ d .^ 2 .- 2 * (6371.0 - 0.001) * 0.95 .* d)
    pot = old_earth(clamp.(r, 0.0, old_r[end]))
    old_path = linear_interpolation(d, pot; extrapolation_bc = Flat())
    new_path = TabulatedInput(d, pot; name = "path", extrapolation = :error, nonnegative = true, register = false)
    nodes, _ = gauss(5, d[1], d[end])
    @test _same(new_path, old_path, vcat(nodes, _nodes_and_midpoints(d)))
end

# SPEC 4.6: day/night splits from the exposure table, signal flux-weighted, backgrounds time-weighted.
_trap(x, y) = sum((x[2:end] .- x[1:end-1]) .* (y[2:end] .+ y[1:end-1]) ./ 2)
# night fraction of a sampled column, independently of the fitter: trapezoids plus the linear value at 0
function _night_fraction_ref(c, e)
    k = findlast(<(0), c)
    e0 = e[k] + (e[k+1] - e[k]) * (0 - c[k]) / (c[k+1] - c[k])
    return _trap([c[1:k]; 0.0], [e[1:k]; e0]) / _trap(c, e)
end
_csv_rows(path) = [parse.(Float64, split(l, ',')) for l in eachline(path)
                   if !isempty(strip(l)) && !startswith(strip(l), "#") && tryparse(Float64, split(l, ',')[1]) !== nothing]

@testset "exposure day/night fractions (SPEC 4.6)" begin
    v2 = joinpath(_TAB_INPUTS, "solarExposure_v2.csv")
    rows = _csv_rows(v2)
    c, et, ef = getindex.(rows, 1), getindex.(rows, 2), getindex.(rows, 3)
    tabs = load_exposure(v2)
    ft, ff = exposure_day_night_fractions(tabs.time), exposure_day_night_fractions(tabs.flux)
    # the fractions match independent trapezoids on the CSV rows
    @test ft.night ≈ _night_fraction_ref(c, et) rtol=1e-12
    @test ff.night ≈ _night_fraction_ref(c, ef) rtol=1e-12
    @test ft.day + ft.night == 1 && ff.day + ff.night == 1
    @test ft.night ≈ quadgk(tabs.time, c[1], 0.0)[1] / quadgk(tabs.time, c[1], 0.0, c[end])[1] rtol=1e-7
    # flux-weighted: uniform in solar longitude (Kepler's second law), so exactly half the year's flux
    # arrives at night
    @test ff.night ≈ 0.5 atol=1e-5
    # time-weighted: reproduces the legacy (time-weighted) inputs/solarExposure.csv
    lrows = _csv_rows(joinpath(_TAB_INPUTS, "solarExposure.csv"))
    lc, le = first.(lrows), last.(lrows)
    @test ft.night ≈ _night_fraction_ref(lc, le) atol=1e-4
    @test ft.night ≈ 0.4978 atol=1e-4
    # shape: ≤ 0.5 % node by node for |cos z| < 0.92, except within ±0.02 of ±cos(φ + ε) (φ = 44.35°,
    # ε = 23.44°): there the density has an integrable (log) peak, the declination turning point, and
    # node values depend on how each table sampled it; there the 0.02-wide bin integrals agree to 2 %.
    lin(xq) = [le[searchsortedlast(lc, q)] + (le[searchsortedlast(lc, q)+1] - le[searchsortedlast(lc, q)]) *
               (q - lc[searchsortedlast(lc, q)]) / (lc[searchsortedlast(lc, q)+1] - lc[searchsortedlast(lc, q)]) for q in xq]
    cusp = cosd(44.35 + 23.44)
    near_cusp = abs.(abs.(c) .- cusp) .< 0.02
    m = (abs.(c) .< 0.92) .& .!near_cusp
    @test maximum(abs.(et[m] ./ lin(c[m]) .- 1)) <= 5e-3
    for s in (-1, 1)
        a, b = s * cusp - 0.01, s * cusp + 0.01
        g = collect(range(a, b; length = 401))
        @test _trap(g, tabs.time(g)) / _trap(g, lin(g)) ≈ 1 atol=2e-2
    end
    # the two weightings really differ (by up to a few % at the cos z extremes), so a swap is visible
    @test 0.02 < maximum(abs.(ef ./ et .- 1)) < 0.05

    # a legacy two-column file is an error that points to the generator; so is a wrong header
    mktempdir() do dir
        legacy = joinpath(dir, "legacy.csv"); write(legacy, "-0.9,0.5\n0.0,0.5\n0.9,0.5\n")
        err = try load_exposure(legacy); nothing catch e; e end
        @test err isa ErrorException && occursin("make_solar_exposure.py", err.msg) && occursin("legacy", err.msg)
        @test_throws ErrorException load_exposure(joinpath(_TAB_INPUTS, "solarExposure.csv"))
        swapped = joinpath(dir, "swapped.csv"); write(swapped, "cosz,flux_weight,time_weight\n-0.9,1,1\n0.9,1,1\n")
        @test_throws ErrorException load_exposure(swapped)
    end

    tab(x, y) = TabulatedInput(x, y; name = "exposure", register = false)
    # a table mirror-symmetric in cos z gives exactly 0.5 (with or without a node at 0)
    xs = [-0.9, -0.5, -0.37, -0.1, -0.002]; ys = [0.48, 0.45, 0.94, 0.5, 0.487]
    for (x, y) in (([xs; -reverse(xs)], [ys; reverse(ys)]), ([xs; 0.0; -reverse(xs)], [ys; 0.4871; reverse(ys)]))
        g = exposure_day_night_fractions(tab(x, y))
        @test g.night === 0.5 && g.day === 0.5
    end
    # the symmetric fixture of the night_sym regression variant: both weightings, exactly 0.5
    sym = load_exposure(joinpath(@__DIR__, "fixtures", "solarExposure_symmetric.csv"))
    @test exposure_day_night_fractions(sym.time).night === 0.5
    @test exposure_day_night_fractions(sym.flux).night === 0.5
    @test sym.time.y == sym.flux.y
    # flat exposure on [-0.8, 0.4]: f_night = 0.8 / 1.2
    @test exposure_day_night_fractions(tab([-0.8, 0.4], [1.0, 1.0])).night ≈ 2 / 3 rtol=1e-15
    # a night-only (or day-only) table cannot fix the split
    @test_throws ErrorException exposure_day_night_fractions(tab([-0.9, 0.0], [1.0, 1.0]))
    @test_throws ErrorException exposure_day_night_fractions(tab([0.0, 0.9], [1.0, 1.0]))
    # reaches both sides but has no exposure on one of them
    @test_throws ErrorException exposure_day_night_fractions(tab([-0.9, 0.0, 0.9], [1.0, 0.0, 0.0]))
    @test_throws ErrorException exposure_day_night_fractions(tab([-0.9, 0.0, 0.9], [0.0, 0.0, 1.0]))
end

# Slow tier: the committed table is what utils/make_solar_exposure.py makes (~5 s, needs numpy).
if (@isdefined(RUN_SLOW) && RUN_SLOW) && success(pipeline(`python3 -c "import numpy"`; stderr = devnull))
    @testset "solar exposure table = generator output (SPEC 4.6)" begin
        mktempdir() do dir
            out = joinpath(dir, "expo.csv")
            @test success(pipeline(`python3 $(joinpath(@__DIR__, "..", "utils", "make_solar_exposure.py")) -o $out`; stdout = devnull))
            new, old = _csv_rows(out), _csv_rows(joinpath(_TAB_INPUTS, "solarExposure_v2.csv"))
            @test length(new) == length(old)
            @test all(isapprox.(reduce(vcat, new), reduce(vcat, old); rtol = 1e-8))
        end
    end
end

@testset "NightOsc Earth model – input table validation" begin
    good = joinpath(_TAB_INPUTS, "EARTH_MODEL_PREM_DENSE.dat")
    @test NightOsc.EarthModel(good; potential_const = MATTER_POTENTIAL_CONST) isa NightOsc.EarthModel
    mktempdir() do dir
        tab(name, lines) = (p = joinpath(dir, name); write(p, join(lines, "\n") * "\n"); p)
        bad = Dict(
            "unsorted"  => ["0.0 13.0 0.47", "0.6 5.0 0.47", "0.5 4.0 0.49", "1.0 3.3 0.49"],
            "duplicate" => ["0.0 13.0 0.47", "0.5 5.0 0.47", "0.5 4.0 0.49", "1.0 3.3 0.49"],
            "nan"       => ["0.0 13.0 0.47", "0.5 NaN 0.47", "1.0 3.3 0.49"],
            "negative"  => ["0.0 13.0 0.47", "0.5 -5.0 0.47", "1.0 3.3 0.49"],
            "neg_ye"    => ["0.0 13.0 0.47", "0.5 5.0 -0.47", "1.0 3.3 0.49"],
            "radius"    => ["0.0 13.0 0.47", "0.5 5.0 0.47", "1.2 3.3 0.49"],
            "columns"   => ["0.0 13.0", "1.0 3.3"],
        )
        for (k, lines) in bad
            @test_throws ErrorException NightOsc.EarthModel(tab("$k.dat", lines); potential_const = MATTER_POTENTIAL_CONST)
        end
    end
end

# A night grid whose lowest edge is the exposure table's first node must build: the zenith-node
# round trip c_of(xi_of(a)) can land 1 ulp below that edge, which an :error table would reject.
@testset "NightOsc exposure query stays inside the table at its lower edge" begin
    earth = NightOsc.EarthModel(joinpath(@__DIR__, "..", "inputs", "EARTH_MODEL_PREM_DENSE.dat");
                                potential_const = MATTER_POTENTIAL_CONST)
    cz, w = read_numeric_table(joinpath(@__DIR__, "..", "inputs", "solarExposure.csv");
                               name = "e", ncols = 2, header = false)
    # lower edges for which the round trip falls below the edge (found by scanning)
    for a0 in (-0.8307118644067797, -0.7866169491525423, -0.7131254237288136)
        keep = cz .> a0 + 1e-9
        expo = TabulatedInput([a0; cz[keep]], [w[1]; w[keep]]; name = "solar exposure",
                              source = "test", extrapolation = :error, register = false)
        @test NightOsc.BinnedP1e(collect(range(a0, 0.0; length = 9)), [0.004, 0.012, 0.02], expo, earth;
                                 phase_const = 1.0) isa NightOsc.BinnedP1e
        @test n_outside(expo) == (0, 0)
    end
end


@testset "SF-III solar-model files (high-Z GS98, low-Z AGSS09met)" begin
    # Published SF-III fluxes at Earth [cm^-2 s^-1] (Zenodo 10822316 v1.2, fluxes_SF3_*.dat):
    # guards against the files being swapped, as the B16 file names are.
    expected = Dict("SF3_GS98" => (b8 = 5.025e6, hep = 7.950e3, z = "high-Z"),
                    "SF3_AGSS09met" => (b8 = 4.135e6, hep = 8.216e3, z = "low-Z"))
    for (name, ex) in expected
        jldopen(joinpath(_TAB_INPUTS, name * ".jld2")) do f
            r, ne = f["radii"], f["n_e"]
            @test occursin(ex.z, f["model"])
            @test f["flux_8B"] == ex.b8 && f["flux_hep"] == ex.hep
            @test r[1] ≈ 5e-4 && r[end] ≈ 0.5 && all(diff(r) .≈ 5e-4)
            @test all(>(0), ne) && issorted(ne; rev = true) && 99 < ne[1] < 102   # N_A/cm^3
            for k in ("prodFractionBoron", "prodFractionHep")
                w = f[k]
                @test length(w) == length(r) && all(>=(0), w)
                @test sum(w) * 5e-4 ≈ 1 rtol = 1e-3                  # dP/d(R/R_sun) on the grid
            end
            # 8B is produced deeper than hep
            mean_r(w) = sum(w .* r) / sum(w)
            @test mean_r(f["prodFractionBoron"]) < mean_r(f["prodFractionHep"])
        end
    end
    # Composition from the profile data, not the metadata: each SF-III n_e profile is closer to the
    # B16 model of the same composition (B16 files: AGSS09_high_z.jld2 is B16-AGSS09met, low-Z, and
    # GS98_low_z.jld2 is B16-GS98, high-Z; the names are swapped).
    ne(name) = jldopen(f -> f["n_e"], joinpath(_TAB_INPUTS, name * ".jld2"))
    dist(a, b) = maximum(abs.(ne(a) ./ ne(b) .- 1))
    @test dist("SF3_GS98", "GS98_low_z") < dist("SF3_GS98", "AGSS09_high_z")
    @test dist("SF3_AGSS09met", "AGSS09_high_z") < dist("SF3_AGSS09met", "GS98_low_z")
    # the high-Z model has the higher core electron density
    @test ne("SF3_GS98")[1] > ne("SF3_AGSS09met")[1]
end

@testset "solar-model input contract (SPEC 8.2)" begin
    for name in ("SF3_GS98", "SF3_AGSS09met", "AGSS09_high_z", "GS98_low_z")
        m, fhep = load_solar_model(joinpath(_TAB_INPUTS, name * ".jld2"))
        @test keys(m) == (:radii, :prodFractionBoron, :prodFractionHep, :n_e, :avgNeBoron, :avgNeHep)
        @test m.avgNeBoron ≈ sum(m.prodFractionBoron .* m.n_e) / sum(m.prodFractionBoron)
        @test (fhep === nothing) == startswith(name, "AGSS09_high_z") | startswith(name, "GS98_low_z")
    end
    r = collect(range(0.01, 0.5; length = 50)); ne = fill(50.0, 50); p = ones(50)
    @test validate_solar_model(r, ne, p, p) === nothing
    @test_throws ErrorException validate_solar_model(r, ne[1:49], p, p)              # lengths
    @test_throws ErrorException validate_solar_model(reverse(r), ne, p, p)           # not increasing
    @test_throws ErrorException validate_solar_model([0.0; r[2:end]], ne, p, p)      # r = 0
    @test_throws ErrorException validate_solar_model(r, [0.0; ne[2:end]], p, p)      # n_e > 0
    @test_throws ErrorException validate_solar_model(r, ne, [-1.0; p[2:end]], p)     # weights >= 0
    @test_throws ErrorException validate_solar_model(r, ne, p, zeros(50))            # not all zero
    @test_throws ErrorException validate_solar_model(r, [NaN; ne[2:end]], p, p)      # finite
end
