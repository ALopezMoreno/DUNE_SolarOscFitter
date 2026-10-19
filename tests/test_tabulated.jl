#=
Tabulated model inputs (SPEC 8.1): validation on construction, an explicit extrapolation policy
per input, never a silent extrapolation, linear interpolation through the nodes, the range used
at setup, and headerless tables that keep their first row.

The ported inputs (⁸B/hep spectra, CC cross section, solar exposure, Earth potential, makePaths
path potential) are compared with their pre-port implementations, which are reproduced below
verbatim from master (a9723b4) on the same files: values must be identical (===) at every node
and every midpoint.
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

    # Solar exposure (master: exposure.jl, CSV.File(header = false), Flat()): first row kept
    expo_file = joinpath(_TAB_INPUTS, "solarExposure.csv")
    df = CSV.File(expo_file, header = false) |> DataFrame
    old_expo = linear_interpolation(df[:, 1], df[:, 2], extrapolation_bc = Flat())
    expo = load_exposure(expo_file)
    @test length(expo.x) == countlines(expo_file) == nrow(df)
    @test expo.x[1] == parse(Float64, split(readline(expo_file), ',')[1])
    @test _same(expo, old_expo, _nodes_and_midpoints(df[:, 1]))
    @test expo.extrapolation === :error
    @test_throws TabulatedDomainError expo(expo.x[1] - 1e-6)

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

# SPEC 4.6: the day/night split of the exposure comes from the exposure table.
@testset "exposure day/night fractions (SPEC 4.6)" begin
    # independently of the fitter: trapezoid on the CSV rows, plus the linear value at cos z = 0
    rows = [parse.(Float64, split(l, ',')) for l in eachline(joinpath(_TAB_INPUTS, "solarExposure.csv")) if !isempty(strip(l))]
    c, e = first.(rows), last.(rows)
    k = findlast(<(0), c)
    e0 = e[k] + (e[k+1] - e[k]) * (0 - c[k]) / (c[k+1] - c[k])
    trap(x, y) = sum((x[2:end] .- x[1:end-1]) .* (y[2:end] .+ y[1:end-1]) ./ 2)
    fn_ref = trap([c[1:k]; 0.0], [e[1:k]; e0]) / trap(c, e)
    f = exposure_day_night_fractions(load_exposure(joinpath(_TAB_INPUTS, "solarExposure.csv")))
    @test f.night ≈ fn_ref rtol=1e-12
    @test f.night ≈ 0.4978 atol=5e-5
    @test f.day + f.night == 1
    @test f.night ≈ quadgk(load_exposure(joinpath(_TAB_INPUTS, "solarExposure.csv")), c[1], 0.0)[1] / quadgk(load_exposure(joinpath(_TAB_INPUTS, "solarExposure.csv")), c[1], 0.0, c[end])[1] rtol=1e-7

    tab(x, y) = TabulatedInput(x, y; name = "exposure", register = false)
    # a table mirror-symmetric in cos z gives exactly 0.5 (with or without a node at 0)
    xs = [-0.9, -0.5, -0.37, -0.1, -0.002]; ys = [0.48, 0.45, 0.94, 0.5, 0.487]
    for (x, y) in (([xs; -reverse(xs)], [ys; reverse(ys)]), ([xs; 0.0; -reverse(xs)], [ys; 0.4871; reverse(ys)]))
        g = exposure_day_night_fractions(tab(x, y))
        @test g.night === 0.5 && g.day === 0.5
    end
    # the shipped symmetric fixture of the night_sym regression variant
    sym = joinpath(@__DIR__, "fixtures", "solarExposure_symmetric.csv")
    @test exposure_day_night_fractions(load_exposure(sym)).night === 0.5
    # flat exposure on [-0.8, 0.4]: f_night = 0.8 / 1.2
    @test exposure_day_night_fractions(tab([-0.8, 0.4], [1.0, 1.0])).night ≈ 2 / 3 rtol=1e-15
    # a night-only (or day-only) table cannot fix the split
    @test_throws ErrorException exposure_day_night_fractions(tab([-0.9, 0.0], [1.0, 1.0]))
    @test_throws ErrorException exposure_day_night_fractions(tab([0.0, 0.9], [1.0, 1.0]))
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

