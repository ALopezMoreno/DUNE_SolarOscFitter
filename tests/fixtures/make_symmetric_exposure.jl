#=
make_symmetric_exposure.jl — tests/fixtures/solarExposure_symmetric.csv, the exposure table of the
regression variant "night_sym" (SPEC 4.6).

The night side (cos z ≤ 0) is the legacy single-weight inputs/solarExposure.csv, with a node added at
cos z = 0 holding its interpolant's value there; the day side is its mirror image. Both weight columns
(time_weight, flux_weight) are this one column, so signal and backgrounds have the night shape of the
legacy file and both splits are exactly 0.5: with it the likelihood must equal the one of the
hard-coded 0.5/0.5 split with the legacy file (bf9ffec), bit for bit.

    julia --project tests/fixtures/make_symmetric_exposure.jl
=#
include(joinpath(@__DIR__, "..", "..", "src", "modelInputs.jl"))

legacy = joinpath(@__DIR__, "..", "..", "inputs", "solarExposure.csv")
cz, w = read_numeric_table(legacy; name = "legacy solar exposure", ncols = 2, header = false)
expo = TabulatedInput(cz, w; name = "legacy solar exposure", source = legacy, units = "cos z",
                      extrapolation = :error, nonnegative = true, register = false)
keep = expo.x .< 0
xn, yn = [expo.x[keep]; 0.0], [expo.y[keep]; Float64(expo.itp(0.0))]
x = [xn; -reverse(xn[1:end-1])]
y = [yn; reverse(yn[1:end-1])]
out = joinpath(@__DIR__, "solarExposure_symmetric.csv")
open(out, "w") do io
    println(io, "# Symmetric exposure for the night_sym regression variant: the legacy inputs/solarExposure.csv")
    println(io, "# night side mirrored onto the day side; both weightings equal. Made by make_symmetric_exposure.jl.")
    println(io, "cosz,time_weight,flux_weight")
    for (a, b) in zip(x, y)
        println(io, repr(a), ",", repr(b), ",", repr(b))
    end
end
t = load_exposure(out)
for col in (t.time, t.flux)
    f = exposure_day_night_fractions(col)
    f.night == 0.5 || error("symmetric table gives f_night = $(f.night)")
end
println("written $out ($(length(x)) rows, f_night = 0.5 for both weightings)")
