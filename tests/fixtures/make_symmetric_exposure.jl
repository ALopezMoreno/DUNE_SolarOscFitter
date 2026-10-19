#=
make_symmetric_exposure.jl — tests/fixtures/solarExposure_symmetric.csv, the exposure table of the
regression variant "night_sym" (SPEC 4.6).

The night side (cos z ≤ 0) is the shipped inputs/solarExposure.csv, with a node added at cos z = 0
holding the shipped interpolant's value there, so the night exposure (exposure_weights, NightOsc) is
the shipped one up to rounding. The day side is its mirror image, so the day/night split is exactly
0.5: with it the likelihood must equal the one before the split came from the exposure file (the
hard-coded 0.5/0.5).

    julia --project tests/fixtures/make_symmetric_exposure.jl
=#
include(joinpath(@__DIR__, "..", "..", "src", "modelInputs.jl"))

expo = load_exposure(joinpath(@__DIR__, "..", "..", "inputs", "solarExposure.csv"))
keep = expo.x .< 0
xn, yn = [expo.x[keep]; 0.0], [expo.y[keep]; Float64(expo.itp(0.0))]
x = [xn; -reverse(xn[1:end-1])]
y = [yn; reverse(yn[1:end-1])]
out = joinpath(@__DIR__, "solarExposure_symmetric.csv")
open(out, "w") do io
    for (a, b) in zip(x, y)
        println(io, repr(a), ",", repr(b))
    end
end
f = exposure_day_night_fractions(load_exposure(out))
f.night == 0.5 || error("symmetric table gives f_night = $(f.night)")
println("written $out ($(length(x)) rows, f_night = $(f.night))")
