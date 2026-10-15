#=
make_fixtures.jl — frozen synthetic detector samples for tests/regression_likelihood.jl.

The real detector and background samples in inputs/ are still changing; the likelihood
regression must not. These samples are small, synthetic and only loosely physical (shapes and
rates are plausible, not DUNE's). They exist to pin the fitter's arithmetic, not to measure
anything. Energies are in GeV, as in the real inputs.

The CSVs are committed. Regenerating them changes every regression value, so only run this
together with `julia --project tests/regression_likelihood.jl <variant> --update`:

    julia --project tests/fixtures/make_fixtures.jl
=#
using CSV, DataFrames, Random

const DIR = @__DIR__
rng = Xoshiro(20261015)

clip(x, lo, hi) = min(max(x, lo), hi)
write_fixture(name, df) = CSV.write(joinpath(DIR, name), df)

# ES signal: recoil energy a random fraction of Etrue, 8 % resolution, selection above 3 MeV
# with 90 % efficiency. Some events reconstruct outside the analysis window on purpose.
function es_signal(n)
    Etrue = 1e-3 .* (2 .+ 18 .* rand(rng, n))
    Ereco = Etrue .* sqrt.(rand(rng, n)) .* (1 .+ 0.08 .* randn(rng, n))
    mask  = (Ereco .> 3e-3) .& (rand(rng, n) .< 0.9)
    DataFrame(Etrue=Etrue, Ereco=Ereco, mask=mask)
end

# ES angular: forward-peaked, broader at low energy.
function es_angular(n)
    Ereco = 1e-3 .* (1 .+ 20 .* rand(rng, n))
    width = 0.05 .+ 0.3e-3 ./ Ereco
    cosθ  = clip.(1 .- abs.(width .* randn(rng, n)), -1.0, 1.0)
    DataFrame(Ereco=Ereco, cos_scatter=cosθ)
end

# CC signal: Ereco ≈ Etrue − 1.5 MeV with 7 % resolution, 80 % selection above 3 MeV.
function cc_signal(n)
    Etrue = 1e-3 .* (2 .+ 19 .* rand(rng, n))
    Ereco = max.(Etrue .- 1.5e-3, 1e-4) .* (1 .+ 0.07 .* randn(rng, n))
    mask  = (Ereco .> 3e-3) .& (rand(rng, n) .< 0.8)
    DataFrame(Etrue=Etrue, Ereco=Ereco, weights=ones(n), mask=mask)
end

# Backgrounds: falling exponential spectra; ES ones carry a `side`, CC ones non-uniform weights.
function es_background(n, slope_MeV, side; weights=false)
    Ereco = 1e-3 .* (1 .+ slope_MeV .* randexp(rng, n))
    df = DataFrame(Ereco=Ereco, mask=rand(rng, n) .< 0.3, side=fill(side, n))
    weights && insertcols!(df, 2, :weights => 0.5 .+ rand(rng, n))
    df
end

function cc_background(n, slope_MeV)
    Ereco = 1e-3 .* (2 .+ slope_MeV .* randexp(rng, n))
    DataFrame(Ereco=Ereco, weights=0.5 .+ 2.5 .* rand(rng, n), mask=rand(rng, n) .< 0.5)
end

write_fixture("es_nue.csv",      es_signal(6000))
write_fixture("es_nuother.csv",  es_signal(6000))
write_fixture("es_angular.csv",  es_angular(6000))
write_fixture("cc_signal.csv",   cc_signal(6000))
write_fixture("es_bg_long.csv",  es_background(3000, 2.0, 0))
write_fixture("es_bg_end.csv",   es_background(3000, 3.0, 1; weights=true))
write_fixture("cc_bg_a.csv",     cc_background(2000, 2.5))
write_fixture("cc_bg_b.csv",     cc_background(2000, 4.0))
println("fixtures written to $DIR")
