using JLD2
using Distributions

include("../src/oscCalc.jl")
include("../src/statsLikelihood.jl")

#energies = collect(range(1, stop=10, length=500))
energies = 10 .^ range(log10(0.1), stop=log10(20), length=1000)

energies *= 1e-3 # MeV not GeV

params = OscillationParameters()
n_e = 103.0

results = Vector{SVector{3,Float64}}()
llhs = Vector{Float64}()

for E_nu in energies
    probs = peanutsProb(params, E_nu, n_e)
    push!(results, probs)
end

# Save energgenerate_poisson_throwsies and results to a JLD2 file
@save "outputs/oscProbs.jld2" energies results


lambda = 32.0
num_throws = 1000
poisson_dist = Poisson(lambda)
throws = rand(poisson_dist, num_throws) .+ 20

for throw in throws
    vecThrow = fill(Float64(throw), 1)
    llh = poissonLogLikelihood(fill(lambda, 1), vecThrow)
    push!(llhs, llh)
end

# Save llh to plot in python
@save "outputs/poissonLlh.jld2" llhs