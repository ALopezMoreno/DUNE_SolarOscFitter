using Logging

# DEBUGGING AND TESTING: Set the logging level to Warn to suppress Info messages
# global_logger(ConsoleLogger(stderr, Logging.Warn))

using LinearAlgebra, Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using Plots, JLD2

include("../src/oscCalc.jl")
include("../src/reweighting.jl")
include("../src/statsLikelihood.jl")


# Set uniform priors in reasonable parameter regions
prior = distprod(
    sin2_th12=Uniform(0.0001, 0.9999),
    sin2_th13=Uniform(0.0001, 0.9999),
    dm2_21=Uniform(1e-8, 2 * 1e-4)
)

# Define Bayesian model
posterior = PosteriorMeasure(likelihood, prior)

# Run MCMC
# Measure elapsed time


samples = bat_sample(posterior, MCMCSampling(mcalg=MetropolisHastings(), nsteps=10^5, nchains=4)).result


println("Truth: $true_par_values")
println("Mode: $(mode(samples))")
println("Mean: $(mean(samples))")
println("Stddev: $(std(samples))")

# Initialize arrays to store the extracted data
sin2_th12 = Float64[]
sin2_th13 = Float64[]
dm2_21 = Float64[]
stepno = Int64[]
chainid = Int32[]  # Assuming chainid is Int32 based on your structure

# Iterate over each sample and extract the desired fields
for sample in samples
    # Extract values from the NamedTuple `v`
    push!(sin2_th12, sample.v.sin2_th12)
    push!(sin2_th13, sample.v.sin2_th13)
    push!(dm2_21, sample.v.dm2_21)
    
    # Extract values from the `info` StructVector
    push!(stepno, sample.info.stepno)
    push!(chainid, sample.info.chainid)
end

@save "outputs/testFit.jld2" sin2_th12 sin2_th13 dm2_21 stepno chainid


# Unshape sample for statistical treatment
unshaped_samples, f_flatten = bat_transform(Vector, samples)
# this can be used to generate a covariance matrix


plot(
    samples,
    mean=false, std=false, globalmode=true, marginalmode=false,
    nbins=50
)

savefig("images/corner.png")