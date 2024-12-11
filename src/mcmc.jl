"""
This script performs Bayesian parameter estimation for neutrino oscillation parameters using Markov Chain 
Monte Carlo (MCMC) sampling. It leverages the BAT.jl library to sample from the posterior distribution 
defined by a likelihood function and prior distributions over the parameters of interest.

Modules and Libraries:
- Utilizes Julia packages such as `LinearAlgebra`, `Statistics`, `Distributions`, `StatsBase`, `BAT`, 
  `DensityInterface`, `IntervalSets`, `Plots`, and `JLD2` for mathematical operations, statistical 
  distributions, Bayesian analysis, plotting, and data storage.
- Includes a setup script from `../src/setup.jl` to configure the fitting process.

Parameters:
- `prior`: A product of distributions defining uniform priors for the squared sine of mixing angles 
  (`sin²θ₁₂`, `sin²θ₁₃`) and the squared mass difference (`Δm²₂₁`).
- `fast`: A boolean flag that determines which likelihood function to use (`likelihood_all_samples_ctr` 
  or `likelihood_all_samples_avg`).

Process:
1. Defines a Bayesian model using the `PosteriorMeasure` with the specified likelihood and prior.
2. Configures MCMC chain parameters, including initialization, burn-in, and convergence settings.
3. Executes MCMC sampling using the Metropolis-Hastings algorithm, logging the number of chains and steps.
4. Extracts parameter samples and metadata (step number and chain ID) from the MCMC results.
5. Saves the extracted data and additional sample information to a JLD2 file for further analysis.

Output:
- A JLD2 file containing arrays of sampled parameter values (`sin²θ₁₂`, `sin²θ₁₃`, `Δm²₂₁`), step numbers, 
  chain IDs, and sample data (`ES_nue`, `ES_nuother`, `CC`).

Note:
- The script assumes the existence of certain global variables such as `mcmcChains`, `mcmcSteps`, 
  `tuningSteps`, `maxTuningAttempts`, `outFile`, and `ereco_data`.
- Logging is set up but commented out; adjust the logging level as needed for debugging.
"""

using Logging

# DEBUGGING AND TESTING: Set the logging level to Warn to suppress Info messages
# global_logger(ConsoleLogger(stderr, Logging.Warn))

using LinearAlgebra, Statistics, Distributions, StatsBase, BAT, DensityInterface, IntervalSets
using Plots, JLD2

include("../src/setup.jl")

# Set uniform priors in reasonable parameter regions
prior = distprod(
    sin2_th12=prior_sin2_th12,
    sin2_th13= prior_sin2_th13, # Uniform(0.0001, 0.1), #Truncated(Normal(0.022, 0.0007), 0.0001, 0.035), # Fix to reactor data for now
    dm2_21=prior_dm2_21
)

# Define Bayesian model
if fast
    posterior = PosteriorMeasure(likelihood_all_samples_ctr, prior)
else
    posterior = PosteriorMeasure(likelihood_all_samples_avg, prior)
end

# Set chain parameters
init = MCMCChainPoolInit(
    init_tries_per_chain=IntervalSets.ClosedInterval(1, 180),  # Example interval
    nsteps_init=1000,
    initval_alg=InitFromTarget()
)


burnin = MCMCMultiCycleBurnin(
    nsteps_per_cycle=tuningSteps,
    max_ncycles=maxTuningAttempts,
    nsteps_final=tuningSteps/10
)

convergence = AssumeConvergence()

println("running $mcmcChains chains with $mcmcSteps steps")
println("tuning will be performed with $tuningSteps steps up to a maximum of $maxTuningAttempts times")

# Run MCMC
@time samples = bat_sample(posterior, MCMCSampling(mcalg=MetropolisHastings(),
                                            nsteps=mcmcSteps,
                                            nchains=mcmcChains,
                                            init=init,
                                            burnin=burnin,
                                            #convergence=convergence
                                            )).result


# println("Truth: $true_par_values")
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

sample_ES_nue = ereco_data.ES_nue
sample_ES_nuother = ereco_data.ES_nuother
sample_CC = ereco_data.CC

@save outFile*"_mcmc.jld2" sin2_th12 sin2_th13 dm2_21 stepno chainid sample_ES_nue sample_ES_nuother sample_CC


# Unshape sample for statistical treatment
# unshaped_samples, f_flatten = bat_transform(Vector, samples)
# this can be used to generate a covariance matrix