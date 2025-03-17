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

# Override mv_proposaldist to inject a custom covariance matrix
function mv_proposaldist(T::Type{<:AbstractFloat}, d::TDist, varndof::Integer)
  df = only(Distributions.params(d))
  μ = fill(zero(T), varndof)
  # Use cov matrix if available, otherwise fall back to the identity
  Σ = (propMatrix !== nothing) ? propMatrix : PDMat(Matrix(I(varndof) * one(T)))
  
  # Construct a multivariate t-distribution with your custom covariance.
  # Use MvTDist from Distributions, which takes for arguments:
  #   - degrees of freedom (as a scalar),
  #   - mean vector, and
  #   - scale matrix (a regular matrix)
  return MvTDist(convert(T, df), μ, Matrix(Σ))
end

# Set prior distributions from config
prior = distprod(
    # priors on mixing parameters
    sin2_th12=prior_sin2_th12,
    sin2_th13= prior_sin2_th13,
    dm2_21=prior_dm2_21,

    # priors on systematic parameters
    integrated_8B_flux=prior_8B_flux

    # priors on nuisance parameters
    
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
    nsteps_init=500,
    initval_alg=InitFromTarget()
)

burninAttempts = max(1, maxTuningAttempts)

burnin = MCMCMultiCycleBurnin(
    nsteps_per_cycle=tuningSteps,
    max_ncycles=burninAttempts,
    nsteps_final=tuningSteps/10
)


convergence = AssumeConvergence()

# Check if there is a covariance matrix for the step proposal function
if propMatrix !== nothing
  # Use BAT's GenericMvTDist for a multivariate t-distribution proposal.
  d = size(propMatrix, 1)
  df = 1.5
  proposal_distribution = TDist(df)
  @logmsg MCMC "found proposal covariance matrix: using MvTDist with df = $df."
else
  # Default to a univariate T distribution if no covariance matrix is provided
  proposal_distribution = TDist(1.0)
  @logmsg MCMC "no proposal covariance matrix given. Running with default parameters"
end


@logmsg MCMC "running $mcmcChains chains with $mcmcSteps steps each."


# Skip tuning if desired
if maxTuningAttempts == 0
  proposal_algorithm = MetropolisHastings(proposal=proposal_distribution, tuning = AdaptiveMHTuning(α=ClosedInterval(0, 1)))
  @logmsg MCMC "skipping the tuning stage."
else
  proposal_algorithm = MetropolisHastings(proposal=proposal_distribution)
  @logmsg MCMC "tuning will be performed with $tuningSteps steps up to a maximum of $maxTuningAttempts times."
end

println(" ")  

# Run MCMC
@time samples = bat_sample(posterior, MCMCSampling(mcalg=proposal_algorithm,
                                            nsteps=mcmcSteps,
                                            nchains=mcmcChains,
                                            init=init,
                                            burnin=burnin,
                                            convergence=convergence
                                            )).result


println(" ")  

@logmsg Setup ("Truth: $true_params")

println(" ")

@logmsg Output "Mode: $(mode(samples))"
@logmsg Output "Mean: $(mean(samples))"
@logmsg Output "Stddev: $(std(samples))"

# Initialize arrays to store the extracted data
sin2_th12 = Float64[]
sin2_th13 = Float64[]
dm2_21 = Float64[]
integrated_8B_flux = Float64[]
stepno = Int64[]
chainid = Int32[]  # Assuming chainid is Int32 based on your structure

# Iterate over each sample and extract the desired fields
for sample in samples
    # Extract values from the NamedTuple `v`
    push!(sin2_th12, sample.v.sin2_th12)
    push!(sin2_th13, sample.v.sin2_th13)
    push!(dm2_21, sample.v.dm2_21)

    # Extract nuisance parameters
    push!(integrated_8B_flux, sample.v.integrated_8B_flux)
    
    # Extract values from the `info` StructVector
    push!(stepno, sample.info.stepno)
    push!(chainid, sample.info.chainid)
end

# sample_ES_nue = ereco_data.ES_nue_night
# sample_ES_nuother = ereco_data.ES_nuother_night
# sample_CC = ereco_data.CC_night

@save outFile*"_mcmc.jld2" sin2_th12 sin2_th13 dm2_21 integrated_8B_flux stepno chainid 
# sample_ES_nue sample_ES_nuother sample_CC

println(" ")
@logmsg Output "$(mcmcChains) output chain with $(size(stepno)) steps saved to : $(outFile)"

# Unshape sample for statistical treatment
# unshaped_samples, f_flatten = bat_transform(Vector, samples)
# this can be used to generate a covariance matrix