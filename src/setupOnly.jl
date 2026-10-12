#=
setupOnly.jl

RunMode "SETUP": build the detectors, the Asimov data and the likelihood, then stop.
Used by tests/regression_likelihood.jl and for interactive work on the likelihood.
=#
using Logging
using LinearAlgebra, Statistics, Distributions, StatsBase, DensityInterface

include(joinpath(@__DIR__, "setup.jl"))
