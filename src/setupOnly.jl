# RunMode SETUP: build the detectors, the Asimov data and the likelihood, then stop.
using Logging
using LinearAlgebra, Statistics, Distributions, StatsBase, DensityInterface

include(joinpath(@__DIR__, "setup.jl"))
