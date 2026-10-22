#=
osc_night.jl — NightOsc: P(ν₁→νₑ) averaged over each (cos z, E_true) bin, two-flavour, for any
radial Earth. Chords are cut into steps with their own potential and slope, and the fast terms are
integrated over the bins in closed form; differentiable (ForwardDiff, Enzyme) in Δm²₂₁, θ₁₂, θ₁₃
and the layer densities. Port of SnuPpy's binned night calculation. night/: earth_model.jl,
kernels.jl, binning.jl (BinnedP1e), ad.jl (derivative rules).
=#
module NightOsc

using DelimitedFiles: readdlm
using Enzyme
using Enzyme: EnzymeRules
using ForwardDiff
using LinearAlgebra
using QuadGK: gauss
using Statistics: median

export EarthModel, PathSegment, build_earth_path, chord_length, p1e_paths, BinnedP1e, binned_p1e

#################################################################################
# Constants
#################################################################################

const EARTH_RADIUS_KM = 6371.0
const DEFAULT_DEPTH_KM = 0.001

# Accuracy settings, validated to ≤ 9e-4 per bin on the fitter grid (also with ±24% layer densities)
const STEP_KM     = 300.0     # longest step in a layer crossing
const XI_NODES    = (8, 3)    # paths per cos z interval: 8 + 3k, k = boundaries crossed
const XI_ORDER    = 6         # points in the zenith interpolation
const XI_PIECE    = 0.1       # longest closed-form zenith piece (in ξ)
const E_PIECE_GEV = 1.0e-3    # longest closed-form energy piece
const MIN_E_NODES = 4         # fewest energy nodes in an interval
const E_ORDER     = 4         # points in the energy interpolation (_interp_loop! assumes 4)

# Energy nodes per MeV for chords crossing k boundaries
NODES_PER_MEV(k) = k <= 4 ? 2 + (k >= 2) : 2 + k ÷ 2

# Earth model and chords
const LAYER_JUMP_FACTOR  = 5.0    # density step > this × median step marks a layer boundary
const JUMP_HALF_WIDTH_KM = 1e-9   # half-width of the step entered at each boundary
const SEGMENT_GL_NODES   = 8      # Gauss-Legendre nodes for a step's average potential and slope
const STEP_SCAN_POINTS   = 41     # cos z samples used to size the steps of each layer

# Series / closed-form switches (shared by _filon and _filon_adj)
const SERIES_LINEAR_MOMENT   = 1e-3   # |κL| in _linear_moment
const SERIES_ROTATION        = 1e-2   # x² in _step
const SERIES_NODE_Z          = 1e-3   # z in _node!
const BLEND_NODE_Z           = 6.0    # z in _node! above which exp(-z²) is negligible
const FILON_CURVATURE_SERIES = 0.05   # phase curvature in _filon
const FILON_STEP_SERIES      = 1e-2   # phase step in _filon

include(joinpath(@__DIR__, "night", "earth_model.jl"))
include(joinpath(@__DIR__, "night", "kernels.jl"))
include(joinpath(@__DIR__, "night", "binning.jl"))
include(joinpath(@__DIR__, "night", "ad.jl"))

end # module NightOsc
