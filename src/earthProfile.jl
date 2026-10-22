# Earth model (load_earth_model in modelInputs.jl) and the cos z grids
include(joinpath(@__DIR__, "modelInputs.jl"))

earth_model = load_earth_model(earthModelFile)
global earth = create_interpolated_model(earth_model)

# cos z grids. The coarse (analysis) grid is piecewise uniform with edges at the core boundaries, so
# no bin straddles a density jump, and covers only the exposure's support. Oscillations are computed
# at N_COSZ_SUB midpoints per coarse bin and block-averaged back.

# cos z of the inner-core and core-mantle boundaries: the two largest jumps of the potential
const _IC_COSZ, _CM_COSZ = let
    r      = earth_model[:radius]      # km, centre→surface
    p      = earth_model[:potential]
    jumps  = abs.(diff(p))
    top2   = sort(sortperm(jumps, rev=true)[1:2])
    r_bdry = [(r[i] + r[i+1]) / 2.0 for i in top2]
    cosz_v = sort([-sqrt(max(0.0, 1.0 - (rb / EARTH_RADIUS_KM)^2)) for rb in r_bdry])
    cosz_v[1], cosz_v[2]   # (inner-core boundary, core-mantle boundary)
end

# Lowest night cos z of the exposure table (cosz_bins.min without solarExposureFile)
const _COSZ_EXP_MIN = let
    cosz_floor = cosz_bins.min
    if @isdefined(solarExposureFile) && isfile(solarExposureFile)
        cz = load_exposure(solarExposureFile).time.x
        night_cosz = filter(c -> c <= 0.0, cz)
        isempty(night_cosz) ? cosz_floor : minimum(night_cosz)
    else
        cosz_floor
    end
end

function _alloc_cosz_bins(n_total, seg_lengths)
    # at least one bin per segment
    n_total >= length(seg_lengths) ||
        error("_alloc_cosz_bins: $n_total bins cannot cover $(length(seg_lengths)) PREM " *
              "segments (need ≥1 each). Lower nBins_cosz_fine or raise nBins_cosz.")
    fracs  = seg_lengths ./ sum(seg_lengths)
    counts = max.(1, floor.(Int, fracs .* n_total))
    rem    = fracs .* n_total .- counts
    for _ in 1:(n_total - sum(counts))
        idx = argmax(rem); counts[idx] += 1; rem[idx] -= 1.0
    end
    counts
end

# Piecewise-uniform coarse bin edges (PREM boundaries on exact edges)
function _piecewise_cosz_edges(seg_breaks, n_per_seg)
    edges = [seg_breaks[1]]
    for i in eachindex(n_per_seg)
        append!(edges, collect(range(seg_breaks[i], seg_breaks[i+1]; length=n_per_seg[i]+1))[2:end])
    end
    edges
end

# PREM-aligned segment breaks within [lo, hi] (mantle/core boundaries land on exact edges).
function _prem_breaks(lo, hi)
    breaks = Float64[lo]
    for b in [_IC_COSZ, _CM_COSZ]          # ascending (most negative first)
        lo < b < hi && push!(breaks, b)
    end
    push!(breaks, hi)
    breaks
end

# Fine cos z edges from `edge` up to 0. Keep p = 1 (uniform): the regeneration oscillates in cos z
# with a frequency set by Δm²₂₁, and graded spacing would distort it.
function _graded_fine_edges(edge, n; p=1.0)
    a = abs(edge)
    [-(a * (j / n)^p) for j in n:-1:0]
end

# Analysis cos z edges: with nBins_cosz_fine > 0 a fine zone [cosz_fine_edge, 0] above a
# PREM-aligned coarse zone, otherwise the uniform-piecewise grid; nBins_cosz bins in total.
const COARSE_COSZ_EDGES = let
    n_tot = cosz_bins.bin_number
    if nBins_cosz_fine > 0 && _COSZ_EXP_MIN < cosz_fine_edge < cosz_bins.max
        n_fine      = min(nBins_cosz_fine, n_tot - 1)
        n_deep      = n_tot - n_fine
        deep_breaks = _prem_breaks(_COSZ_EXP_MIN, cosz_fine_edge)
        deep_edges  = _piecewise_cosz_edges(deep_breaks, _alloc_cosz_bins(n_deep, diff(deep_breaks)))
        fine_edges  = _graded_fine_edges(cosz_fine_edge, n_fine)
        vcat(deep_edges, fine_edges[2:end])    # share the cosz_fine_edge node
    else
        breaks = _prem_breaks(_COSZ_EXP_MIN, cosz_bins.max)
        _piecewise_cosz_edges(breaks, _alloc_cosz_bins(n_tot, diff(breaks)))
    end
end

# cosz, cosz_calc and the exposure weights all assume this
@assert length(COARSE_COSZ_EDGES) == cosz_bins.bin_number + 1 "COARSE_COSZ_EDGES has $(length(COARSE_COSZ_EDGES)) edges, expected $(cosz_bins.bin_number + 1)"

# const for type stability (Enzyme)
const N_COSZ_BINS = cosz_bins.bin_number

# 2 only where the coarse grid alone would undersample the slowest oscillation
const N_COSZ_SUB = cosz_bins.bin_number < 40 ? 2 : 1

global cosz_calc = let
    cz = Float64[]
    for i in 1:cosz_bins.bin_number
        lo, hi = COARSE_COSZ_EDGES[i], COARSE_COSZ_EDGES[i+1]
        w = hi - lo
        append!(cz, [lo + (k - 0.5) * w / N_COSZ_SUB for k in 1:N_COSZ_SUB])
    end
    cz
end

global cosz = [(COARSE_COSZ_EDGES[i] + COARSE_COSZ_EDGES[i+1]) / 2 for i in 1:cosz_bins.bin_number]