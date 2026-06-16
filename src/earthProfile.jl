using DelimitedFiles   # For reading Earth model data files
using Interpolations   # For creating interpolated density functions

# Earth physical constants
const EARTH_RADIUS_KM = 6371.0  # Earth radius in kilometers

function load_earth_model(file_path::String)
    """
    Load Earth density model from a data file.
    
    Expected file format: whitespace-delimited columns
    Column 1: Radius factor (fraction of Earth radius)
    Column 2: Density (g/cm³)
    Column 3: Electron fraction
    
    Returns:
    Dictionary with radius (km) and matter potential arrays
    """
    # Load the data from file (assumes whitespace delimited)
    data = readdlm(file_path)
    
    # Extract and convert data columns
    # Small factor (1.0000000001) added to avoid numerical issues at boundaries
    radius = data[:, 1] .* EARTH_RADIUS_KM .* 1.0000000001
    density = data[:, 2]        # Density in g/cm³
    e_fraction = data[:, 3]     # Electron fraction
    
    # Compute neutrino matter potential
    # Factor 1.52588e-4 converts density×electron_fraction to neutrino potential
    potential = density .* e_fraction .* 1.52588e-4
    
    return Dict(:radius => radius, :potential => potential)
end

function create_interpolated_model(earth_model::Dict)
    """
    Create an interpolated function for the Earth matter potential.
    
    Uses linear interpolation between data points with error checking
    for out-of-range requests.
    
    Returns:
    Interpolated function that can be evaluated at any radius
    """
    x = earth_model[:radius]    # Radius values (km)
    y = earth_model[:potential] # Matter potential values
    
    # Create linear interpolation function
    # extrapolation_bc=Throw() ensures error for out-of-range values
    linear_interp = linear_interpolation(x, y; extrapolation_bc=Throw())
    
    return linear_interp
end

# Load and process Earth model
earth_model = load_earth_model(earthModelFile)
global earth = create_interpolated_model(earth_model)

# Set up zenith angle arrays for neutrino path calculations.
#
# Coarse grid is piecewise-uniform with bin edges at PREM discontinuities so no
# bin straddles a density jump.  Bins are allocated only within the compact
# support of the solar exposure (the range of cos(zenith) the detector actually
# sees); PREM boundaries outside that support are dropped.
#
# N_COSZ_SUB fine midpoints per coarse bin are used for oscillation computation,
# then block-averaged back to coarse resolution.  N_COSZ_SUB=2 is needed only
# when nBins_cosz < 40; at ≥ 40 the coarse grid already meets the Nyquist
# requirement set by the slowest oscillation in the fit range.

# Detect the two largest density discontinuities from the loaded potential profile.
# potential = density × e_fraction × const, so its jumps track density jumps.
# Returns the two boundary cosz values sorted ascending (most negative = deepest first).
const _IC_COSZ, _CM_COSZ = let
    r      = earth_model[:radius]      # km, centre→surface
    p      = earth_model[:potential]
    jumps  = abs.(diff(p))
    top2   = sort(sortperm(jumps, rev=true)[1:2])   # indices of two largest jumps
    r_bdry = [(r[i] + r[i+1]) / 2.0 for i in top2]
    cosz_v = sort([-sqrt(max(0.0, 1.0 - (rb / EARTH_RADIUS_KM)^2)) for rb in r_bdry])
    cosz_v[1], cosz_v[2]   # (inner-core boundary, core-mantle boundary)
end

# Lower bound of the exposure compact support on the night side.
# Falls back to cosz_bins.min for scripts that don't set solarExposureFile.
const _COSZ_EXP_MIN = let
    cosz_floor = cosz_bins.min
    if @isdefined(solarExposureFile) && isfile(solarExposureFile)
        raw = readdlm(solarExposureFile, ',')
        night_cosz = filter(c -> c <= 0.0, Float64.(raw[:, 1]))
        isempty(night_cosz) ? cosz_floor : minimum(night_cosz)
    else
        cosz_floor
    end
end

function _alloc_cosz_bins(n_total, seg_lengths)
    # max.(1, …) forces ≥1 bin per segment, so a zone with fewer bins than PREM
    # segments would overflow (sum(counts) > n_total) and the correction loop below,
    # whose range goes empty, could not claw it back → wrong total edge count.
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

# Fine cos(z) edges from `edge` (negative) up to 0. The day-night regeneration is an
# OSCILLATION in cos(z) whose frequency encodes Δm²₂₁; estimating a frequency cleanly
# requires UNIFORM sampling, so the fine zone is uniform (p=1) — graded spacing is a
# varying sample rate that would distort the frequency content. (p>1 horizon-weights the
# bins; kept as an option but not recommended for this measurement.)
function _graded_fine_edges(edge, n; p=1.0)
    a = abs(edge)
    [-(a * (j / n)^p) for j in n:-1:0]     # increasing: edge … 0; p=1 ⇒ uniform spacing
end

# Coarse cos(z) edges. With nBins_cosz_fine>0: a graded fine zone in [cosz_fine_edge, 0]
# (horizon-refined for the day-night first-peak) plus a PREM-aligned coarse zone below;
# otherwise the legacy uniform-piecewise grid. Total stays at cosz_bins.bin_number.
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

# The coarse grid must have exactly nBins_cosz+1 edges, else cosz, cosz_calc and
# exposure_weights desync (silent mis-binning / shape errors downstream).
@assert length(COARSE_COSZ_EDGES) == cosz_bins.bin_number + 1 "COARSE_COSZ_EDGES has $(length(COARSE_COSZ_EDGES)) edges, expected $(cosz_bins.bin_number + 1)"

# N_COSZ_SUB=2 only when the coarse grid is too sparse to meet Nyquist alone.
const N_COSZ_SUB = cosz_bins.bin_number < 40 ? 2 : 1

# Fine grid: N_COSZ_SUB midpoints per coarse bin
global cosz_calc = let
    cz = Float64[]
    for i in 1:cosz_bins.bin_number
        lo, hi = COARSE_COSZ_EDGES[i], COARSE_COSZ_EDGES[i+1]
        w = hi - lo
        append!(cz, [lo + (k - 0.5) * w / N_COSZ_SUB for k in 1:N_COSZ_SUB])
    end
    cz
end

# Coarse bin centres for analysis-resolution arrays
global cosz = [(COARSE_COSZ_EDGES[i] + COARSE_COSZ_EDGES[i+1]) / 2 for i in 1:cosz_bins.bin_number]