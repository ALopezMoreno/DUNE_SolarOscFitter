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

const _IC_COSZ = -sqrt(max(0.0, 1.0 - (1221.0 / EARTH_RADIUS_KM)^2))  # inner-core boundary
const _CM_COSZ = -sqrt(max(0.0, 1.0 - (3480.0 / EARTH_RADIUS_KM)^2))  # core-mantle boundary

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
    fracs  = seg_lengths ./ sum(seg_lengths)
    counts = max.(1, floor.(Int, fracs .* n_total))
    rem    = fracs .* n_total .- counts
    for _ in 1:(n_total - sum(counts))
        idx = argmax(rem); counts[idx] += 1; rem[idx] -= 1.0
    end
    counts
end

# Segment breaks: exposure lower bound + any PREM boundaries inside the support.
const _SEG_BREAKS = let
    breaks = Float64[_COSZ_EXP_MIN]
    for b in [_IC_COSZ, _CM_COSZ]          # sorted ascending (most negative first)
        if _COSZ_EXP_MIN < b < cosz_bins.max
            push!(breaks, b)
        end
    end
    push!(breaks, cosz_bins.max)
    breaks
end
const _SEG_LENGTHS   = diff(_SEG_BREAKS)
const _N_COARSE_SEGS = _alloc_cosz_bins(cosz_bins.bin_number, _SEG_LENGTHS)

# Piecewise-uniform coarse bin edges (PREM boundaries on exact edges)
function _piecewise_cosz_edges(seg_breaks, n_per_seg)
    edges = [seg_breaks[1]]
    for i in eachindex(n_per_seg)
        append!(edges, collect(range(seg_breaks[i], seg_breaks[i+1]; length=n_per_seg[i]+1))[2:end])
    end
    edges
end

const COARSE_COSZ_EDGES = _piecewise_cosz_edges(_SEG_BREAKS, _N_COARSE_SEGS)

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