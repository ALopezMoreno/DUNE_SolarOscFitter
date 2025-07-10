#=
earthProfile.jl

Earth density profile loading and processing for neutrino oscillation calculations.
This module loads the Earth's density structure and creates interpolated functions
for calculating matter effects during neutrino propagation through Earth.

Key Features:
- Loading of Earth density models from data files
- Conversion of density profiles to neutrino matter potentials
- Linear interpolation for continuous density functions
- Zenith angle binning for path calculations
- Support for different Earth model parameterizations

The Earth matter potential is crucial for calculating neutrino oscillations
during nighttime propagation when neutrinos pass through Earth's interior.

Author: [Author name]
=#

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
    linear_interp = LinearInterpolation(x, y; extrapolation_bc=Throw())
    
    return linear_interp
end

# Load and process Earth model
earth_model = load_earth_model(earthModelFile)
global earth = create_interpolated_model(earth_model)

# Set up zenith angle arrays for neutrino path calculations
# High-resolution array for detailed calculations
global cosz_calc = collect(range(cosz_bins.min, stop=cosz_bins.max, length=cosz_bins.bin_number * 3))
# Analysis-resolution array for final binning
global cosz = collect(range(cosz_bins.min, stop=cosz_bins.max, length=cosz_bins.bin_number))