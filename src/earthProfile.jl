using DelimitedFiles   # For reading .dat file
using Interpolations   # For linear interpolation

const EARTH_RADIUS_KM = 6371.0


function load_earth_model(file_path::String)
    # Load the data from file (assumes whitespace delimited)
    data = readdlm(file_path)
    
    # Julia arrays are 1-indexed; assume data columns: column 1 => radius factor, 2 => density, 3 => e_fraction.
    radius = data[:, 1] .* EARTH_RADIUS_KM .* 1.0000000001
    density = data[:, 2]
    e_fraction = data[:, 3]
    
    # Compute potential as density * e_fraction * 1.52588e-4
    potential = density .* e_fraction .* 1.52588e-4
    
    return Dict(:radius => radius, :potential => potential)
end


function create_interpolated_model(earth_model::Dict)
    x = earth_model[:radius]
    y = earth_model[:potential]

    # Create linear interpolation using LinearInterpolation from Interpolations.jl.
    # The extrapolation_bc=Throw() ensures an error is raised if an out-of-range value is requested.
    linear_interp = LinearInterpolation(x, y; extrapolation_bc=Throw())

    return linear_interp
end

earth_model = load_earth_model(earthModelFile)
global earth = create_interpolated_model(earth_model)

# we set the angles for the paths before and after re-binning
global cosz_calc = collect(range(cosz_bins.min, stop=cosz_bins.max, length=cosz_bins.bin_number * 5))
global cosz = collect(range(cosz_bins.min, stop=cosz_bins.max, length=cosz_bins.bin_number))