using StatsBase
using LinearAlgebra

function extract_dataframes(filepaths::Vector{String})::Vector{DataFrame}
    return [CSV.File(fp) |> DataFrame for fp in filepaths]
end

function create_histogram(dataRaw, bin_info; weightsRaw=nothing, normalise=true)
    # Extract values from the named tuple
    bin_number = bin_info.bin_number
    min_val = bin_info.min
    max_val = bin_info.max

    # Clean empty values
    data = filter(!ismissing, dataRaw)

    bins = get(bin_info, :bins, nothing)
    
    if bins === nothing
        # Create bins if 'bins' field is not present
        println("Failed to get bins from bin_info. Generating uniform binning between extremal values.")
        bins = range(min_val, stop=max_val, length=bin_number + 1)
    else
        # Ensure the provided bins have the correct number of edges
        if length(bins) != bin_number + 1
            error("Provided bins do not match the expected number of bin edges.")
        end
    end

    # Check if optional weights are provided and have valid length
    if weightsRaw !== nothing
        weights = filter(!ismissing, weightsRaw)
        weights = coalesce.(weights, 0)
        if length(data) != length(weights)
            error("Weights vector length must match the length of data.")
        end
        # Create histogram using weights
        hist = fit(Histogram, data, Weights(weights), bins)
    else
        # Create unweighted histogram
        hist = fit(Histogram, data, bins)
    end

    # Normalize the histogram
    if normalise == true
        total_count = sum(hist.weights)
        bin_heights = hist.weights ./ total_count
    else
        bin_heights = hist.weights
    end

    # Calculate the central value of each bin
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in 1:length(bins)-1]

    return bin_heights, bin_centers
end