#=
histHelpers.jl

Histogram creation and manipulation utilities for the Solar Oscillation Fitter.
This module provides functions for creating histograms from Monte Carlo data,
handling weighted samples, and rebinning operations.

Key Features:
- DataFrame extraction from CSV files
- Weighted and unweighted histogram creation
- Flexible binning with custom bin edges
- Histogram rebinning for different energy resolutions
- Support for missing data handling

These utilities are used throughout the analysis to process Monte Carlo
samples and create energy distributions for different detection channels.

Author: [Author name]
=#

using StatsBase     # For histogram fitting and statistical operations
using LinearAlgebra # For linear algebra operations

# Utility function to load multiple CSV files as DataFrames
function extract_dataframes(filepaths::Vector{String})::Vector{DataFrame}
    """Load CSV files from a list of file paths and return as DataFrames"""
    return [CSV.File(fp) |> DataFrame for fp in filepaths]
end


function create_histogram(dataRaw, bin_info; weights=nothing, normalise=true)
    """
    Create a histogram from data with optional weights and normalization.
    
    Arguments:
    - dataRaw: Raw data vector (may contain missing values)
    - bin_info: Named tuple with binning information (bin_number, min, max, optionally bins)
    - weights: Optional weight vector for weighted histograms
    - normalise: Whether to normalize histogram to unit area
    
    Returns:
    - bin_heights: Histogram bin contents (normalized or raw counts)
    - bin_centers: Center positions of each bin
    """
    
    # Extract binning parameters from the named tuple
    bin_number = bin_info.bin_number
    min_val = bin_info.min
    max_val = bin_info.max

    # Remove missing values from data
    data = filter(!ismissing, dataRaw)

    # Get bin edges (either provided or generate uniform bins)
    bins = get(bin_info, :bins, nothing)
    
    if bins === nothing
        # Create uniform bins if not provided
        println("Failed to get bins from bin_info. Generating uniform binning between extremal values.")
        bins = range(min_val, stop=max_val, length=bin_number + 1)
    else
        # Validate provided bins
        if length(bins) != bin_number + 1
            error("Provided bins do not match the expected number of bin edges.")
        end
    end

    # Handle optional weights
    if weights !== nothing
        # Clean weights (remove missing, replace with 0)
        processed_weights = filter(!ismissing, weights)
        processed_weights = coalesce.(processed_weights, 0)
        
        # Validate weight vector length
        if length(data) != length(processed_weights)
            error("Weights vector length must match the length of data.")
        end
        
        # Create weighted histogram
        hist = fit(Histogram, data, Weights(processed_weights), bins)
    else
        # Create unweighted histogram
        hist = fit(Histogram, data, bins)
    end

    # Apply normalization if requested
    if normalise == true
        total_count = sum(hist.weights)
        bin_heights = hist.weights ./ total_count
    else
        bin_heights = hist.weights
    end

    # Calculate bin center positions
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in 1:length(bins)-1]

    return bin_heights, bin_centers
end


function rebin_histogram(old_edges, old_weights, new_edges)
    """
    Rebin a histogram from one binning scheme to another.
    
    This function redistributes histogram weights from old bins to new bins
    based on the fractional overlap between bin ranges.
    
    Arguments:
    - old_edges: Bin edges of the original histogram
    - old_weights: Weights/contents of the original histogram bins
    - new_edges: Bin edges for the target rebinned histogram
    
    Returns:
    - new_weights: Rebinned histogram weights
    """
    
    # Initialize new histogram with zeros
    new_weights = zeros(eltype(old_weights), length(new_edges) - 1)
    
    # Loop over each new bin
    for (i, (new_left, new_right)) in enumerate(zip(new_edges[1:end-1], new_edges[2:end]))
        # Loop over each old bin to find overlaps
        for (j, (old_left, old_right)) in enumerate(zip(old_edges[1:end-1], old_edges[2:end]))
            # Calculate overlap between new bin [new_left, new_right) and old bin [old_left, old_right)
            overlap_left = max(new_left, old_left)
            overlap_right = min(new_right, old_right)
            overlap_width = max(0, overlap_right - overlap_left)
            old_bin_width = old_right - old_left

            # Add proportional contribution if bins overlap
            if overlap_width > 0
                fraction = overlap_width / old_bin_width
                new_weights[i] += old_weights[j] * fraction
            end
        end
    end
    return new_weights
end