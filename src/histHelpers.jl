using StatsBase
using LinearAlgebra

function extract_dataframes(filepaths::Vector{String})::Vector{DataFrame}
    return [CSV.File(fp) |> DataFrame for fp in filepaths]
end


"""
    ereco_edges(b) -> Vector{Float64}

Finite reco-energy bin edges of `(bin_number, min, max[, threshold])`: uniform, with the second edge
exactly at the threshold when given (threshold_binning).
"""
function ereco_edges(b)
    e = collect(range(b.min, b.max, length = b.bin_number + 1))
    hasproperty(b, :threshold) && (e[2] = b.threshold)
    return e
end

# Relative MC variance per bin of a weighted histogram, Σw²/(Σw)² (0 for empty bins).
function mc_relative_variance(E, w, bin_info)
    s1, _ = create_histogram(E, bin_info, weights=w,       normalise=false)
    s2, _ = create_histogram(E, bin_info, weights=w .^ 2,  normalise=false)
    return Float64[s > 0 ? q / s^2 : 0.0 for (s, q) in zip(s1, s2)]
end

"""
    create_histogram(dataRaw, bin_info; weights = nothing, normalise = true) -> (heights, centres)

Histogram of `dataRaw` (missing values dropped) on `bin_info.bins`, or uniform bins if absent;
optionally weighted, normalised to unit area by default.
"""
function create_histogram(dataRaw, bin_info; weights=nothing, normalise=true)
    
    bin_number = bin_info.bin_number
    min_val = bin_info.min
    max_val = bin_info.max

    data = filter(!ismissing, dataRaw)

    bins = get(bin_info, :bins, nothing)
    
    if bins === nothing
        println("Failed to get bins from bin_info. Generating uniform binning between extremal values.")
        bins = range(min_val, stop=max_val, length=bin_number + 1)
    else
        if length(bins) != bin_number + 1
            error("Provided bins do not match the expected number of bin edges.")
        end
    end

    if weights !== nothing
        processed_weights = filter(!ismissing, weights)
        processed_weights = coalesce.(processed_weights, 0)
        
        if length(data) != length(processed_weights)
            error("Weights vector length must match the length of data.")
        end
        
        hist = fit(Histogram, data, Weights(processed_weights), bins)
    else
        hist = fit(Histogram, data, bins)
    end

    if normalise == true
        total_count = sum(hist.weights)
        bin_heights = hist.weights ./ total_count
    else
        bin_heights = hist.weights
    end

    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in 1:length(bins)-1]

    return bin_heights, bin_centers
end


"""
    rebin_histogram(old_edges, old_weights, new_edges) -> new_weights

Redistribute bin contents onto new edges in proportion to the overlap of the bins.
"""
function rebin_histogram(old_edges, old_weights, new_edges)
    
    new_weights = zeros(eltype(old_weights), length(new_edges) - 1)
    
    for (i, (new_left, new_right)) in enumerate(zip(new_edges[1:end-1], new_edges[2:end]))
        for (j, (old_left, old_right)) in enumerate(zip(old_edges[1:end-1], old_edges[2:end]))
            overlap_left = max(new_left, old_left)
            overlap_right = min(new_right, old_right)
            overlap_width = max(0, overlap_right - overlap_left)
            old_bin_width = old_right - old_left

            if overlap_width > 0
                fraction = overlap_width / old_bin_width
                new_weights[i] += old_weights[j] * fraction
            end
        end
    end
    return new_weights
end