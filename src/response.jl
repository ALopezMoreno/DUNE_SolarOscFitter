using StatsBase
using LinearAlgebra

# THIS SHOULD NOT LIVE HERE FOREVER!
function create_histogram(data, bin_info)
    # Extract values from the named tuples
    bin_number = bin_info.bin_number
    min_val = bin_info.min
    max_val = bin_info.max

    # Create bins
    bins = range(min_val, stop=max_val, length=bin_number + 1)

    # Create histogram
    hist = fit(Histogram, data, bins)

    # Normalize the histogram
    total_count = sum(hist.weights)
    bin_heights = hist.weights ./ total_count

    # Calculate the central value of each bin
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in 1:length(bins)-1]

    return bin_heights, bin_centers
end


function create_response_matrix(data, bin_info)
    # Extract values from the named tuples
    e_true = data.e_true
    e_reco = data.e_reco
    bin_number = bin_info.bin_number
    min_val = bin_info.min
    max_val = bin_info.max

    # Create bins
    bins = range(min_val, stop=max_val, length=bin_number + 1)

    # Initialize the contribution matrix
    contribution_matrix = zeros(Float64, bin_number, bin_number)

    # Map e_true and e_reco to their respective bins and fill the contribution matrix
    @inbounds for (true_val, reco_val) in zip(e_true, e_reco)
        # Check if the values are within the specified bounds
        if min_val <= true_val <= max_val && min_val <= reco_val <= max_val
            true_bin = searchsortedfirst(bins, true_val)
            reco_bin = searchsortedfirst(bins, reco_val)

            # Adjust reco_bin if it exceeds bin_number
            if reco_bin > bin_number
                reco_bin = bin_number
            end

            # Increment the contribution matrix
            contribution_matrix[true_bin, reco_bin] += 1
        end
    end

    # Normalize each row to sum to 1
    for i in 1:bin_number
        row_sum = sum(contribution_matrix[i, :])
        if row_sum > 0
            contribution_matrix[i, :] /= row_sum
        end
    end

    return contribution_matrix
end


# Load simulations and create response matrices
df2_nue = CSV.File(nue_filepath) |> DataFrame
df2_other = CSV.File(other_filepath) |> DataFrame
df2_CC = CSV.File(CC_filepath) |> DataFrame

nue_ES_sample = (e_true=df2_nue.Enu, e_reco=df2_nue.Ereco)
other_ES_sample = (e_true=df2_other.Enu, e_reco=df2_other.Ereco)
CC_sample = (e_true=df2_CC.Enu, e_reco=df2_CC.Ereco)

nue_ES_nue_response = create_response_matrix(nue_ES_sample, bins)
other_ES_response = create_response_matrix(other_ES_sample, bins)
CC_response = create_response_matrix(CC_sample, bins)

# Save in named tuple
ES_response = (nue=nue_ES_nue_response, nuother=other_ES_response)
responseMatrices = (ES=ES_response, CC=CC_response)