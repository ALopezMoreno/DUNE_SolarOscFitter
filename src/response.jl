include("../src/histHelpers.jl")

function create_response_matrix(data, bin_info_etrue, bin_info_ereco)
    # Extract values from the named tuples and replace zeros
    e_true = ifelse.(e_true .== 0, 1e-9, e_true)
    e_reco = ifelse.(e_reco .== 0, 1e-9, e_reco)

    # Extract bin information for e_true
    bin_number_etrue = bin_info_etrue.bin_number
    min_val_etrue = bin_info_etrue.min
    max_val_etrue = bin_info_etrue.max

    # Extract bin information for e_reco
    bin_number_ereco = bin_info_ereco.bin_number
    min_val_ereco = bin_info_ereco.min
    max_val_ereco = bin_info_ereco.max

    # Create bins for e_true and e_reco
    bins_etrue = range(min_val_etrue, stop=max_val_etrue, length=bin_number_etrue + 1)
    bins_ereco = bin_info_ereco.bins # These should always already exist!!!

    # Initialize the contribution matrix
    contribution_matrix = zeros(Float64, bin_number_etrue, bin_number_ereco)

    # Map e_true and e_reco to their respective bins and fill the contribution matrix
    @inbounds for (true_val, reco_val) in zip(e_true, e_reco)
        # Check if the values are within the specified bounds
        if min_val_etrue <= true_val <= max_val_etrue && min_val_ereco <= reco_val <= max_val_ereco
            true_bin = searchsortedfirst(bins_etrue, true_val)
            reco_bin = searchsortedfirst(bins_ereco, reco_val)

            # Adjust true_bin and reco_bin if they exceed their respective bin numbers
            if true_bin > bin_number_etrue
                true_bin = bin_number_etrue
            end
            if reco_bin > bin_number_ereco
                reco_bin = bin_number_ereco
            end

            # Increment the contribution matrix
            contribution_matrix[true_bin, reco_bin] += 1
        end
    end

    # Normalize each row to sum to 1
    for i in 1:bin_number_etrue
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

# Extend ereco bins from -inf to +inf:
bins_ES = collect(range(Ereco_bins_ES.min, stop=Ereco_bins_ES.max, length=Ereco_bins_ES.bin_number + 1))
bins_CC = collect(range(Ereco_bins_CC.min, stop=Ereco_bins_CC.max, length=Ereco_bins_CC.bin_number + 1))

bins_ES[1] = -Inf
bins_ES[end] = Inf

bins_CC[1] = -Inf
bins_CC[end] = Inf

global Ereco_bins_ES_extended = (bin_number=Ereco_bins_ES.bin_number, min=Ereco_bins_ES.min, max=Ereco_bins_ES.max, bins=bins_ES)
global Ereco_bins_CC_extended = (bin_number=Ereco_bins_CC.bin_number, min=Ereco_bins_CC.min, max=Ereco_bins_CC.max, bins=bins_CC)


nue_ES_sample = (e_true=df2_nue.Enu, e_reco=df2_nue.Ereco)
other_ES_sample = (e_true=df2_other.Enu, e_reco=df2_other.Ereco)
CC_sample = (e_true=df2_CC.Enu, e_reco=df2_CC.Ereco)

nue_ES_nue_response = create_response_matrix(nue_ES_sample, Etrue_bins, Ereco_bins_ES_extended)
other_ES_response = create_response_matrix(other_ES_sample, Etrue_bins, Ereco_bins_ES_extended)
CC_response = create_response_matrix(CC_sample, Etrue_bins, Ereco_bins_CC_extended)

# Save in named tuple
ES_response = (nue=nue_ES_nue_response, nuother=other_ES_response)
responseMatrices = (ES=ES_response, CC=CC_response)


