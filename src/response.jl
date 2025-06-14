include("../src/histHelpers.jl")

function create_response_matrix(data, bin_info_etrue, bin_info_ereco)
    # Extract values from the named tuples and replace zeros
    e_true = ifelse.(data.e_true .== 0, 1e-9, data.e_true)
    e_reco = ifelse.(data.e_reco .== 0, 1e-9, data.e_reco)

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
df_nue = CSV.File(nue_filepath) |> DataFrame
df_nuother = CSV.File(other_filepath) |> DataFrame
df_CC = CSV.File(CC_filepath) |> DataFrame

# Extend ereco bins from -inf to +inf:
bins_ES = collect(range(Ereco_bins_ES.min, stop=Ereco_bins_ES.max, length=Ereco_bins_ES.bin_number + 1))
bins_CC = collect(range(Ereco_bins_CC.min, stop=Ereco_bins_CC.max, length=Ereco_bins_CC.bin_number + 1))

bins_ES[1] = -Inf
bins_ES[end] = Inf

bins_CC[1] = -Inf
bins_CC[end] = Inf

global Ereco_bins_ES_extended = (bin_number=Ereco_bins_ES.bin_number, min=Ereco_bins_ES.min, max=Ereco_bins_ES.max, bins=bins_ES)
global Ereco_bins_CC_extended = (bin_number=Ereco_bins_CC.bin_number, min=Ereco_bins_CC.min, max=Ereco_bins_CC.max, bins=bins_CC)

nue_ES_sample = (e_true=df_nue.Etrue[df_nue.mask], e_reco=df_nue.Eraw[df_nue.mask])
other_ES_sample = (e_true=df_nuother.Etrue[df_nuother.mask], e_reco=df_nuother.Eraw[df_nuother.mask])
CC_sample = (e_true=df_CC.Etrue[df_CC.mask], e_reco=df_CC.Eraw[df_CC.mask])

nue_ES_nue_response = create_response_matrix(nue_ES_sample, Etrue_bins, Ereco_bins_ES_extended)
other_ES_response = create_response_matrix(other_ES_sample, Etrue_bins, Ereco_bins_ES_extended)
CC_response = create_response_matrix(CC_sample, Etrue_bins, Ereco_bins_CC_extended)

# Save in named tuple
ES_response = (nue=nue_ES_nue_response, nuother=other_ES_response)
responseMatrices = (ES=ES_response, CC=CC_response)

ES_nue_selection, _ = create_histogram(df_nue.Eraw[df_nue.mask], Ereco_bins_ES_extended, normalise=false)
ES_nuother_selection, _ = create_histogram(df_nuother.Eraw[df_nuother.mask], Ereco_bins_ES_extended, normalise=false)
CC_selection, _ = create_histogram(df_CC.Eraw[df_CC.mask], Ereco_bins_CC_extended, normalise=false)

ES_nue_total, ES_nue_bin_centers = create_histogram(df_nue.Eraw, Ereco_bins_ES_extended, normalise=false)
ES_nuother_total, ES_nuother_bin_centers = create_histogram(df_nuother.Eraw, Ereco_bins_ES_extended, normalise=false)
CC_total, CC_bin_centers = create_histogram(df_CC.Eraw, Ereco_bins_CC_extended, normalise=false)

global ES_nue_eff = @. ifelse(ES_nue_total == 0, 0.0, ES_nue_selection / ES_nue_total)
global ES_nuother_eff = @. ifelse(ES_nuother_total == 0, 0.0, ES_nuother_selection / ES_nuother_total)
global CC_eff = @. ifelse(CC_total == 0, 0.0, CC_selection / CC_total)