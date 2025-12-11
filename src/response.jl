#=
response.jl

Detector response matrix creation for the Solar Oscillation Fitter.
This module processes Monte Carlo simulation data to create response matrices
that map true neutrino energies to reconstructed energies, accounting for
detector resolution and efficiency effects.

Key Features:
- Response matrix creation from MC truth and reconstruction data
- Energy binning and histogram processing
- Detection efficiency calculations (selection + reconstruction)
- Support for different neutrino interaction channels (ES, CC)
- Extended energy binning with infinite bounds for edge effects

The response matrices are essential for converting theoretical predictions
at true energy into observable distributions at reconstructed energy.

Author: [Author name]
=#

include("../src/histHelpers.jl")

function create_response_matrix(data, bin_info_x, bin_info_y)
    """
    Create a detector response matrix from Monte Carlo simulation data.
    
    The response matrix R[i,j] gives the probability that a neutrino with
    true energy in bin i will be reconstructed with energy in bin j.
    
    Arguments:
    - data: Named tuple with x and y vectors
    - bin_info_x: x-axis binning specification
    - bin_info_y: y-axis binning specification
    
    Returns:
    - contribution_matrix: Normalized response matrix (rows sum to 1)
    """
    # Replace zero energies with small values to avoid binning issues
    x = ifelse.(data.x .== 0, 1e-9, data.x)
    y = ifelse.(data.y .== 0, 1e-9, data.y)

    # Extract true energy binning parameters
    bin_number_x = bin_info_x.bin_number
    min_val_x = bin_info_x.min
    max_val_x = bin_info_x.max

    # Extract reconstructed energy binning parameters
    bin_number_y = bin_info_y.bin_number
    min_val_y = bin_info_y.min
    max_val_y = bin_info_y.max

    # Create/get bin edges
    bins_x = hasproperty(bin_info_x, :bins) ?
         bin_info_x.bins :
         range(min_val_x, stop = max_val_x, length = bin_number_x + 1)

    bins_y = hasproperty(bin_info_y, :bins) ?
         bin_info_y.bins :
         range(min_val_y, stop = max_val_y, length = bin_number_y + 1)

    # Initialize the response matrix
    contribution_matrix = zeros(Float64, bin_number_x, bin_number_y)

    # Fill the response matrix by binning MC events
    @inbounds for (x_val, y_val) in zip(x, y)
        # Check if values are within analysis bounds
        if min_val_x <= x_val <= max_val_x && min_val_y <= y_val <= max_val_y
            # Find appropriate bins
            true_bin = searchsortedfirst(bins_x, x_val)
            reco_bin = searchsortedfirst(bins_y, y_val)

            # Handle edge cases for bin indices
            if true_bin > bin_number_x
                true_bin = bin_number_x
            end
            if reco_bin > bin_number_y
                reco_bin = bin_number_y
            end

            # Increment the response matrix element
            contribution_matrix[true_bin, reco_bin] += 1
        end
    end

    # Normalize each row to create probability matrix
    # Each row represents P(y | x) e.g. P(E_reco | E_true) for a given true energy bin
    for i in 1:bin_number_x
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
df_ES_angular = CSV.File(angular_filepath) |> DataFrame
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

nue_ES_sample = (x=df_nue.Etrue, y=df_nue.Ereco)
other_ES_sample = (x=df_nuother.Etrue, y=df_nuother.Ereco)
angular_ES_sample = (x=df_ES_angular.Ereco, y=df_ES_angular.cos_scatter)
CC_sample = (x=df_CC.Etrue, y=df_CC.Ereco)

nue_ES_response = create_response_matrix(nue_ES_sample, Etrue_bins, Ereco_bins_ES_extended)
nuother_ES_response = create_response_matrix(other_ES_sample, Etrue_bins, Ereco_bins_ES_extended)

# transpose angular response because we want Ereco always on the same axis
angular_ES_response = create_response_matrix(angular_ES_sample, Ereco_bins_ES_extended, cos_scatter_bins)'

CC_response = create_response_matrix(CC_sample, Etrue_bins, Ereco_bins_CC_extended)

angular_BG_response = fill(1 / cos_scatter_bins.bin_number, size(angular_ES_response))

# Save in named tuple
ES_response = (nue=nue_ES_response, nuother=nuother_ES_response, angular=angular_ES_response)
responseMatrices = (ES=ES_response, CC=CC_response, BG=(angular=angular_BG_response,))

# Helper function to safely get weights if they exist
function get_weights(df, mask=nothing)
    if hasproperty(df, :Weights)
        return mask !== nothing ? df.Weights[mask] : df.Weights
    else
        return nothing
    end
end

# Get selection efficiencies
ES_nue_selection, _ = create_histogram(
    df_nue.Ereco[df_nue.mask], 
    Ereco_bins_ES_extended, 
    normalise=false, 
    weights=get_weights(df_nue, df_nue.mask)
)

ES_nuother_selection, _ = create_histogram(
    df_nuother.Ereco[df_nuother.mask], 
    Ereco_bins_ES_extended,
    normalise=false,
    weights=get_weights(df_nuother, df_nuother.mask)
)

CC_selection, CC_selected_bin_centers = create_histogram(
    df_CC.Ereco[df_CC.mask],
    Ereco_bins_CC_extended,
    normalise=false,
    weights=get_weights(df_CC, df_CC.mask)
)

# For total histograms (no mask)
ES_nue_total, ES_nue_bin_centers = create_histogram(
    df_nue.Ereco,
    Ereco_bins_ES_extended,
    normalise=false,
    weights=get_weights(df_nue)
)

ES_nuother_total, ES_nuother_bin_centers = create_histogram(
    df_nuother.Ereco,
    Ereco_bins_ES_extended,
    normalise=false,
    weights=get_weights(df_nuother)
)

CC_total, CC_bin_centers = create_histogram(
    df_CC.Ereco,
    Ereco_bins_CC_extended,
    normalise=false,
    weights=get_weights(df_CC)
)

global ES_nue_selection_eff = @. ifelse(ES_nue_total == 0, 0.0, ES_nue_selection / ES_nue_total)
global ES_nuother_selection_eff = @. ifelse(ES_nuother_total == 0, 0.0, ES_nuother_selection / ES_nuother_total)
global CC_selection_eff = @. ifelse(CC_total == 0, 0.0, CC_selection / CC_total)

# get reco efficiencies from histFile
#TO DO. FOR THE MOMENT ASSUME 90% FLAT FOR CC (ES ALREADY INCLUDED IN RECO FILES)

global ES_nue_reco_eff = fill(1., Ereco_bins_ES.bin_number)
global ES_nuother_reco_eff = fill(1., Ereco_bins_ES.bin_number)
global CC_reco_eff = fill(0.9, Ereco_bins_CC.bin_number)

# Total efficiency is the product of the two:
global ES_nue_eff = ES_nue_selection_eff .* ES_nue_reco_eff
global ES_nuother_eff = ES_nuother_selection_eff .* ES_nuother_reco_eff
global CC_eff = CC_selection_eff .* CC_reco_eff


#=
myP = plot(CC_bin_centers.*1000, CC_eff, 
    seriestype = :steppre,
    linewidth = 2,
    label = "CC Efficiency",
    xlabel = "Recoil Energy (MeV)",
    ylabel = "Efficiency",
    title = "CC Selection Efficiency"
)

# --- Plot 2: Total vs. Selected Events (step histograms) ---
plot_events = plot(CC_bin_centers.*1000, 
    log10.(replace(CC_total, 0 => 1e-3)),  # Replace zeros for log scale
    seriestype = :steppre, 
    linewidth = 2,
    label = "CC Total Events",
    xlabel = "Recoil Energy (MeV)",
    ylabel = "Counts (log scale)",
    title = "CC Events: Total vs. Selected",
    color = :blue,
)

plot!(plot_events, CC_selected_bin_centers.*1000, 
    log10.(replace(CC_selection, 0 => 1e-3)),  # Replace zeros for log scale
    seriestype = :steppre, 
    linewidth = 2,
    label = "CC Selected Events", ylims=[-2,6]
)

# --- Display both plots ---
display(myP)  # Show efficiency plot
sleep(1000)

#display(plot_events)  # Show events plot
=#