include("../src/histHelpers.jl")

# Extract efficiency file as dataframes. The function expects arrays of file names and outputs an array of dataframes
df_ES_eff = extract_dataframes([ES_efficiency_filepath])[1]
df_CC_eff = extract_dataframes([CC_efficiency_filepath])[1]

ES_selection, _ = create_histogram(df_ES_eff.Ereco, Ereco_bins_ES_extended, weights = df_ES_eff.Mask, normalise=false)
CC_selection, _ = create_histogram(df_CC_eff.Ereco, Ereco_bins_CC_extended, weights = df_CC_eff.Mask, normalise=false)

ES_total, ES_bin_centers = create_histogram(df_ES_eff.Ereco, Ereco_bins_ES_extended, normalise=false)
CC_total, CC_bin_centers = create_histogram(df_CC_eff.Ereco, Ereco_bins_CC_extended, normalise=false)

global ES_eff = ES_selection ./ ES_total
global CC_eff = CC_selection ./ CC_total


################### DEBUGGING ######################
# using Plots
# max_val = maximum([maximum(CC_selection), maximum(CC_total)])


# Create the plot with ylim set from 0 to 1.2 times the maximum value
# plt = plot(CC_bin_centers, CC_eff,
#     marker = :o,
#     xlabel = "CC Bin Centers",
#     ylabel = "Value",
#     title = "CC Selection and Efficiency",
#     label = "CC Selection",
#     ylim = (0, 1.2)
# )


# Display the plot
# display(plt)
# sleep(20)
# exit()