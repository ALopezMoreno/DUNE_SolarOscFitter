module PropagationDebug

using Plots

export debug_plot_CC_backgrounds,
       debug_heatmap_response_CC,
       debug_heatmap_oscprobs,
       debug_heatmap_CC_night

"""
    debug_plot_CC_backgrounds(backgrounds; E_min = 5, E_max = 25)

Stacked bar chart of CC backgrounds (e.g. gammas + neutrons).
`backgrounds` is expected to have a field `.CC` which is an
indexable collection of background components.
"""
function debug_plot_CC_backgrounds(backgrounds; E_min::Real = 5, E_max::Real = 25)
    @assert !isempty(backgrounds.CC) "backgrounds.CC is empty"

    n_bins = length(backgrounds.CC[1])
    bins   = range(E_min, E_max, length = n_bins)

    bar(
        bins,
        backgrounds.CC[1],
        label = "Gammas",
        title = "Stacked Bar Chart of CC Backgrounds",
        xlabel = "Reco energy bin",
        ylabel = "Counts",
    )

    if length(backgrounds.CC) ≥ 2
        bar!(
            bins,
            backgrounds.CC[2],
            label = "Neutrons",
            bar_position = :stack,
        )
    end

    display(current())
end

"""
    debug_heatmap_response_CC(responseMatrices)

Heatmap of the CC response matrix (transpose, matching your original debug).
"""
function debug_heatmap_response_CC(responseMatrices)
    pt = heatmap(
        responseMatrices.CC',
        title = "Response Matrices CC'",
        xlabel = "Columns",
        ylabel = "Rows",
        aspect_ratio = :equal,
    )
    display(pt)
end

"""
    debug_heatmap_oscprobs(oscProbs_nue_8B_night_large; title_suffix = "")

Heatmap of the fine-binned νₑ 8B night oscillation probabilities.
Call this from wherever you still have `oscProbs_nue_8B_night_large`.
"""
function debug_heatmap_oscprobs(oscProbs_nue_8B_night_large; title_suffix::AbstractString = "")
    t = "Oscillation probabilities νₑ 8B night"
    if !isempty(title_suffix)
        t *= " – " * title_suffix
    end

    myP = heatmap(
        oscProbs_nue_8B_night_large,
        xlabel = "Column Index",
        ylabel = "Row Index",
        title  = t,
        aspect_ratio = :equal,
    )
    display(myP)
end

"""
    debug_heatmap_CC_night(eventRate_CC_night)

Heatmap of night-time CC event rates vs cos(zenith) and E_reco.
"""
function debug_heatmap_CC_night(eventRate_CC_night)
    myPlot = heatmap(
        eventRate_CC_night[:, 1:end],
        title = "CC Night Event Rate",
        xlabel = "E_reco bin",
        ylabel = "cos(θₙ) bin",
        aspect_ratio = :equal,
    )
    display(myPlot)
end

end # module
