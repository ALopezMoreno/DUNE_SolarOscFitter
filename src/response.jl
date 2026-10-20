include(joinpath(@__DIR__, "histHelpers.jl"))

"""
    create_response_counts(data, bin_info_x, bin_info_y) -> Matrix{Float64}

MC event counts n[i, j] per (x bin, y bin), before row normalisation. Kept for the MC
statistical uncertainty of the predictions (src/likelihoods/mcstat.jl).
"""
create_response_counts(data, bin_info_x, bin_info_y) = _response_matrix(data, bin_info_x, bin_info_y; normalise=false)

create_response_matrix(data, bin_info_x, bin_info_y) = _response_matrix(data, bin_info_x, bin_info_y; normalise=true)

function _response_matrix(data, bin_info_x, bin_info_y; normalise::Bool)
    """
    Response matrix R[i, j] = P(reco bin j | true bin i) from MC events `data` (x = true, y = reco);
    rows sum to 1.
    """
    # zero energies would fall below the first edge
    x = ifelse.(data.x .== 0, 1e-9, data.x)
    y = ifelse.(data.y .== 0, 1e-9, data.y)

    bin_number_x = bin_info_x.bin_number
    min_val_x = bin_info_x.min
    max_val_x = bin_info_x.max

    bin_number_y = bin_info_y.bin_number
    min_val_y = bin_info_y.min
    max_val_y = bin_info_y.max

    bins_x = hasproperty(bin_info_x, :bins) ?
         bin_info_x.bins :
         range(min_val_x, stop = max_val_x, length = bin_number_x + 1)

    bins_y = hasproperty(bin_info_y, :bins) ?
         bin_info_y.bins :
         range(min_val_y, stop = max_val_y, length = bin_number_y + 1)

    contribution_matrix = zeros(Float64, bin_number_x, bin_number_y)

    # Range check on the bin edges: with ±Inf outer edges, out-of-window events go in the end bins.
    lo_x, hi_x = first(bins_x), last(bins_x)
    lo_y, hi_y = first(bins_y), last(bins_y)
    @inbounds for (x_val, y_val) in zip(x, y)
        if lo_x <= x_val <= hi_x && lo_y <= y_val <= hi_y
            true_bin = min(searchsortedlast(bins_x, x_val), bin_number_x)
            reco_bin = min(searchsortedlast(bins_y, y_val), bin_number_y)

            contribution_matrix[true_bin, reco_bin] += 1
        end
    end

    normalise || return contribution_matrix

    for i in 1:bin_number_x
        row_sum = sum(contribution_matrix[i, :])
        if row_sum > 0
            contribution_matrix[i, :] /= row_sum
        end
    end

    return contribution_matrix
end

# Signal responses use unweighted counts: the meaning of signal `weights` is not defined yet, so
# non-unit weights are ignored with a warning.
function warn_if_signal_weights_ignored(label, df, path)
    "weights" in names(df) || return nothing
    w = skipmissing(df.weights)
    if !all(==(1), w)
        @warn "$label signal sample $path has non-unit weights (range $(extrema(w))); " *
              "they are IGNORED: response matrices and efficiencies use unweighted counts."
    end
    return nothing
end

"""
    build_response_matrices(det) -> (responseMatrices, Ereco_bins_ES_extended, Ereco_bins_CC_extended)

Response matrices of detector `det`, with `.eff` (selection × reco efficiencies), `.bins` and the
MC counts `.mc` (likelihoods/mcstat.jl).
"""
function build_response_matrices(det)
    Ereco_bins_ES           = det.Ereco_bins_ES
    Ereco_bins_CC           = det.Ereco_bins_CC
    cos_scatter_bins        = det.cos_scatter_bins
    angular_cos_cut         = det.angular_cos_cut
    inclusive_analysis      = det.inclusive_analysis
    semi_inclusive_analysis = det.semi_inclusive_analysis

    df_nue      = CSV.File(det.nue_filepath)      |> DataFrame
    df_nuother  = CSV.File(det.other_filepath)    |> DataFrame
    df_ES_angular = CSV.File(det.angular_filepath) |> DataFrame
    df_CC       = CSV.File(det.CC_filepath)       |> DataFrame
    warn_if_signal_weights_ignored("ES nue",     df_nue,     det.nue_filepath)
    warn_if_signal_weights_ignored("ES nuother", df_nuother, det.other_filepath)
    warn_if_signal_weights_ignored("CC",         df_CC,      det.CC_filepath)

    bins_ES = ereco_edges(Ereco_bins_ES)
    bins_CC = ereco_edges(Ereco_bins_CC)
    bins_ES[1] = -Inf;  bins_ES[end] = Inf
    bins_CC[1] = -Inf;  bins_CC[end] = Inf
    Ereco_bins_ES_extended = merge(Ereco_bins_ES, (bins=bins_ES,))
    Ereco_bins_CC_extended = merge(Ereco_bins_CC, (bins=bins_CC,))

    nue_ES_sample    = (x=df_nue.Etrue,        y=df_nue.Ereco)
    other_ES_sample  = (x=df_nuother.Etrue,    y=df_nuother.Ereco)
    angular_ES_sample = (x=df_ES_angular.Ereco, y=df_ES_angular.cos_scatter)
    CC_sample        = (x=df_CC.Etrue,         y=df_CC.Ereco)

    nue_ES_response     = create_response_matrix(nue_ES_sample,   Etrue_bins, Ereco_bins_ES_extended)
    nuother_ES_response = create_response_matrix(other_ES_sample, Etrue_bins, Ereco_bins_ES_extended)
    angular_ES_response = create_response_matrix(angular_ES_sample, Ereco_bins_ES_extended, cos_scatter_bins)'
    CC_response         = create_response_matrix(CC_sample, Etrue_bins, Ereco_bins_CC_extended)

    # Angular background response — uniform, then apply forward-hemisphere cut
    N = cos_scatter_bins.bin_number
    edges   = range(cos_scatter_bins.min, cos_scatter_bins.max, length=N + 1)
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    ang_mask   = centers .>= angular_cos_cut
    n_allowed  = count(ang_mask)
    angular_BG_response = zeros(size(angular_ES_response))
    angular_BG_response[ang_mask, :] .= 1.0 / n_allowed

    ES_response = (nue=nue_ES_response, nuother=nuother_ES_response, angular=angular_ES_response)

    # Selection efficiencies
    ES_nue_sel, _    = create_histogram(df_nue.Ereco[df_nue.mask],         Ereco_bins_ES_extended, normalise=false)
    ES_nue_tot, _    = create_histogram(df_nue.Ereco,                      Ereco_bins_ES_extended, normalise=false)
    ES_nuoth_sel, _  = create_histogram(df_nuother.Ereco[df_nuother.mask], Ereco_bins_ES_extended, normalise=false)
    ES_nuoth_tot, _  = create_histogram(df_nuother.Ereco,                  Ereco_bins_ES_extended, normalise=false)
    CC_sel, _        = create_histogram(df_CC.Ereco[df_CC.mask],           Ereco_bins_CC_extended, normalise=false)
    CC_tot, _        = create_histogram(df_CC.Ereco,                       Ereco_bins_CC_extended, normalise=false)

    ES_nue_sel_eff  = @. ifelse(ES_nue_tot  == 0, 0.0, ES_nue_sel  / ES_nue_tot)
    ES_nuoth_sel_eff = @. ifelse(ES_nuoth_tot == 0, 0.0, ES_nuoth_sel / ES_nuoth_tot)
    CC_sel_eff       = @. ifelse(CC_tot       == 0, 0.0, CC_sel       / CC_tot)

    ES_nue_eff    = ES_nue_sel_eff  .* fill(1.0, Ereco_bins_ES.bin_number)
    ES_nuother_eff = ES_nuoth_sel_eff .* fill(1.0, Ereco_bins_ES.bin_number)
    CC_eff         = CC_sel_eff

    eff_tuple = (ES_nue=ES_nue_eff, ES_nuother=ES_nuother_eff, CC=CC_eff)

    # MC counts behind the responses and efficiencies (likelihoods/mcstat.jl)
    mc_tuple = (
        ES_nue     = (counts = create_response_counts(nue_ES_sample,   Etrue_bins, Ereco_bins_ES_extended), tot = Float64.(ES_nue_tot)),
        ES_nuother = (counts = create_response_counts(other_ES_sample, Etrue_bins, Ereco_bins_ES_extended), tot = Float64.(ES_nuoth_tot)),
        CC         = (counts = create_response_counts(CC_sample,       Etrue_bins, Ereco_bins_CC_extended), tot = Float64.(CC_tot)),
        ang_mask   = ang_mask,
    )
    bins_tuple = (ES=Ereco_bins_ES_extended, CC=Ereco_bins_CC_extended, cos_scatter=cos_scatter_bins)

    if inclusive_analysis || semi_inclusive_analysis
        # CC in ES reco bins: the forward inclusive hemisphere
        CC_inclusive_response = create_response_matrix(CC_sample, Etrue_bins, Ereco_bins_ES_extended)
        CC_incl_sel, _ = create_histogram(df_CC.Ereco[df_CC.mask], Ereco_bins_ES_extended,
                                          normalise=false)
        CC_incl_tot, _ = create_histogram(df_CC.Ereco, Ereco_bins_ES_extended,
                                          normalise=false)
        CC_incl_sel_eff = @. ifelse(CC_incl_tot == 0, 0.0, CC_incl_sel / CC_incl_tot)
        CC_incl_eff     = CC_incl_sel_eff
        eff_tuple = merge(eff_tuple, (CC_incl=CC_incl_eff,))
        mc_tuple  = merge(mc_tuple, (CC_incl = (counts = create_response_counts(CC_sample, Etrue_bins, Ereco_bins_ES_extended),
                                                tot = Float64.(CC_incl_tot)),))

        if semi_inclusive_analysis
            # CC forward/backward split, CC assumed uniform in cos θ (ang_mask = forward bins)
            Ncos       = cos_scatter_bins.bin_number
            f_CC_above = count(ang_mask) / Ncos
            f_CC_below = 1.0 - f_CC_above

            # Mis-ID response, ES → CC in two steps: the ES response (no efficiency), then the CC
            # response with the ES reco energy as pseudo-true energy, for backward ES events only.
            N_ES        = Ereco_bins_ES.bin_number
            # centres of the finite edges (bins_ES has ±Inf outer edges)
            ES_edges    = ereco_edges(Ereco_bins_ES)
            ES_centers  = 0.5 .* (ES_edges[1:end-1] .+ ES_edges[2:end])
            Etrue_edges = collect(range(Etrue_bins.min, Etrue_bins.max,
                                        length = Etrue_bins.bin_number + 1))

            CC_response_as_ereco = zeros(N_ES, Ereco_bins_CC.bin_number)
            for j in 1:N_ES
                etrue_bin = min(searchsortedlast(Etrue_edges, ES_centers[j]),
                                Etrue_bins.bin_number)
                if etrue_bin >= 1
                    CC_response_as_ereco[j, :] = CC_response[etrue_bin, :]
                end
            end

            backward_frac = vec(sum(angular_ES_response[.!ang_mask, :], dims=1))  # (N_Ereco_ES,)

            # (N_Etrue × N_Ereco_CC) backward mis-ID response
            nue_ES_backward     = nue_ES_response    .* backward_frac'
            nuother_ES_backward = nuother_ES_response .* backward_frac'
            CC_misID_nue_response     = nue_ES_backward    * CC_response_as_ereco
            CC_misID_nuother_response = nuother_ES_backward * CC_response_as_ereco

            responseMatrices = (ES=ES_response, CC=CC_response,
                                CC_inclusive=CC_inclusive_response,
                                CC_misID_nue=CC_misID_nue_response,
                                CC_misID_nuother=CC_misID_nuother_response,
                                CC_split=(above=f_CC_above, below=f_CC_below),
                                BG=(angular=angular_BG_response,),
                                eff=eff_tuple, bins=bins_tuple)
        else
            # pure inclusive
            responseMatrices = (ES=ES_response, CC=CC_response,
                                CC_inclusive=CC_inclusive_response,
                                BG=(angular=angular_BG_response,),
                                eff=eff_tuple, bins=bins_tuple)
        end
    else
        responseMatrices = (ES=ES_response, CC=CC_response,
                            BG=(angular=angular_BG_response,),
                            eff=eff_tuple, bins=bins_tuple)
    end

    responseMatrices = merge(responseMatrices, (mc = mc_tuple,))
    return responseMatrices, Ereco_bins_ES_extended, Ereco_bins_CC_extended
end


