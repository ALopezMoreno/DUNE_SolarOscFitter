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

include(joinpath(@__DIR__, "histHelpers.jl"))

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
            # Find appropriate bins (searchsortedlast gives left-inclusive bin index)
            true_bin = min(searchsortedlast(bins_x, x_val), bin_number_x)
            reco_bin = min(searchsortedlast(bins_y, y_val), bin_number_y)

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

# Helper function to safely get weights if they exist
function get_weights(df, mask=nothing)
    if hasproperty(df, :weights)
        return mask !== nothing ? df.weights[mask] : df.weights
    else
        return nothing
    end
end

"""
    build_response_matrices(det) -> (responseMatrices, Ereco_bins_ES_extended, Ereco_bins_CC_extended)

Build detector response matrices for detector `det`. Uses shared global `Etrue_bins`.
`responseMatrices` includes `.eff` (selection×reco efficiencies) and `.bins`
(Ereco/cos-scatter bin specs) so propagation functions need no separate globals.
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

    # Extend ereco bins from -Inf to +Inf
    bins_ES = collect(range(Ereco_bins_ES.min, stop=Ereco_bins_ES.max, length=Ereco_bins_ES.bin_number + 1))
    bins_CC = collect(range(Ereco_bins_CC.min, stop=Ereco_bins_CC.max, length=Ereco_bins_CC.bin_number + 1))
    bins_ES[1] = -Inf;  bins_ES[end] = Inf
    bins_CC[1] = -Inf;  bins_CC[end] = Inf
    Ereco_bins_ES_extended = (bin_number=Ereco_bins_ES.bin_number, min=Ereco_bins_ES.min, max=Ereco_bins_ES.max, bins=bins_ES)
    Ereco_bins_CC_extended = (bin_number=Ereco_bins_CC.bin_number, min=Ereco_bins_CC.min, max=Ereco_bins_CC.max, bins=bins_CC)

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

    # ── TEMPORARY efficiency boost ────────────────────────────────────────────
    # The ES selection mask is too stringent above ~10 MeV.  To study high-energy
    # sensitivity without retuning the selection, we artificially raise the masking
    # efficiency to a 90% plateau by 11.5 MeV using a smooth Hermite step.
    #
    # Affected quantities (inclusive / semi-inclusive modes only):
    #   • ES_nue_eff / ES_nuother_eff  → ES signal rates in the inclusive likelihood
    #   • CC_incl_eff (boosted below)  → forward CC rates projected onto ES bins
    # Exclusive-mode ES rates (eff.ES_nue / eff.ES_nuother used outside the
    # inclusive block) are unaffected because CC_eff is not boosted here.
    #
    # Remove this block once the selection efficiency is properly tuned at high energy.
    if inclusive_analysis || semi_inclusive_analysis
        _es_edges   = collect(range(Ereco_bins_ES.min, Ereco_bins_ES.max, length=Ereco_bins_ES.bin_number+1))
        _es_centers = 0.5 .* (_es_edges[1:end-1] .+ _es_edges[2:end])
        _t      = clamp.((_es_centers .- 0.010) ./ (0.0115 - 0.010), 0.0, 1.0)
        _smooth = @. _t^2 * (3 - 2*_t)   # Hermite smooth-step: 0 at 10 MeV, 1 at 11.5 MeV
        ES_nue_eff     = @. ES_nue_eff     + _smooth * (0.9 - ES_nue_eff)
        ES_nuother_eff = @. ES_nuother_eff + _smooth * (0.9 - ES_nuother_eff)
    end

    eff_tuple = (ES_nue=ES_nue_eff, ES_nuother=ES_nuother_eff, CC=CC_eff)
    bins_tuple = (ES=Ereco_bins_ES_extended, CC=Ereco_bins_CC_extended, cos_scatter=cos_scatter_bins)

    if inclusive_analysis || semi_inclusive_analysis
        # CC mapped to ES bins — used for the forward inclusive hemisphere in both modes
        CC_inclusive_response = create_response_matrix(CC_sample, Etrue_bins, Ereco_bins_ES_extended)
        CC_incl_sel, _ = create_histogram(df_CC.Ereco[df_CC.mask], Ereco_bins_ES_extended,
                                          normalise=false)
        CC_incl_tot, _ = create_histogram(df_CC.Ereco, Ereco_bins_ES_extended,
                                          normalise=false)
        CC_incl_sel_eff = @. ifelse(CC_incl_tot == 0, 0.0, CC_incl_sel / CC_incl_tot)
        CC_incl_eff     = CC_incl_sel_eff
        # TEMP: forward CC masking efficiency boost — same Hermite ramp as ES above.
        # Applies to the inclusive CC→ES-bin projection (forward hemisphere in semi-inclusive).
        # _smooth is computed in the temporary block above; both blocks must be removed together.
        CC_incl_eff = @. CC_incl_eff + _smooth * (0.9 - CC_incl_eff)
        eff_tuple = merge(eff_tuple, (CC_incl=CC_incl_eff,))

        if semi_inclusive_analysis
            # ── CC angular split fractions (uniform-CC assumption) ────────────
            # ang_mask selects FORWARD bins (centers >= angular_cos_cut), already computed above.
            Ncos       = cos_scatter_bins.bin_number
            f_CC_above = count(ang_mask) / Ncos   # fraction of CC in forward hemisphere
            f_CC_below = 1.0 - f_CC_above         # fraction of CC in backward hemisphere

            # ── Mis-ID response: two-step ES→CC chain ────────────────────────
            # Step 1: nue/nuother_ES_response (normalised, NO efficiency) encodes the
            #   100%-efficiency ES reco pass.  ES_nue_eff is deliberately NOT applied here.
            # Step 2: CC_response reindexed to ES reco bin centres encodes the CC reco pass
            #   treating e_reco_ES as pseudo-true energy (the stated approximation).
            # Backward fraction (per Ereco_ES bin) selects only backward-scattered ES events.
            N_ES        = Ereco_bins_ES.bin_number
            ES_centers  = 0.5 .* (bins_ES[1:end-1] .+ bins_ES[2:end])
            Etrue_edges = collect(range(Etrue_bins.min, Etrue_bins.max,
                                        length = Etrue_bins.bin_number + 1))

            # Reindex CC response rows by ES reco bin centres
            CC_response_as_ereco = zeros(N_ES, Ereco_bins_CC.bin_number)
            for j in 1:N_ES
                etrue_bin = min(searchsortedlast(Etrue_edges, ES_centers[j]),
                                Etrue_bins.bin_number)
                if etrue_bin >= 1
                    CC_response_as_ereco[j, :] = CC_response[etrue_bin, :]
                end
            end

            # Backward angular fraction per Ereco_ES bin
            backward_frac = vec(sum(angular_ES_response[.!ang_mask, :], dims=1))  # (N_Ereco_ES,)

            # Scale ES response columns by backward fraction, then fold with CC_response_as_ereco.
            # Result: (N_Etrue × N_Ereco_CC) matrix encoding the full backward mis-ID chain.
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
            # Pure inclusive (existing path unchanged)
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

    return responseMatrices, Ereco_bins_ES_extended, Ereco_bins_CC_extended
end


