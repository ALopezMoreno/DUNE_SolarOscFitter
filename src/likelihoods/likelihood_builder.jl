###################################
# --- ES sample contributions --- #
###################################

function llh_ES_poisson(d::LikelihoodInputs, parameters, rates)::Float64
    idx  = d.index_ES
    nObs = d.nObserved

    loglh_ES_day = poissonLogLikelihood(
        rates.ES_day[idx:end],
        nObs.ES_day[idx:end],
    )

    loglh_ES_night = sum(
        poissonLogLikelihood(row[idx:end], obs_row[idx:end])
        for (row, obs_row) in zip(eachrow(rates.ES_night), eachrow(nObs.ES_night))
    )

    return loglh_ES_day + loglh_ES_night
end

function llh_ES_poisson_perbin(d::LikelihoodInputs, parameters, rates;
                               below_threshold::Bool = false)

    idx  = d.index_ES
    nObs = d.nObserved

    n_bins = length(nObs.ES_day)

    # Allocate full outputs filled with zeros (zero contribution below threshold)
    llh_ES_day_bins   = zeros(n_bins)               # same length as ES_day
    llh_ES_night_bins = zeros(size(nObs.ES_night))  # same shape as ES_night

    mask = below_threshold ? (1:n_bins) : (idx:n_bins)

    @views begin
        llh_ES_day_bins[mask] .= perbin_poissonLogLikelihood(
            rates.ES_day[mask],
            nObs.ES_day[mask]
        )

        llh_ES_night_bins[:, mask] .= perbin_poissonLogLikelihood(
            rates.ES_night[:, mask],
            nObs.ES_night[:, mask]
        )
    end

    return (ES_day = llh_ES_day_bins, ES_night = llh_ES_night_bins)
end

function llh_ES_angle(d::LikelihoodInputs, parameters, rates)::Float64
    idx  = d.index_ES
    nObs = d.nObserved

    # DAY: rates.ES_day and nObs.ES_day are (Ncos, NE)
    # → restrict in energy (dimension 2) and flatten over all cos bins + energy bins
    loglh_ES_day = poissonLogLikelihood(
        vec(rates.ES_day[:, idx:end]),
        vec(nObs.ES_day[:, idx:end]),
    )

    # NIGHT: rates.ES_night and nObs.ES_night are (Ncos, NE, Nnight)
    # → loop over the night bins (dim=3), restrict in energy, and sum log-likelihoods
    loglh_ES_night = sum(
        poissonLogLikelihood(
            vec(rate_slice[:, idx:end]),
            vec(obs_slice[:, idx:end]),
        )
        for (rate_slice, obs_slice) in zip(
            eachslice(rates.ES_night, dims = 3),
            eachslice(nObs.ES_night,  dims = 3),
        )
    )

    return loglh_ES_day + loglh_ES_night
end

function llh_ES_angle_perbin(d::LikelihoodInputs, parameters, rates;
                             below_threshold::Bool = false)

    idx  = d.index_ES
    nObs = d.nObserved

    n_bins = size(nObs.ES_day, 2)  # bins are along 2nd dimension

    # Allocate full outputs filled with zeros (zero contribution below threshold)
    llh_ES_day_bins   = zeros(size(nObs.ES_day))        # same shape as ES_day
    llh_ES_night_bins = zeros(size(nObs.ES_night))      # same shape as ES_night

    mask = below_threshold ? (1:n_bins) : (idx:n_bins)

    @views begin
        llh_ES_day_bins[:, mask] .= perbin_poissonLogLikelihood(
            rates.ES_day[:, mask],
            nObs.ES_day[:, mask]
        )

        llh_ES_night_bins[:, mask, :] .= perbin_poissonLogLikelihood(
            rates.ES_night[:, mask, :],
            nObs.ES_night[:, mask, :]
        )
    end

    return (ES_day = llh_ES_day_bins, ES_night = llh_ES_night_bins)
end

function llh_ES_angle_conditional(d::LikelihoodInputs, parameters, angle_rates)::Float64
    idx  = d.index_ES
    nObs = d.nObserved

    loglh_ES_angle = sum(
    conditional_poissonLogLikelihood(col, obs_col)
    for (col, obs_col) in zip(eachcol(angle_rates[:, idx:end]),
                              eachcol(nObs.ES_angular[:, idx:end]))
    )

    return loglh_ES_angle
end

## NOTE: NO PERBIN CONTRIBUTION ON THIS SO FAR


###################################
# --- CC sample contributions --- #
###################################

function llh_CC_poisson(d::LikelihoodInputs, parameters, rates)::Float64
    idx  = d.index_CC
    nObs = d.nObserved

    loglh_CC_day = poissonLogLikelihood(
        rates.CC_day[idx:end],
        nObs.CC_day[idx:end],
    )

    loglh_CC_night = sum(
        poissonLogLikelihood(row[idx:end], obs_row[idx:end])
        for (row, obs_row) in zip(eachrow(rates.CC_night), eachrow(nObs.CC_night))
    )

    return loglh_CC_day + loglh_CC_night
end

function llh_CC_poisson_perbin(d::LikelihoodInputs, parameters, rates;
                               below_threshold::Bool = false)

    idx  = d.index_CC
    nObs = d.nObserved

    n_bins = length(nObs.CC_day)

    # Initialize outputs with zeros
    llh_CC_day_bins   = zeros(n_bins)
    llh_CC_night_bins = zeros(size(nObs.CC_night))

    # Decide which bins to compute
    mask = below_threshold ? (1:n_bins) : (idx:n_bins)

    @views begin
        llh_CC_day_bins[mask] .= perbin_poissonLogLikelihood(
            rates.CC_day[mask],
            nObs.CC_day[mask]
        )

        llh_CC_night_bins[:, mask] .= perbin_poissonLogLikelihood(
            rates.CC_night[:, mask],
            nObs.CC_night[:, mask]
        )
    end

    return (CC_day = llh_CC_day_bins, CC_night = llh_CC_night_bins)
end


# NOT READY: FOR TESTING ONLY
# NOTE: NO PER-BIN CALCULATION EITHER
function llh_CC_barlowBeeston(
    d::LikelihoodInputs,
    parameters,
    rates,
    σ_matrix
)::Float64

    idx  = d.index_CC
    nObs = d.nObserved

    # --- Day contribution still Poisson ---
    loglh_CC_day = poissonLogLikelihood(
        rates.CC_day[idx:end],
        nObs.CC_day[idx:end],
    )

    # --- Night contribution with Barlow–Beeston ---
    loglh_CC_night = sum(
        barlowBeestonLogLikelihood(
            row[idx:end],
            obs_row[idx:end],
            σ_row[idx:end]
        )
        for (row, obs_row, σ_row) in zip(
            eachrow(rates.CC_night),
            eachrow(nObs.CC_night),
            eachrow(σ_matrix)
        )
    )

    return loglh_CC_day + loglh_CC_night
end


###################################
# --- Put everything together --- #
###################################
function make_likelihood(
    d::LikelihoodInputs;
    use_ES::Bool = true,
    use_CC::Bool = true,
    ES_llh::Function = llh_ES_poisson,
    CC_llh::Function = llh_CC_poisson,
    uncertainty_ratio_CC_night = nothing,   # needed only for Barlow-Beeston
    debug::Bool = false
)
    return function (parameters)
        # bounds
        check_earth_norm_bounds(parameters) || return -Inf

        # propagate MC once
        rates = expected_rates(d, parameters)

        # optional debugging
        if debug
            print_negatives_1d(rates.CC_day, parameters)
            print_negatives_2d(rates.CC_night, parameters)
        end

        loglh = 0.0

        # === ES contribution ==========================================
        if use_ES
            if angular_reco
                loglh += llh_ES_angle(d, parameters, rates)
            else
                loglh += ES_llh(d, parameters, rates)
            end
        end

        # === CC contribution ==========================================
        if use_CC
            if CC_llh === llh_CC_barlowBeeston
                if uncertainty_ratio_CC_night === nothing
                    error("uncertainty_ratio_CC_night must be provided for Barlow-Beeston systematics")
                end
                loglh += CC_llh(d, parameters, rates, uncertainty_ratio_CC_night)
            else
                loglh += CC_llh(d, parameters, rates)
            end
        end

        return loglh
    end
end

function make_perbin_likelihood(
    d::LikelihoodInputs;
    rates = nothing,
    use_ES::Bool = true,
    use_CC::Bool = true,
    ES_llh::Function = llh_ES_poisson,              # kept for signature parity
    CC_llh::Function = llh_CC_poisson,              # kept for signature parity
    uncertainty_ratio_CC_night = nothing,           # needed only for Barlow-Beeston
    debug::Bool = false
)
    # Helper: build a fixed-shape zero/-Inf template based on current data shapes.
    # Uses d.nObserved because it defines the binning/shape of the likelihood terms.
    function _template(fillval::Float64)
        idxES = d.index_ES
        idxCC = d.index_CC
        nObs  = d.nObserved

        # ES shapes depend on whether angular reco is enabled.
        # - non-angular: ES_day is Vector, ES_night is Matrix
        # - angular:     ES_day is Matrix, ES_night is 3D Array

        ###### THIS BLOCK IGNORES BINS BELOW THE THRESHOLD ######

        #=
        ES_day_shape   = angular_reco ? size(@view nObs.ES_day[:, idxES:end]) :
                                        size(@view nObs.ES_day[idxES:end])
        ES_night_shape = angular_reco ? size(@view nObs.ES_night[:, idxES:end, :]) :
                                        size(@view nObs.ES_night[:, idxES:end])

        CC_day_shape   = size(@view nObs.CC_day[idxCC:end])
        CC_night_shape = size(@view nObs.CC_night[:, idxCC:end])
        =#
        ##########################################################

        ###### THIS BLOCK COUNTS BINS BELOW THE THRESHOLD ######

        ES_day_shape   = angular_reco ? size(nObs.ES_day) :
                                        size(nObs.ES_day)
        ES_night_shape = angular_reco ? size(nObs.ES_night) :
                                        size(nObs.ES_night)

        CC_day_shape   = size(nObs.CC_day)
        CC_night_shape = size(nObs.CC_night)

        #########################################################

        ES_day   = fill(fillval, ES_day_shape...)
        ES_night = fill(fillval, ES_night_shape...)
        CC_day   = fill(fillval, CC_day_shape...)
        CC_night = fill(fillval, CC_night_shape...)

        return (ES_day=ES_day, ES_night=ES_night, CC_day=CC_day, CC_night=CC_night)
    end

    # Cache a zero template so we don't re-allocate shapes every call.
    # (Still allocates new arrays per call via `copy`; you can reuse and overwrite if desired.)
    zero_tpl = _template(0.0)

    return function (parameters; rates=nothing)
        # bounds: if invalid, return fixed-shape -Inf arrays
        check_earth_norm_bounds(parameters) || return _template(-Inf)

        # propagate MC
        local _rates = rates === nothing ? expected_rates(d, parameters) : rates

        # optional debugging
        if debug
            print_negatives_1d(_rates.CC_day, parameters)
            print_negatives_2d(_rates.CC_night, parameters)
        end

        # Start with zeros in the correct fixed output shape
        # (copy so callers can mutate without affecting the cached template)
        out = (ES_day=copy(zero_tpl.ES_day),
               ES_night=copy(zero_tpl.ES_night),
               CC_day=copy(zero_tpl.CC_day),
               CC_night=copy(zero_tpl.CC_night))

        # === ES contribution ==========================================
        if use_ES
            es = angular_reco ? llh_ES_angle_perbin(d, parameters, _rates) :
                                llh_ES_poisson_perbin(d, parameters, _rates)

            # Add into output (ensures fixed fields are always present)
            out.ES_day   .+= es.ES_day
            out.ES_night .+= es.ES_night
        end

        # === CC contribution ==========================================
        if use_CC
            cc = if CC_llh === llh_CC_barlowBeeston
                uncertainty_ratio_CC_night === nothing &&
                    error("uncertainty_ratio_CC_night must be provided for Barlow-Beeston systematics")
                llh_CC_barlowBeeston_perbin(d, parameters, _rates, uncertainty_ratio_CC_night)
            else
                # Default Poisson per-bin
                llh_CC_poisson_perbin(d, parameters, _rates)
            end

            out.CC_day   .+= cc.CC_day
            out.CC_night .+= cc.CC_night
        end

        return out
    end
end