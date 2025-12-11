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

# NOT READY: FOR TESTING ONLY
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
