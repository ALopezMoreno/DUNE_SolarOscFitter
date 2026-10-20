###################################
# --- ES sample contributions --- #
###################################

# Poisson log-likelihood of the night rows from energy bin `idx` on. Explicit loops: Enzyme rejects
# zipping the active rates with the constant data.
function _night_poisson_sum(rates_night::AbstractMatrix, obs_night::AbstractMatrix, idx::Int)
    n_E = size(rates_night, 2)
    acc = poissonLogLikelihood(view(rates_night, 1, idx:n_E), view(obs_night, 1, idx:n_E))
    for k in 2:size(rates_night, 1)
        acc += poissonLogLikelihood(view(rates_night, k, idx:n_E), view(obs_night, k, idx:n_E))
    end
    return acc
end

# Angular form: rates (n_cos, n_E, n_cosz); one Poisson term per cos z slice.
function _night_angular_poisson_sum(rates_night::AbstractArray{<:Any,3}, obs_night::AbstractArray{<:Any,3}, idx::Int)
    n_E = size(rates_night, 2)
    acc = poissonLogLikelihood(vec(rates_night[:, idx:n_E, 1]), vec(obs_night[:, idx:n_E, 1]))
    for k in 2:size(rates_night, 3)
        acc += poissonLogLikelihood(vec(rates_night[:, idx:n_E, k]), vec(obs_night[:, idx:n_E, k]))
    end
    return acc
end

function llh_ES_poisson(d::LikelihoodInputs, parameters, rates)
    idx  = d.index_ES
    nObs = d.nObserved

    loglh_ES_day = poissonLogLikelihood(
        rates.ES_day[idx:end],
        nObs.ES_day[idx:end],
    )

    loglh_ES_night = _night_poisson_sum(rates.ES_night, nObs.ES_night, idx)

    return loglh_ES_day + loglh_ES_night
end

function llh_ES_poisson_perbin(d::LikelihoodInputs, parameters, rates;
                               below_threshold::Bool = false)

    idx  = d.index_ES
    nObs = d.nObserved

    n_bins = length(nObs.ES_day)

    llh_ES_day_bins   = zeros(n_bins)               # zero below threshold
    llh_ES_night_bins = zeros(size(nObs.ES_night))

    # the overflow bin (to +Inf) is left out of the diagnostic: no signal, it dominates the scores
    mask = below_threshold ? (1:n_bins) : (idx:n_bins-1)

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

function llh_ES_angle(d::LikelihoodInputs, parameters, rates)
    idx  = d.index_ES
    nObs = d.nObserved

    # day (Ncos, NE), night (Ncos, NE, Nnight)
    loglh_ES_day = poissonLogLikelihood(
        vec(rates.ES_day[:, idx:end]),
        vec(nObs.ES_day[:, idx:end]),
    )

    loglh_ES_night = _night_angular_poisson_sum(rates.ES_night, nObs.ES_night, idx)

    return loglh_ES_day + loglh_ES_night
end

function llh_ES_angle_perbin(d::LikelihoodInputs, parameters, rates;
                             below_threshold::Bool = false)

    idx  = d.index_ES
    nObs = d.nObserved

    n_bins = size(nObs.ES_day, 2)

    llh_ES_day_bins   = zeros(size(nObs.ES_day))
    llh_ES_night_bins = zeros(size(nObs.ES_night))

    mask = below_threshold ? (1:n_bins) : (idx:n_bins-1)

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


###################################
# --- CC sample contributions --- #
###################################

function llh_CC_poisson(d::LikelihoodInputs, parameters, rates)
    idx  = d.index_CC
    nObs = d.nObserved

    loglh_CC_day = poissonLogLikelihood(
        rates.CC_day[idx:end],
        nObs.CC_day[idx:end],
    )

    loglh_CC_night = _night_poisson_sum(rates.CC_night, nObs.CC_night, idx)

    return loglh_CC_day + loglh_CC_night
end

function llh_CC_poisson_perbin(d::LikelihoodInputs, parameters, rates;
                               below_threshold::Bool = false)

    idx  = d.index_CC
    nObs = d.nObserved

    n_bins = length(nObs.CC_day)

    llh_CC_day_bins   = zeros(n_bins)
    llh_CC_night_bins = zeros(size(nObs.CC_night))

    # overflow bin excluded
    mask = below_threshold ? (1:n_bins) : (idx:n_bins-1)

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


# Testing only, no per-bin form.
function llh_CC_barlowBeeston(
    d::LikelihoodInputs,
    parameters,
    rates,
    σ_matrix
)

    idx  = d.index_CC
    nObs = d.nObserved

    loglh_CC_day = poissonLogLikelihood(
        rates.CC_day[idx:end],
        nObs.CC_day[idx:end],
    )

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

# -Inf in the parameters' number type, so the return type stays concrete (BAT rejects a Union).
_neg_inf(parameters::NamedTuple) = convert(promote_type(Float64, map(typeof, values(parameters))...), -Inf)
_neg_inf(parameters) = -Inf
function make_likelihood(
    d::LikelihoodInputs;
    use_ES::Bool = true,
    use_CC::Bool = true,
    ES_llh = llh_ES_poisson,
    CC_llh = llh_CC_poisson,
    uncertainty_ratio_CC_night = nothing,   # needed only for Barlow-Beeston
    debug::Bool = false
)
    return function (parameters; precomputed_osc=nothing)
        check_earth_norm_bounds(parameters) || return _neg_inf(parameters)

        rates = expected_rates(d, parameters; precomputed_osc=precomputed_osc)

        if debug
            print_negatives_1d(rates.CC_day, parameters)
            print_negatives_2d(rates.CC_night, parameters)
        end

        loglh = zero(_neg_inf(parameters))

        # === ES contribution ==========================================
        if use_ES
            loglh += ES_llh(d, parameters, rates)
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
    ES_llh = llh_ES_poisson_perbin,
    CC_llh = llh_CC_poisson_perbin,
    uncertainty_ratio_CC_night = nothing,           # needed only for Barlow-Beeston
    debug::Bool = false
)
    # Arrays shaped like d.nObserved, filled with `fillval`
    function _template(fillval::Float64)
        nObs = d.nObserved
        ES_day   = fill(fillval, size(nObs.ES_day)...)
        ES_night = fill(fillval, size(nObs.ES_night)...)
        CC_day   = fill(fillval, size(nObs.CC_day)...)
        CC_night = fill(fillval, size(nObs.CC_night)...)
        return (ES_day=ES_day, ES_night=ES_night, CC_day=CC_day, CC_night=CC_night)
    end

    zero_tpl = _template(0.0)

    return function (parameters; rates=nothing)
        check_earth_norm_bounds(parameters) || return _template(-Inf)

        local _rates = rates === nothing ? expected_rates(d, parameters) : rates

        if debug
            print_negatives_1d(_rates.CC_day, parameters)
            print_negatives_2d(_rates.CC_night, parameters)
        end

        out = (ES_day=copy(zero_tpl.ES_day),
               ES_night=copy(zero_tpl.ES_night),
               CC_day=copy(zero_tpl.CC_day),
               CC_night=copy(zero_tpl.CC_night))

        # === ES contribution ==========================================
        if use_ES
            es = ES_llh(d, parameters, _rates)

            out.ES_day   .+= es.ES_day
            out.ES_night .+= es.ES_night
        end

        # === CC contribution ==========================================
        if use_CC
            cc = if CC_llh === llh_CC_barlowBeeston
                uncertainty_ratio_CC_night === nothing &&
                    error("uncertainty_ratio_CC_night must be provided for Barlow-Beeston systematics")
                error("llh_CC_barlowBeeston_perbin is not yet implemented")
            else
                CC_llh(d, parameters, _rates)
            end

            out.CC_day   .+= cc.CC_day
            out.CC_night .+= cc.CC_night
        end

        return out
    end
end