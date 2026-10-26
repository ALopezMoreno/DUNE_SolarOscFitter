#=
priors.jl — prior distribution of every fit parameter, shared by RunMode MCMC and LLH.
=#

"""
    build_priors() -> Dict{Symbol,Any}

Prior of every free parameter, from the config globals and each detector's background nuisances
(setup.jl must have run). Earth layers get their marginals (truncated normals), as the LLH scan uses
them; the MCMC samples them jointly as one MvNormal (`bat_priors`).
"""
function build_priors()
    # Construct a dictionary of priors
    priors = Dict{Symbol,Any}(
      # miximg parameters
      :sin2_th12 => prior_sin2_th12,
      :sin2_th13 => prior_sin2_th13,
      :dm2_21 => prior_dm2_21,

      # HEP discovery
      :integrated_HEP_flux => prior_HEP_flux,

      # systematic parameters
      :integrated_8B_flux => prior_8B_flux,
      # CC xsec shape: omit when fix_cc_xsec=true so they are not sampled.
      # propagation_osc.jl falls back to the Asimov true values in that case.
      (fix_cc_xsec ? () : (:cc_xsec_norm => prior_cc_xsec_norm,
                            :cc_xsec_tilt => prior_cc_xsec_tilt,
                            :cc_xsec_curv => prior_cc_xsec_curv))...
    )


    # Conditionally add nuisance parameters
    if earthUncertainty
      # Marginals per layer; the MCMC uses the joint MvNormal (bat_priors)
      means = mean(earth_normalisation_prior)
      covmat = cov(earth_normalisation_prior)
      n = length(means)
      for i in 1:n
        priors[Symbol("earth_norm_$i")] = truncated(Normal(means[i], sqrt(covmat[i, i])), 0.0, 2.0)
      end
    end

    # Add per-detector background nuisance parameters (names prefixed by detector name)
    for (dname, out) in detector_outputs
        det = detector_configs[dname]
        if det.ES_mode
            for (i, dist) in enumerate(out.ES_bg_norms_pars)
                priors[Symbol("$(dname)_ES_bg_norm_$i")] = dist
            end
        end
        if det.CC_mode && !det.inclusive_analysis
            for (i, dist) in enumerate(out.CC_bg_norms_pars)
                priors[Symbol("$(dname)_CC_bg_norm_$i")] = dist
            end
        end
    end
    return priors
end

# BAT samples the Earth layer scales as K independent standard normals `earth_z`; the likelihood reads
# earth_norm = μ + L z (L L' = the prior covariance): the full MvNormal of the Earth prior file. BAT's
# own transform of a full-covariance MvNormal is not Enzyme-differentiable (BAT 4.0.9).
const EARTH_NORM_NAMES = earthUncertainty ?
    Tuple(Symbol("earth_norm_$i") for i in eachindex(earth_normalisation_true)) : ()
const EARTH_MU = earthUncertainty ? Vector{Float64}(mean(earth_normalisation_prior)) : Float64[]
const EARTH_L  = earthUncertainty ? Matrix(cholesky(Symmetric(Matrix(cov(earth_normalisation_prior)))).L) : zeros(0, 0)

"""
    bat_priors(priors) -> Dict{Symbol,Any}

The priors as BAT samples them: the `earth_norm_i` marginals replaced by one `earth_z`, K independent
standard normals that `flat_params` maps onto the Earth prior's MvNormal (SPEC 2.6).
"""
function bat_priors(priors)
    earthUncertainty || return priors
    p = Dict{Symbol,Any}(k => v for (k, v) in priors if k ∉ EARTH_NORM_NAMES)
    p[:earth_z] = MvNormal(zeros(length(EARTH_MU)), Diagonal(ones(length(EARTH_MU))))
    return p
end

# μᵢ + Σⱼ Lᵢⱼ zⱼ, plain loop (Enzyme)
function _earth_norm(z, i)
    acc = zero(eltype(z)) + EARTH_MU[i]
    for j in 1:i
        acc += EARTH_L[i, j] * z[j]
    end
    return acc
end

"""
    flat_params(p) -> NamedTuple

BAT parameters with `earth_z` mapped to the Earth layer scales `earth_norm_1…K`, as the likelihood, the
chain files and the LLH scan name them. Type-stable (K is a constant).
"""
flat_params(p::NamedTuple{N}) where {N} = :earth_z in N ?
    merge(Base.structdiff(p, NamedTuple{(:earth_z,)}),
          NamedTuple{EARTH_NORM_NAMES}(ntuple(i -> _earth_norm(p.earth_z, i), Val(length(EARTH_NORM_NAMES))))) : p

"""
    bat_params(p) -> NamedTuple

Inverse of `flat_params`: the Earth layer scales `earth_norm_1…K` mapped back to `earth_z`.
"""
function bat_params(p::NamedTuple)
    isempty(EARTH_NORM_NAMES) && return p
    rest = Base.structdiff(p, NamedTuple{EARTH_NORM_NAMES})
    return merge(rest, (earth_z = LowerTriangular(EARTH_L) \ (Float64[p[k] for k in EARTH_NORM_NAMES] .- EARTH_MU),))
end

"""
    build_posterior(llh, priors=build_priors()) -> (prior, posterior)

BAT prior (`bat_priors`) and posterior of the log-likelihood `llh` of flat parameters.
"""
function build_posterior(llh, priors=build_priors())
    prior = distprod(; bat_priors(priors)...)
    posterior = PosteriorMeasure(let f = llh; logfuncdensity(p -> f(flat_params(p))) end, prior)
    return prior, posterior
end
