#=
priors.jl — prior distribution of every fit parameter, shared by RunMode MCMC and LLH.
=#

"""
    build_priors() -> Dict{Symbol,Any}

Prior of every free parameter, from the config globals and each detector's background nuisances
(setup.jl must have run). Earth layers get independent truncated normals; mcmc.jl adds the
correlation term of a correlated covariance to the log-density.
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
      # Independent truncated normals per layer; correlations are added to the log-density below.
      means = mean(earth_normalisation_prior)
      covmat = cov(earth_normalisation_prior)
      n = length(means)
      for i in 1:n
        priors[Symbol("earth_norm_$i")] = truncated(Normal(means[i], sqrt(covmat[i, i])), 0.0, 2.0)
      end
      # param_bounds = Dict(:earth_norm => (0.0, 2.0)) # We want these to be fixed between 0 and 2
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
