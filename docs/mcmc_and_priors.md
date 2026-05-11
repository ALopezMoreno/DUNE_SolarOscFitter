# MCMC and Priors

This document describes the Hamiltonian Monte Carlo setup, prior distributions, and batch sampling strategy. The implementation is in `src/mcmc.jl`.

---

## 1. BAT.jl setup

Sampling uses BAT.jl's `TransformedMCMC` with a Hamiltonian Monte Carlo kernel:

```julia
proposal_algorithm = HamiltonianMC(
    termination = GeneralisedNoUTurn(max_depth = 4)
)
hmc_context = BATContext(ad = ADSelector(:ForwardDiff))
```

- **NUTS with max_depth = 4**: the No-U-Turn Sampler adaptively selects the number of leapfrog steps per proposal, up to 2⁴ = 16 steps. Increasing `max_depth` improves mixing for correlated posteriors at the cost of more likelihood evaluations per step.
- **ForwardDiff AD**: gradients are computed by forward-mode automatic differentiation. The full pipeline from oscillation parameters to log-likelihood is differentiable (see [oscillation_physics.md §8](oscillation_physics.md#8-forwarddiff-gradient-flow)).

These are module-level globals in `mcmc.jl`. To change HMC tuning, edit only that file.

---

## 2. Prior distributions

All priors are defined in `mcmc.jl` and constructed from the distributions in `config.yaml` via `string_to_distribution()`.

### Oscillation parameters

| Parameter | Config key | Typical prior | Physics motivation |
|-----------|-----------|---------------|-------------------|
| sin²θ₁₂ | `prior_sin2_th12` | `Uniform(0.15, 0.5)` | LMA region |
| sin²θ₁₃ | `prior_sin2_th13` | `truncated(Normal(0.022, 0.0007), 0.005, 0.035)` | Reactor experiment constraint |
| Δm²₂₁ (eV²) | `prior_dm2_21` | `Uniform(2e-5, 3e-4)` | Solar + KamLAND range |

### Solar flux parameters

| Parameter | Config key | Typical prior | Physics motivation |
|-----------|-----------|---------------|-------------------|
| ⁸B flux (ν/cm²/s) | `prior_8B_flux` | `truncated(Normal(5.25e6, 0.28e6), 1.5e6, 1e7)` | SNO NC measurement (5.3% uncertainty) |
| HEP flux (ν/cm²/s) | `prior_HEP_flux` | `Uniform(0, 2e4)` | Conservative upper bound |

### Earth density systematics

When `earth_potential_uncertainties: true`, one nuisance parameter per Earth layer is added:

```julia
priors[:earth_norm_i] = truncated(Normal(μᵢ, √Cᵢᵢ), 0.0, 2.0)
```

where `μᵢ` and `Cᵢᵢ` come from the covariance matrix in `earth_normalisation_prior_file`. Only the diagonal is used because BAT.jl does not yet support multivariate Gaussian priors. This approximation treats each Earth layer as independent, which is conservative — the true posterior for Earth density is correlated across layers. Update when BAT adds `MvNormal` prior support.

### Background nuisance parameters

For each background with `ES_bg_par_counts[i] = 1` (or CC equivalent):

```julia
priors[:ES_bg_norm_i] = truncated(Normal(norm, norm × sys), 0.0, 2 × norm)
```

The number of background nuisance parameters in the fit is determined at runtime by `ES_bg_par_counts` and `CC_bg_par_counts`, so the prior dict grows automatically when more backgrounds have systematics enabled.

---

## 3. Batch MCMC

Long chains are accumulated in memory-efficient batches to avoid OOM errors on clusters:

```julia
for batch in 1:n_batches
    result = bat_sample(posterior, MCMCSampling(..., nsteps = batchSteps), ...)
    JLD2.jldopen(outFile, "a+") do f
        f["batch_$(batch)"] = result
    end
    chain_state = get_chain_state(result)
end
```

Each batch of `batchSteps` steps (default 50) is appended as `batch_N` inside a single JLD2 file. The Python utilities (`posteriorHelpers.py`) handle both the flat (legacy) and batched formats transparently.

**Resuming runs:** Setting `prevFile` in config loads the serialised `_chainState.bin` from a previous run and passes it as the initial state, so chains continue from where they stopped rather than re-tuning from scratch.

**Important:** Always delete the output JLD2 file before starting a new run. JLD2 append semantics can silently mix batches from different runs into one file.

---

## 4. Tuning

BAT.jl's adaptive tuning runs for up to `maxTuningAttempts` rounds of `nTuning` steps each. During tuning, the step size (ε) and mass matrix (M) are adapted to achieve an acceptance rate near the HMC target (~0.8). The tuning state is discarded before sampling begins.

Setting `maxTuningAttempts: 0` skips tuning entirely (faster, but efficiency may be poor unless a good `proposal_matrix` is provided via config).

**Mass matrix warm-start:** If `proposal_matrix` is set in config, a pre-computed covariance matrix from a previous chain is loaded as the initial mass matrix. Generate one with:

```bash
python3 utils/makeOutputCovariance.py outputs/my_chain
```

---

## 5. Reproducibility

The random seed in `mcmc.jl` is commented out, so runs are not reproducible by default. To fix a seed:

```julia
# In mcmc.jl, add to the BATContext:
hmc_context = BATContext(ad = ADSelector(:ForwardDiff), rng = StableRNG(42))
```

Use `StableRNGs.jl` for cross-platform reproducibility.
