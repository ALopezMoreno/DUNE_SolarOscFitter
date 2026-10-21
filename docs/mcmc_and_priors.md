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
| ⁸B flux (ν/cm²/s) | `prior_8B_flux` | `truncated(Normal(5.25e6, 0.20e6), 1.5e6, 1e7)` | SNO NC measurement, PRC 88 025501: 5.25 ± 0.16 (stat) ± 0.12 (syst, symmetrised), in quadrature 3.8% |
| HEP flux (ν/cm²/s) | `prior_HEP_flux` | `Uniform(0, 2e4)` | Conservative upper bound |

### CC cross-section shape parameters

Unless `fix_cc_xsec: true`, three nuisance parameters describe the CC cross-section shape relative to the nominal table (`cc_xsec_file`, default MARLEY 2.0.0):

| Parameter | Config key | Default prior | Role |
|-----------|-----------|---------------|------|
| `cc_xsec_norm` | `prior_cc_xsec_norm` | `truncated(Normal(1.0, 0.1), 0.1, 2.0)` | Overall scale |
| `cc_xsec_tilt` | `prior_cc_xsec_tilt` | `Normal(0.0, 0.1)` | Spectral slope: `exp(tilt·p₁)`, `p₁ = L − ⟨L⟩` |
| `cc_xsec_curv` | `prior_cc_xsec_curv` | `Normal(0.0, 0.05)` | Curvature: `exp(curv·p₂)`, `p₂ = L² − a·L − b` |

The cross section is `σ(E) = σ_nominal(E) · norm · exp(tilt·p₁ + curv·p₂)` per E_true bin, with `L = ln(E/E_pivot)` (`E_pivot` is the rate-weighted mean energy; it only fixes the origin of L). `p₁`, `p₂` are monic polynomials in L made orthogonal (Gram–Schmidt) to 1 and to each other under the weight `w_i` = unoscillated CC ⁸B rate in E_true bin i (nominal σ × ⁸B flux × exposure, a setup-time constant): `⟨p₁⟩ = ⟨p₂⟩ = ⟨p₁p₂⟩ = 0`, `⟨f⟩ = Σ w f / Σ w`. Consequently, for the unoscillated CC ⁸B rate `N = Σ w·shape`, `∂ln N/∂tilt = ∂ln N/∂curv = 0` at `tilt = curv = 0` and the tilt–curv cross term vanishes: tilt and curv change the shape, not (to first order) the rate, which is the job of norm. This is first order only (`∂²ln N/∂tilt² = ⟨p₁²⟩ ≠ 0`), and it holds for the unoscillated ⁸B spectrum, not for the oscillated, efficiency-weighted one. Being monic, curv is the coefficient of `(ln E)²`; the slope `d ln σ/d ln E = tilt + curv·(2L − a)` equals tilt where `L = a/2` (about 9 MeV for the default binning), or everywhere when curv = 0. So the same prior strings as in the old basis (`exp(tilt·L + curv·L²)`) induce a slightly different prior on σ(E): old chains with free tilt/curv are not comparable. Setup logs `E_pivot`, `⟨L⟩`, `a`, `b` per detector (`cc_xsec_shape_basis` in `src/xsec.jl`; SPEC 3.3).

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

Each batch of `batchSteps` steps (200, set in `mcmc.jl`) is appended as `batch_N` inside a single JLD2 file. The batch size only sets the checkpoint/output cadence — tuning state persists across batches. The Python utilities (`posteriorHelpers.py`) handle both the flat (legacy) and batched formats transparently. The file also holds the run record as the string `run_config` (see [configuration_reference.md](configuration_reference.md), "Run record"); readers only look at `batch_*` and `derived_*` keys.

**Resuming runs:** Setting `prevFile` in config loads the serialised `_chainState.bin` from a previous run and passes it as the initial state, so chains continue from where they stopped rather than re-tuning from scratch. On resume, `warmup_process()` runs a throwaway likelihood evaluation first to warm the heap — without it, large resumed runs hit a severe GC collapse.

**Important:** Always delete the output JLD2 file before starting a new run. JLD2 append semantics can silently mix batches from different runs into one file.

---

## 4. Tuning

BAT.jl's adaptive tuning runs for up to `maxTuningAttempts` rounds of `nTuning` steps each. BAT samples in a pretransformed space (PriorToNormal: every prior becomes a standard normal) and puts an affine map z ↦ A z + b on top, which plays the role of the mass matrix. The step size is tuned by dual averaging (StepSizeAdaptor), and A by BAT's RAM rule. Both keep adapting through the sampling batches; a resumed run continues from the saved tuner state.

The RAM rule was written for random-walk Metropolis and does not learn the posterior shape for HMC: it rescales A along each trajectory by (acceptance − 0.8) while the step size adapts at the same time, and the overall scale of A and the step size are interchangeable, so nothing fixes the shape. On a toy posterior with a ×100 spread of widths it ended at A ≈ identity from a cold start, and a correct starting A was gone within ~50 init steps. The tuned scales of chain 1 are logged after tuning ("Chain 1 tuned transform scales"); values near 1 mean A is still the prior shape. When posterior widths differ a lot from the prior widths, NUTS then needs long trajectories (often the full 2^`hmc_max_depth` gradients per step).

Setting `maxTuningAttempts: 0` skips tuning entirely.

**Mass-matrix warm start:** set `warm_start_chain` to the `outFile` of an earlier run (same fitted parameters and priors; the likelihood may differ slightly, e.g. a changed binning). Its samples are mapped into the pretransformed space, b and A are set to their mean and Cholesky-factored covariance, A is kept fixed (only the step size is tuned; with `maxTuningAttempts: 0` only during init and sampling), and the chains start from the earlier run's draws. A resumed run keeps the fixed A.

Measured (2026-10-16, not a test): on the toy above, effective samples per 1000 gradients went from 4 to 125. On HD CC (10% cross-section prior, 16 parameters, `hmc_max_depth: 5`, 3 chains × ~290 steps), warm-starting from a cold run of the same length took the smallest ESS from 6 to 272, ESS per 1000 gradients from 0.2 to 15, max R-hat from 1.21 to 1.006, and gradients per step from 32 (the cap) to 20.

Because A stays fixed, a previous run that under-covers the posterior (too narrow along some direction) fixes that shape for the whole new run. The chain is still valid, only less efficient. Warm-start again from the new run if its posterior differs much from the old one.

---

## 5. Reproducibility

The random seed in `mcmc.jl` is commented out, so runs are not reproducible by default. To fix a seed:

```julia
# In mcmc.jl, add to the BATContext:
hmc_context = BATContext(ad = ADSelector(:ForwardDiff), rng = StableRNG(42))
```

Use `StableRNGs.jl` for cross-platform reproducibility.
