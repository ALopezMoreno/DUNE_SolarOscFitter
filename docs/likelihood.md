# Likelihood

This document describes how predicted event rates are compared to observed data to form the log-likelihood. The implementation is split across three files:

- `src/likelihoods/likelihood_statistical.jl` — the statistical formulas
- `src/likelihoods/likelihood_builder.jl` — assembles components into a single function
- `src/likelihoods/likelihood_main.jl` — wraps for BAT.jl and constructs the per-bin diagnostic version

---

## 1. Assembly (`likelihood_builder.jl`)

**Function:** `make_likelihood(d, ...)` (`likelihood_builder.jl:216–263`)

`d` is a `LikelihoodInputs` struct containing:
- `nObserved` — the Asimov dataset (fixed; produced by `setup.jl`)
- `index_ES`, `index_CC` — energy threshold bin indices

At each MCMC step:

1. **Earth norm bounds check** — if any `earth_norm` parameter is outside [0, 1.3], return `−∞` immediately (prevents propagation into unphysical Earth density territory).
2. **Call `propagateSamples`** with current parameter values to get predicted event rates.
3. **Accumulate log-likelihood contributions** from enabled channels.

**Channel flags:**

```julia
use_ES = ES_mode
use_CC = CC_mode && !inclusive_analysis
```

When `inclusiveMode: true`, CC events are folded into the ES likelihood; the separate CC branch is disabled.

**Likelihood selector:**

- Standard ES: `llh_ES_poisson` (Poisson over energy bins)
- Angular ES: `llh_ES_angle` (Poisson over every (cos θ_scatter, E_reco) bin, day and each night zenith bin; see §3)
- Standard CC: `llh_CC_poisson`
- CC with MC stats: `llh_CC_barlowBeeston`

---

## 2. Poisson log-likelihood

**Function:** `poissonLogLikelihood(nExpected, nMeasured)` (`likelihood_statistical.jl:1–41`)

The standard extended Poisson:

```
log L = Σᵢ [nᵢ · log(eᵢ) − eᵢ] + const
```

Rearranged as a deviance (so that log L = 0 at the true point):

```
−log L = Σᵢ term_i

where:
  term_i = eᵢ − nᵢ + nᵢ · log(nᵢ / eᵢ)   if nᵢ > 0
  term_i = eᵢ                               if nᵢ = 0
```

**Regularisation for HMC:**

Both `eᵢ` and `nᵢ` are floored at ε = 10⁻¹²:

```julia
e_reg = max(e, ε)
m_reg = max(m, ε)
term  = e - m + m * (log(m_reg) - log(e_reg))   # for m > 0
```

When `e → 0` and `m > 0`, `log(ε) ≈ −27.6` produces a large, smooth penalty. A previous implementation used a hard-wall `1e9 · (1 − e/ε)²` term which created a gradient discontinuity that could trap HMC near `e = 0`; this was replaced with the log-floor approach.

---

## 3. Angular ES log-likelihood

**Function:** `llh_ES_angle` (`likelihood_builder.jl`)

With `use_scattering_info: true` the ES rates are binned in (cos θ_scatter, E_reco), by day and per
night zenith bin, and `llh_ES_angle` sums the Poisson log-likelihood (§2) over all of those bins
above the energy threshold.

This already contains the angular shape information. For Poisson counts nⱼ with expectations μⱼ in
the angle bins j of one energy bin, with N = Σⱼ nⱼ and M = Σⱼ μⱼ,

```
∏ⱼ Poisson(nⱼ | μⱼ) = Poisson(N | M) × Multinomial(n₁…n_k | N, μⱼ / M)
```

so the 2-D Poisson log-likelihood is exactly the energy-spectrum term plus the conditional
(shape-only) angular term. A separate conditional likelihood would count the angular shape twice
if added, and would discard the rate and spectrum if used alone. It would only be the right tool
if the angular normalisation per energy bin were not trusted; the code no longer contains one
(removed in October 2026, see git history).

---

## 3b. MC statistical uncertainty (`mc_stat_likelihood`)

`src/likelihoods/mcstat.jl`. With `mc_stat_likelihood: DembinskiAbdelmotteleb` or `BarlowBeeston`,
each channel's predictions are scaled by a factor β per E_reco bin, shared by the day bin and all
night bins of that energy (`mc_stat_day_night: shared`), and profiled in closed form against the
bin's effective MC count a = M²/Var_MC(M), computed once at the Asimov truth from the signal
response counts and the background weights. Method: Dembinski & Abdelmotteleb,
[arXiv:2206.12346](https://arxiv.org/abs/2206.12346); tests in `tests/test_mcstat.jl` and the
regression variants `night_da`, `night_bb`, `night_da_indep`.

## 4. Barlow–Beeston log-likelihood (old CC-night-only function, not used)

Superseded by §3b; `llh_CC_barlowBeeston` is kept only as the formula reference it was.

**Function:** `barlowBeestonLogLikelihood(nExpected, nMeasured, sigmaVar)` (`likelihood_statistical.jl:149–207`)

Extends the Poisson likelihood to account for finite MC statistics in the background prediction. `sigmaVar[i] = σᵢ / μᵢ` is the fractional uncertainty in the expected count for bin i.

**Formula:** Profile over a hidden nuisance parameter βᵢ (true background scale):

```
β = ½ · (1 − e·s² + √[(e·s² − 1)² + 4·m·s²])

log L_BB = β·e − m + m·log(m / (β·e)) + (β−1)² / (2s²)
```

When `s = 0` (no MC uncertainty), this reduces to the standard Poisson formula with `β = 1`.

**Regularisation at degenerate counts:**

When `e = 0` and `m > 0`, or `m = 0` and `e > 0`, the BB formula is undefined. A half-count substitution is applied:
```julia
e = 0.5   # if e == 0 and m > 0
m = 0.5   # if m == 0 and e > 0
```
The half count acts as the effective 1σ scale of an empty Poisson bin (in the Gaussian approximation an n = 0 bin would get σ = √n = 0, i.e. infinite confidence), keeping the mismatch penalty finite and reasonably sized. These substitutions are intentional and will be revisited once a proper background floor is in place.

---

## 5. Per-bin diagnostics (`make_perbin_likelihood`)

**Function:** `make_perbin_likelihood(d, ...)` (`likelihood_builder.jl:265–338`)

Returns a function that, given a parameter point, produces the per-bin log-likelihood contribution arrays (same shape as the observed data) rather than a scalar sum. Used by `RunMode: derived` to append bin-level diagnostics to chain files for plotting by `utils/plotBinDiagnostics.py`.

The formulas are identical to the aggregate case; output is arrays of per-bin deviance values filled with 0 below the energy threshold and `−Inf` for disabled channels.

---

## 6. BAT.jl integration (`likelihood_main.jl`)

The Julia function produced by `make_likelihood` is wrapped in `logfuncdensity(...)` for BAT.jl when `RunMode == "MCMC"`:

```julia
likelihood_all_samples = logfuncdensity(make_likelihood(ereco_data))
```

BAT calls this function once per HMC leapfrog step. ForwardDiff propagates gradients through the full call chain (see [oscillation_physics.md](oscillation_physics.md) §8).
