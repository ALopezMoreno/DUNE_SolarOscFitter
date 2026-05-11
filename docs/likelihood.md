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
- Angular ES: `llh_ES_angle` (conditional multinomial over scatter-angle bins within each energy bin)
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

## 3. Conditional (shape-only) log-likelihood

**Function:** `conditional_poissonLogLikelihood(nExpected, nMeasured)` (`likelihood_statistical.jl:72–145`)

Used for angular reconstruction (`llh_ES_angle`). Compares the *shape* of the scatter-angle distribution within each reconstructed energy bin, marginalising over the total count.

**Derivation:** For Poisson counts with expected rates μⱼ in angular bin j within a fixed energy bin:

```
P(c = j | b) = μⱼ / Σⱼ μⱼ       (conditional probabilities)
```

The conditional log-likelihood:

```
log L_cond = Σⱼ nⱼ · log(μⱼ / Σⱼ μⱼ)
           = Σⱼ nⱼ · [log μⱼ − log(Σⱼ μⱼ)]
```

Implemented as a KL-divergence term that cancels the data-only constant:

```julia
llh -= n * (log(n / Σn) - log(μ / Σμ))     # for n > 0, μ > 0
llh += -1e9                                  # for n > 0, μ = 0 (observed where forbidden)
```

**Edge cases:**
- `Σμ = 0`, `Σn = 0`: returns 0 (no constraint)
- `Σμ = 0`, `Σn > 0`: returns `−1e9` (large penalty)

This is added to the Poisson log-likelihood for the total-count term to form the full joint likelihood `P(b) · P(c|b)`.

**Known limitation:** When `Σμ = 0` and `Σn = 0`, the function returns 0 rather than a penalty. This is intentional pending the addition of a background floor — once a floor is in place, `Σμ` will never be exactly 0 and this case will not arise.

---

## 4. Barlow–Beeston log-likelihood

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
This approximates the Wald lower-bound credible interval. These substitutions are intentional and will be revisited once a proper background floor is in place (see `CLAUDE.md` → Known limitations).

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
