# Oscillation Physics

This document describes how neutrino oscillation probabilities are computed, from parameter input to final probability arrays. The implementation lives in `src/oscillations/osc.jl` and is driven by `src/propagation/propagation_osc.jl`.

---

## 1. Oscillation parameters

**Struct:** `oscPars{T<:Real}` (`osc.jl:33–47`)

```julia
struct oscPars{T<:Real}
    Δm²₂₁ :: T    # solar mass splitting (eV²)
    θ₁₂   :: T    # solar mixing angle (radians)
    θ₁₃   :: T    # reactor mixing angle (radians)
    Δm²₃₁ :: T    # atmospheric splitting (eV², default 2.5e-3)
    m₀    :: T    # lightest mass (eV, default 1e-9)
    θ₂₃   :: T    # atmospheric angle (radians, default π/4)
    δCP   :: T    # CP phase (radians, default -1.611)
end
```

The type parameter `T<:Real` is what enables automatic differentiation: when the HMC sampler calls the likelihood with `ForwardDiff.Dual` numbers, every downstream computation inherits that type and the gradient flows all the way back to the oscillation parameters. For plain `Float64` (LLH scan mode) Julia specialises to identical machine code.

The constructor (`osc.jl:44–47`) calls `promote_type` over all three required inputs so mixing a `Float64` with a `Dual` promotes both to `Dual`.

**Conversion from config parameters** (`propagation_osc.jl:30–36`, `get_mixing_parameters`):

Config stores `sin²θ`, not `θ`. Conversion:

```
θ₁₂ = asin(√(clamp(sin²θ₁₂, 0, 1)))
θ₁₃ = asin(√(clamp(sin²θ₁₃, 0, 1)))
```

Clamping prevents NaN from AD perturbations that push `sin²θ` slightly outside [0,1].

---

## 2. PMNS mixing matrix

**Function:** `get_PMNS(params)` (`osc.jl:57–64`)

The PMNS matrix is factored as U = U₁ × U₂ × U₃:

| Factor | Rotation plane | Matrix |
|--------|---------------|--------|
| U₁ | 2–3 | θ₂₃ (atmospheric) |
| U₂ | 1–3 | θ₁₃, δCP (reactor + CP) |
| U₃ | 1–2 | θ₁₂ (solar) |

Implemented as `SMatrix{3,3}` (StaticArrays) so the 3×3 algebra is stack-allocated. The `Complex{T}` entries carry Dual number parts when differentiation is active.

---

## 3. Solar day probability (MSW effect)

**Function:** `mswProb(energy, mixingPars, n_e)` (`osc.jl:77–90`)

Computes P(νₑ→νₑ) for neutrinos propagating adiabatically through solar matter. Reference: Wolfenstein (1978), Mikheyev & Smirnov (1985); θ₁₃ correction from Barger et al.

### Formula

The MSW matter potential for νₑ forward scattering:

```
A_cc = 2√2 · G_F · N_e · E · cos²θ₁₃
```

where `G_F = 5.4489e-5` (Fermi constant in units of N_e/N_A, so that `G_F · N_e` gives a potential in eV²/GeV when energy is in GeV).

The ratio of matter to vacuum potential:

```
β = A_cc / Δm²₂₁
```

β = 1 at the MSW resonance (maximal mixing in matter). For solar 8B neutrinos at typical production densities, β ≫ 1 (strongly suppressed νₑ at production, adiabatic level-crossing).

Modified cosine of twice the solar mixing angle in matter:

```
cos(2θ₁₂ᵐ) = (cos(2θ₁₂) − β) / √[(cos(2θ₁₂) − β)² + sin²(2θ₁₂)]
```

Survival probability combining the adiabatic 1-2 system with the θ₁₃ correction:

```
P(νₑ→νₑ) = ½ · (1 − sin²θ₁₃)² · (1 + cos(2θ₁₂) · cos(2θ₁₂ᵐ)) + sin⁴θ₁₃
```

### Key variables in code

| Variable | Code name | Meaning |
|----------|-----------|---------|
| A_cc | `Acc` | Matter potential (eV²/GeV × GeV = eV²) |
| β | `beta` | Matter/vacuum ratio |
| cos(2θ₁₂ᵐ) | `c2th12m` | Modified mixing in matter (returned for use in night calculation) |
| sin²θ₁₃ | `s2th13` | Reactor mixing probability |
| P(νₑ→νₑ) | `probs` | Survival probability |

**Output:** `(probs, c2th12m)` — both are returned because `c2th12m` is reused in the night correction.

### Assumptions

- Adiabatic propagation: density changes on scales ≫ oscillation length. Valid throughout the Sun.
- θ₁₃ enters only as a perturbative correction to the 1-2 system.
- The 1-3 and 2-3 subsystems decouple at solar densities.

---

## 4. LMA matter angle

**Function:** `LMA_angle(energy, mixingPars, N_e)` (`osc.jl:96–101`)

Computes the effective 1-2 mixing angle in matter at a single production density:

```
β = 2√2 · G_F · cos²θ₁₃ · N_e · E / Δm²₂₁
θ₁₂ᵐ = ½ · arccos[(cos(2θ₁₂) − β) / √[(cos(2θ₁₂) − β)² + sin²(2θ₁₂)]]
```

The `arccos` argument is clamped to [−1, 1] to handle floating-point rounding at extremes. Used in the night-time correction formula (Section 6) when computing the solar mixing angle at a specific production radius.

---

## 5. Production-region integration (solar day probability)

Solar neutrinos are not produced at a single point. They are created across a range of solar radii with a production PDF (`solarModel.prodFractionBoron` or `prodFractionHep`).

### Slow integration: `osc_prob_day` (`osc.jl:104–120`)

Evaluates `mswProb` at every density point in `solarModel.n_e` (a grid over solar radii), then averages weighted by the production fraction:

```
P_day(E) = Σᵢ prodFraction[i] · P(E, n_e[i]) / Σᵢ prodFraction[i]
```

The broadcasting `mswProb.(E, Ref(mixingPars), solarModel.n_e')` evaluates at all (energy × density) pairs simultaneously.

### Fast average: `osc_prob_day_fast` (`osc.jl:123–132`)

Uses pre-computed production-weighted average electron densities:
- `solarModel.avgNeBoron` for 8B
- `solarModel.avgNeHep` for HEP

Single call to `mswProb(energy, params, n_e)` with the scalar average density. O(1) vs. O(n_radii). Selected when `fastFit: true` in config.

**Active path:** `fast=true` → `osc_prob_day_fast`; `fast=false` → `osc_prob_day`

---

## 6. Day→Night correction (IYSW formula)

**Functions:** `osc_prob_both_slow`, `osc_prob_both_fast` (`osc.jl:139–207`)

The night-time probability relates to the day probability via the Ioannisian–Yu–Smirnov–Wyler (IYSW) formula (Phys.Lett. B 643, 2006):

```
P_night(path, E) = P_day(E) + ΔP(path, E)

ΔP = cos²θ₁₃ · cos(2θ₁₂_sol) · (P₁ₑ(path, E) − P₀)

where:
  θ₁₂_sol = effective mixing angle at neutrino production point (from LMA_angle)
  P₁ₑ     = P(νₑ→ν₁) accumulated through Earth paths (computed separately)
  P₀      = cos²θ₁₂ · cos²θ₁₃  (baseline with no Earth effect)
```

`P_night` has shape `(n_paths, n_energies)` where `n_paths` = number of zenith angle bins (one Earth trajectory each).

### Slow vs. fast

The **slow** version (`osc_prob_both_slow`) integrates ΔP over the production region, applying a different `θ₁₂_sol` for each solar radius. More accurate for large MSW effects.

The **fast** version (`osc_prob_both_fast`) uses a single average density, consistent with `osc_prob_day_fast`.

### Validity

The IYSW formula is accurate to ≲1% for solar oscillation parameters. It breaks down for Earth-core-crossing paths where matter effects are large — an acceptable approximation for DUNE's zenith angle coverage.

---

## 7. Earth propagation

`P₁ₑ(path, E)` — the probability that a νₑ arriving at Earth is in mass eigenstate ν₁ — is the key input to the IYSW formula. It is computed by `setup_earth_propagation` (`propagation_osc.jl:52–74`) and cached before the inner MCMC loop.

Three backends exist, selected by config flags:

### 7a. BargerOsc (analytical eigenvalues)

**Modules:** `BargerOsc.Slow` and `BargerOsc.Fast` (`osc.jl:223–350`)

Computes eigenvalues of the matter Hamiltonian analytically using the cubic formula, then builds Lagrange projection matrices.

**Matter Hamiltonian:**
```
H_matter = H_vac + V,   V[1,1] = ±√2 · G_F · N_e · E · cos²θ₁₃
```
(+ for neutrinos, − for antineutrinos; all diagonal elements shifted to preserve tracelessness)

**Eigenvalue computation** (`get_eigen`, `osc.jl:232–277`):

The Barger–Liu–Marfatia cubic formula gives three eigenvalues λ₁, λ₂, λ₃ via a trigonometric substitution:

```
a = α/3,   brac = √(α² − 3β)/3
θ⁰ = arccos[(2α³ − 9αβ + 27γ) / (2·brac³)]

λ₁ = −2·brac·cos(θ⁰/3)      + (m²₁ − α/3)
λ₂ = −2·brac·cos((θ⁰+2π)/3) + (m²₁ − α/3)
λ₃ = −2·brac·cos((θ⁰+4π)/3) + (m²₁ − α/3)
```

where α, β, γ are functions of vacuum mass splittings and the matter potential. The arccos argument is clamped to [−1+ε, 1−ε] (`osc.jl:251`).

**Projection matrices** (Lagrange formula):

```
Pᵢ = ∏_{j≠i} (H − λⱼ·I) / ∏_{j≠i} (λᵢ − λⱼ)
```

**Oscillation along a segment of length l:**

```
U_segment = Σᵢ Pᵢ · exp(i · 2.534 · (l/E) · λᵢ)
```

where the constant 2.534 converts units (km, GeV, eV²) to a dimensionless phase.

**Fast variant** pre-computes `get_H()` for every unique density in the lookup table once per energy, then indexes into the pre-computed matrices per path segment rather than re-diagonalising.

### 7b. NumOsc (numerical diagonalisation)

**Modules:** `NumOsc.Slow` and `NumOsc.Fast` (`osc.jl:357–480`)

Builds the full 3×3 matter Hamiltonian and diagonalises it numerically via Julia's `eigen()`.

**Oscillation kernel** (`osc_kernel`, `osc.jl:360–364`):

```
P(l) = U · diag(exp(i · const · (l/E) · λᵢ)) · U†
```

Products along the path are accumulated segment by segment. The final amplitude matrix is squared element-wise to get oscillation probabilities.

**NumOsc.Fast** pre-allocates a reusable lookup buffer indexed by density layer index, and only extracts the [1,1] amplitude (νₑ→ν₁) rather than building the full 3×3 output. This is the hot path when `fastFit: true`.

**Selecting between BargerOsc and NumOsc:** Currently the code dispatches based on which `osc_prob_earth` is in scope at include time (determined by which submodule is loaded). In practice, `NumOsc.Fast` is the default for `fast=true`.

### 7c. nuFast (external C++ library)

A compiled C++ library (`src/oscillations/libnufast_earth.so`) from the NuFast-Earth project. Provides the same `osc_prob_both_fast` interface. Selected when `nuFast: true` in config. Incompatible with Julia multithreading.

### Config flags summary

| `nuFast` | `fastFit` | Active path |
|----------|-----------|-------------|
| `false` | `false` | NumOsc.Slow — integrate over production region, full Earth diagonalisation |
| `false` | `true` | NumOsc.Fast — average solar density, pre-allocated Earth lookup |
| `true` | (ignored) | External nuFast C++ library |

---

## 8. ForwardDiff gradient flow

HMC requires the gradient of the log-likelihood with respect to all sampled parameters. This is provided by ForwardDiff automatic differentiation.

**What carries Dual numbers:**

- All fields of `oscPars{T}` when constructed from sampled Dual parameters
- `get_PMNS` → `SMatrix{3,3,Complex{T}}` entries carry Dual parts
- `mswProb` → `probs::T`, `c2th12m::T`
- `osc_prob_day_fast` / `osc_prob_both_fast` → arrays of `T`
- The full event rate pipeline (`compute_oscillated_samples`, `apply_*_response`) — all multiplications preserve `T`
- The likelihood formulas (`poissonLogLikelihood`, etc.) — scalar `T` accumulation

**What does NOT differentiate:**

- `earth_lookup` — pre-tabulated density values, treated as constants
- `solarModel` — solar structure model, fixed
- Response matrices — built once from MC, fixed
- Bin edges, exposure weights — detector geometry constants

ForwardDiff operates in forward mode over all N sampled parameters simultaneously (N ≈ 5–30 depending on config), computing the full gradient in a single likelihood evaluation pass.

---

## 9. Call graph (one likelihood evaluation)

```
likelihood(params)                         [likelihood_main.jl]
  │
  ├─ get_mixing_parameters(params)          sin²θ → θ; constructs oscPars{T}
  │
  ├─ setup_earth_propagation(E_calc, ...)   [propagation_osc.jl]
  │    └─ osc_prob_earth(E_calc, ...)       P₁ₑ matrix  (n_paths × n_E_fine)
  │         └─ matter_osc_per_e(...)        per-energy kernel products
  │
  ├─ normalize_backgrounds(raw_bgs, params) apply nuisance scale factors
  │
  ├─ compute_oscillation_probabilities(...)  [propagation_osc.jl]
  │    ├─ [nuFast path]  osc_prob_both_fast(E_calc, mixingPars, lookup, ...)
  │    ├─ [fast path]    osc_prob_both_fast(E_calc, P₁ₑ, mixingPars, solarModel)
  │    │    ├─ mswProb(E, params, avgNe)     P_day scalar
  │    │    └─ IYSW correction               P_night matrix
  │    └─ [slow path]    osc_prob_both_slow(...)
  │         ├─ mswProb.(E, params, n_e_grid) P_day at each solar radius
  │         └─ IYSW integrated over production region
  │
  ├─ compute_oscillated_samples(...)         multiply Etrue histograms × oscProbs
  │
  ├─ compute_ES_event_rates(...)             fold through response matrix + efficiency
  ├─ compute_CC_event_rates(...)
  │
  └─ poissonLogLikelihood(expected, observed)
```

---

## 10. Key variable reference

| Variable | Shape | Units | Meaning |
|----------|-------|-------|---------|
| `beta` (`β`) | scalar/vector | dimensionless | Matter/vacuum potential ratio |
| `Acc` | scalar/vector | eV² | MSW charged-current potential |
| `c2th12m` | scalar/vector | dimensionless | cos(2θ₁₂) in solar matter |
| `solarAngle` | scalar/vector | radians | Effective θ₁₂ at production point |
| `P₀` | scalar | [0,1] | Baseline probability (no Earth effect) = cos²θ₁₂·cos²θ₁₃ |
| `matrix_p_1e` / `oscProbs_1e` | (n_paths, n_E_fine) | [0,1] | P(νₑ→ν₁) through Earth |
| `prob_day` | (n_E,) | [0,1] | P(νₑ→νₑ) from Sun, daytime |
| `prob_night` | (n_paths, n_E) | [0,1] | P(νₑ→νₑ) day+Earth correction |
| `oscProbs.nue_8B_day` | (n_Etrue_bins,) | [0,1] | Binned day νₑ prob, 8B |
| `oscProbs.nue_8B_night` | (n_paths_bins, n_Etrue_bins) | [0,1] | Binned night νₑ prob, 8B |
| `E_calc` | (n_E_fine,) | GeV | Fine energy grid for oscillation calc |
| `earth_lookup` | (n_layers,) | g/cm³ | Average densities per Earth layer |
| `earth_paths` | vector of Path | — | Neutrino trajectories for each cosz bin |

---

## 11. Approximations and validity

| Approximation | Where | Validity |
|--------------|-------|---------|
| Adiabatic MSW (no level jumps) | `mswProb` | Valid in Sun; breaks at sharp density discontinuities |
| IYSW connection formula | `osc_prob_both_*` | ≲1% error; breaks for core-crossing paths |
| Constant density per Earth segment | Earth propagation | ~1–2% per segment; PREM profile used |
| θ₁₃ perturbative in matter | `mswProb` | Good for solar νₑ: sin²θ₁₃ ≈ 0.022 |
| Production-weighted average density (fast) | `osc_prob_day_fast` | Introduces ~1% error vs. full integration |
| Unitarity: P(ν_other) = 1 − P(νₑ) | `propagation_osc.jl:140–144` | Exact in 3-flavor; breaks for sterile mixing |
| 3-flavor only (main code) | All of osc.jl | 4-flavor extension exists in `nu4NumOsc` module |
