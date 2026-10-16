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

where `G_F = 5.396969e-5` (Fermi constant in units of N_e/N_A, so that `G_F · N_e` gives a potential in eV²/GeV when energy is in GeV).

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

## 4. Matter angle at production

`mswProb` returns, besides P(νₑ→νₑ), the cosine of twice the 1-2 mixing angle in matter at the production density:

```
β = 2√2 · G_F · cos²θ₁₃ · N_e · E / Δm²₂₁
cos 2θ₁₂ᵐ = (cos 2θ₁₂ − β) / √[(cos 2θ₁₂ − β)² + sin² 2θ₁₂]
```

This is the factor that multiplies the night-time regeneration term (Section 6).

---

## 5. Production-region integration (solar day probability)

Solar neutrinos are not produced at a single point. They are created across a range of solar radii with a production PDF (`solarModel.prodFractionBoron` or `prodFractionHep`).

**Function:** `osc_prob_day_c2m(E, mixingPars, solarModel; process, fast)` returns P_day(E) and the production-averaged ⟨cos 2θ₁₂ᵐ⟩(E).

- `fast=false` evaluates `mswProb` on the full density grid `solarModel.n_e` and averages with the production fraction: P_day(E) = Σᵢ prodFraction[i] · P(E, n_e[i]) / Σᵢ prodFraction[i]. The same weights are used for ⟨cos 2θ₁₂ᵐ⟩. This is exact for the night term, which is linear in cos 2θ₁₂ᵐ.
- `fast=true` evaluates both at the production-weighted mean density (`solarModel.avgNeBoron`, `avgNeHep`): one call instead of n_radii.

`fastFit` selects between them.

---

## 6. Day→Night correction (IYSW formula)

**Functions:** `osc_prob_both_slow`, `osc_prob_both_fast` (`osc.jl:139–207`)

The night-time probability relates to the day probability via the Ioannisian–Yu–Smirnov–Wyler (IYSW) formula (Phys.Lett. B 643, 2006):

```
P_night(path, E) = P_day(E) + ΔP(path, E)

ΔP = cos²θ₁₃ · cos(2θ₁₂_sol) · (P₁ₑ(path, E) − P₀)

where:
  θ₁₂_sol = effective mixing angle at neutrino production point (cos 2θ₁₂ᵐ from mswProb)
  P₁ₑ     = P(νₑ→ν₁) accumulated through Earth paths (computed separately)
  P₀      = cos²θ₁₂ · cos²θ₁₃  (baseline with no Earth effect)
```

`P_night` has shape `(n_paths, n_energies)` where `n_paths` = number of zenith angle bins (one Earth trajectory each).

### Slow vs. fast

The **slow** version (`osc_prob_both_slow`) integrates ΔP over the production region, applying a different `θ₁₂_sol` for each solar radius. More accurate for large MSW effects.

The **fast** version (`osc_prob_both_fast`) uses a single average density, consistent with `osc_prob_day_c2m(...; fast=true)`.

These two functions serve the legacy per-path night calculation (`nightOsc: false`). With NightOsc, `binned_oscillation_probabilities` applies the same formula to the bin-averaged P̄₁ₑ (Section 7d).

### Validity

The IYSW formula is accurate to ≲1% for solar oscillation parameters. It breaks down for Earth-core-crossing paths where matter effects are large — an acceptable approximation for DUNE's zenith angle coverage.

---

## 7. Earth propagation

`P₁ₑ(path, E)` — the probability that a ν₁ arriving at Earth is detected as νₑ — is the key input to the IYSW formula. It is computed by `setup_earth_propagation` (`propagation_osc.jl`) at every likelihood evaluation.

By default (`nightOsc: true`) it is computed by NightOsc (7d), which returns P₁ₑ already averaged over each analysis (cos z, E_true) bin. The per-path 3-flavour NumOsc calculation (7b) is the legacy alternative (`nightOsc: false`).

The choice is made once, in `build_osc_setup` (`src/propagation/osc_setup.jl`), and stored in the type of the constant `OSC::OscSetup{Night, EarthUnc, FastSolar}`. `setup_earth_propagation` and `compute_oscillation_probabilities` dispatch on `OSC.night` (`NightOscNight` or `NumOscNight{Fast}`), so the likelihood stays type-stable. The builder also checks the configuration against the rest of the analysis, and fails at setup if any check fails:
- the engine grid matches the analysis cos z and E_true bins, with two `E_calc` points per bin;
- the engine's per-bin exposure matches `exposure_weights` up to one common factor;
- the number of Earth layers matches the Earth-uncertainty prior;
- makePaths and NightOsc cross the same layers in the same order (`check_layer_agreement`), so `earth_norm_i` scales the same layer in both calculations.

### 7d. NightOsc (default): bin-averaged P₁ₑ

**Files:** module `NightOsc` in `src/oscillations/osc_night.jl`, which holds the accuracy settings and series switches, plus `src/oscillations/night/`:
- `earth_model.jl`: Earth model and chords;
- `kernels.jl`: step propagator, `p1e_paths`, Filon quadrature and its adjoint;
- `binning.jl`: `BinnedP1e` setup and evaluation;
- `ad.jl`: ForwardDiff and Enzyme rules.

It is a port of SnuPpy's `core/osc/binned_night.py` and its kernels. The engine (`OSC.night.engine`) is built once at setup from `COARSE_COSZ_EDGES`, the E_true bin edges, `exposure_intp` and the Earth model file.

**Why.** The legacy calculation samples one path per cos z bin (or two) and two energies per E_true bin. On long paths the regeneration oscillates many times within one bin, so two samples alias it. Against a dense reference on the HD CC grid (20 cos z × 30 E_true bins, 1–21 MeV), the legacy P₁ₑ is off by up to 0.04–0.06 in single bins (median 2×10⁻³), which is comparable to the regeneration amplitude itself. NightOsc is within 1.5×10⁻⁴ at nominal density and 6×10⁻⁴ with scaled layers (median ~10⁻⁵).

**Earth model.** Any radial table (r/R⊕, density, Y_e). Layer boundaries are the density jumps larger than `LAYER_JUMP_FACTOR` (5) × the median step, as in `makePaths.jl`, so layer indices (1 = outermost) match `earth_norm_i`. This is checked at setup against makePaths. On `EARTH_MODEL_PREM_DENSE.dat` the two agree on every layer sequence, with crossing lengths within 1.2 km. On the sparse `EARTH_MODEL_PREM.dat`, makePaths fails to assign layers and NightOsc finds an extra boundary at the surface. Chords are split at exact line–sphere intersections and into steps of at most 300 km. Each step carries its own Gauss–Legendre average potential and linear slope; there is no shared per-layer potential.

**Propagation.** Two-flavour reduction: P₁ₑ = c₁₃²·P₂ν with the potential scaled by c₁₃². Earth effects on ν₃ are suppressed by 2EV/Δm²₃₁ ~ 10⁻³, and θ₂₃, δCP and Δm²₃₁ drop out. Each step is propagated exactly at its average potential. Its linear slope is added to first order in the interaction picture and exponentiated, so each step stays unitary.

**Bin averages.**
- cos z is split at the layer tangencies. Within each interval, ξ = √((r_top² − r_min²)/(r_top² − r_bot²)) removes the √ cusp at the tangency. Paths sit at 8 + 3k Gauss–Legendre nodes in ξ, where k is the number of boundaries crossed.
- Energy nodes are uniform: 2 per MeV for k ≤ 1, 3 per MeV for k = 2–4.
- P₁ₑ is split into a smooth part plus one term per step boundary, C_b e^{−iψ_b}: the first-order adiabatic amplitude created at that boundary, times its matter phase to the detector. Only the slowly varying C_b, ψ_b and the smooth part are interpolated. Each term is integrated over the energy bin and the cos z bin in closed form (Filon quadrature), weighted by the exposure per unit cos z.

The result is the exposure-weighted cos z average of the flat E_true average in each bin, shape `(n_cosz, n_Etrue)`, at analysis resolution. `binned_oscillation_probabilities` then forms

```
P_night[k, j] = P̄_day[j] + c₁₃² · ⟨cos2θ₁₂ᵐ⟩[j] · (P̄₁ₑ[k, j] − c₁₂²c₁₃²)
```

where P̄_day and ⟨cos2θ₁₂ᵐ⟩ are averaged over the two `E_calc` points of each bin (they vary slowly).

**Factorisation error.** The exact bin average of P_night is P̄_day + c₁₃²·⟨cos2θ₁₂ᵐ·(P₁ₑ − c₁₂²c₁₃²)⟩. The formula above replaces ⟨cos2θ₁₂ᵐ·P₁ₑ⟩ by ⟨cos2θ₁₂ᵐ⟩·P̄₁ₑ, dropping the within-bin covariance of the two. It also takes P̄_day and ⟨cos2θ₁₂ᵐ⟩ from two points rather than the full bin. Measured on the HD grid (0.5 MeV bins, 1–21 MeV):
- Δm²₂₁ = 6×10⁻⁵: worst bin 9×10⁻⁵, against a day-night difference up to 6×10⁻²;
- Δm²₂₁ = 2×10⁻⁵ (prior edge): 2.7×10⁻⁴;
- Δm²₂₁ = 1.5×10⁻⁴: 2.6×10⁻⁵.

The largest errors are at low energy, where cos2θ₁₂ᵐ changes fastest. All are below NightOsc's own bin-averaging accuracy. `tests/test_osc_night.jl` bounds the error at 5×10⁻⁴. `fastFit` now only selects the solar production average (mean density or full integration, `osc_prob_day_c2m`). The Earth part is the same in both modes.

**Earth-density uncertainty.** `earth_norm_i` multiplies the potential and slope of every step in layer i. This is exact, with no expansion, so the 12% prior is fine.

**Accuracy.** Validated on the fitter grids against dense sampling (64–128 cos z midpoints per bin, energies resolving 0.3 rad), for Δm²₂₁ in [2×10⁻⁵, 3×10⁻⁴], sin²θ₁₂ in [0.25, 0.45], and layer densities scaled by up to ±24%:
- worst bin ≤ 9×10⁻⁴ over 0.1–25 MeV;
- ≤ 4.4×10⁻⁴ above 3 MeV.

SnuPpy's zenith rule (6 + 2k) reaches 3×10⁻³ in mantle bins once layer densities are scaled, hence 8 + 3k here.

**Propagator against 3 flavours.** `p1e_paths` at 300 km linear-slope steps agrees with the full 3-flavour `Osc.NumOsc.Slow` solver at 10 km constant steps to ≤ 5×10⁻⁴, for 3–15 MeV, Δm²₂₁ in [2×10⁻⁵, 1.5×10⁻⁴] and cos z down to −0.99. At equal 10 km steps the difference, which is the two-flavour reduction alone, is ≤ 2×10⁻⁴. δCP does not change it.

**Tests.** `tests/test_osc_night.jl` checks:
- bin averages against dense sampling;
- `p1e_paths` against NumOsc;
- layer agreement with makePaths, and the exposure check;
- the factorisation error;
- the gradient against finite differences;
- the cached Jacobian rule against plain duals;
- the Enzyme rule against ForwardDiff;
- the unsafe Float64-word fast paths against the generic loops;
- that per-task stores are released with their engine.

`tests/regression_likelihood.jl` (run with `JULIA_REGRESSION=1`) compares the full likelihood, for NightOsc and the legacy 3-flavour path, with stored reference values. With `--grad`, it also compares the Enzyme gradient with ForwardDiff.

**Cost and gradients.** Setup ~3 s. Per call, on the HD CC grid, single thread:
- value ~14 ms;
- value + Jacobian ~150 ms with Earth uncertainty (Δm²₂₁, θ₁₂, θ₁₃ and the 5 layers the grid crosses);
- value + Jacobian ~70 ms without Earth uncertainty.

`NightOsc.binned_p1e` evaluates this Jacobian once per parameter point. It caches the result per task, because BAT runs each chain in its own task, and applies the chain rule to the incoming duals. The cost therefore does not grow with the number of sampled parameters or ForwardDiff chunk passes. A full 16-parameter likelihood gradient costs ~165–180 ms, against ~31 ms for the legacy calculation. Working arrays are preallocated per task, so a call allocates < 0.2 MB. Workspaces and the Jacobian cache live in a per-task `WeakKeyDict` keyed by the engine, so an engine that is no longer used is freed together with its arrays.

### 7b. NumOsc (legacy 3-flavour per-path calculation)

(Section 7a, an analytical-eigenvalue Barger backend, was removed in October 2026. It no longer ran, and nothing called it. The nuFast C++ backend, formerly 7c, was removed at the same time.)

**Modules:** `Osc.NumOsc.Slow` and `Osc.NumOsc.Fast` (`osc.jl`)

Builds the full 3×3 matter Hamiltonian and diagonalises it numerically via Julia's `eigen()`.

**Oscillation kernel** (`osc_kernel`):

```
P(l) = U · diag(exp(i · const · (l/E) · λᵢ)) · U†
```

Products along the path are accumulated segment by segment, one constant potential per layer crossing. The final amplitude matrix is squared element-wise to get oscillation probabilities.

**NumOsc.Fast** pre-allocates a reusable lookup buffer indexed by density layer index, and only extracts the [1,1] amplitude (νₑ→ν₁) rather than building the full 3×3 output. It is selected with `nightOsc: false, fastFit: true`. `NumOsc.Slow` uses each segment's own average density and ignores `earth_norm_i`.

NumOsc is also the independent reference for NightOsc's propagator in `tests/test_osc_night.jl`.

### 4-state propagation (not wired into the fit)

`Osc.nu4NumOsc` propagates four states: the three active flavours plus one sterile. It has its own `oscPars`, with Δm²₄₁ and θ₁₄, θ₂₄, θ₃₄. Neutral-current potentials are included, with n_n = 0.7 n_e. Day-time propagation uses a decoherent (averaged) kernel. It is kept as the starting point for light-sterile or heavy-neutral-lepton studies. Large Δm²₄₁ gives the averaged, non-unitary limit. HNL production thresholds and decays are not modelled.

### Config flags summary

| `nightOsc` | `fastFit` | Active path |
|------------|-----------|-------------|
| `true` (default) | `false` | NightOsc Earth crossing; solar production region integrated |
| `true` (default) | `true` | NightOsc Earth crossing; solar mean production density |
| `false` | `false` | Legacy: NumOsc.Slow — integrate over production region, full Earth diagonalisation (ignores `earth_norm_i`) |
| `false` | `true` | Legacy: NumOsc.Fast — average solar density, pre-allocated Earth lookup |

---

## 8. ForwardDiff gradient flow

HMC requires the gradient of the log-likelihood with respect to all sampled parameters. This is provided by ForwardDiff automatic differentiation.

**What carries Dual numbers:**

- All fields of `oscPars{T}` when constructed from sampled Dual parameters
- `get_PMNS` → `SMatrix{3,3,Complex{T}}` entries carry Dual parts
- `mswProb` → `probs::T`, `c2th12m::T`
- `osc_prob_day_c2m` / `osc_prob_both_fast` → arrays of `T`
- The full event rate pipeline (`compute_oscillated_samples`, `apply_*_response`) — all multiplications preserve `T`
- The likelihood formulas (`poissonLogLikelihood`, etc.) — scalar `T` accumulation

**What does NOT differentiate:**

- `earth_lookup` — pre-tabulated density values, treated as constants
- NightOsc's chords, nodes and interpolation weights — built once in Float64. Only Δm²₂₁, θ₁₂, θ₁₃ and `earth_norm_i` enter its per-call kernel, which is generic in the number type
- `solarModel` — solar structure model, fixed
- Response matrices — built once from MC, fixed
- Bin edges, exposure weights — detector geometry constants

ForwardDiff operates in forward mode over all N sampled parameters simultaneously (N ≈ 5–30 depending on config), so its cost grows with N.

**Reverse mode (`ad_backend: "Enzyme"`, the default with `nightOsc: true`).** The whole likelihood is differentiable by Enzyme in reverse mode, whose cost does not depend on N. This is the backend to use with many Earth-density parameters. On the HD CC config, Enzyme agrees with ForwardDiff to ~1e-13. Timings on one thread (October 2026): one likelihood evaluation 19 ms, of which NightOsc is 99.8 %; Enzyme gradient ~85 ms; ForwardDiff gradient ~200 ms, which grows with the number of NightOsc inputs. The first gradient call compiles for 2–3 minutes per process.

What makes it work (keep these properties when editing the likelihood path):
- Settings read by the likelihood are `const` (`E_calc`, `exposure_weights`, `N_COSZ_BINS`, `CC_XSEC_NORM_TRUE`, and `OSC`, whose type carries the night backend, the Earth-uncertainty switch and `fastFit`), and `LikelihoodInputs` has concrete field types.
- Detector modes are type parameters (`DetFlags`), so `det_flags.ES_mode` etc. are compile-time constants and the unused channel branches disappear.
- No logging macros (they contain `try`/`catch`), dynamic splats (`vcat(x...)`), `Adjoint` wrappers of constant matrices, or generators/zips mixing constant data with differentiable values inside the likelihood: those are written as loops or moved into `@noinline` helpers.
- NightOsc has a custom Enzyme rule (`_p1e_vec`). Its forward pass (`_p1e_forward_store!`) computes the primal while keeping each path's stage-1 output in per-task workspaces, and records the point (`:forward_x` in the task store). The reverse pass, `_p1e_vjp_manual`, computes wᵀ∂P/∂x:
  - with Earth-layer normalisations (the usual case) it reuses that forward state when the record names the same point (without them the forward pass records nothing and the reverse pass always recomputes); otherwise (another NightOsc call came in between, or a different point) it recomputes the forward pass and re-records. Any plain primal call clears the record, so state is never reused at the wrong point, also with several NightOsc calls per evaluation;
  - it applies hand-written adjoints of stages 3 (zenith integrals) and 2 (energy integrals); the Filon integral is linear in its amplitudes, so its adjoint is the phase-dependent weights plus three phase derivatives, `_filon_adj`;
  - it runs one small Enzyme reverse pass per (path, energy node) for stage 1 (`_node!`).

  Measured on the HD CC grid: ~60–65 ms with the forward state reused (~75 ms recomputing), about 3–4× one evaluation, allocating 0.1 MB. About half of it is the nested per-node Enzyme passes. The earlier whole-kernel nested Enzyme pass took ~190 ms and 50 MB. The cost does not grow with the number of Earth-density parameters. The nested Enzyme calls go through `invokelatest`, which avoids an Enzyme 0.13 nested-compilation bug. `tests/test_osc_night.jl` checks the rule against ForwardDiff.
- Do not use Enzyme's runtime-activity mode: on this code it wrote derivative values into the observed (Asimov) data.

---

## 9. Call graph (one likelihood evaluation)

```
likelihood(params)                         [likelihood_main.jl]
  │
  ├─ get_mixing_parameters(params)          sin²θ → θ; constructs oscPars{T}
  │
  ├─ setup_earth_propagation(E_calc, ...)   [propagation_osc.jl]
  │    ├─ [nightOsc] binned_p1e(OSC.night.engine, Δm², θ₁₂, θ₁₃, earth_norms)
  │    │                                    P̄₁ₑ matrix (n_cosz × n_Etrue), bin-averaged
  │    └─ [legacy] osc_prob_earth(E_calc, ...)  P₁ₑ matrix  (n_paths × n_E_fine)
  │         └─ matter_osc_per_e(...)        per-energy kernel products
  │
  ├─ normalize_backgrounds(raw_bgs, params) apply nuisance scale factors
  │
  ├─ compute_oscillation_probabilities(...)  [propagation_osc.jl]
  │    ├─ [nightOsc]     binned_oscillation_probabilities(E_calc, mixingPars, solarModel, P̄₁ₑ)
  │    │    ├─ osc_prob_day_c2m(E_calc, ...)  P_day, ⟨cos2θ₁₂ᵐ⟩, block-averaged to E_true bins
  │    │    └─ IYSW correction               P_night (n_cosz × n_Etrue)
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
| Constant density per Earth segment | Legacy Earth propagation | ~1–2% per segment; PREM profile used |
| Two-flavour Earth crossing, linear potential per ≤300 km step | NightOsc | ≤ 5×10⁻⁴ vs 3-flavour NumOsc at 10 km steps (two-flavour reduction alone ≤ 2×10⁻⁴) |
| Interpolated bin averages | NightOsc | ≤ 9×10⁻⁴ per bin (≤ 4.4×10⁻⁴ above 3 MeV) vs dense sampling |
| ⟨cos2θ₁₂ᵐ⟩ and P_day averaged separately from P̄₁ₑ within an E_true bin | `binned_oscillation_probabilities` | ≤ 9×10⁻⁵ per bin at Δm²₂₁ = 6×10⁻⁵, ≤ 2.7×10⁻⁴ at 2×10⁻⁵ (§7d) |
| θ₁₃ perturbative in matter | `mswProb` | Good for solar νₑ: sin²θ₁₃ ≈ 0.022 |
| Production-weighted average density (fast) | `osc_prob_day_c2m(...; fast=true)` | Introduces ~1% error vs. full integration |
| Unitarity: P(ν_other) = 1 − P(νₑ) | `propagation_osc.jl:140–144` | Exact in 3-flavor; breaks for sterile mixing |
| 3-flavor only (main code) | All of osc.jl | 4-flavor extension exists in `nu4NumOsc` module |
