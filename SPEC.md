# SPEC — validation criteria

What the fitter must satisfy, stated so that each item can be tested. Derivations and formulas are in
[docs/](docs/index.md); this file only lists checks and tolerances, with a link to the source of each.
Item numbers are stable: code, tests and docs refer to them (e.g. "SPEC 3.3").

## Status legend

| Mark | Tag | Meaning |
|:---:|---|---|
| 🟢 | **[tested]** | Covered by a test (file named). |
| 🟡 | **[untested]** | No test yet; a candidate for one. |
| 🔴 | **[known to fail]** | A current bug is open against it (see the 2026-10-15 audit). |
| 🟣 | **[inferred]** | Drafted from the code or docs and not yet confirmed by the physicist. Confirm or correct. |
| ⚠️ | note | Code, inputs or docs currently disagree with the item; see the note under it. |

Two marks on one item (e.g. 🟢🟡) mean part of it is tested and part is not; the text says which.

## Overview

| Section | 🟢 | 🟡 | 🟣 | 🔴 |
|---|:---:|:---:|:---:|:---:|
| [1. Oscillation probabilities](#1-oscillation-probabilities) | 3 | 3 | 2 | – |
| [2. Earth model](#2-earth-model) | 6 | 2 | – | – |
| [3. Cross sections](#3-cross-sections) | 5 | – | – | – |
| [4. Detector response and event rates](#4-detector-response-and-event-rates) | 9 | 1 | 2 | – |
| [5. Likelihood](#5-likelihood) | 7 | 3 | 1 | – |
| [6. Gradients](#6-gradients) | 3 | – | – | – |
| [7. Sampling and I/O](#7-sampling-and-io) | 2 | 3 | 1 | – |
| [8. Model inputs](#8-model-inputs) | 2 | – | – | – |

Partly tested items count in both columns.

<details>
<summary><b>All items at a glance</b> (click to expand)</summary>

| Item | Status | Requirement |
|---|:---:|---|
| [1.1](#11) | 🟡 | Probabilities in [0, 1], unitarity |
| [1.2](#12) | 🟡🟣 | Vacuum limit |
| [1.3](#13) | 🟢 | NightOsc bin averages vs dense sampling, < 1e-3 |
| [1.4](#14) | 🟢 | NightOsc vs 3-flavour NumOsc, < 1e-3 |
| [1.5](#15) | 🟢 | Day-night factorisation error, < 5e-4 |
| [1.6](#16) | 🟡 | Convergence in zenith steps and E_calc |
| [1.7](#17) | 🟣 | Legacy vs NightOsc, ≤ 0.06 per bin |
| [2.1](#21) | 🟡⚠️ | Nominal potential reproduces the Earth table |
| [2.2](#22) | 🟢 | 6 sorted layers; segment lengths sum to the chord |
| [2.3](#23) | 🟢 | makePaths and NightOsc cross the same layers |
| [2.4](#24) | 🟢 | Uncrossed layer has zero gradient |
| [2.5](#25) | 🟢 | `earth_norm_i` outside [0, 2] gives −Inf |
| [2.6](#26) | 🟢🟡 | Earth-norm priors use the full covariance |
| [2.7](#27) | 🟢 | Parameter names sort numerically |
| [3.1](#31) | 🟢 | ES couplings (BKS95, PDG 2024), 5e-4 |
| [3.2](#32) | 🟢 | ES σ = 0 below threshold |
| [3.2a](#32a) | 🟢 | ES recoil quadrature vs analytic, 1e-12 |
| [3.3](#33) | 🟢 | CC shape nuisances: orthogonal basis |
| [3.4](#34) | 🟢 | CC nominal table (MARLEY 2.0.0), never extrapolated |
| [4.1](#41) | 🟢 | Response rows sum to 1; threshold binning |
| [4.2](#42) | 🟢 | One norm per ES background |
| [4.3](#43) | 🟢 | Fixed background = free norm at nominal |
| [4.3a](#43a) | 🟢 | Background MC weights in selected and total counts |
| [4.3b](#43b) | 🟢 | Normalisation conventions are config keys |
| [4.3c](#43c) | 🟢 | Semi-inclusive: each ES event in one channel |
| [4.3d](#43d) | 🟣 | Signal-sample weights (meaning undefined) |
| [4.3e](#43e) | 🟢 | Target counts per kt |
| [4.4](#44) | 🟢 | `expected_rates` shapes |
| [4.4a](#44a) | 🟢 | Input samples checked at setup |
| [4.5](#45) | 🟡🟣⚠️ | Event-level inputs; invariance to order and re-binning |
| [5.1](#51) | 🟢 | Poisson log L |
| [5.2](#52) | 🟢 | Barlow–Beeston → Poisson |
| [5.3](#53) | 🟢🟡 | Total = ES + CC; LLH scan |
| [5.4](#54) | 🟡 | Asimov: maximum at the truth |
| [5.5](#55) | 🟢 | Full-likelihood regression, rtol 1e-9 |
| [5.5a](#55a) | 🟢🟣 | MC statistical uncertainty |
| [5.6](#56) | 🟡 | Per-bin log L sums to the total minus overflow |
| [5.7](#57) | 🟢 | Derived mode, every detector |
| [5.8](#58) | 🟢 | Python loaders fail rather than fall back |
| [6.1](#61) | 🟢 | ForwardDiff vs finite differences, 1e-5 |
| [6.2](#62) | 🟢 | Cached Jacobian rule vs dual numbers, 1e-9 |
| [6.3](#63) | 🟢 | Enzyme vs ForwardDiff, ~1e-13 |
| [7.1](#71) | 🟢🟡 | Resume never loses batches |
| [7.2](#72) | 🟡 | Seeds fix the chains |
| [7.3](#73) | 🟡 | Coverage of credible intervals |
| [7.4](#74) | 🟢🟣 | Warm start |
| [8.1](#81) | 🟢 | Input-table contract |
| [8.2](#82) | 🟢 | Solar-model inputs |

</details>

---

## 1. Oscillation probabilities

> 📖 Source: [oscillation_physics.md](docs/oscillation_physics.md)

<a id="11"></a>
#### 🟡 1.1 · Unitarity and range
P(νe→νe) + P(νe→ν_other) = 1 and every probability lies in **[0, 1]**, for day and night, over the full
prior range of (sin²θ₁₂, sin²θ₁₃, Δm²₂₁) and E_true in the analysis window.
- ν_other is built as 1 − P_ee in `propagation_osc.jl`, so the real check is 0 ≤ P_ee ≤ 1.

<a id="12"></a>
#### 🟡🟣 1.2 · Vacuum limit
With zero matter potential, the night P₁ₑ equals the vacuum |U_e1|² and the day-night difference
vanishes.

<a id="13"></a>
#### 🟢 1.3 · NightOsc bin averages vs dense sampling
Agree to **< 1e-3 per bin** (measured ≤ 9e-4; ≤ 4.4e-4 above 3 MeV).
> ✅ `test_osc_night.jl` · "bin averages match dense sampling"

<a id="14"></a>
#### 🟢 1.4 · NightOsc vs the 3-flavour solver
NightOsc (two-flavour, ≤ 300 km linear-potential steps) agrees with the 3-flavour NumOsc solver to
**< 1e-3** (measured ≤ 5e-4).
> ✅ "step propagator matches the 3-flavour NumOsc solver"

<a id="15"></a>
#### 🟢 1.5 · Day-night factorisation
The error from averaging ⟨cos2θ₁₂ᵐ⟩ and P_day separately is **< 5e-4 per bin**.
> ✅ "day-night factorisation error is small"

<a id="16"></a>
#### 🟡 1.6 · Convergence
Refining the zenith steps (8+3k rule) and the E_calc points does not change P₁ₑ beyond the tolerance
in 1.3, including with layer densities scaled by ±24%.
- Untested as a test: measured during the NightOsc port; `utils/checkOscGridConvergence.jl` is the
  manual check.

<a id="17"></a>
#### 🟣 1.7 · Legacy vs NightOsc
Legacy (`nightOsc: false`) and NightOsc agree to **≤ 0.06 in any single bin** (median ~2e-3). This is a
known accuracy limit of the legacy method, not a requirement to tighten. *(Inferred bound.)*

---

## 2. Earth model

> 📖 Source: [oscillation_physics.md §7](docs/oscillation_physics.md)

<a id="21"></a>
#### 🟡⚠️ 2.1 · Nominal potential
With all `earth_norm_i = 1`, the potential reproduces PREM (`inputs/EARTH_MODEL_PREM_DENSE.dat`)
exactly.

> [!WARNING]
> The file is not PREM (input audit, 2026-10-18): it has no crust or ocean (3.38 g/cm³ up to the
> surface) and its layer boundaries are shifted by up to 15 km. The item currently means "reproduces
> the shipped table". The Earth modelling is being reworked; restate this item then.

<a id="22"></a>
#### 🟢 2.2 · Layers and chords
The model has **6 layers** with sorted boundaries; path segment lengths sum to the chord length; paths
start and end in layer 1.
> ✅ "Earth model and chords"

<a id="23"></a>
#### 🟢 2.3 · Same layers in both propagators
makePaths and NightOsc cross the same layers in the same order, so `earth_norm_i` scales the same layer
in both.
> ✅ "Earth layers agree with makePaths"; also checked at setup.

<a id="24"></a>
#### 🟢 2.4 · Uncrossed layer
A layer not crossed by any path in the grid has **exactly zero gradient**.
> ✅ (inner core)

<a id="25"></a>
#### 🟢 2.5 · Norm range
`earth_norm_i` outside **[0, 2]** gives log L = −Inf.
> ✅ `test_likelihood_correctness.jl`

<a id="26"></a>
#### 🟢🟡 2.6 · Earth-norm prior covariance
Earth-norm priors use the full covariance of `earth_normalisation_prior_file`.
- 🟢 Independent normals plus `earth_prior_correlation` (added in `mcmc.jl` when the covariance is not
  diagonal) equal the MvNormal up to a constant (`test_likelihood_correctness.jl`).
- 🟡 The current covariance file is diagonal, so the term has never run inside BAT or under Enzyme.

> [!IMPORTANT]
> Run `regression_likelihood.jl night --grad` with a correlated covariance file before using one.

<a id="27"></a>
#### 🟢 2.7 · Parameter-name order
Parameter names sort numerically (`earth_norm_2` before `earth_norm_10`).
> ✅ `test_response_and_backgrounds.jl`

---

## 3. Cross sections

> 📖 Source: [event_rate_pipeline.md](docs/event_rate_pipeline.md)

<a id="31"></a>
#### 🟢 3.1 · ES couplings
ES couplings follow BKS95 App. A: at T → 0, ν_μ e **g_V = −0.0395** and **g_A = −0.5063** (PDG 2024
Table 10.3) to **5e-4**.
> ✅ `test_es_xsec.jl`

<a id="32"></a>
#### 🟢 3.2 · ES threshold
ES cross sections are **exactly 0** (never negative) below the TPC threshold.
> ✅

<a id="32a"></a>
#### 🟢 3.2a · ES recoil quadrature
The recoil quadrature equals the analytic integral of dσ/dT for constant couplings (**1e-12**); I(T)
matches BKS95 Eq. A5; σ₀ = 2 G_F² m_e² (ħc)²/π.
> ✅ `test_es_xsec.jl`

<a id="33"></a>
#### 🟢 3.3 · CC shape nuisances (orthogonal basis)

$$\sigma(E) = \sigma_\text{nominal}(E)\cdot \text{norm}\cdot \exp\big(\text{tilt}\, p_1(L) + \text{curv}\, p_2(L)\big),\qquad L = \ln(E/E_\text{pivot})$$

applied per E_true bin to CC ⁸B and CC hep, day and night.

- **Basis.** p₁ = L − ⟨L⟩ and p₂ = L² − a·L − b are monic and orthogonal to 1 and to each other under
  the weight w_i = unoscillated CC ⁸B rate in E_true bin i (σ_nominal × ⁸B flux × exposure; not the
  oscillated or efficiency-weighted spectrum): ⟨p₁⟩ = ⟨p₂⟩ = ⟨p₁p₂⟩ = 0, with ⟨f⟩ = Σ w f / Σ w.
- **Nominal point.** At `cc_xsec_tilt = cc_xsec_curv = 0` and `cc_xsec_norm = 1` the CC cross section
  equals the nominal table (3.4) **exactly**; norm is a pure multiplicative factor.
- **Orthogonality, first order only.** For N = Σ_i w_i·shape_i, ∂ln N/∂tilt = ∂ln N/∂curv = 0 at
  tilt = curv = 0 (to **1e-10** of the natural scale ⟨p²⟩^½), and the cross term
  ∂²ln N/∂tilt∂curv = ⟨p₁p₂⟩ = 0. ∂²ln N/∂tilt² = ⟨p₁²⟩ and ∂²ln N/∂curv² = ⟨p₂²⟩ are not zero, so a
  finite tilt or curv does shift N at second order.
- **Meaning of tilt and curv.** ln(σ/σ_nominal) is linear in ln E with slope tilt when curv = 0
  (monotone tilt), and quadratic in ln E with second derivative 2·curv. In general the slope is
  tilt + curv·(2L − a), equal to tilt at L = a/2.
- **Degenerate weights.** With weight in only two E_true bins p₂ vanishes there and curv is
  unconstrained by the rate (setup warns); with weight in fewer than two bins setup errors; non-finite
  weights are an error.

> ✅ `test_cc_xsec.jl` (synthetic ⁸B-like weights) and `derived_mode.jl` (the setup weights of the
> fixture detectors, and the propagation applying the basis), via `cc_xsec_shape_checks.jl`.

<a id="34"></a>
#### 🟢 3.4 · CC nominal cross section
The ν_e–⁴⁰Ar total cross section is the table in `cc_xsec_file`, by default **MARLEY 2.0.0**
(HF-CRPA + Bhattacharya 2009, 0–30 MeV, provenance in the file header).
- Every tabulated point is used (the first included), and values agree with the MARLEY output.
- σ = 0 below the threshold (**3.7–3.9 MeV**); σ ≥ 0 and strictly increasing above it.
- Interpolation is linear; evaluating outside the table is an **error, never an extrapolation**.
  `range_Etrue` of every shipped config lies inside the table (setup checks it).
- Tables without a header row, with non-finite or negative values, or with unsorted or repeated
  energies are rejected.
- The run record pins the file read (resolved path and SHA-256), also when run from outside the
  repository.

> ✅ `test_cc_xsec.jl`

---

## 4. Detector response and event rates

> 📖 Source: [event_rate_pipeline.md](docs/event_rate_pipeline.md)

<a id="41"></a>
#### 🟢 4.1 · Response matrix and threshold binning
- Each response-matrix row (fixed E_true) **sums to 1** over all E_reco, with events reconstructed
  outside the window in the first/last (±∞) bins; they are not redistributed into the window.
  Selection efficiency is applied per E_reco bin. *(`test_response_and_backgrounds.jl`)*
- Binning is built from the analysis threshold: `nBins_Ereco` uniform fitted bins span
  [`Ereco_min`, range max], plus one underflow bin (−∞, `Ereco_min`) that is never fitted; the last
  fitted bin also holds the overflow.
- The threshold is **exactly the lower edge of bin 2** (no rounding, also for `Ereco_min = 0`), setup's
  threshold index is 2 (an error otherwise), and an event at the threshold is fitted.
  *(`test_config_and_backgrounds.jl`, `derived_mode.jl`)*

> ✅ `test_response_and_backgrounds.jl`, `test_config_and_backgrounds.jl`, `derived_mode.jl`

<a id="42"></a>
#### 🟢 4.2 · ES background norms
Each ES background has its own normalisation parameter `ES_bg_norm_i`.
> ✅

<a id="43"></a>
#### 🟢 4.3 · Fixed vs free background
A background with `systematics: 0` equals the same background with a free norm evaluated at its
nominal value.
> ✅

<a id="43a"></a>
#### 🟢 4.3a · Background MC weights
Background MC weights enter selected and total counts alike: each bin holds
(selected weighted events)/(generated events) × exposure × area × flux.
> ✅

<a id="43b"></a>
#### 🟢 4.3b · Normalisation conventions
Normalisation conventions (module mass, generated events, face areas, CC reference flux) are config
keys, not code constants.
> ✅

<a id="43c"></a>
#### 🟢 4.3c · Semi-inclusive mode
Each ES event is in exactly one channel (forward → ES, backward → CC as mis-ID). Requires
`use_scattering_info` (enforced at config parse). The mis-ID chain maps every ES reco bin through its
finite centre.
> ✅ (centres)

<a id="43d"></a>
#### 🟣 4.3d · Signal-sample weights
Meaning not yet defined (TODO.md); non-unit weights are ignored with a warning.

<a id="43e"></a>
#### 🟢 4.3e · Target counts
Target counts per kt from the LAr molar mass (39.948 g/mol), Z = 18 and the ⁴⁰Ar abundance
(99.604 %): **2.7135e32 electrons**, **1.5015e31 ⁴⁰Ar**.
> ✅

<a id="44"></a>
#### 🟢 4.4 · `expected_rates` shapes
`expected_rates` returns arrays with the documented shapes: day `(n_Ereco,)`, night
`(n_cosz, n_Ereco)`.
> ✅ "expected_rates – wiring & shapes"

<a id="44a"></a>
#### 🟢 4.4a · Input-sample checks at setup
Input samples are checked at setup (`src/inputChecks.jl`):
- A per-file table is logged. Its "in window" counts selected events at or above the threshold (what
  the fit uses).
- Warnings are raised for:
  - pre-binned samples (about one row per distinct value, on a regular grid with step ≥ 10 keV;
    quantised or Float32 event samples are not flagged);
  - signal and ES background samples with every event selected (efficiency 1);
  - byte-identical input files.
- **Not detected:** flattened 2-D histograms (values repeat across the second axis), histograms with
  bins under 10 keV, log-spaced histograms.
- A per-bin CSV of data, signal and each background component at the truth is written per detector
  and channel; its components add up to the total background at the truth and its data column equals
  the Asimov data.

> ✅ `test_input_checks.jl`, `derived_mode.jl`

<a id="45"></a>
#### 🟡🟣⚠️ 4.5 · Event-level inputs
Inputs are event-level samples; rates are invariant (within MC statistics) to the input event order
and to re-binning the analysis grid.

> [!WARNING]
> The production `config.yaml` uses pre-binned CC neutron and gamma background files (200 keV
> histograms), which alias into a ±20% sawtooth per analysis bin (input audit, 2026-10-18). Setup warns
> (4.4a) and runs anyway. New productions are expected.

---

## 5. Likelihood

> 📖 Source: [likelihood.md](docs/likelihood.md)

<a id="51"></a>
#### 🟢 5.1 · Poisson log L
Poisson log L matches the reference formula; equals 0 for n_exp = n_obs; is finite for zero counts.
> ✅ `test_likelihood_correctness.jl`

<a id="52"></a>
#### 🟢 5.2 · Barlow–Beeston limit
Barlow–Beeston reduces to Poisson when the MC uncertainty → 0.
> ✅

<a id="53"></a>
#### 🟢🟡 5.3 · Channel composition and LLH scan
- 🟢 Total log L = ES + CC when both channels are on. *("make_likelihood – ES/CC composition")*
- 🟡 `llhScan.jl` evaluates `total_llh` once per grid point: a conditional scan (other parameters at the
  Asimov truth) storing 2·log L and its conditioning point, which `plotLLH.py` marks. Untested beyond
  a smoke run.

<a id="54"></a>
#### 🟡 5.4 · Asimov maximum
Asimov: at the true parameters the log L is at its maximum and the expected counts equal the data.

<a id="55"></a>
#### 🟢 5.5 · Full-likelihood regression
The full-likelihood value at fixed points matches `tests/regression_reference.toml` (**rtol 1e-9**).
- Variants: `night`, `legacy`, `night_da`, `night_bb`, `night_da_indep`.
- Points: the Asimov truth, two displaced points (`near`, `slow`: oscillation parameters and Earth
  norms moved), and `xsec` (the CC cross-section nuisances moved, 3.3).
- Frozen synthetic samples (`tests/fixtures/`): ES and CC, two detectors, fixed and free backgrounds,
  semi-inclusive angular ES. It does not depend on the real (changing) inputs.

> ✅ `regression_likelihood.jl` (opt-in, `JULIA_REGRESSION=1`)

<a id="55a"></a>
#### 🟢🟣 5.5a · MC statistical uncertainty
`mc_stat_likelihood` (`src/likelihoods/mcstat.jl`, notes in `docs/mcstat/mcstat.pdf`).

🟢 Tested (`test_mcstat.jl`, `derived_mode.jl`, regression variants `night_da`, `night_bb`,
`night_da_indep`):
- the closed-form scale β is the minimum of the profiled likelihood, per bin and for a shared
  day + night group (against brute-force minimisation);
- the per-bin form equals MaCh3's `GetTestStatLLH` (DA and Conway);
- a → ∞ gives Poisson; Asimov data give β = 1 and an unchanged likelihood at the truth; profiling never
  lowers the likelihood below Poisson;
- the signal MC variance agrees with brute-force resampling of the MC; the background relative
  variance is Σw²/(Σw)²;
- the prediction rebuilt from its components equals the Asimov data in every E_reco bin and mode
  (plain, angular, inclusive, semi-inclusive);
- per-bin pieces add up to the channel likelihood; Enzyme agrees with ForwardDiff;
- predictions significantly below 0 raise an error on every path (incl. a = Inf bins and the per-bin
  diagnostic), as in plain Poisson.

Design choices:
- What is frozen: the effective count a (the relative MC uncertainty) at the truth; MaCh3's default
  freezes Σw² (the absolute one) instead.
- `shared` is exact for backgrounds and an approximation for signal; `independent` dilutes the
  day/night (Δm²₂₁) information.
- Conway's form pins β = 0 for N = 0 and M > a (zero gradient there).

🟣 Not modelled (inferred small, unverified): the same CC MC in both semi-inclusive channels; MC
variance of the mis-ID and scattering-angle responses; per-component shape fluctuations within one
E_reco bin.

<a id="56"></a>
#### 🟡 5.6 · Per-bin log L
Per-bin log L (`make_perbin_likelihood`, a diagnostic) summed over bins equals the total log L minus
the overflow (last, +∞) energy bin, which the diagnostic leaves out by design (it holds little signal
and dominated the bin-importance scores). `per_bin_llh` returns one result per detector.

<a id="57"></a>
#### 🟢 5.7 · Derived mode
Derived mode processes every active detector; each detector's quantities equal what the
single-detector version computed for it (all 88 keys of the first detector identical to the previous
code, 2026-10-15).
> ✅ `tests/derived_mode.jl`, on a synthetic weighted chain with three detectors (ES+CC,
> semi-inclusive angular, pure inclusive), against values computed directly: per-step asymmetries;
> posterior-predictive means, variances and 1/3σ bands; background and signal-only arrays; per-bin
> log-L means, variances and correlations; CC background components summing to the total; best-fit
> oscillation map; no CC keys without a CC channel.

<a id="58"></a>
#### 🟢 5.8 · Python loaders
The Python loaders (`utils/posteriorHelpers.py`) return the requested detector's derived values and
**fail, rather than fall back**, when a file cannot provide that detector, when chains resolve to
different detectors, or when a per-step derived array does not match the chain length (chain resumed
after deriving).
> ✅ (values and unknown detector, in `derived_mode.jl`)

---

## 6. Gradients

<a id="61"></a>
#### 🟢 6.1 · ForwardDiff vs finite differences
NightOsc ForwardDiff gradient matches central finite differences to **1e-5** (relative).
> ✅ "ForwardDiff gradient matches finite differences"

<a id="62"></a>
#### 🟢 6.2 · Cached Jacobian rule
The cached Jacobian rule equals direct dual numbers under any ForwardDiff chunk size (**1e-9**).
> ✅

<a id="63"></a>
#### 🟢 6.3 · Enzyme vs ForwardDiff
- The Enzyme reverse-mode gradient of the full likelihood matches ForwardDiff (**~1e-13**) and leaves
  the primal value unchanged.
- The reverse rule's reuse of the primal's forward state gives exactly the VJP of a fresh pass, for any
  order of calls at different points within one gradient.

> ✅ Unit level in `test_osc_night.jl` (reuse equality, stale-state cases, three NightOsc calls per
> gradient against ForwardDiff); full likelihood via `regression_likelihood.jl night --grad` (manual;
> at the `near` point with the CC cross-section nuisances moved; 4.5e-16 on the fixtures, 2026-10-19).

---

## 7. Sampling and I/O

> 📖 Source: [mcmc_and_priors.md](docs/mcmc_and_priors.md)

<a id="71"></a>
#### 🟢🟡 7.1 · Resume
- 🟢 Resuming a run into the same output file never loses or overwrites existing batches
  (`next_batch_name`; end-to-end resume smoke run 2026-10-15).
- 🟡 A failed write stops the run (`saveBatch` rethrows).

<a id="72"></a>
#### 🟡 7.2 · Seeds
Seeds fix the chains: two runs with the same config and seed give identical samples.

<a id="73"></a>
#### 🟡 7.3 · Coverage
On Asimov or toy data, the 68%/95% credible intervals cover the true values at the nominal rate.
*(Needs toy-data generation.)*

<a id="74"></a>
#### 🟢🟣 7.4 · Warm start
🟢 `warm_start_chain` (`test_warm_start.jl`, through `_build_samplingalg` and `rebuild_live_state`):
- the mass matrix (BAT's affine transform in the PriorToNormal space) is the mean and covariance of the
  previous run's samples (weights = multiplicities) mapped into that space;
- samples past the current prior support, or clamped onto a boundary, are dropped (an error if more
  than 1%);
- the transform stays unchanged through init, burn-in, sampling and a resume (`prevFile`), and the
  chains start from previous draws;
- a previous run missing a fitted parameter is refused.

🟣 BAT's default RAM tuning does not learn the posterior shape for HMC: it rescales A along each
trajectory by (p_accept − 0.8) while the step size adapts too, and the two are degenerate. On a toy it
ended at A ≈ identity, from a cold start and from a correct warm start (gone within ~50 init steps).
*(Toy study 2026-10-16; not a test.)*

---

## 8. Model inputs

> 📖 Source: [event_rate_pipeline.md](docs/event_rate_pipeline.md#tabulated-inputs-srctabulatedjl-srcmodelinputsjl)

<a id="81"></a>
#### 🟢 8.1 · Input-table contract
Every tabulated model input (⁸B and hep spectra, CC cross section, solar exposure, Earth potential,
the potential along a makePaths chord) is built through one helper, `TabulatedInput`
(`src/tabulated.jl`), which:
- **validates on construction:** rejects tables with non-finite values, unsorted or repeated nodes,
  fewer than two points, mismatched lengths, or (where the quantity requires it) negative values, with
  an error naming the input and its source. The NightOsc Earth model, which keeps its own
  interpolation, rejects the same defects in its table;
- **interpolates linearly** through the nodes (exact at the nodes; at in-domain points identical to
  Interpolations.jl's `linear_interpolation` on the same nodes);
- **never extrapolates silently:** each input has an explicit policy (`:error` by default, `:flat`,
  `:zero`); under `:error` a query outside the table raises an error naming the input, the queried
  value and the domain; queries outside the table are counted under every policy;
- **logs its range:** records the range queried, and setup logs each input's domain against that range
  once.

Further:
- Queries computed from coordinates that should land on a table edge (the NightOsc zenith nodes at the
  lowest night cos z) are clamped to the bin range, so a rounding error never reaches outside the
  table; an exposure file must cover [lowest night cos z, 0].
- All shipped inputs use `:error`. Headerless tables keep their first row; a table that requires a
  header (CC) refuses a numeric first row.
- Porting the inputs onto the helper left every value unchanged (nodes and midpoints identical to the
  previous implementation; the likelihood regression is unchanged).

> ✅ `test_tabulated.jl` (and `test_cc_xsec.jl` for the CC table)

<a id="82"></a>
#### 🟢 8.2 · Solar-model inputs
A solar-model file (`solar_model_file`, JLD2) holds `radii` [R/R☉], `n_e` [N_A/cm³] and the ⁸B and hep
production weights `prodFractionBoron`, `prodFractionHep`, one per radius. At load (`load_solar_model`):
- **validated:** radii finite, strictly increasing, in (0, 1]; n_e finite and > 0; weights finite,
  ≥ 0, not all zero, lengths matching; errors name the file;
- **normalised by the fitter:** the weights may be dP/dr or per-zone fractions; every use divides by
  their sum;
- **hep flux:** the Asimov hep flux is the config's `true_integrated_HEP_flux`; when the file stores the
  model's own prediction (`flux_hep`) and the two differ by more than 1%, setup warns.

Shipped files and their content, checked against the profile data (not only metadata):

| File | Model | Composition |
|---|---|---|
| `SF3_GS98.jld2` (default) | SF-III / B23, GS98 (Zenodo 10822316) | high-Z |
| `SF3_AGSS09met.jld2` | SF-III / B23, AGSS09met | low-Z |
| `GS98_low_z.jld2` (legacy) | B16-GS98 | **high-Z** (name swapped) |
| `AGSS09_high_z.jld2` (legacy, regression fixture) | B16-AGSS09met | **low-Z** (name swapped) |

The SF-III files are built by `utils/make_solar_model_inputs.jl` from the source tables, whose SHA-256
they store, and equal the source columns (n_e = 10^log_ne, nu_8B, nu_hep) on (0, 0.5] R☉.

> ✅ `test_tabulated.jl` ("SF-III solar-model files", "solar-model input contract")

---

## Planned (Phase 2, not yet in the code)

> [!NOTE]
> Earth-density parametrisations beyond per-layer scaling (boundary shifts, local polynomial
> perturbations, large-scale per-layer perturbations). Each must reduce to the nominal Earth model at
> zero perturbation (2.1), be finite and physical (density > 0) at the prior edges, and stay
> Enzyme-differentiable (6.3). Requirements to be written by the physicist before implementation.
