# SPEC — validation criteria

What the fitter must satisfy, stated so that each item can be tested. Derivations and formulas are in
[docs/](docs/index.md); this file only lists checks and tolerances, with a link to the source of each.

Status tags:
- **[tested]** covered by a test (file named).
- **[untested]** no test yet; a candidate for one.
- **[known to fail]** a current bug is open against it (see the 2026-10-15 audit).
- **[inferred]** drafted from the code or docs and not yet confirmed by the physicist. Confirm or correct.

## 1. Oscillation probabilities ([oscillation_physics.md](docs/oscillation_physics.md))

1.1 Unitarity: P(νe→νe) + P(νe→ν_other) = 1 and every probability lies in [0, 1], for day and night,
    over the full prior range of (sin²θ₁₂, sin²θ₁₃, Δm²₂₁) and E_true in the analysis window.
    **[untested]** (ν_other is built as 1 − P_ee in `propagation_osc.jl`, so the real check is 0 ≤ P_ee ≤ 1.)

1.2 Vacuum limit: with zero matter potential, the night P₁ₑ equals the vacuum |U_e1|² and the
    day-night difference vanishes. **[untested] [inferred]**

1.3 NightOsc bin averages agree with dense sampling to < 1e-3 per bin (measured ≤ 9e-4; ≤ 4.4e-4
    above 3 MeV). **[tested]** `test_osc_night.jl` "bin averages match dense sampling".

1.4 NightOsc (two-flavour, ≤300 km linear-potential steps) agrees with the 3-flavour NumOsc solver to
    < 1e-3 (measured ≤ 5e-4). **[tested]** "step propagator matches the 3-flavour NumOsc solver".

1.5 Day-night factorisation error (⟨cos2θ₁₂ᵐ⟩ and P_day averaged separately) < 5e-4 per bin.
    **[tested]** "day-night factorisation error is small".

1.6 Convergence: refining the zenith steps (8+3k rule) and E_calc points does not change P₁ₑ beyond
    the tolerance in 1.3, including with layer densities scaled by ±24%. **[untested as a test]**
    (measured during the NightOsc port; `utils/checkOscGridConvergence.jl` is the manual check).

1.7 Legacy (`nightOsc: false`) and NightOsc agree to ≤ 0.06 in any single bin (median ~2e-3).
    This is a known accuracy limit of the legacy method, not a requirement to tighten. **[inferred bound]**

## 2. Earth model ([oscillation_physics.md §7](docs/oscillation_physics.md))

2.1 With all `earth_norm_i = 1`, the potential reproduces PREM (`inputs/EARTH_MODEL_PREM_DENSE.dat`)
    exactly. **[untested]**

2.2 The model has 6 layers with sorted boundaries; path segment lengths sum to the chord length;
    paths start and end in layer 1. **[tested]** "Earth model and chords".

2.3 makePaths and NightOsc cross the same layers in the same order, so `earth_norm_i` scales the same
    layer in both. **[tested]** "Earth layers agree with makePaths"; also checked at setup.

2.4 A layer not crossed by any path in the grid has exactly zero gradient. **[tested]** (inner core).

2.5 `earth_norm_i` outside [0, 2] gives log L = −Inf. **[tested]** `test_likelihood_correctness.jl`.

2.6 Earth-norm priors use the full covariance of `earth_normalisation_prior_file`.
    **[known to fail]** `mcmc.jl` uses independent normals (diagonal only); documented limitation.

2.7 Parameter names sort numerically (`earth_norm_2` before `earth_norm_10`). **[known to fail]**
    `_params_by_prefix` sorts lexicographically; latent while there are fewer than 10 layers.

## 3. Cross sections ([event_rate_pipeline.md](docs/event_rate_pipeline.md))

3.1 ES couplings follow BKS95 App. A: at T→0, ν_μ e g_V = −0.0395 and g_A = −0.5063 (PDG 2024
    Table 10.3) to 5e-4. **[tested]** `test_es_xsec.jl`.

3.2 ES cross sections are exactly 0 (never negative) below the TPC threshold. **[tested]**

3.3 CC shape systematics: at `cc_xsec_tilt = cc_xsec_curv = 0` and `cc_xsec_norm = 1` the CC cross
    section equals the MARLEY nominal; tilt is orthogonal to norm and curvature to both (flux-weighted,
    about E_pivot). **[untested] [inferred]**

## 4. Detector response and event rates ([event_rate_pipeline.md](docs/event_rate_pipeline.md))

4.1 Each response-matrix row (fixed E_true) sums to the selection efficiency over all E_reco
    *including* under/overflow; out-of-window events are not redistributed into the window.
    **[known to fail]** `create_response_matrix` drops out-of-window events and renormalises
    (ES rates ~10–15% high). The fix requires regenerating the regression reference.

4.2 Each ES background has its own normalisation parameter `ES_bg_norm_i`. **[known to fail]**
    `norm_index` is never incremented in `propagation_bg.jl`.

4.3 A background with `systematics: 0` gets the same area/exposure conversion as one with systematics.
    **[known to fail]** sys=0 skips `ES_conversions` (~1e8 too small); no current config uses it.

4.4 `expected_rates` returns arrays with the documented shapes: day `(n_Ereco,)`, night
    `(n_cosz, n_Ereco)`. **[tested]** "expected_rates – wiring & shapes".

4.5 Inputs are event-level samples; rates are invariant (within MC statistics) to the input event order
    and to re-binning the analysis grid. **[untested] [inferred]**

## 5. Likelihood ([likelihood.md](docs/likelihood.md))

5.1 Poisson log L matches the reference formula; equals 0 for n_exp = n_obs; is finite for zero counts.
    **[tested]** `test_likelihood_correctness.jl`.

5.2 Barlow–Beeston reduces to Poisson when the MC uncertainty → 0. **[tested]**

5.3 Total log L = ES + CC when both channels are on. **[tested]** "make_likelihood – ES/CC composition".
    **[known to fail]** in `llhScan.jl`, which counts it twice.

5.4 Asimov: at the true parameters the log L is at its maximum and the expected counts equal the data.
    **[untested]**

5.5 The full-likelihood value at fixed points matches `tests/regression_reference.toml` (rtol 1e-9)
    for both night variants. **[tested]** `regression_likelihood.jl` (opt-in, `JULIA_REGRESSION=1`).

5.6 Per-bin log L summed over bins equals total log L, for one and for several detectors.
    **[known to fail]** with more than one detector (`per_bin_llh`).

## 6. Gradients

6.1 NightOsc ForwardDiff gradient matches central finite differences to 1e-5 (relative).
    **[tested]** "ForwardDiff gradient matches finite differences".

6.2 The cached Jacobian rule equals direct dual numbers under any ForwardDiff chunk size (1e-9).
    **[tested]**

6.3 The Enzyme reverse-mode gradient of the full likelihood matches ForwardDiff (~1e-13) and leaves
    the primal value unchanged. **[tested]** unit level in `test_osc_night.jl`; full likelihood via
    `regression_likelihood.jl night --grad` (manual).

## 7. Sampling and I/O ([mcmc_and_priors.md](docs/mcmc_and_priors.md))

7.1 Resuming a run into the same output file never loses or overwrites existing batches.
    **[known to fail]** the batch counter restarts at 1 and the write error is only logged.

7.2 Seeds fix the chains: two runs with the same config and seed give identical samples. **[untested]**

7.3 Coverage: on Asimov or toy data, the 68%/95% credible intervals cover the true values at the
    nominal rate. **[untested]** (needs toy-data generation.)

## Planned (Phase 2, not yet in the code)

Earth-density parametrisations beyond per-layer scaling (boundary shifts, local polynomial
perturbations, large-scale per-layer perturbations). Each must reduce to PREM at zero perturbation
(2.1), be finite and physical (density > 0) at the prior edges, and stay Enzyme-differentiable
(6.3). Requirements to be written by the physicist before implementation.
