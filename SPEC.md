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
    **[tested]** independent normals plus `earth_prior_correlation` (added in `mcmc.jl` when the
    covariance is not diagonal) equal the MvNormal up to a constant (`test_likelihood_correctness.jl`).
    The current covariance file is diagonal, so the term has never run inside BAT or under Enzyme:
    run `regression_likelihood.jl night --grad` with a correlated file before using one. **[untested]**

2.7 Parameter names sort numerically (`earth_norm_2` before `earth_norm_10`). **[tested]**
    `test_response_and_backgrounds.jl`.

## 3. Cross sections ([event_rate_pipeline.md](docs/event_rate_pipeline.md))

3.1 ES couplings follow BKS95 App. A: at T→0, ν_μ e g_V = −0.0395 and g_A = −0.5063 (PDG 2024
    Table 10.3) to 5e-4. **[tested]** `test_es_xsec.jl`.

3.2 ES cross sections are exactly 0 (never negative) below the TPC threshold. **[tested]**

3.2a The recoil quadrature equals the analytic integral of dσ/dT for constant couplings (1e-12);
    I(T) matches BKS95 Eq. A5; σ₀ = 2 G_F² m_e² (ħc)²/π. **[tested]** `test_es_xsec.jl`.

3.3 CC shape systematics: at `cc_xsec_tilt = cc_xsec_curv = 0` and `cc_xsec_norm = 1` the CC cross
    section equals the MARLEY nominal; tilt is orthogonal to norm and curvature to both (flux-weighted,
    about E_pivot). **[untested] [inferred]**

## 4. Detector response and event rates ([event_rate_pipeline.md](docs/event_rate_pipeline.md))

4.1 Each response-matrix row (fixed E_true) sums to 1 over all E_reco, with events reconstructed
    outside the window in the first/last (±∞) bins; they are not redistributed into the window.
    Selection efficiency is applied per E_reco bin. **[tested]** `test_response_and_backgrounds.jl`.
    Binning is built from the analysis threshold: `nBins_Ereco` uniform fitted bins span
    [`Ereco_min`, range max], plus one underflow bin (−∞, `Ereco_min`) that is never fitted; the
    last fitted bin also holds the overflow. The threshold is exactly the lower edge of bin 2 (no
    rounding, also for `Ereco_min = 0`), setup's threshold index is 2 (an error otherwise), and an
    event at the threshold is fitted. **[tested]** (`test_config_and_backgrounds.jl`, `derived_mode.jl`)

4.2 Each ES background has its own normalisation parameter `ES_bg_norm_i`. **[tested]**

4.3 A background with `systematics: 0` equals the same background with a free norm evaluated at its
    nominal value. **[tested]**

4.3a Background MC weights enter selected and total counts alike: each bin holds
    (selected weighted events)/(generated events) × exposure × area × flux. **[tested]**

4.3b Normalisation conventions (module mass, generated events, face areas, CC reference flux) are
    config keys, not code constants. **[tested]**

4.3c Semi-inclusive mode: each ES event is in exactly one channel (forward → ES, backward → CC as
    mis-ID). Requires `use_scattering_info` (enforced at config parse). The mis-ID chain maps
    every ES reco bin through its finite centre. **[tested]** (centres)

4.3e Target counts per kt from the LAr molar mass (39.948 g/mol), Z = 18 and the ⁴⁰Ar abundance
    (99.604 %): 2.7135e32 electrons, 1.5015e31 ⁴⁰Ar. **[tested]**

4.3d Signal-sample weights: meaning not yet defined (TODO.md); non-unit weights are ignored with a
    warning. **[inferred]**

4.4 `expected_rates` returns arrays with the documented shapes: day `(n_Ereco,)`, night
    `(n_cosz, n_Ereco)`. **[tested]** "expected_rates – wiring & shapes".

4.4a Input samples are checked at setup (`src/inputChecks.jl`): a per-file table is logged, and
    warnings are raised for pre-binned samples (about one row per distinct value, on a regular grid
    with step ≥ 10 keV; quantised or Float32 event samples are not flagged), signal and ES background
    samples with every event selected (efficiency 1), and byte-identical input files. Not detected:
    flattened 2-D histograms (values repeat across the second axis), histograms with bins under
    10 keV, log-spaced histograms. The table's
    "in window" counts selected events at or above the threshold (what the fit uses). A per-bin CSV of data, signal and
    each background component at the truth is written per detector and channel; its components add
    up to the total background at the truth and its data column equals the Asimov data.
    **[tested]** `test_input_checks.jl`, `derived_mode.jl`.

4.5 Inputs are event-level samples; rates are invariant (within MC statistics) to the input event order
    and to re-binning the analysis grid. **[untested] [inferred]**

## 5. Likelihood ([likelihood.md](docs/likelihood.md))

5.1 Poisson log L matches the reference formula; equals 0 for n_exp = n_obs; is finite for zero counts.
    **[tested]** `test_likelihood_correctness.jl`.

5.2 Barlow–Beeston reduces to Poisson when the MC uncertainty → 0. **[tested]**

5.3 Total log L = ES + CC when both channels are on. **[tested]** "make_likelihood – ES/CC composition".
    `llhScan.jl` evaluates `total_llh` once per grid point: a conditional scan (other parameters at
    the Asimov truth) storing 2·log L and its conditioning point, which `plotLLH.py` marks.
    **[untested]** beyond a smoke run.

5.4 Asimov: at the true parameters the log L is at its maximum and the expected counts equal the data.
    **[untested]**

5.5 The full-likelihood value at fixed points matches `tests/regression_reference.toml` (rtol 1e-9)
    for both night variants. **[tested]** `regression_likelihood.jl` (opt-in, `JULIA_REGRESSION=1`),
    on frozen synthetic samples (`tests/fixtures/`): ES and CC, two detectors, fixed and free
    backgrounds, semi-inclusive angular ES. It does not depend on the real (changing) inputs.

5.5a MC statistical uncertainty (`mc_stat_likelihood`, src/likelihoods/mcstat.jl, notes in
    docs/mcstat/mcstat.pdf). **[tested]** `test_mcstat.jl`, `derived_mode.jl`, regression variants
    `night_da`, `night_bb`, `night_da_indep`:
    - the closed-form scale β is the minimum of the profiled likelihood, per bin and for a shared
      day + night group (against brute-force minimisation);
    - the per-bin form equals MaCh3's `GetTestStatLLH` (DA and Conway);
    - a → ∞ gives Poisson; Asimov data give β = 1 and an unchanged likelihood at the truth;
      profiling never lowers the likelihood below Poisson;
    - the signal MC variance agrees with brute-force resampling of the MC; the background
      relative variance is Σw²/(Σw)²;
    - the prediction rebuilt from its components equals the Asimov data in every E_reco bin and
      mode (plain, angular, inclusive, semi-inclusive);
    - per-bin pieces add up to the channel likelihood; Enzyme agrees with ForwardDiff.
    What is frozen: the effective count a (the relative MC uncertainty) at the truth; MaCh3's
    default freezes Σw² (the absolute one) instead. `shared` is exact for backgrounds and an
    approximation for signal; `independent` dilutes the day/night (Δm²₂₁) information.
    Not modelled **[inferred small, unverified]**: the same CC MC in both semi-inclusive channels;
    MC variance of the mis-ID and scattering-angle responses; per-component shape fluctuations
    within one E_reco bin. Conway's form pins β = 0 for N = 0 and M > a (zero gradient there).
    Predictions significantly below 0 raise an error on every path (incl. a = Inf bins and the
    per-bin diagnostic), as in plain Poisson. **[tested]**

5.6 Per-bin log L (`make_perbin_likelihood`, a diagnostic) summed over bins equals the total log L
    minus the overflow (last, +∞) energy bin, which the diagnostic leaves out by design (it holds
    little signal and dominated the bin-importance scores). `per_bin_llh` returns one result per
    detector. **[untested]**

5.7 Derived mode processes every active detector; each detector's quantities equal what the
    single-detector version computed for it (all 88 keys of the first detector identical to the
    previous code, 2026-10-15). **[tested]** `tests/derived_mode.jl`, on a synthetic weighted chain
    with three detectors (ES+CC, semi-inclusive angular, pure inclusive), against values computed
    directly: per-step asymmetries; posterior-predictive means, variances and 1/3σ bands;
    background and signal-only arrays; per-bin log-L means, variances and correlations; CC
    background components summing to the total; best-fit oscillation map; no CC keys without a
    CC channel.

5.8 The Python loaders (`utils/posteriorHelpers.py`) return the requested detector's derived values
    and fail, rather than fall back, when a file cannot provide that detector, when chains resolve
    to different detectors, or when a per-step derived array does not match the chain length
    (chain resumed after deriving). **[tested]** (values and unknown detector, in `derived_mode.jl`)

## 6. Gradients

6.1 NightOsc ForwardDiff gradient matches central finite differences to 1e-5 (relative).
    **[tested]** "ForwardDiff gradient matches finite differences".

6.2 The cached Jacobian rule equals direct dual numbers under any ForwardDiff chunk size (1e-9).
    **[tested]**

6.3 The Enzyme reverse-mode gradient of the full likelihood matches ForwardDiff (~1e-13) and leaves
    the primal value unchanged. **[tested]** unit level in `test_osc_night.jl`; full likelihood via
    `regression_likelihood.jl night --grad` (manual; 5.2e-16 on the fixtures, 2026-10-16).
    The reverse rule's reuse of the primal's forward state gives exactly the VJP of a fresh pass,
    for any order of calls at different points within one gradient. **[tested]**
    (`test_osc_night.jl`: reuse equality, stale-state cases, three NightOsc calls per gradient
    against ForwardDiff)

## 7. Sampling and I/O ([mcmc_and_priors.md](docs/mcmc_and_priors.md))

7.1 Resuming a run into the same output file never loses or overwrites existing batches.
    **[tested]** (`next_batch_name`; end-to-end resume smoke run 2026-10-15). A failed write stops
    the run (`saveBatch` rethrows). **[untested]**

7.2 Seeds fix the chains: two runs with the same config and seed give identical samples. **[untested]**

7.3 Coverage: on Asimov or toy data, the 68%/95% credible intervals cover the true values at the
    nominal rate. **[untested]** (needs toy-data generation.)

## Planned (Phase 2, not yet in the code)

Earth-density parametrisations beyond per-layer scaling (boundary shifts, local polynomial
perturbations, large-scale per-layer perturbations). Each must reduce to PREM at zero perturbation
(2.1), be finite and physical (density > 0) at the prior edges, and stay Enzyme-differentiable
(6.3). Requirements to be written by the physicist before implementation.
