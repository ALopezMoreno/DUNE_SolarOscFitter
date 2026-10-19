# TODO

Deferred work, agreed with the maintainer. Remove items when done.

## Next up

The maintainer's queue, in priority order; the master session works from here (see CLAUDE.md,
"Parallel work"). Status: `todo` · `running (branch)` · `review (branch)` · `blocked: reason` ·
`done (commit)`. Only the maintainer adds items or changes their order; a `done` item stays until
the maintainer has seen it.

- [x] `done (master 9299488)` **Save the resolved config with each run.** Write the config as parsed (after all
      defaults) into the output, next to the chain, so a chain records what produced it.
- [x] `done (master 8a0c0dd)` **Reject unknown config keys.** Error (or warn) on top-level and `detectors:` keys
      that readConfig.jl does not read, so a typo does not silently fall back to the default.
- [ ] `todo` *(low priority)* **Config inheritance.** `base:` key (a file or a list) resolved in
      `load_parameters`: deep-merge dicts, later wins, lists replaced; flat configs unchanged.
      Test: an override file resolves to the same dict as the equivalent flat file. Then convert
      `configs/` and retire the sed scripts there.
- [ ] `blocked: audit delivered 2026-10-18, maintainer to pick fixes (temp_claude/reports_2026-10)` **Model inputs as tabulated data, interpolated.** Every model input (solar models,
      fluxes, cross sections, Earth model, exposure, ...) should come from tabulated data and be
      turned into a smooth function by interpolation, done in a robust, predictable way. Verify
      that this is the case: list each input, its source table, how it is interpolated or
      extrapolated, and any hard-coded values or ad-hoc fits. Report before changing anything.
- [x] `done (master 66061b9)` **MARLEY 2.0.0 nominal CC cross section** (0–30 MeV table,
      no extrapolation). Chains from before are stale.
- [x] `done (master 0946cbd)` **Tabulated-input helper.** One loader for input tables:
      validation, explicit extrapolation policy (error by default), domain vs used range logged at
      setup. No change to the likelihood value. Exposure files must cover [lowest night cos z, 0].
- [ ] `study: the maintainer runs it (orthogonal basis on master e55e57b)` **CC cross-section nuisances for a DUNE constraining-power study.**
      Goal: joint ES+CC fit to see how well DUNE constrains σ_CC. MARLEY (S. Gardiner, Oct 2026):
      physics-motivated uncertainties are ~1 year away; use ad hoc variations for now.
      - Done on branch self-warm-start (2026-10-18, maintainer's choice): keep
        norm·exp(tilt·L + curv·L²), with wide default priors (tilt Normal(0, 0.3), curv Normal(0, 0.5)).
      - Dropped: an energy shift of σ(E). Over the 8B CC rate it is ≈ one fixed combination of
        norm/tilt/curv: 0.3 MeV ≈ (−0.11, +0.23, −0.30), with 3% of its shape left over for E_true 7–16
        MeV. A detector energy scale (on E_reco, shared by ES and CC) is a different parameter; see
        Statistics.
      - Done (master e55e57b, 2026-10-19): tilt/curv in an orthogonal basis, SPEC 3.3 [tested];
        regression gained an "xsec" point; chains from another basis warn when reused.
      - Then the study: Asimov fits with ES only, CC only and ES+CC, cross section fixed vs free;
        posterior/prior widths, and the data's Fisher information per cross-section direction
        (prior-independent).
- [ ] `todo: discuss with Steven Gardiner first` **More realistic CC cross-section handles** (later,
      replacing or adding to tilt/curv):
      - Split σ = σ_Fermi + s_GT·σ_GT(E). The Fermi transition to the IAS has B(F) = N − Z = 4, nearly
        model-independent; the uncertainty sits in the GT strength: (p,n) vs β-decay tension ~10%,
        QRPA vs shell model above 8 MeV.
      - MARLEY 2.0 per-transition B(F)/B(GT) errors → covariance → eigenmodes (one carries ~96%,
        nearly a normalisation).
      - One B09 → B98 data-set template.
      - Fake-data check: Asimov from B98 fitted with the nominal model; bias on θ₁₂, Δm²₂₁ and hep.
      Inputs: temp_claude/reports_2026-10/marley/.
- [ ] `todo` **Review the diagnostic plots in utils/ now that the likelihood is differentiable.**
      Parameter pulls, per-bin diagnostics (utils/plotBinDiagnostics.py) and similar: where they
      come from finite differences or ±1σ re-evaluations, use the exact gradient/Hessian instead
      (e.g. impacts, Fisher-matrix widths, ∂(per-bin LLH)/∂θ).
- [ ] `checking (night-fraction 68f5ac3)` **Day/night split from the exposure file.** Replace the
      hard-coded 0.5/0.5 by f_night = ∫_{cos z<0} exposure / ∫ exposure (0.4978 for the shipped
      file), for signal and backgrounds; A_DN per unit exposure. Changes the likelihood.
      Still to check: how solarExposure.csv was made (site latitude; time-weighted only, or with
      the seasonal 1/r² flux weighting, which would bias A_DN by ~0.8% if missing).
      2026-10-19: the CSV is time-weighted (Kepler orbit reproduces 0.4978). That is right for the
      backgrounds; the signal needs 1/r² weighting, which by Kepler's 2nd law is uniform in solar
      longitude: f_night = 0.5000 and a cos z shape ±2.8% different at the extremes. Proposal: one
      exposure table with a time column (backgrounds) and a flux column (signal split and shape),
      generated by a Kepler-correct script. Approved 2026-10-19; utils/make_solar_exposure.py written
      (time column reproduces the CSV to 0.1–0.2%, flux night fraction 0.500000); plumbing done (WIP
      commits); final runs pending: NightOsc exposure guard trips at the ±0.377 peak (0.3%), decision: measure
      P1e effect, then relax that guard to 1e-2 if negligible; then slow+regression, --update, --grad. (parchita's sun_exposure
      double counts the eccentricity: uniform longitude × 1/r².)
- [ ] `review (solar-models-sf3)` **SF-III solar models.** inputs/SF3_GS98.jld2 (high-Z) and
      inputs/SF3_AGSS09met.jld2 (low-Z) from Zenodo 10822316, as parchita. The old B16 files have
      swapped names. Default: high-Z GS98 (maintainer, 2026-10-19), hep Asimov flux 7950.
- [ ] `todo` **Solar spectra and fastFit.** 8B shape from Winter 2006 with its full tail (fitter's
      file cut at ~14.3 MeV, ~1% harder mid-range), hep from Bahcall 1997 (both in PEANUTS/Data).
      Changes the likelihood. (fastFit hep −4.5%: not to be fixed separately; the maintainer: once a
      fast production-averaging exists it is always used, except for BSM fits.)
- [ ] `blocked: the maintainer will rework the Earth modelling (2026-10-18)` **New Earth model.** Look at
      /home/lopezmore/Desktop/EarthModels and include that Earth model in the fitter. Do not
      start until the maintainer says so.
      After the Earth update, check `earth_uncertainty_file`: it is a required config key, but the
      code only prints it and the file it names (inputs/earth_uncertainty_4.5e-5.jld2) does not
      exist; the Earth prior comes from `earth_normalisation_prior_file`. Remove it (as an obsolete,
      warn-only key) or give it a real use.
- [ ] `blocked: after the new Earth model (maintainer, 2026-10-18)` **Self warm start** (branch
      `self-warm-start`, unmerged): mass matrix from the run's own tuning when no warm_start_chain is
      given. On HD CC (2026-10-18) it gave no overall gain over a cold start (min ESS 6 vs 8; only
      Δm²₂₁ mixed 3.6× better); revisit, possibly with repeated estimation windows, once the Earth
      model changes.
- [ ] `blocked: after the new Earth model (maintainer, 2026-10-18)` **Oscillation file and grid
      evaluation** (branch `osc-config`, worktree .claude/worktrees/osc-config, WIP, not checked):
      `oscillation:` → configs/osc/standard.yaml (engine, tolerance, grids, `nightosc:` settings),
      RunMode EVALUATE (event-weighted rate error, Fisher widths, cost; grids and NightOsc settings).
      Found on HD CC: 40 E_true bins mispredict ~5e-4 of events at Δm²₂₁ ≈ 5e-5 (2e-3 at the prior
      minimum, 5% smearing); the NightOsc defaults are ~100× more accurate than 1e-3 needs, and
      paths 4+1k, nodes ×0.5, 1200 km steps would be ~8× faster at 1.2e-4. Before changing any
      default: check with scaled layer densities, gradients and posteriors; regression update.
      Δm²₂₁ information still rising with more cos z bins at 100. Redo the scans on the new
      Earth model.

## Input format (parked)

- [ ] **Decided (maintainer, 2026-10-18): samples carry per-event `weights`; no `mask`.** Weights are
      needed for reweighting events. Still to settle before implementing:
      - what a signal weight means (selection × reweighting?);
      - how the selection efficiency gets in. Today it is masked/all events per E_true bin
        (src/response.jl), so the sample either keeps rejected events at weight 0 or comes with
        the generated count per E_true.
      Then use the weights in the response matrices and efficiencies, drop `mask` from the format
      and the input checks, and write it into SPEC.md. Until then, non-unit signal weights are
      ignored with a warning.
- [ ] Ask the sample producer: was the selection applied to the HD radiological sample? (All 360k
      events have mask = true, which CC background files may legitimately have, but they give
      ~4e9 events/kt-year in the fit window, 1e5 × the signal.) Why is `smeared_2.5_radiological.csv` identical to the
      unsmeared file? The smeared neutron/gamma files are pre-binned (0.2 MeV grid).
- [ ] ES background face convention: `side` = 0 gets both long faces (2·12·64 m²), 1 both end caps.
      `data_file_Nbkg_..._Bottom.csv` has side = 0. Is the bottom face meant to count as two long
      faces? (Ask the sample producer.)
- [ ] HD in config.yaml lists 3 ES background files but 2 norms/systematics (the third is dropped
      with a warning; HD runs CC-only, so no effect today).

## Before using a correlated Earth prior

- [ ] Run `julia --project tests/regression_likelihood.jl night --grad` with a correlated
      covariance file (the correlation term in mcmc.jl has only been tested in isolation).

## Statistics (on hold, ask before starting)

- [x] MC statistical uncertainty in the likelihood: `mc_stat_likelihood` (Dembinski–Abdelmotteleb
      or Conway), shared day/night factor; docs/mcstat/mcstat.pdf. Open: it is off by default;
      decide whether to switch it on for production fits once the inputs are event-level (the
      pre-binned HD neutron/gamma files make their MC uncertainty meaningless).
- [ ] Energy-scale and resolution systematics. (An unfinished energy-scale sketch, `src/responseSys.jl`,
      was removed as dead code on 2026-10-15; see `git log --all -- src/responseSys.jl`.)
- [ ] R-hat / ESS convergence diagnostics; toy-data coverage tests.
