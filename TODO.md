# TODO

Deferred work, agreed with the maintainer. Remove items when done.

## Input-file checks at load time

Catch bad or misunderstood input samples before they reach a fit. All of these would have
flagged the HD CC inputs of May/June 2026.

- [ ] Warn if a sample looks pre-binned: Ereco values on a regular grid, or far fewer rows than
      expected (the fitter needs event-level samples; pre-binned input aliases when re-binned).
- [ ] Warn if every event has `mask = true` in a background sample (selection probably not applied).
- [ ] Warn if two input files of a detector are byte-identical (e.g. a "smeared" file that was
      never smeared).
- [ ] Print a per-file table at setup: rows, selected rows, weight range, events in the fit window.
- [ ] Save a per-bin signal / per-background-component breakdown for each channel at setup
      (the table built by hand on 2026-10-15), so a new background version shows its effect at once.

## Input format (parked)

- [ ] Define in SPEC.md what `weights` means in signal and background samples (events per
      kt-year? cross-section weights?) and which selection `mask` represents. Until then,
      non-unit weights in signal samples are ignored with a warning (src/response.jl).
- [ ] Ask the sample producer: was the selection applied to the HD radiological sample (all
      360k events have mask = true)? Why is `smeared_2.5_radiological.csv` identical to the
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

- [ ] Barlow–Beeston for limited MC statistics on the backgrounds (radiological weights 5.3e4
      per event). `barlowBeestonLogLikelihood` exists; check how it is wired in.
- [ ] Energy-scale and resolution systematics. (An unfinished energy-scale sketch, `src/responseSys.jl`,
      was removed as dead code on 2026-10-15; see `git log --all -- src/responseSys.jl`.)
- [ ] R-hat / ESS convergence diagnostics; toy-data coverage tests.
