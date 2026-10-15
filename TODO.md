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

## Statistics (on hold, ask before starting)

- [ ] Barlow–Beeston for limited MC statistics on the backgrounds (radiological weights 5.3e4
      per event). `barlowBeestonLogLikelihood` exists; check how it is wired in.
- [ ] Energy-scale and resolution systematics.
- [ ] R-hat / ESS convergence diagnostics; toy-data coverage tests.
