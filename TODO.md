# TODO

Deferred work, agreed with the maintainer. Remove items when done.

## Analysis threshold (decide)

- [ ] The fit starts at the first Ereco bin whose LOWER edge is above `Ereco_min` (setup.jl:
      `findfirst(edge > threshold)`), so a bin starting exactly at the threshold is excluded. With
      HD's `Ereco_min_CC: 5` and 0.8 MeV bins from 1 MeV, the fit starts at 5.8 MeV, not 5.0.
      Changing it to `>=` changes the likelihood (regression reference update).

## Input format (parked)

- [ ] Define in SPEC.md what `weights` means in signal and background samples (events per
      kt-year? cross-section weights?) and which selection `mask` represents. Until then,
      non-unit weights in signal samples are ignored with a warning (src/response.jl).
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

- [ ] Barlow–Beeston for limited MC statistics on the backgrounds (radiological weights 5.3e4
      per event). `barlowBeestonLogLikelihood` exists; check how it is wired in.
- [ ] Energy-scale and resolution systematics. (An unfinished energy-scale sketch, `src/responseSys.jl`,
      was removed as dead code on 2026-10-15; see `git log --all -- src/responseSys.jl`.)
- [ ] R-hat / ESS convergence diagnostics; toy-data coverage tests.
