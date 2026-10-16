# TODO

Deferred work, agreed with the maintainer. Remove items when done.

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

- [x] MC statistical uncertainty in the likelihood: `mc_stat_likelihood` (Dembinski–Abdelmotteleb
      or Conway), shared day/night factor; docs/mcstat/mcstat.pdf. Open: it is off by default;
      decide whether to switch it on for production fits once the inputs are event-level (the
      pre-binned HD neutron/gamma files make their MC uncertainty meaningless).
- [ ] Energy-scale and resolution systematics. (An unfinished energy-scale sketch, `src/responseSys.jl`,
      was removed as dead code on 2026-10-15; see `git log --all -- src/responseSys.jl`.)
- [ ] R-hat / ESS convergence diagnostics; toy-data coverage tests.
