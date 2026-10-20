# Tutorial: an idealised DUNE VD fit

A complete, self-contained fit you can run in under an hour, on idealised detector inputs. It
uses the same physics, priors and model inputs as the production `config.yaml`, split into
subconfigs that the run configs build on with `base:`.

## The idealised detector

| | |
|---|---|
| Detector | one vertical-drift module, 10 kt fiducial, 10 years (100 kt-yr) |
| Channels | ES (ν e → ν e) and CC (ν_e ⁴⁰Ar → e⁻ ⁴⁰K*), fitted as separate channels |
| Threshold | E_reco > 5 MeV in both channels |
| ES E_reco | electron kinetic energy T, smeared by 10% (Gaussian σ/E), all events selected |
| CC E_reco | electron kinetic energy T_e = E_ν − 1.504 MeV − E_x, smeared by 10%, all events selected |
| Backgrounds | one per channel, flat in E_reco from 0 up to where the signal ends (14.5 MeV ES, 12.3 MeV CC); S/B = 0.5 above threshold, i.e. background = 2 × signal |
| Not used | ES angular information, MC-statistics likelihood |

The ES recoil spectrum is drawn from the fitter's own dσ/dT (BKS95, `src/xsec.jl`). For CC, the
⁴⁰K* level is drawn in proportion to B · p_e E_e F(Z, E_e) (allowed transitions, B(F) and B(GT)
from MARLEY's Bhattacharya 2009 table in `marley_levels_Bhattacharya2009.dat`). The total CC cross
section is still MARLEY 2.0.0, as in production; the level table only decides how the neutrino
energy is shared between the electron and the nucleus. The expected numbers of events are about
35 000 ES and 105 000 CC signal events above threshold, plus twice as many background events.

## Files

```
tutorial_example/
├── make_inputs.jl            builds inputs/ and calibrates the backgrounds (step 0)
├── marley_levels_Bhattacharya2009.dat
├── inputs/                   generated, not in git
├── configs/                  the pieces of one full configuration
│   ├── run_control.yaml        sections 1 and 5 of config.yaml: outFile, RunMode, backends, seed
│   ├── physics.yaml            sections 2 and 3: Asimov truth and priors
│   ├── sampler.yaml            section 4: chains and steps (shortened)
│   ├── model_inputs.yaml       sections 6 and 7: E_true / cos z binning, solar, flux and Earth models
│   ├── detectors/VD.yaml       section 8: the idealised VD module
│   └── tutorial.yaml           base: all of the above; a complete config on its own
├── 01_setup.yaml             base: configs/tutorial.yaml, RunMode SETUP
├── 02_mcmc.yaml              base: configs/tutorial.yaml, the joint ES + CC fit
├── 03_mcmc_ES_only.yaml      base: 02_mcmc.yaml, singleChannel ES
├── 04_mcmc_CC_only.yaml      base: 02_mcmc.yaml, singleChannel CC
├── 05_llh_scan.yaml          base: configs/tutorial.yaml, RunMode LLH
└── 06_derived.yaml           base: 02_mcmc.yaml, RunMode derived on the step-2 chain
```

### How `base:` works

A config lists the files it builds on under `base:`, as one path or a list, relative to its own
directory. The bases are merged in order, and the file's own keys go on top. Mappings merge key
by key at every depth, so `03_mcmc_ES_only.yaml` changes one setting of the VD block and keeps
the rest:

```yaml
base: 02_mcmc.yaml
outFile: "outputs/tutorial/mcmc_ES_only"
detectors:
  VD:
    singleChannel: "ES"
```

Scalars and lists are replaced whole, and a key cannot be deleted. Every run writes
`<outFile>_config.yaml`. Its `config:` section is the merged result, a flat config you can rerun
as it is, and its provenance lists each base file with its SHA-256. Details are in
[docs/configuration_reference.md](../docs/configuration_reference.md).

## Running it

Run everything from the repository root.

**0. Build the inputs** (about a minute):

```bash
julia --project tutorial_example/make_inputs.jl
```

This writes six event-level CSVs (500 000 events each, energies in GeV) to
`tutorial_example/inputs/`. It then runs the fitter's SETUP twice. The first run finds where each
channel's signal ends: the E_reco bin edge below which 99.9% of the signal above threshold lies.
The backgrounds are then rewritten flat up to that point. The second run sets the two background
normalisations in `configs/detectors/VD.yaml` so that S/B = 0.5 holds. The sampling is seeded, so
the values in git are reproduced. Rerun the script whenever you change the threshold, the
exposure, the E_reco range, the resolution or the truth. `--no-calibrate` only writes the samples,
reusing the endpoints recorded in `VD.yaml`. Use it when the inputs are missing, for example in a
new checkout, and the detector settings are unchanged.

**1. Set up the likelihood** (about a minute):

```bash
julia -t auto --project src/readConfig.jl tutorial_example/01_setup.yaml
```

Look in the log for the event counts, `S/B 0.5` per channel and the day-night asymmetries.
`outputs/tutorial/setup_VD_{ES,CC}_breakdown.csv` hold signal and background per E_reco bin.
Four warnings, "every event ... passes the selection", are expected: this is an ideal detector
with a selection efficiency of 1.

**2. The joint fit** (about 30 minutes on 4 threads, 3.5 GB of memory):

```bash
julia -t 4 --project src/readConfig.jl tutorial_example/02_mcmc.yaml
python3 utils/plotOutput.py outputs/tutorial/mcmc
```

The log warns that NUTS trajectories hit the tree-depth cap. That is the production default
`hmc_max_depth: 4`, which trades effective sample size for speed. Add `hmc_max_depth: 6` to a run
config to see the difference.

**3. One channel at a time**, to see what each contributes:

```bash
julia -t 4 --project src/readConfig.jl tutorial_example/03_mcmc_ES_only.yaml
julia -t 4 --project src/readConfig.jl tutorial_example/04_mcmc_CC_only.yaml
python3 utils/compareOutput.py -a outputs/tutorial/mcmc_ES_only -b outputs/tutorial/mcmc_CC_only \
    --labels "ES only" "CC only" -o outputs/tutorial/ES_vs_CC.pdf
```

**4. A likelihood scan** (RunMode LLH, 10 × 10 points per parameter pair):

```bash
julia -t auto --project src/readConfig.jl tutorial_example/05_llh_scan.yaml
python3 utils/plotLLH.py outputs/tutorial/scan
```

**5. Derived quantities and per-bin diagnostics** from the step-2 chain:

```bash
julia -t auto --project src/readConfig.jl tutorial_example/06_derived.yaml
```

Outputs are not overwritten cleanly. Delete a run's files (`outputs/tutorial/<name>*`) before
running it again.

## Things to try

- **Threshold or resolution.** Change `Ereco_min_ES/CC` in `configs/detectors/VD.yaml` or
  `RESOLUTION` in `make_inputs.jl`, then rerun step 0 so that S/B is 0.5 again.
- **Exposure.** Change `CC_exposure` and `ES_exposure` (module-years), then rerun the calibration.
- **A new variant** needs one short file: `base: 02_mcmc.yaml` plus the keys that change, for
  example `fix_cc_xsec: true`, `mc_stat_likelihood: "DembinskiAbdelmotteleb"` or a different
  `true_dm2_21`.
- **ES direction.** `use_scattering_info: true` turns on the ES angular likelihood. The ideal
  angular sample has no angular smearing, so the background separation it gives is unrealistically
  strong.

## Limitations of the idealisation

- Every event is reconstructed and selected, with no energy-scale or resolution systematics.
- CC uses discrete ⁴⁰K* levels: no continuum and no de-excitation γ energy in E_reco. The Fermi
  function is the non-relativistic one.
- Backgrounds are flat, end where the signal ends, and their shape is known exactly. Each has a 2%
  normalisation prior.
- Above the endpoint the bins have no background, so the hep tail sits on zero background and
  the hep sensitivity comes out optimistic.
- The background MC is thin near its endpoint (CC: a few events in the last bins). This matters
  only with an MC-statistics likelihood.
