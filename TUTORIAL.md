# Tutorial

This tutorial fits θ₁₂, θ₁₃, Δm²₂₁ and the solar fluxes using one idealised DUNE vertical-drift
module (10 kt for 10 years) with both channels:

- ES (ν e → ν e) and CC (ν_e ⁴⁰Ar);
- a 5 MeV threshold;
- the electron energy smeared by 10%;
- flat backgrounds at S/B = 0.5.

Everything it needs can be built from a fresh clone. The details of the inputs and configs are in
[tutorial_example/README.md](tutorial_example/README.md).

Run every command from the repository root.

## 1. Install

Install Julia 1.12.3 and Python 3.10 or later, and set them up as in [README.md](README.md#installation):

```bash
julia --project=./ -e "using Pkg; Pkg.instantiate()"
pip install matplotlib numpy pandas scipy h5py seaborn mplhep cmasher cmap \
            networkx corner Pillow termcolor peanuts pyyaml
```

LaTeX is optional: without it, plot labels use matplotlib's mathtext (see README).

## 2. Build the inputs (about 2 minutes)

```bash
julia --project tutorial_example/make_inputs.jl
```

This writes the event samples to `tutorial_example/inputs/` (not in git) and sets the background
normalisations.

## 3. Check that everything runs (about 2 minutes)

```bash
julia -t 2 --project src/readConfig.jl tutorial_example/08_quick_test.yaml
```

This is a short fit with the fast settings, meant only to show that nothing is broken.

Plot what goes into the fit before running it (about a minute). `--plot` builds the likelihood
and plots instead of running. The plots cover the fluxes and cross sections, the true-energy,
signal and background spectra, the response matrices, the day-night asymmetry and the raw samples:

```bash
julia -t auto --project src/readConfig.jl tutorial_example/02_mcmc.yaml --plot   # → images/mcmc_inputs_VD.pdf
```

## 4. Run the fit

Choose one of the two configs:

| Config | Time on 4 threads | Use it for |
|---|---|---|
| `tutorial_example/07_mcmc_fast.yaml` | ~4.5 minutes | test fits; approximate night calculation and coarser grid (precision ~1e-2), short chains |
| `tutorial_example/02_mcmc.yaml` | ~25 minutes | the full calculation |

```bash
julia -t 4 --project src/readConfig.jl tutorial_example/02_mcmc.yaml
```

The chains go to `outputs/tutorial/mcmc.jld2`, or `mcmc_fast.jld2` for the fast config. Before
running a config again, delete its old outputs (`outputs/tutorial/mcmc*`).

## 5. Make the image

```bash
python3 utils/plotOutput.py outputs/tutorial/mcmc -o outputs/tutorial/corner
```

This writes `outputs/tutorial/corner.pdf`: a corner plot of sin²θ₁₂, sin²θ₁₃ and Δm²₂₁, drawn
against the current KamLAND, solar and JUNO contours. Use `outputs/tutorial/mcmc_fast` for the fast
fit. For the quick check of step 3, add `-b 0`: its chains are shorter than the default burn-in of
100 steps.

```bash
python3 utils/plotOutput.py outputs/tutorial/quick_test -b 0 -o outputs/tutorial/quick_corner
```

Other options: `-e` adds corner plots with the hep and ⁸B fluxes and the day-night asymmetry;
`--help` lists the rest.

## Next steps

[tutorial_example/README.md](tutorial_example/README.md) also covers:

- ES-only and CC-only fits, and how to compare them;
- a likelihood scan;
- derived quantities;
- how the configs build on each other with `base:`;
- things to change.
