# SolarOscFitter — Developer Documentation

SolarOscFitter is a Bayesian analysis framework for measuring solar neutrino oscillation parameters (θ₁₂, θ₁₃, Δm²₂₁) with the DUNE detector. It propagates neutrino oscillation probabilities — including MSW matter effects in the Sun and Earth — through detector response matrices, builds an extended Poisson likelihood over reconstructed energy and angle bins, and samples the posterior via Hamiltonian Monte Carlo.

---

## Analysis chain

```
config.yaml
    │
    ▼
readConfig.jl          parse YAML → ~40 global variables
    │
    ▼
setup.jl               load solar model, Earth profile, MC samples, response matrices,
    │                  backgrounds; generate Asimov data; build likelihood function
    │
    ▼  (once per likelihood evaluation)
propagateSamples()
    ├── get_mixing_parameters()        sin²θ → θ in radians
    ├── setup_earth_propagation()      compute P(νe→ν₁) through Earth at fine resolution
    ├── normalize_backgrounds()        apply nuisance-parameter scale factors to BG MC
    ├── compute_oscillation_probs()    day/night × 8B/HEP × νe/νother   [osc.jl]
    ├── compute_oscillated_samples()   multiply unoscillated Etrue histograms × oscProbs
    └── compute_*_event_rates()        fold through response matrices → predicted counts
            │
            ▼
    likelihood_all_samples()           compare predicted vs. Asimov counts → log L
            │
            ▼
    BAT.jl HMC sampler                posterior over (θ₁₂, θ₁₃, Δm²₂₁, fluxes, nuisances)
```

---

## Documentation files

| File | What it covers |
|------|---------------|
| [oscillation_physics.md](oscillation_physics.md) | PMNS matrix, MSW effect, Earth matter propagation (three backends), ForwardDiff flow |
| [event_rate_pipeline.md](event_rate_pipeline.md) | MC inputs → response matrices → oscillated samples → final event counts per bin |
| [likelihood.md](likelihood.md) | Poisson, conditional (shape-only), and Barlow-Beeston likelihood formulas; assembly |
| [mcmc_and_priors.md](mcmc_and_priors.md) | HMC setup, prior definitions, batch MCMC, Earth systematics approximation |
| [configuration_reference.md](configuration_reference.md) | Every `config.yaml` key: type, default, effect |

For operational instructions (how to run, Python utilities, known limitations) see [CLAUDE.md](../CLAUDE.md).
