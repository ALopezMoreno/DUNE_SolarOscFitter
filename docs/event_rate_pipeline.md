# Event Rate Pipeline

This document traces the full path from raw Monte Carlo input files to the predicted event counts per bin that enter the likelihood. Each stage's inputs, transformation, and outputs are described with array shapes and units.

---

## Overview

```
CSV files (MC truth)
    │
    ▼
response.jl          Build response matrices P(Ereco | Etrue)
    │
    ▼
backgrounds.jl       Load BG MC, compute efficiencies, set up nuisance priors
    │
    ▼
setup.jl             Assemble Asimov true_params; call propagateSamples once
    │
    ┌──────────────────────────── propagateSamples() ─────────────────────────────┐
    │                                                                             │
    │  propagation_osc.jl    Oscillation probabilities  (Etrue grid)              │
    │       ↓                                                                     │
    │  propagation_osc.jl    compute_oscillated_samples()                         │
    │       ↓                                                                     │
    │  propagation_bg.jl     normalize_backgrounds()                              │
    │       ↓                                                                     │
    │  propagation_reco.jl   apply response matrices  → event rates per Ereco bin │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
likelihood_builder.jl   Compare predicted vs. Asimov counts → log L
```

---

## Stage 1: Response matrix construction (`src/response.jl`)

**Purpose:** Convert raw MC (Etrue, Ereco) pairs into a row-normalised transfer matrix P(Ereco | Etrue).

**Inputs:** CSV files with columns `Etrue`, `Ereco`, `mask` (bool selection), optional `Weights`.

**Function:** `create_response_matrix(data, bin_info_x, bin_info_y)` (`response.jl:24–88`)

**Algorithm:**

1. Zero-energy protection: replace `Etrue == 0` with `1e-9` to avoid binning failures.
2. Build 2D histogram over `(Etrue bin, Ereco bin)` pairs:
   ```
   For each event (x_true, y_reco) within analysis range:
     true_bin = searchsortedlast(bins_x, x_true)    ← left-inclusive binning
     reco_bin = searchsortedlast(bins_y, y_reco)
     contribution_matrix[true_bin, reco_bin] += weight (or 1)
   ```
   `searchsortedlast` returns the last edge index ≤ value, giving the correct bin number for left-inclusive intervals [a_i, a_{i+1}).
3. Row-normalise: each row i becomes P(Ereco | Etrue_i), summing to 1.

**Matrices built:**

| Name | Shape | Description |
|------|-------|-------------|
| `nue_ES_response` | (Etrue, Ereco_ES) | P(Ereco | Etrue) for νₑ ES interactions |
| `nuother_ES_response` | (Etrue, Ereco_ES) | P(Ereco | Etrue) for νμ,τ ES interactions |
| `CC_response` | (Etrue, Ereco_CC) | P(Ereco | Etrue) for CC interactions |
| `angular_ES_response` | (cos_scatter, Ereco_ES) | P(cos_scatter | Ereco) for ES angular reconstruction |
| `angular_BG_response` | (cos_scatter, Ereco_ES) | Background angular distribution (uniform over allowed bins) |
| `CC_inclusive_response` | (Etrue, Ereco_ES) | P(Ereco_ES | Etrue) for CC folded into ES bins (inclusive mode) |

**Angular cut** (`response.jl:125–140`, also `propagation_reco.jl:128–142`):

```
mask[i] = (cos_scatter_center[i] >= angular_cos_cut)
angular_BG_response[mask, :] = 1 / count(mask)   # uniform over forward hemisphere
angular_BG_response[~mask, :] = 0
```

`angular_cos_cut` is set via `ES_cos_cut` in `config.yaml` (default −1.0 = no cut; 0.0 = forward hemisphere only).

**Detection efficiencies:**

Selection efficiency per Ereco bin is computed from the MC mask:
```
eff[i] = (events passing mask in bin i) / (all events in bin i)
```
This is already folded into the input MC files (detection efficiency = 1 after cuts). For CC, a flat 0.9 reconstruction efficiency is applied on top (hardcoded; TODO: read from histFile).

**Output:** `responseMatrices` named tuple stored as a global.

---

## Stage 2: Background loading (`src/backgrounds.jl`)

**Purpose:** Load background MC samples, compute per-bin expected rates, and configure systematic nuisance parameters.

### ES backgrounds

For each background file:

1. Create Ereco histogram (weighted or unweighted MC).
2. Compute selection efficiency per bin:
   ```
   eff[i] = selected_events[i] / total_events[i]   (0 if total == 0)
   ```
   The input files already incorporate detector efficiency; this eff is the analysis cut efficiency only.
3. Compute attenuation factor to normalise to the MC sample size:
   ```
   attenuation = sum(total_events) / 50e6
   ```
4. Final rate in analysis period:
   ```
   ES_bg[j] = histogram[j] × detection_time × eff[j] × ES_normalisation × attenuation
   ```

### CC backgrounds

CC MC is pre-normalised to 1 kt-year of exposure:
```
CC_bg[j] = selected_histogram[j] × 10 × CC_normalisation
```
The factor of 10 scales from the 1 kt-year MC normalisation to the 10 kt-year analysis exposure. Update this factor if the target exposure changes.

### Systematic uncertainties

For each background with a non-zero systematic fraction `sys`:
- True normalization `norm` and prior `TruncatedNormal(norm, norm×sys, [0, 2×norm])` are pushed onto `ES_bg_norms_true` and `ES_bg_norms_pars`.
- `ES_bg_par_counts[i] = 1` signals that this background is fitted; `= 0` means it is fixed.
- These priors feed into the MCMC parameter vector as `ES_bg_norm_1`, `ES_bg_norm_2`, …

**Globals produced:** `ES_bg`, `CC_bg`, `ES_sides`, `ES_bg_par_counts`, `CC_bg_par_counts`, `ES_bg_norms_true`, `ES_bg_norms_pars`, and CC equivalents.

---

## Stage 3: Asimov data and initialization (`src/setup.jl`)

`setup.jl` calls `propagateSamples` once with the true parameter values to generate the Asimov dataset (the "observed" data used throughout the analysis). The resulting arrays are stored as globals and passed to the likelihood function as fixed observed counts.

**Energy threshold indices:**

```julia
index_ES = findfirst(x -> x > E_threshold.ES, Ereco_bins_ES_extended.bins)
index_CC = findfirst(x -> x > E_threshold.CC, Ereco_bins_CC_extended.bins)
```

If either returns `nothing`, an error is raised (the threshold exceeds all bins — check config). All likelihood computations use `rates[index:end]` to exclude low-energy bins.

**Day-night asymmetry** (computed at Asimov point, stored as `true_params`):

```
ES_Ntot = sum(ES_night[:, index_ES:end]) − 0.5 × ES_bg_aboveThreshold
ES_Dtot = sum(ES_day[index_ES:end])      − 0.5 × ES_bg_aboveThreshold

A_ES = 2 × (ES_Dtot − ES_Ntot) / (ES_Dtot + ES_Ntot)
A_eff_ES = 2 × (ES_Dtot − ES_Ntot) / (ES_Dtot + ES_Ntot + 2 × ES_bg_aboveThreshold)
```

The `0.5 × BG` subtracts the background's 50% day/50% night contribution so the asymmetry is signal-only.

---

## Stage 4: Oscillated sample production (`src/propagation/propagation_osc.jl`)

**Function:** `compute_oscillated_samples(unoscillatedSample, params, oscProbs)` (`propagation_osc.jl:159–196`)

**Inputs:**

| Field | Shape | Description |
|-------|-------|-------------|
| `unoscillatedSample.ES_nue_8B` | (Etrue,) | νₑ ES events from 8B flux (truth-energy histogram) |
| `unoscillatedSample.ES_nuother_8B` | (Etrue,) | νμ,τ ES events from 8B flux |
| `unoscillatedSample.ES_nue_hep` | (Etrue,) | νₑ ES events from HEP flux |
| `unoscillatedSample.ES_nuother_hep` | (Etrue,) | νμ,τ ES events from HEP flux |
| `unoscillatedSample.CC_8B`, `.CC_hep` | (Etrue,) | CC events |
| `oscProbs.nue_8B_day` | (Etrue_bins,) | P(νₑ survive) day, 8B |
| `oscProbs.nue_8B_night` | (cosz_bins, Etrue_bins) | P(νₑ survive) night, 8B |
| … | … | HEP and νother analogues |

**Transformation (ES channel):**

```
ES.nue_day   = ES_nue_8B .* oscProbs.nue_8B_day  .* flux_8B
             + ES_nue_hep .* oscProbs.nue_hep_day .* flux_HEP

ES.nue_night = (ES_nue_8B'  .* (flux_8B'  .* oscProbs.nue_8B_night  .* exposure_weights))
             + (ES_nue_hep' .* (flux_HEP  .* oscProbs.nue_hep_night .* exposure_weights))
```

Shape of `ES.nue_day`: `(Etrue_bins,)`.  
Shape of `ES.nue_night`: `(cosz_bins, Etrue_bins)` — one row per zenith-angle bin.

`exposure_weights` (shape `(cosz_bins,)`) accounts for the actual fraction of sidereal time the Sun spends in each zenith band. The night-time broadcasting `flux' .* oscProbs.night .* exposure_weights` applies the correct weight to each cosz row.

---

## Stage 5: Response matrix application (`src/propagation/propagation_reco.jl` + `propagation_core.jl`)

### Helper functions (`propagation_core.jl`)

**`apply_day_response(osc, response, eff; scale=0.5)`** (`propagation_core.jl:73–75`):

```
result = scale × (response' × osc) .× eff
```

- `response'` is `(Ereco × Etrue)` — transpose maps Etrue → Ereco
- Matrix-vector product sums over Etrue: expected counts per Ereco bin
- `.× eff` applies per-bin detection efficiency
- `scale = 0.5` accounts for 50% day/night duty cycle

Output shape: `Vector(Ereco_bins)`.

**`apply_night_response(osc_rows, response, eff; scale=0.5)`** (`propagation_core.jl:77–79`):

For each cosz row of `osc_rows`:
```
row_result = scale × (row' × response) .× eff'
```
then `vcat(...)` stacks all rows.

Output shape: `Matrix(cosz_bins, Ereco_bins)`.

### Standard ES rates (`compute_ES_event_rates`, `propagation_reco.jl:21–38`)

```
rate_ES_day   = (apply_day_response(nue_day, R_nue, eff_nue)
               + apply_day_response(nuother_day, R_nuother, eff_nuother))
               + 0.5 × BG_ES

rate_ES_night = (apply_night_response(nue_night, R_nue, eff_nue)
               + apply_night_response(nuother_night, R_nuother, eff_nuother))
               + 0.5 × (BG_ES' .× exposure_weights)
```

| Output | Shape | Description |
|--------|-------|-------------|
| `rate_ES_day` | `(Ereco_ES,)` | Expected ES counts in day |
| `rate_ES_night` | `(cosz_bins, Ereco_ES)` | Expected ES counts per zenith band at night |

### Angular ES rates (`compute_ES_angular_event_rates`, `propagation_reco.jl:101–150`)

Used when `angular_reco: true`. Projects 1D event rates onto the cos(scatter angle) axis.

**Day (2D):**
```
rate_day_angular[cos_scatter, Ereco] =
    R_angular[cos_scatter, Ereco] × rate_ES_day[Ereco]
  + R_BG_angular[cos_scatter, Ereco] × 0.5 × BG_ES[Ereco]
```

**Night (3D):**
```
rate_night_angular[cos_scatter, Ereco, cosz] =
    reshape(R_angular, Ncos, NEreco, 1) × reshape(rate_ES_night', 1, NEreco, Ncosz)
  + reshape(R_BG_angular, Ncos, NEreco, 1) × reshape(exposure_weights' × BG_ES, 1, NEreco, Ncosz) × 0.5
```

**Angular cut** applied after: `rate_day_angular .*= mask`, `rate_night_angular .*= reshape(mask, :, 1, 1)`.

| Output | Shape | Description |
|--------|-------|-------------|
| `rate_ES_day_angular` | `(cos_scatter, Ereco_ES)` | Day ES counts vs. scatter angle |
| `rate_ES_night_angular` | `(cos_scatter, Ereco_ES, cosz)` | Night ES counts per zenith and scatter angle |

### CC rates (`compute_CC_event_rates`, `propagation_reco.jl:173–183`)

Identical structure to standard ES, using `CC_response` and `CC_eff`:

```
rate_CC_day   = apply_day_response(CC.day, R_CC, eff_CC) + 0.5 × BG_CC
rate_CC_night = apply_night_response(CC.night, R_CC, eff_CC) + 0.5 × (BG_CC' .× exposure_weights)
```

| Output | Shape | Description |
|--------|-------|-------------|
| `rate_CC_day` | `(Ereco_CC,)` | Expected CC counts in day |
| `rate_CC_night` | `(cosz_bins, Ereco_CC)` | Expected CC counts per zenith band at night |

### Inclusive mode (`compute_CC_inclusive_event_rates`, `propagation_reco.jl:200–222`)

When `inclusiveMode: true`, CC neutrinos are reconstructed into ES energy bins using `CC_inclusive_response`:

```
cc_incl_day   = apply_day_response(CC.day, R_CC_incl, eff_CC_incl)
cc_incl_night = apply_night_response(CC.night, R_CC_incl, eff_CC_incl)
```

These are added to `rate_ES_day` and `rate_ES_night` in `propagation_main.jl`. CC backgrounds are absorbed into the ES background. The separate CC likelihood branch is disabled (`use_CC = false`).

In angular mode, CC inclusive events are distributed uniformly across scatter-angle bins (isotropic, via `R_BG_angular`).

---

## Stage 6: Background normalization (`src/propagation/propagation_bg.jl`)

**Function:** `normalize_backgrounds(raw_backgrounds, params)` (`propagation_bg.jl:32–73`)

At each likelihood call, nuisance parameters may have changed. This function recomputes the background vector from scratch using the current nuisance parameter values.

**ES:** For each background with `ES_bg_par_counts[i] != 0`:
```
factor = ES_conversions(side, params.ES_bg_norm_i)
       = area_cm² × norm_value
BG_ES += raw_ES[i] × factor
```

Detector face areas:
- `side = 0`: 2 × 12 m × 64 m long faces = 153 600 cm²
- `side = 1`: 2 × 12 m × 12 m end caps = 28 800 cm²
- `side = -1` (or other): sum of both = 182 400 cm²

**CC:** For each CC background with `CC_bg_par_counts[i] != 0`:
```
factor = CC_conversions(params.CC_bg_norm_i)
       = norm_value / 2.2e-6
BG_CC += raw_CC[i] × factor
```
The reference flux 2.2×10⁻⁶ n/cm²/s is what the CC MC was generated at. Dividing by this and multiplying by the fitted norm converts to arbitrary flux.

---

## Stage 7: Final event counts entering the likelihood

After `propagateSamples` returns, the likelihood function receives:

| Array | Shape | Used by |
|-------|-------|---------|
| `eventRate_ES_day` | `(Ereco_ES,)` or `(cos_scatter, Ereco_ES)` | ES day likelihood |
| `eventRate_ES_night` | `(cosz, Ereco_ES)` or `(cos_scatter, Ereco_ES, cosz)` | ES night likelihood |
| `eventRate_CC_day` | `(Ereco_CC,)` | CC day likelihood |
| `eventRate_CC_night` | `(cosz, Ereco_CC)` | CC night likelihood |
| `BG_ES` | `(Ereco_ES,)` | Barlow-Beeston uncertainty |
| `BG_CC` | `(Ereco_CC,)` | Barlow-Beeston uncertainty |

The likelihood slices each array at `[index_ES:end]` or `[index_CC:end]` to apply the analysis energy threshold, then calls the appropriate statistical formula (see [likelihood.md](likelihood.md)).

---

## Array shape summary

| Stage | ES day | ES night | CC day | CC night |
|-------|--------|----------|--------|----------|
| Unoscillated samples | `(Etrue,)` | `(cosz_calc, Etrue)` | `(Etrue,)` | `(cosz_calc, Etrue)` |
| After oscillation | `(Etrue,)` | `(cosz_bins, Etrue)` | `(Etrue,)` | `(cosz_bins, Etrue)` |
| After response fold (standard) | `(Ereco_ES,)` | `(cosz_bins, Ereco_ES)` | `(Ereco_CC,)` | `(cosz_bins, Ereco_CC)` |
| After response fold (angular) | `(cos_scatter, Ereco_ES)` | `(cos_scatter, Ereco_ES, cosz)` | — | — |

Note that `cosz_calc` (fine zenith grid for oscillation computation) is block-averaged down to `cosz_bins` (analysis zenith bins) via `block_average` before the response fold.
