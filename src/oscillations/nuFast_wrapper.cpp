#include "nuFast_wrapper.h"
#include "NuFastEarth.h"

// A density model that scales an existing Earth_Density by shell
class ScaledEarth : public PREM_NDiscontinuityLayer {
public:
    std::vector<double> scales;  // one per shell

    ScaledEarth(int n_inner_core_discontinuities,
                int n_outer_core_discontinuities,
                int n_inner_mantle_discontinuities,
                int n_outer_mantle_discontinuities,
                const double* n_vec,
                std::size_t nN)
        : PREM_NDiscontinuityLayer(n_inner_core_discontinuities,
                                   n_outer_core_discontinuities,
                                   n_inner_mantle_discontinuities,
                                   n_outer_mantle_discontinuities),
          scales(nN)
    {
        for (size_t i = 0; i < nN; ++i)
            scales[i] = n_vec[i];
    }

    double rhoYe(double r) override
    {
        // Get the original PREM density
        double base_val = PREM_NDiscontinuityLayer::rhoYe(r);

        // Outside Earth -> same behaviour
        if (r < 0 || r > r_E)
            return base_val;

        // Use the discontinuities that PREM_NDiscontinuityLayer already set up
        int idx = 0;
        while (idx < n_discontinuities && r > discontinuities[idx])
            ++idx;
        if (idx >= n_discontinuities)
            idx = n_discontinuities - 1;

        double scale = (idx < (int)scales.size()) ? scales[idx] : 1.0;
        return base_val * scale;
    }
};

extern "C" {

struct NuFastContext {
    Probability_Engine engine;
    std::size_t nE;
    std::size_t nCosz;
    Earth_Density* earth;

    NuFastContext() : nE(0), nCosz(0), earth(nullptr) {}
};

using NuFastEarthHandle = void*;


NuFastEarthHandle nufast_earth_create(
    const double* energies, std::size_t nE,
    const double* coszs,    std::size_t nCosz
) {
    try {
        auto* ctx = new NuFastContext();

        // Save grid sizes
        ctx->nE    = nE;
        ctx->nCosz = nCosz;

        // Build the std::vectors NuFast expects
        std::vector<double> E(energies, energies + nE);
        std::vector<double> C(coszs,    coszs    + nCosz);

        ctx->engine.Set_Spectra(E, C);

        // Set default Earth model
        ctx->earth = new PREM_NDiscontinuityLayer(1, 1, 1, 3);
        ctx->engine.Set_Earth(2, ctx->earth);

        ctx->engine.Set_rhoYe_Sun(100.0 * (2.0/3.0));

        return static_cast<NuFastEarthHandle>(ctx);
    } catch (...) {
        return nullptr;
    }
}


void nufast_earth_probs(
    NuFastEarthHandle handle,
    const OscPars* pars,
    const double* n_vec,
    std::size_t nN,
    double* out_probs_day,
    double* out_probs_night
) {
    auto* ctx    = static_cast<NuFastContext*>(handle);
    auto& engine = ctx->engine;

    // Update osc params
    engine.Set_Oscillation_Parameters(
        pars->s12sq,
        pars->s13sq,
        pars->s23sq,
        pars->delta,
        pars->dmsq21,
        pars->dmsq31,
        true
    );
    
    // If n_vec is full, modify layer densities
    if (nN == 6) {
        auto* base = new PREM_NDiscontinuityLayer(1, 1, 1, 3);

        // wrap it with ScaledEarth
        if (ctx->earth) {
            delete ctx->earth;
        }
        ctx->earth = new ScaledEarth(1, 1, 1, 3, n_vec, nN);
        engine.Set_Earth(2, ctx->earth);
    }

    // Get the full probability grids
    auto probs_day = engine.Get_Solar_Day_Probabilities();
    auto probs_night = engine.Get_Solar_Night_Probabilities();

    // Extract P_ee (0,0)

    size_t idx_night = 0;
    for (size_t j = 0; j < ctx->nCosz; ++j) { 
        for (size_t i = 0; i < ctx->nE; ++i) { 
            if (j == 0) {
                out_probs_day[i] = probs_day[i].arr[0][0];
            }
            // Night: linear index for (row=j, col=i) in (nCosz, nE) column-major:
            out_probs_night[j + i * ctx->nCosz] = probs_night[i][j].arr[0][0];
        }
    }
}


void nufast_earth_destroy(NuFastEarthHandle handle)
{
    if (!handle) return;
    auto* engine = static_cast<Probability_Engine*>(handle);
    delete engine;
}

}  // extern "C"