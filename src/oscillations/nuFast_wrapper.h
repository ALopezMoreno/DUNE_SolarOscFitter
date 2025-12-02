// nufast_earth_wrapper.h
#pragma once
#include <cstddef>
#include <vector>

extern "C" {

// Match oscillation parameters from julia:
struct OscPars {
    double s12sq;
    double s13sq;
    double s23sq;
    double delta;   
    double dmsq21; 
    double dmsq31;  
};

// Opaque pointer to the engine
typedef void* NuFastEarthHandle;

// Initialise a NuFast-Earth engine
NuFastEarthHandle nufast_earth_create(
    const double* energies, std::size_t nE,
    const double* coszs,    std::size_t nCosz
);

// Get probabilities
void nufast_earth_probs(
    NuFastEarthHandle handle,
    const OscPars* pars,
    const double* n_vec,
    std::size_t nN,
    double* out_probs_day,
    double* out_probs_night
);

// Destroy the engine and free memory
void nufast_earth_destroy(NuFastEarthHandle handle);

}  // extern "C"