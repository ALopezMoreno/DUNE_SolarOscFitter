module nuFastOsc

using ..Osc: oscPars          # import your physics parameter struct
using Base: @ccall

# Path to the shared library built by your Makefile
const libnufast = joinpath(@__DIR__, "libnufast_earth.so")

# C handle type – matches `typedef void* NuFastEarthHandle;`
const NuFastHandle = Ptr{Cvoid}

# Global state inside this module (not in Main)
const nufast_engine = Ref{NuFastHandle}(C_NULL)
const nE_global     = Ref{Csize_t}(0)
const nCosz_global  = Ref{Csize_t}(0)

# C-compatible version of oscPars
struct COscPars
    s12sq::Cdouble
    s13sq::Cdouble
    s23sq::Cdouble
    delta::Cdouble
    dmsq21::Cdouble
    dmsq31::Cdouble
end

# Convert Osc.oscPars → COscPars
function to_cpars(p::oscPars)::COscPars
    COscPars(
        sin(p.θ₁₂)^2,
        sin(p.θ₁₃)^2,
        sin(p.θ₂₃)^2,
        p.δCP,
        p.Δm²₂₁,
        p.Δm²₃₁
    )
end

"""
    init_engines(E::AbstractVector, cosz::AbstractVector)

Create the NuFast engines once, and cache the handle + grid sizes.
Call this exactly once after you know `E_calc` and `cosz_calc`.
"""
const n_engines = Threads.nthreads()
const nufast_engines = Vector{NuFastHandle}(undef, n_engines)

function init_engines(E_calc, cosz_calc)
    Threads.@threads for t in 1:n_engines
        Evec  = collect(Float64, E_calc)
        Cvec  = collect(Float64, cosz_calc)
        nE    = length(Evec)
        nCosz = length(Cvec)

        GC.@preserve Evec Cvec begin
            handle = @ccall libnufast.nufast_earth_create(
                pointer(Evec)::Ptr{Cdouble}, Csize_t(nE)::Csize_t,
                pointer(Cvec)::Ptr{Cdouble}, Csize_t(nCosz)::Csize_t
            )::NuFastHandle
            handle == C_NULL && error("nufast_earth_create returned NULL")
            nufast_engines[t] = handle
        end
    end
end

"""
    shutdown_engines()

Destroy the NuFast engines and reset the handle.
"""
function shutdown_engines()
    for t in 1:length(nufast_engines)
        handle = nufast_engines[t]
        if handle != C_NULL
            @ccall libnufast.nufast_earth_destroy(
                handle::NuFastHandle
            )::Cvoid
            nufast_engines[t] = C_NULL
        end
    end
    return nothing
end

"""
    osc_prob_both_fast(E, params, lookup_density, paths; anti=false, n_vec=[])

Compute P_ee(E, cosz) day and night using the already-initialised NuFast engine.

- `E` is accepted to match the signature of your other backends, but the
  actual energy grid is the one used when calling `init_engines`.
- `n_vec` is either empty or length 6; it is passed through to C++ to
  rescale the Earth model.
"""
function osc_prob_both_fast(
    E::AbstractVector{<:Real},
    params::oscPars,
    lookup_density,
    paths;
    anti::Bool = false,
    n_vec = []
)
    tid = Threads.threadid()
    handle = nufast_engines[tid]

    cpars  = to_cpars(params)
    nvec64 = collect(Float64, n_vec)
    nN     = length(nvec64)

    nE    = Int(nE_global[])
    nCosz = Int(nCosz_global[])

    P_ee_day   = Array{Float64}(undef, nE)
    P_ee_night = Array{Float64}(undef, nCosz, nE)

    GC.@preserve nvec64 P_ee_day P_ee_night cpars begin
        @ccall libnufast.nufast_earth_probs(
            handle::NuFastHandle,
            Ref(cpars)::Ref{COscPars},
            pointer(nvec64)::Ptr{Cdouble},
            Csize_t(nN)::Csize_t,
            pointer(P_ee_day)::Ptr{Cdouble},
            pointer(P_ee_night)::Ptr{Cdouble}
        )::Cvoid
    end

    return P_ee_day, P_ee_night
end

end # module nuFastOsc
