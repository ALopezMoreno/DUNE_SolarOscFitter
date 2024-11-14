using StaticArrays
include("../src/oscCalc.jl")

mutable struct NuSpectrum
    ETrue8B::Vector{Float64}
    events8B_es::Vector{Float64}
    events8B_cc::Vector{Float64}

    ETrueHep::Vector{Float64}
    eventsHep_es::Vector{Float64}
    eventsHep_cc::Vector{Float64}

    oscweights8B::Union{Vector{Float64}, Nothing}
    oscweightsHep::Union{Vector{Float64}, Nothing}

    events_es_oscillated_nue::Union{Vector{Float64}, Nothing}
    events_es_oscillated_other::Union{Vector{Float64}, Nothing}
    events_cc_oscillated::Union{Vector{Float64}, Nothing}

    function NuSpectrum(ETrue8B, events8B_es, events8B_cc, ETrueHep, eventsHep_es, eventsHep_cc)
        new(ETrue8B, events8B_es, events8B_cc, ETrueHep, eventsHep_es, eventsHep_cc, nothing, nothing, nothing, nothing, nothing)
    end
end


# Getter functions
function get_ETrue8B(s::NuSpectrum)
    return s.ETrue8B
end

function get_events8B_es(s::NuSpectrum)
    return s.events8B_es
end

function get_events8B_cc(s::NuSpectrum)
    return s.events8B_cc
end

function get_ETrueHep(s::NuSpectrum)
    return s.ETrueHep
end

function get_eventsHep_es(s::NuSpectrum)
    return s.eventsHep_es
end

function get_eventsHep_cc(s::NuSpectrum)
    return s.eventsHep_cc
end

function get_oscweights8B(s::NuSpectrum)
    return s.oscweights8B
end

function get_oscweightsHep(s::NuSpectrum)
    return s.oscweightsHep
end

function get_events_es_oscillated_nue(s::NuSpectrum)
    return s.events_es_oscillated_nue
end

function get_events_es_oscillated_other(s::NuSpectrum)
    return s.events_es_oscillated_other
end

function get_events_cc_oscillated(s::NuSpectrum)
    return s.events_cc_oscillated
end

# Setter functions
function set_ETrue8B!(s::NuSpectrum, new_value::Vector{Float64})
    s.ETrue8B = new_value
end

function set_events8B_es!(s::NuSpectrum, new_value::Vector{Float64})
    s.events8B_es = new_value
end

function set_events8B_cc!(s::NuSpectrum, new_value::Vector{Float64})
    s.events8B_cc = new_value
end

function set_ETrueHep!(s::NuSpectrum, new_value::Vector{Float64})
    s.ETrueHep = new_value
end

function set_eventsHep_es!(s::NuSpectrum, new_value::Vector{Float64})
    s.eventsHep_es = new_value
end

function set_eventsHep_cc!(s::NuSpectrum, new_value::Vector{Float64})
    s.eventsHep_cc = new_value
end

function set_oscweightsHep!(s::NuSpectrum, new_value::Vector{Float64})
    s.oscweightsHep = new_value
end

function set_oscweights8B!(s::NuSpectrum, new_value::Vector{Float64})
    s.oscweights8B = new_value
end

function set_events_es_oscillated_nue!(s::NuSpectrum, new_value::Vector{Float64})
    s.events_es_oscillated_nue = new_value
end

function set_events_es_oscillated_other!(s::NuSpectrum, new_value::Vector{Float64})
    s.events_es_oscillated_other = new_value
end

function set_events_cc_oscillated!(s::NuSpectrum, new_value::Vector{Float64})
    s.events_cc_oscillated = new_value
end

function oscReweight!(s::NuSpectrum, oscpars)
    # Calculate the oscillation weights for 8B and Hep using solarSurfaceProbs
    s.oscweights8B = [solarSurfaceProbs(oscpars, e, process="8B") for e in s.ETrue8B]
    s.oscweightsHep = [solarSurfaceProbs(oscpars, e, process="hep") for e in s.ETrueHep]

    # Update the values of the oscillated fluxes
    @inbounds set_events_cc_oscillated!(s, s.events8B_cc .* s.oscweights8B .+ s.eventsHep_cc .* s.oscweightsHep)
    @inbounds set_events_es_oscillated_nue!(s, s.events8B_es .* s.oscweights8B .+ s.eventsHep_es .* s.oscweightsHep)
    @inbounds set_events_es_oscillated_other!(s, s.events8B_es .* (1 .- s.oscweights8B) .+ s.eventsHep_es .* (1 .- s.oscweightsHep))
end