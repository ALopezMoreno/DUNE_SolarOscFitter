using Printf
using Logging

# Custom levels for the analysis stages; Info prints as MCMC
const MCMC = LogLevel(1)
const Setup = LogLevel(2)
const Output = LogLevel(3)

function sci_notation(x; digits=4) 
    if x == 0
        return "0"
    else
        exponent = floor(Int, log10(abs(x)))
        mantissa = x / 10.0^exponent
        return "$(round(mantissa, digits=digits)) × 10^$exponent" 
    end
end  

# Level names for ColoredLogger only (a Base.show method would change every package's output)
function level_name(level::LogLevel)
    level == Setup         && return "Setup"
    level == MCMC          && return "MCMC"
    level == Output        && return "Output"
    level == Logging.Info  && return "MCMC"
    level == Logging.Warn  && return "Warn"
    level == Logging.Error && return "Error"
    return string(Int32(level.level))
end

struct ColoredLogger <: AbstractLogger
    min_level::LogLevel
end

function Logging.handle_message(logger::ColoredLogger, level, message, _module, group, id,
    file, line; kwargs...)

    use_level = (level == Logging.Info) ? MCMC : level

    # 256-colour palette
    light_blue = 45
    bright_green = 46
    orange = 208
    red = 196

    if use_level == Setup
        printstyled("=> ", color=light_blue, bold=true)
        printstyled(level_name(level), ": ", color=light_blue, bold=true)
        printstyled(message, color=:default, bold=false)
        println()
    elseif use_level == MCMC
        printstyled("=> ", color=:yellow, bold=true)
        printstyled(level_name(level), ": ", color=:yellow, bold=true)
        printstyled(message, color=:default, bold=false)
        println()
    elseif use_level == Output
        printstyled("=> ", color=bright_green, bold=true)
        printstyled(level_name(level), ": ", color=bright_green, bold=true)
        printstyled(message, color=bright_green, bold=false)
        println()
    elseif use_level == Logging.Warn
        printstyled("=> ", color=orange, bold=true)
        printstyled(level_name(level), ": ", color=orange, bold=true)
        printstyled(message, color=orange, bold=true)
        println()
    elseif use_level == Logging.Error
        printstyled("=> ", color=red, bold=true)
        printstyled(level_name(level), ": ", color=red, bold=true)
        printstyled(message, color=red, bold=true)
        println()
    else
        printstyled("=> ", color=:default)
        printstyled(level_name(level), ": ", color=:default)
        printstyled(message, color=:default)
        println()
    end
end

function Logging.shouldlog(logger::ColoredLogger, level, _module, group, id)
    return true
end

Logging.min_enabled_level(logger::ColoredLogger) = min(MCMC, Setup, Output, Logging.Info)

Logging.catch_exceptions(logger::ColoredLogger) = false

ColoredLogger() = ColoredLogger(MCMC)

global_logger(ColoredLogger())
