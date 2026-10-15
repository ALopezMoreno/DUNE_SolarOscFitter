using Printf
using Logging

# Define custom log levels for different analysis stages
# We keep built-in Info, Warn, Error but override Info output behavior
const MCMC = LogLevel(1)    # For MCMC sampling progress and diagnostics
const Setup = LogLevel(2)   # For initialization and configuration messages  
const Output = LogLevel(3)  # For final results and output generation

# Utility function to format numbers in scientific notation
function sci_notation(x; digits=4) 
    if x == 0
        return "0"
    else
        exponent = floor(Int, log10(abs(x)))
        mantissa = x / 10.0^exponent
        return "$(round(mantissa, digits=digits)) × 10^$exponent" 
    end
end  

# Names of the custom and built-in levels, for ColoredLogger only. (This used to be a
# Base.show method for LogLevel, which changed how every package's log levels print.)
function level_name(level::LogLevel)
    level == Setup         && return "Setup"
    level == MCMC          && return "MCMC"
    level == Output        && return "Output"
    level == Logging.Info  && return "MCMC"   # Info is shown as MCMC for consistency
    level == Logging.Warn  && return "Warn"
    level == Logging.Error && return "Error"
    return string(Int32(level.level))
end

struct ColoredLogger <: AbstractLogger
    min_level::LogLevel
end

# Required method for AbstractLogger interface
function Logging.handle_message(logger::ColoredLogger, level, message, _module, group, id,
    file, line; kwargs...)

    # Convert Info level to MCMC for consistency
    use_level = (level == Logging.Info) ? MCMC : level

    # Define colors using 256-color palette
    light_blue = 45    # Light blue in 256-color palette
    bright_green = 46  # Bright green in 256-color palette
    orange = 208       # Orange in 256-color palette
    red = 196          # Red in 256-color palette

    # Direct printing with printstyled
    if use_level == Setup
        printstyled("=> ", color=light_blue, bold=true)
        printstyled(level_name(level), ": ", color=light_blue, bold=true)
        printstyled(message, color=:default, bold=false)
        println()  # Newline
    elseif use_level == MCMC
        printstyled("=> ", color=:yellow, bold=true)
        printstyled(level_name(level), ": ", color=:yellow, bold=true)
        printstyled(message, color=:default, bold=false)
        println()  # Newline
    elseif use_level == Output
        printstyled("=> ", color=bright_green, bold=true)
        printstyled(level_name(level), ": ", color=bright_green, bold=true)
        printstyled(message, color=bright_green, bold=false)
        println()  # Newline
    elseif use_level == Logging.Warn
        printstyled("=> ", color=orange, bold=true)
        printstyled(level_name(level), ": ", color=orange, bold=true)
        printstyled(message, color=orange, bold=true)
        println()  # Newline
    elseif use_level == Logging.Error
        printstyled("=> ", color=red, bold=true)
        printstyled(level_name(level), ": ", color=red, bold=true)
        printstyled(message, color=red, bold=true)
        println()  # Newline
    else
        printstyled("=> ", color=:default)
        printstyled(level_name(level), ": ", color=:default)
        printstyled(message, color=:default)
        println()  # Newline
    end
end

# Required method for AbstractLogger interface - Important change here!
function Logging.shouldlog(logger::ColoredLogger, level, _module, group, id)
    # Allow our custom levels and standard levels
    return true
end

# Required method for AbstractLogger interface
Logging.min_enabled_level(logger::ColoredLogger) = min(MCMC, Setup, Output, Logging.Info)

# Required method for AbstractLogger interface
Logging.catch_exceptions(logger::ColoredLogger) = false

# Constructor with default minimum level
ColoredLogger() = ColoredLogger(MCMC)

# Set up the logger
global_logger(ColoredLogger())
