#=
readConfig.jl

Configuration file parser and main entry point for the Solar Oscillation Fitter.
This script reads YAML configuration files and sets up global parameters for
the analysis, then dispatches to the appropriate analysis mode.

Key Features:
- YAML configuration file parsing
- Memory usage tracking and profiling
- Support for multiple run modes (MCMC, LLH scan, derived variables)
- Automatic parameter validation and type conversion
- Comprehensive logging and settings export

Run Modes:
- "MCMC": Bayesian parameter estimation with MCMC sampling
- "LLH": Likelihood scanning over parameter space
- "derived": Post-processing of MCMC chains for derived quantities

Author: [Author name]
=#

using YAML          # For configuration file parsing
using Printf        # For formatted output
using JLD2          # For data file I/O
using PDMats        # For positive definite matrices
using Plots         # For memory usage plotting
using Distributions # For prior distributions

# Memory monitoring utilities

# Function to retrieve process RSS (Resident Set Size) in MB by parsing /proc/self/status
function get_rss_mb()
    for line in eachline("/proc/self/status")
         if startswith(line, "VmRSS:")
             # Expected line format: "VmRSS:   123456 kB"
             parts = split(line)
             value_kb = parse(Float64, parts[2])
             return value_kb / 1024  # Convert kB to MB
         end
    end
    return 0.0
end

# Function to track memory usage over time.
# Samples memory usage every `interval` seconds and returns two vectors: memory usage and elapsed times,
# along with a function to stop the tracker.
function track_memory(interval::Float64=0.1)
    mem_usages = Float64[]
    times = Float64[]
    start_time = time()
    stop_tracker = false
    tracker_task = @async begin
        while !stop_tracker
            current_mem = get_rss_mb()
            push!(mem_usages, current_mem)
            push!(times, time() - start_time)
            sleep(interval)
        end
    end
    # Return the vectors and a closure to stop the tracker.
    return mem_usages, times, ()->(stop_tracker = true; wait(tracker_task))
end


include("../src/logger.jl")

# Function to load parameters from the YAML file
function load_parameters(yaml_file::String)
    try
        params = YAML.load_file(yaml_file)
        return params
    catch e
        println("Error loading YAML file: $e")
        return nothing
    end
end

# Functiont to extract proposal covariance matrix
function load_proposal_matrix(config)
    global propMatrix
    if haskey(config, "proposal_matrix")
        jld2_file_path = config["proposal_matrix"]
        try
            # Matrix must be positive-definite
            propMatrix = PDMat(JLD2.load(jld2_file_path, "posterior_cov"))
        catch e
            println("An error occurred while loading the JLD2 file: ", e)
            propMatrix = nothing
        end
    else
        propMatrix = nothing
    end
end

# Function to convert string to distribution
function string_to_distribution(dist_str)
    return eval(Meta.parse(dist_str))
end

# Function to construct argument strings for the called script
function construct_arguments(params::Dict)
    args = []
    for (key, value) in params
        if isa(value, AbstractString)
            push!(args, "--$key=$(value)")
        elseif isa(value, Bool)
            push!(args, "--$key=$(value ? "true" : "false")")
        elseif isa(value, Number)
            push!(args, "--$key=$(value)")
        elseif isa(value, AbstractVector)
            vector_string = "[" * join(value, ",") * "]"
            push!(args, "--$key=$vector_string")
        else
            println("Unsupported parameter type for key: $key")
        end
    end
    return join(args, " ")
end

# Function to load priors for earth potential uncertainty nuisance parameters
function load_earth_normalisation_prior(jld2_file::String)
    try
        # Load data from the JLD2 file
        data = JLD2.load(jld2_file)
        # Extract covariance matrix and central value array (using designated keys)
        covariance = data["covariance"]
        central = data["true"]
        
        # Create a multivariate normal distribution as the prior distribution
        global earth_normalisation_prior = MvNormal(central, covariance) 
        # Set the global variable for central values
        global earth_normalisation_true = central
        return earth_normalisation_prior
    catch e
        println("Error loading earth normalisation prior from file ", jld2_file, ": ", e)
        global earth_normalisation_prior = nothing
        global earth_normalisation_true = nothing
        return nothing
    end
end

function save_settings_to_file(filename::String)
    # Open the file for writing
    open(filename, "w") do file
        # Write header
        write(file, "=== Configuration Settings ===\n\n")

        # General settings
        write(file, "----- General -----\n")
        write(file, "Fast mode: $fast\n")
        write(file, "Use nuFast (overrides Fast mode to *true*): $nuFast\n")
        write(file, "Earth uncertainty enabled: $earthUncertainty\n")
        write(file, "Single channel mode: $singleChannel\n\n")

        # Solar and Earth models
        write(file, "----- Solar & Earth Models -----\n")
        write(file, "Solar model file: $solarModelFile\n")
        write(file, "Flux model file: $flux_file_path\n")
        write(file, "Earth model file: $earthModelFile\n")
        write(file, "Earth uncertainty file: $earthUncertaintyFile\n\n")

        # Binning
        write(file, "----- Binning -----\n")
        write(file, "Etrue bins: $(Etrue_bins)\n")
        write(file, "Ereco bins (ES): $(Ereco_bins_ES)\n")
        write(file, "Ereco bins (CC): $(Ereco_bins_CC)\n")
        write(file, "Energy thresholds (ES, CC): $E_threshold\n")
        write(file, "cos(z) bins: $cosz_bins\n\n")
        write(file, "cos(s) bins: $cos_scatter_bins\n\n")

        # MCMC parameters
        write(file, "----- MCMC -----\n")
        write(file, "Steps: $mcmcSteps\n")
        write(file, "Chains: $mcmcChains\n")
        write(file, "Tuning steps: $tuningSteps\n")
        write(file, "Max tuning attempts: $maxTuningAttempts\n\n")

        # Priors
        write(file, "----- Priors -----\n")
        write(file, "Prior sin²θ₁₂: $(prior_sin2_th12)\n")
        write(file, "Prior sin²θ₁₃: $(prior_sin2_th13)\n")
        write(file, "Prior Δm²₂₁: $(prior_dm2_21)\n")
        write(file, "Prior ⁸B flux: $(prior_8B_flux)\n")
        write(file, "Prior HEP flux: $(prior_HEP_flux)\n\n")

        # True values (Asimov)
        write(file, "----- True Values (Asimov) -----\n")
        write(file, "True sin²θ₁₂: $sin2_th12_true\n")
        write(file, "True sin²θ₁₃: $sin2_th13_true\n")
        write(file, "True Δm²₂₁: $dm2_21_true\n")
        write(file, "True ⁸B flux: $integrated_8B_flux_true\n")
        write(file, "True HEP flux: $integrated_HEP_flux_true\n\n")

        # Background files
        write(file, "----- Backgrounds & Efficiencies -----\n")
        write(file, "ES background files: $ES_filepaths_BG\n")
        write(file, "CC background files: $CC_filepaths_BG\n")
        write(file, "ES background norms: $ES_bg_norms\n")
        write(file, "CC background norms: $CC_bg_norms\n")
        write(file, "ES background systematics: $ES_bg_sys\n")
        write(file, "CC background systematics: $CC_bg_sys\n")
    end
end


# Main script logic
function main()
    # Check if a command-line argument is provided
    if length(ARGS) > 0
        # Use the first argument as the path to the YAML file
        yaml_file = ARGS[1]
    else
        # Fallback to a default path or handle the error
        println("No configuration file provided. Defaulting to config.yaml")
        yaml_file = joinpath(@__DIR__, "..", "config.yaml")
    end

    config = load_parameters(yaml_file)

    if isnothing(config)
        println("Failed to load config parameters. Exiting.")
        return
    end

    ##########################################
    #### Set inputs from config YAML file ####
    ##########################################

    # Output prefix
    global outFile = config["outFile"]

    # Solar fluxes
    global solarModelFile = config["solar_model_file"]
    global flux_file_path = config["flux_model_file"]

    # Earth model file
    global earthModelFile = config["earth_model_file"]
    global earthUncertaintyFile = config["earth_uncertainty_file"]

    # Solar angle exposure over time
    global solarExposureFile = config["solar_exposure_file"]

    # Reconstruction samples
    global nue_filepath = config["reconstruction_sample_ES_nue"]
    global other_filepath = config["reconstruction_sample_ES_nuother"]
    global angular_filepath = config["reconstruction_sample_ES_angle"]
    global CC_filepath = config["reconstruction_sample_CC"]

    # MC normalisation
    global ES_normalisation = config["ES_exposure"]
    global CC_normalisation = config["CC_exposure"]

    # Background MC
    global ES_filepaths_BG = config["ES_background_files"]
    global CC_filepaths_BG = config["CC_background_files"]

    # Background normalisations (MC gets normalised to 1)
    global ES_bg_norms = config["ES_background_normalisations"]
    global CC_bg_norms = config["CC_background_normalisations"]

    # Systematic uncertainties on the background rates (constant over bins)
    global ES_bg_sys = config["ES_background_systematics"]
    global CC_bg_sys = config["CC_background_systematics"]

    # Binning
    global Etrue_bins = (bin_number=config["nBins_Etrue"], min=config["range_Etrue"][1]*1e-3, max=config["range_Etrue"][2]*1e-3)
    global Ereco_bins_ES = (bin_number=config["nBins_Ereco_ES"], min=config["range_Ereco_ES"][1]*1e-3, max=config["range_Ereco_ES"][2]*1e-3)  # The first and last bins are exended ad infinitum
    global Ereco_bins_CC = (bin_number=config["nBins_Ereco_CC"], min=config["range_Ereco_CC"][1]*1e-3, max=config["range_Ereco_CC"][2]*1e-3)  # The first and last bins are exended ad infinitum
    global E_threshold = (ES=config["Ereco_min_ES"]*1e-3, CC=config["Ereco_min_CC"]*1e-3)
    global cosz_bins = (bin_number=config["nBins_cosz"], min=-1, max=0)
    global cos_scatter_bins = (bin_number=config["nBins_cos_scatter"], min=0, max=1)

    # MCMC parameters
    load_proposal_matrix(config)
    global mcmcSteps = config["nSteps"]
    global mcmcChains = config["nChains"]
    global tuningSteps = config["nTuning"]
    global maxTuningAttempts = config["maxTuningAttempts"]

    # Prior distributions
    global prior_sin2_th12 = string_to_distribution(config["prior_sin2_th12"])
    global prior_sin2_th13 = string_to_distribution(config["prior_sin2_th13"])
    global prior_dm2_21 = string_to_distribution(config["prior_dm2_21"])
    global prior_8B_flux = string_to_distribution(config["prior_8B_flux"])
    global prior_HEP_flux = string_to_distribution(config["prior_HEP_flux"])
    load_earth_normalisation_prior(config["earth_normalisation_prior_file"])

    # LLH parameters
    global llhBins = config["llh_bins"]

    # Asimov values
    global sin2_th12_true = config["true_sin2_th12"]
    global sin2_th13_true = config["true_sin2_th13"]
    global dm2_21_true = config["true_dm2_21"]
    global integrated_8B_flux_true = mean(prior_8B_flux)
    global integrated_HEP_flux_true = config["true_integrated_HEP_flux"]

    # Fast mode?
    global fast = config["fastFit"]
    # use NuFast?
    global nuFast = config["nuFast"]

    # Single channel fit?
    global singleChannel = get(config, "singleChannel", false)
    global CC_mode, ES_mode
    if singleChannel == false
        CC_mode = true
        ES_mode = true
    elseif singleChannel == "CC"
        CC_mode = true
        ES_mode = false
    elseif singleChannel == "ES"
        CC_mode = false
        ES_mode = true
    else
        error("Invalid value for singleChannel: $singleChannel. Expected false, \"CC\", or \"ES\".")
    end

    # Angular likelihood info?
    global angular_reco = config["use_scattering_info"]

    # Uncertainties?
    global earthUncertainty = config["earth_potential_uncertainties"]

    # Previous file?
    global prevFile = haskey(config, "prevFile") ? config["prevFile"] : nothing

    allowed_modes = ["LLH", "MCMC", "derived", "PROFILE"]
    run_mode = config["RunMode"]

    if run_mode ∉ allowed_modes
        error("Invalid RunMode: '$run_mode'. Must be one of: $(join(allowed_modes, ", "))")
    end
    
    # Determine which script to run based on RunMode
    script_to_run = Dict(
        "LLH" => "llhScan.jl",
        "MCMC" => "mcmc.jl",
        "derived" => "derive_variables_from_chain.jl",
        "PROFILE" => "profiling.jl"
    )[run_mode]

    script_path = joinpath(@__DIR__, script_to_run)  # Reference script in the same directory

    # include the corresponding script
    include(script_path)

    @logmsg Output ("$(outFile)_configSettings.txt")
    save_settings_to_file("$(outFile)_configSettings.txt")
end



# --- Begin Memory Tracking & Main Execution ---
# 
# @printf("Starting memory tracking...\n")
# mem_usages, times, stop_tracking = track_memory(0.1)

# @printf("Executing main()...\n")
main()   # Execution of the main function

# @printf("Stopping memory tracking...\n")
# Stop the memory tracker and wait for the task to finish.
# stop_tracking()

# --- Plot Memory Usage Over Time ---
#=
@printf("Plotting memory usage over time...\n")
usage_plot = plot(times, mem_usages, xlabel="Time (s)", ylabel="Memory (MB)", 
                  title="Memory Usage During Execution", legend=false)
savefig(usage_plot, "memory_usage.png")
display(usage_plot)

# --- Memory Breakdown by Global Objects ---
@printf("Collecting memory breakdown for global objects...\n")
global_names = names(Main, all=true)
object_sizes = Dict{Symbol,Float64}()

for name in global_names

        try
            obj = getfield(Main, name)
            # Compute size in MB using ObjectSizes.osize (if ObjectSizes is available)
            # Otherwise, use Base.summarysize. Here we use Base.summarysize for simplicity
            object_sizes[name] = Base.summarysize(obj) / 1024^2
        catch
            object_sizes[name] = 0.0
        end
end

# Create vectors for plotting: convert the Symbol keys to Strings.
labels = map(string, collect(keys(object_sizes)))
sizes = collect(values(object_sizes))

@printf("Plotting memory breakdown by global object...\n")

breakdown_plot = bar(labels, sizes,
                      xlabel="Global Object", ylabel="Memory (MB)", 
                      title="Memory Breakdown by Global Object", legend=false,
                      xticks=:(auto), xrotation=45)
savefig(breakdown_plot, "memory_object_breakdown.png")
display(breakdown_plot)

@printf("Memory profiling complete. See 'memory_usage.png' and 'memory_object_breakdown.png' for results.\n")


=#

# Execution of the main function
# main()
