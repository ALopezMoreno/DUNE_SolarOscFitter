using YAML
using Printf

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
    global SolarModelFile = config["solar_model_file"]

    global flux_file_path = config["flux_model_file"]

    # Reconstruction samples
    global nue_filepath = config["reconstruction_sample_ES_nue"]
    global other_filepath = config["reconstruction_sample_ES_nuother"]
    global CC_filepath = config["reconstruction_sample_CC"]

    # MC normalisation
    global ES_normalisation = config["ES_flux_normalisation"]
    global CC_normalisation = config["CC_flux_normalisation"]

    # Background MC
    global ES_nue_filepath_BG = config["reconstruction_sample_ES_nue"]
    global ES_nuother_filepath_BG = config["reconstruction_sample_ES_nuother"]
    global CC_filepath_BG = config["CC_background_file"]

    global bins = (bin_number=config["nBins"], min=config["energy_range"][1]*1e-3, max=config["energy_range"][2]*1e-3)
    global E_threshold = (ES=config["emin_ES"]*1e-3, CC=config["emin_CC"]*1e-3)

    # MCMC parameters
    global mcmcSteps = config["nSteps"]
    global mcmcChains = config["nChains"]
    global tuningSteps = config["nTuning"]
    global maxTuningAttempts = config["maxTuningAttempts"]

    # LLH parameters
    global llhBins = config["llh_bins"]

    # Asimov values
    global sin2_th12_true = config["true_sin2_th12"]
    global sin2_th13_true = config["true_sin2_th13"]
    global dm2_21_true = config["true_dm2_21"]

    # Fast mode?
    global fast = config["fastFit"]

    # Determine which script to call based on LLH
    script_to_run = config["LLH"] ? "llhScan.jl" : "mcmc.jl"
    script_path = joinpath(@__DIR__, script_to_run)  # Reference script in the same directory

    # include the corresponding script
    include(script_path)
end

# Run the main function
main()
