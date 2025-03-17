using YAML
using Distributions
using Printf
using JLD2
using PDMats

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
    global CC_filepath = config["reconstruction_sample_CC"]

    # MC normalisation
    global ES_normalisation = config["ES_flux_normalisation"]
    global CC_normalisation = config["CC_flux_normalisation"]

    # Background MC
    global ES_nue_filepath_BG = config["reconstruction_sample_ES_nue"]
    global ES_nuother_filepath_BG = config["reconstruction_sample_ES_nuother"]
    global CC_filepath_BG = config["CC_background_file"]
    global CC_bg_norm = config["CC_background_normalisation"]

    # Binning
    global Etrue_bins = (bin_number=config["nBins_Etrue"], min=config["range_Etrue"][1]*1e-3, max=config["range_Etrue"][2]*1e-3)
    global Ereco_bins_ES = (bin_number=config["nBins_Ereco_ES"], min=config["range_Ereco_ES"][1]*1e-3, max=config["range_Ereco_ES"][2]*1e-3)  # The first and last bins are exended ad infinitum
    global Ereco_bins_CC = (bin_number=config["nBins_Ereco_CC"], min=config["range_Ereco_CC"][1]*1e-3, max=config["range_Ereco_CC"][2]*1e-3)  # The first and last bins are exended ad infinitum
    global E_threshold = (ES=config["Ereco_min_ES"]*1e-3, CC=config["Ereco_min_CC"]*1e-3)
    global cosz_bins = (bin_number=config["nBins_cosz"], min=-1, max=0)

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
    load_earth_normalisation_prior(config["earth_normalisation_prior_file"])

    # LLH parameters
    global llhBins = config["llh_bins"]

    # Asimov values
    global sin2_th12_true = config["true_sin2_th12"]
    global sin2_th13_true = config["true_sin2_th13"]
    global dm2_21_true = config["true_dm2_21"]
    global integrated_8B_flux_true = mean(prior_8B_flux)

    # Fast mode?
    global fast = config["fastFit"]

    # Uncertainties?
    global earthUncertainty = config["earth_potential_uncertainties"]

    # Determine which script to call based on LLH
    script_to_run = config["LLH"] ? "llhScan.jl" : "mcmc.jl"
    script_path = joinpath(@__DIR__, script_to_run)  # Reference script in the same directory

    # include the corresponding script
    include(script_path)
end

# Run the main function
main()
