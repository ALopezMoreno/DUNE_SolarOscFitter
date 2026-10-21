using YAML          # For configuration file parsing
using Printf        # For formatted output
using JLD2          # For data file I/O
using PDMats        # For positive definite matrices
using Plots         # For memory usage plotting
using Distributions # For prior distributions
using DataStructures: OrderedDict   # detector order = active_detectors order

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


include(joinpath(@__DIR__, "core.jl"))        # DEFAULT_CC_XSEC_FILE
include(joinpath(@__DIR__, "logger.jl"))
include(joinpath(@__DIR__, "runRecord.jl"))   # <outFile>_config.yaml: resolved config + provenance

# Load a config file with its `base:` chain resolved (src/configFiles.jl); `nothing` on error.
function load_parameters(yaml_file::String)
    try
        params = resolve_config(yaml_file)
        return params
    catch e
        println("Error loading YAML file: $e")
        return nothing
    end
end

# Functiont to extract proposal covariance matrix
function load_proposal_matrix(config)
    # Nothing reads a proposal matrix: the HMC sampler tunes its own metric. Say so instead of
    # loading (and, on failure, silently dropping) a file that has no effect.
    global propMatrix = nothing
    if haskey(config, "proposal_matrix")
        @warn "proposal_matrix ($(config["proposal_matrix"])) is set but not used by the HMC sampler; ignoring it. " *
              "To start from a previous run's posterior, set warm_start_chain to that run's outFile."
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
function load_earth_normalisation_prior(jld2_file::String; required::Bool=true)
    if !required
        # Not used by the fit; load it if possible, but never stop the run over it.
        try
            return load_earth_normalisation_prior(jld2_file; required=true)
        catch e
            isfile(jld2_file) && @warn "Could not read $jld2_file (unused: earth_potential_uncertainties is off): $e"
            global earth_normalisation_prior = nothing
            global earth_normalisation_true = nothing
            return nothing
        end
    end
    # Errors propagate: with earth_potential_uncertainties on, a missing or malformed prior
    # file must stop the run, not leave `nothing` to fail somewhere later.
    data = JLD2.load(jld2_file)
    covariance = data["covariance"]
    central = data["true"]
    global earth_normalisation_prior = MvNormal(central, covariance)
    global earth_normalisation_true = central
    return earth_normalisation_prior
end

function mc_stat_key(block, defaults, name)
    v = String(get(block, "mc_stat_likelihood", get(defaults, "mc_stat_likelihood", "Poisson")))
    v in ("Poisson", "BarlowBeeston", "DembinskiAbdelmotteleb") ||
        error("Detector $name: mc_stat_likelihood must be Poisson, BarlowBeeston or DembinskiAbdelmotteleb (got $v)")
    return v
end

function mc_stat_dn_key(block, defaults, name)
    v = String(get(block, "mc_stat_day_night", get(defaults, "mc_stat_day_night", "shared")))
    v in ("shared", "independent") ||
        error("Detector $name: mc_stat_day_night must be shared or independent (got $v)")
    return v
end

"""
    threshold_binning(block, channel, name) -> (bin_number, min, max)   [GeV]

Reco-energy binning built from the analysis threshold: `nBins_Ereco_<ch>` uniform bins of width
w = (range_max − Ereco_min)/nBins span [Ereco_min, range_max], plus one bin below them. The
response and background code stretch the outer bins to ±∞, so the lowest bin is the underflow
(−∞, Ereco_min), never fitted, and the highest also collects the overflow. The threshold is
always a bin edge, and the fit starts exactly at it. `range_Ereco_<ch>[1]` is not used.
"""
function threshold_binning(block, ch, name)
    thr = Float64(block["Ereco_min_$ch"])
    hi  = Float64(block["range_Ereco_$ch"][2])
    n   = Int(block["nBins_Ereco_$ch"])
    thr < hi || error("Detector $name: Ereco_min_$ch = $thr MeV must be below range_Ereco_$ch[2] = $hi MeV")
    n >= 1   || error("Detector $name: nBins_Ereco_$ch must be ≥ 1")
    w = (hi - thr) / n
    return (bin_number = n + 1, min = (thr - w) * 1e-3, max = hi * 1e-3, threshold = thr * 1e-3)
end

# ── Known config keys ─────────────────────────────────────────────────────────
# Every key read from the YAML must be listed here; validate_config_keys() rejects anything else,
# so a misspelled key cannot silently fall back to its default. Adding a key: read it in
# parse_detector_block (per detector) or main() (top level) AND add it to the matching list.

# Keys of one `detectors:` entry, read by parse_detector_block (also accepted at the top level of
# a flat config without `detectors:`).
const DETECTOR_CONFIG_KEYS = Set([
    "singleChannel", "inclusiveMode", "semiInclusiveMode", "use_scattering_info", "ES_cos_cut",
    "ES_exposure", "CC_exposure",
    "reconstruction_sample_ES_nue", "reconstruction_sample_ES_nuother",
    "reconstruction_sample_ES_angle", "reconstruction_sample_CC",
    "ES_background_files", "CC_background_files",
    "ES_background_normalisations", "CC_background_normalisations",
    "ES_background_systematics", "CC_background_systematics",
    "Ereco_min_ES", "Ereco_min_CC", "range_Ereco_ES", "range_Ereco_CC",
    "nBins_Ereco_ES", "nBins_Ereco_CC", "nBins_cos_scatter",
    "module_mass_kt", "ES_bg_generated_events", "ES_bg_face_areas_m2", "CC_bg_reference_flux",
    "mc_stat_likelihood", "mc_stat_day_night",
])

# Top-level keys, read by main().
const TOP_LEVEL_CONFIG_KEYS = Set([
    # run control
    "outFile", "prevFile", "warm_start_chain", "RunMode", "fastFit", "nightOsc", "ad_backend", "hmc_max_depth",
    "mcmc_seed",
    # Asimov is the only data mode: `Asimov: false` is rejected below; data_file is reserved for
    # real data and not read yet (both documented in docs/configuration_reference.md).
    "Asimov", "data_file",
    # input files
    "solar_model_file", "flux_model_file", "earth_model_file", "earth_uncertainty_file",
    "earth_normalisation_prior_file", "solar_exposure_file", "cc_xsec_file",
    # shared binning
    "TPC_threshold", "nBins_Etrue", "range_Etrue", "nBins_cosz", "nBins_cosz_fine", "cosz_fine_edge",
    # detectors
    "detectors", "active_detectors", "mc_stat_likelihood", "mc_stat_day_night",
    # MCMC / LLH / derived
    "nSteps", "nChains", "nTuning", "maxTuningAttempts", "llh_bins", "thinning",
    # priors and nuisances
    "prior_sin2_th12", "prior_sin2_th13", "prior_dm2_21", "prior_8B_flux", "prior_HEP_flux",
    "fix_cc_xsec", "prior_cc_xsec_norm", "prior_cc_xsec_tilt", "prior_cc_xsec_curv",
    "earth_potential_uncertainties", "CC_xsec_scale",
    # Asimov truth
    "true_sin2_th12", "true_sin2_th13", "true_dm2_21", "true_8B_flux", "true_integrated_HEP_flux",
    "true_cc_xsec_norm",
])

# Known but obsolete top-level keys: accepted with a warning (or an error, for a value that can no
# longer be honoured) where they are read in main() / load_proposal_matrix.
const OBSOLETE_CONFIG_KEYS = Set(["proposal_matrix", "binnedNight", "nuFast"])

function _levenshtein(a::AbstractString, b::AbstractString)
    a, b = collect(lowercase(a)), collect(lowercase(b))
    prev = collect(0:length(b))
    for i in 1:length(a)
        cur = similar(prev); cur[1] = i
        for j in 1:length(b)
            cur[j+1] = min(prev[j+1] + 1, cur[j] + 1, prev[j] + (a[i] == b[j] ? 0 : 1))
        end
        prev = cur
    end
    return prev[end]
end

_closest_key(key, known) = isempty(known) ? nothing : argmin(k -> (_levenshtein(key, k), k), collect(known))
_did_you_mean(key, known) = (k = _closest_key(key, known); k === nothing ? "" : " (did you mean '$k'?)")

"""
    validate_config_keys(config)

Error on any config key that is not read by the fitter (top level or inside a `detectors:` entry),
naming the key, its section and the closest known key. Obsolete keys are allowed here (they warn
where they are read). Also checks that every name in `active_detectors` has a `detectors:` entry.
"""
function validate_config_keys(config::AbstractDict)
    multi = haskey(config, "detectors")
    top_known = multi ? union(TOP_LEVEL_CONFIG_KEYS, OBSOLETE_CONFIG_KEYS) :
                        union(TOP_LEVEL_CONFIG_KEYS, OBSOLETE_CONFIG_KEYS, DETECTOR_CONFIG_KEYS)
    problems = String[]
    function check(dict, known, section, other_known, other_hint)
        for key in sort(collect(string.(keys(dict))))
            key in known && continue
            msg = "unknown key '$key' in $section"
            # Suggest from both sections: a misspelled key may also be in the wrong one.
            closest = _closest_key(key, union(known, other_known))
            if key in other_known
                msg *= " ($other_hint)"
            elseif closest in other_known && !(closest in known)
                msg *= " (did you mean '$closest'? That is $other_hint)"
            else
                msg *= " (did you mean '$closest'?)"
            end
            push!(problems, msg)
        end
    end
    check(config, top_known, "the top level", DETECTOR_CONFIG_KEYS,
          "a per-detector key: with a `detectors:` section it must go inside each detector entry")
    if multi
        dets = config["detectors"]
        dets isa AbstractDict || error("config: `detectors:` must be a mapping of detector name => settings")
        for (name, block) in dets
            block isa AbstractDict || error("config: detectors.$name must be a mapping of settings")
            check(block, DETECTOR_CONFIG_KEYS, "detectors.$name", TOP_LEVEL_CONFIG_KEYS,
                  "a top-level key: move it out of the detector entry")
        end
        active = get(config, "active_detectors", String[])
        if !(active isa AbstractVector)
            push!(problems, "active_detectors must be a list, e.g. [HD] (got $(repr(active)))")
        elseif isempty(active)
            push!(problems, "active_detectors is empty: name at least one entry of detectors:")
        else
            for name in unique(active)
                count(==(name), active) > 1 && push!(problems, "active_detectors lists '$name' more than once")
            end
            for name in active
                haskey(dets, name) || push!(problems,
                    "active_detectors names '$name', which has no entry under detectors:" *
                    _did_you_mean(string(name), string.(keys(dets))))
            end
        end
    end
    isempty(problems) || error("Invalid config:\n  " * join(problems, "\n  "))
    # Obsolete keys: proposal_matrix and binnedNight warn where they are read; nuFast: true errors
    # in main(), so only the harmless `nuFast: false` is reported here.
    get(config, "nuFast", true) == false &&
        @warn "config key nuFast is obsolete (the nuFast backend was removed); please delete it"
    # Asimov is the only data mode; `Asimov: false` used to be silently ignored.
    asimov = get(config, "Asimov", true)
    asimov === false &&
        error("Asimov: false (fit to real data from data_file) is not implemented; set Asimov: true")
    asimov === true || error("Asimov must be true or false (got $(repr(asimov)))")
    return nothing
end

"""
    parse_detector_block(block, name) -> NamedTuple

Parse one entry from the `detectors:` YAML section (or the flat top-level config
for backward compatibility) into a per-detector named tuple consumed by setup.jl.
"""
function parse_detector_block(block::Dict, name::String; defaults::Dict = Dict{String,Any}())
    sc = get(block, "singleChannel", false)
    cc_mode = (sc == false || sc == "CC")
    es_mode = (sc == false || sc == "ES")
    inclusive_analysis      = Bool(get(block, "inclusiveMode", false))
    semi_inclusive_analysis = Bool(get(block, "semiInclusiveMode", false))
    if semi_inclusive_analysis && inclusive_analysis
        error("inclusiveMode and semiInclusiveMode cannot both be true for detector $name")
    end
    if semi_inclusive_analysis && (!cc_mode || !es_mode)
        error("semiInclusiveMode requires singleChannel: false for detector $name")
    end
    # The forward/backward split needs the scattering angle: without it every ES event stays in
    # the ES channel while the backward ones are also added to CC as mis-ID (counted twice).
    if semi_inclusive_analysis && !Bool(block["use_scattering_info"])
        error("semiInclusiveMode requires use_scattering_info: true for detector $name")
    end
    return (
        name                    = name,
        ES_normalisation        = Float64(block["ES_exposure"]),
        CC_normalisation        = Float64(block["CC_exposure"]),
        nue_filepath            = block["reconstruction_sample_ES_nue"],
        other_filepath          = block["reconstruction_sample_ES_nuother"],
        angular_filepath        = block["reconstruction_sample_ES_angle"],
        CC_filepath             = block["reconstruction_sample_CC"],
        ES_filepaths_BG         = block["ES_background_files"],
        CC_filepaths_BG         = block["CC_background_files"],
        ES_bg_norms             = Float64.(block["ES_background_normalisations"]),
        CC_bg_norms             = Float64.(block["CC_background_normalisations"]),
        ES_bg_sys               = Float64.(block["ES_background_systematics"]),
        CC_bg_sys               = Float64.(block["CC_background_systematics"]),
        Ereco_bins_ES           = threshold_binning(block, "ES", name),
        Ereco_bins_CC           = threshold_binning(block, "CC", name),
        E_threshold             = (ES=block["Ereco_min_ES"]*1e-3, CC=block["Ereco_min_CC"]*1e-3),
        cos_scatter_bins        = (bin_number=Int(block["nBins_cos_scatter"]), min=0.0, max=1.0),
        CC_mode                 = cc_mode,
        ES_mode                 = es_mode,
        singleChannel           = sc,
        angular_reco            = Bool(block["use_scattering_info"]),
        angular_cos_cut         = Float64(get(block, "ES_cos_cut", -1.0)),
        inclusive_analysis      = inclusive_analysis,
        semi_inclusive_analysis = semi_inclusive_analysis,
        # Normalisation conventions of the MC inputs (defaults reproduce the earlier
        # hard-coded values; see docs/configuration_reference.md).
        module_mass_kt          = Float64(get(block, "module_mass_kt", 10.0)),
        ES_bg_generated_events  = Float64(get(block, "ES_bg_generated_events", 50e6)),
        ES_bg_face_areas_m2     = Float64.(get(block, "ES_bg_face_areas_m2", [2 * 12.0 * 64.0, 2 * 12.0 * 12.0])),
        CC_bg_reference_flux    = Float64(get(block, "CC_bg_reference_flux", 2.2e-6)),
        # MC statistical uncertainty of the predictions (src/likelihoods/mcstat.jl); top-level
        # keys set the default for every detector.
        mc_stat_likelihood      = mc_stat_key(block, defaults, name),
        mc_stat_day_night       = mc_stat_dn_key(block, defaults, name),
    )
end

function save_settings_to_file(filename::String)
    # Open the file for writing
    open(filename, "w") do file
        # Write header
        write(file, "=== Configuration Settings ===\n\n")

        # General settings
        write(file, "----- General -----\n")
        write(file, "Fast mode: $fast\n")
        write(file, "Night Earth crossing (NightOsc): $nightOsc\n")
        write(file, "AD backend: $ad_backend\n")
        write(file, "MCMC seed: $(something(mcmc_seed, "random"))\n")
        write(file, "Earth uncertainty enabled: $earthUncertainty\n")
        write(file, "CC xsec file: $cc_xsec_file\n")
        write(file, "CC xsec scale: $CC_xsec_scale\n\n")

        # Solar and Earth models
        write(file, "----- Solar & Earth Models -----\n")
        write(file, "Solar model file: $solarModelFile\n")
        write(file, "Flux model file: $flux_file_path\n")
        write(file, "Earth model file: $earthModelFile\n")
        write(file, "Earth uncertainty file: $earthUncertaintyFile\n\n")

        # Shared binning
        write(file, "----- Shared Binning -----\n")
        write(file, "TPC threshold: $(tpc_threshold*1e3) MeV\n")
        write(file, "Etrue bins: $(Etrue_bins)\n")
        write(file, "cos(z) bins: $cosz_bins\n\n")

        # Per-detector settings
        write(file, "----- Detectors -----\n")
        for (dname, det) in detector_configs
            write(file, "  [$dname]\n")
            write(file, "    singleChannel: $(det.singleChannel)\n")
            write(file, "    inclusiveMode: $(det.inclusive_analysis)\n")
            write(file, "    semiInclusiveMode: $(det.semi_inclusive_analysis)\n")
            write(file, "    ES_exposure: $(det.ES_normalisation)\n")
            write(file, "    CC_exposure: $(det.CC_normalisation)\n")
            write(file, "    Ereco bins (ES): $(det.Ereco_bins_ES)\n")
            write(file, "    Ereco bins (CC): $(det.Ereco_bins_CC)\n")
            write(file, "    E_threshold: $(det.E_threshold)\n")
            write(file, "    cos_scatter_bins: $(det.cos_scatter_bins)\n")
            write(file, "    ES background files: $(det.ES_filepaths_BG)\n")
            write(file, "    CC background files: $(det.CC_filepaths_BG)\n")
            write(file, "    ES background norms: $(det.ES_bg_norms)\n")
            write(file, "    CC background norms: $(det.CC_bg_norms)\n")
        end
        write(file, "\n")

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
    end
end


"""
    write_input_report(prefix, detectors)

`--plot` (plotMode): plot each detector's inputs from the SETUP outputs at `prefix` with
utils/plotInputs.py, into `<prefix>_<det>.pdf`. PYTHON chooses the interpreter (default python3).
"""
function write_input_report(prefix, detectors)
    python = get(ENV, "PYTHON", "python3")
    for d in detectors
        run(`$python $(joinpath(@__DIR__, "..", "utils", "plotInputs.py")) $prefix --detector $d -o $(prefix)_$(d)`)
        @logmsg Output "Input plots: $(prefix)_$(d).pdf"
    end
end

# Main script logic
function main()
    # --plot (plotMode): plot instead of run. The likelihood is built (RunMode SETUP) under the
    # prefix <outFile>_inputs, the inputs of each detector are plotted, and no fit is run.
    global plotMode = "--plot" in ARGS
    args = filter(!=("--plot"), ARGS)
    if length(args) > 0
        # Use the first argument as the path to the YAML file
        yaml_file = args[1]
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
    validate_config_keys(config)   # unknown (e.g. misspelled) keys are an error, not a default
    if plotMode
        config["RunMode"] = "SETUP"
        config["outFile"] = string(config["outFile"], "_inputs")
        isfile(config["outFile"] * "_config.yaml") &&
            error("plotMode: $(config["outFile"])_* already exist; delete the previous plots first")
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

    # CC ν_e–⁴⁰Ar total cross-section table (read and validated in xsec.jl)
    global cc_xsec_file = String(get(config, "cc_xsec_file", DEFAULT_CC_XSEC_FILE))

    # True TPC detection threshold (GeV) — lower integration limit of the ES cross-section.
    # Must match the MC generation threshold. Distinct from the Ereco_min_ES analysis cut.
    global tpc_threshold = Float64(get(config, "TPC_threshold", 0.2)) * 1e-3

    # Shared binning (same for all detectors)
    global Etrue_bins = (bin_number=config["nBins_Etrue"], min=config["range_Etrue"][1]*1e-3, max=config["range_Etrue"][2]*1e-3)
    global cosz_bins  = (bin_number=config["nBins_cosz"], min=-1, max=0)
    # Horizon-refined cos(z) binning: nBins_cosz_fine graded bins in [cosz_fine_edge, 0]
    # (concentrated where the day-night regeneration first-peak lives), the rest PREM-aligned.
    # cosz_fine_edge ≈ first-peak cos(z) for the slowest frequency (prior_dm2_21 lower bound).
    global nBins_cosz_fine = get(config, "nBins_cosz_fine", 0)   # 0 ⇒ legacy uniform-piecewise grid
    global cosz_fine_edge  = get(config, "cosz_fine_edge", -0.12)

    # Per-detector configuration blocks
    if haskey(config, "detectors")
        active = get(config, "active_detectors", sort(collect(keys(config["detectors"]))))
        global detector_configs = OrderedDict{String,Any}(
            name => parse_detector_block(config["detectors"][name], name; defaults = config)
            for name in active
        )
    else
        # Backward-compatible fallback: read flat keys as a single "default" detector
        global detector_configs = OrderedDict{String,Any}(
            "default" => parse_detector_block(config, "default")
        )
    end

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
    global fix_cc_xsec = Bool(get(config, "fix_cc_xsec", false))
    global prior_cc_xsec_norm = string_to_distribution(
        get(config, "prior_cc_xsec_norm", "truncated(Normal(1.0, 0.1), 0.1, 2.0)")
    )
    global prior_cc_xsec_tilt = string_to_distribution(
        get(config, "prior_cc_xsec_tilt", "Normal(0.0, 0.1)")
    )
    global prior_cc_xsec_curv = string_to_distribution(
        get(config, "prior_cc_xsec_curv", "Normal(0.0, 0.05)")
    )
    load_earth_normalisation_prior(config["earth_normalisation_prior_file"];
                                   required = Bool(config["earth_potential_uncertainties"]))

    # LLH parameters
    global llhBins = config["llh_bins"]

    # Derived parameters
    global thinning = config["thinning"]

    # Asimov values
    global sin2_th12_true = config["true_sin2_th12"]
    global sin2_th13_true = config["true_sin2_th13"]
    global dm2_21_true = config["true_dm2_21"]
    global integrated_8B_flux_true = Float64(get(config, "true_8B_flux", mean(prior_8B_flux)))
    global integrated_HEP_flux_true = config["true_integrated_HEP_flux"]
    global cc_xsec_norm_true = Float64(get(config, "true_cc_xsec_norm", 1.0))

    # Fast mode?
    global fast = config["fastFit"]
    # The nuFast C++ backend was removed (Oct 2026); old configs may still carry the key.
    get(config, "nuFast", false) && error("nuFast: true is no longer supported (the nuFast backend was removed); " *
                                          "remove the key or set it to false")
    # Night-time Earth crossing: NightOsc (exact cos z / E_true bin averages, own-segment
    # potentials). false restores the legacy 3-flavour per-path calculation (Osc.NumOsc).
    if haskey(config, "binnedNight")
        haskey(config, "nightOsc") && error("config sets both nightOsc and its old name binnedNight; keep nightOsc")
        @warn "config key binnedNight is now nightOsc; please rename it"
    end
    global nightOsc = Bool(get(config, "nightOsc", get(config, "binnedNight", true)))
    # Automatic differentiation for HMC: "Enzyme" (reverse mode through NightOsc's custom rule:
    # cost independent of the number of parameters, ~2x faster than ForwardDiff on HD with 8
    # NightOsc inputs, more with more Earth parameters; needs nightOsc) or "ForwardDiff" (cost grows
    # with the number of NightOsc inputs). Default: Enzyme with nightOsc, ForwardDiff without.
    global ad_backend_defaulted = !haskey(config, "ad_backend")
    global ad_backend = String(get(config, "ad_backend", nightOsc ? "Enzyme" : "ForwardDiff"))
    ad_backend in ("ForwardDiff", "Enzyme") ||
        error("ad_backend must be \"ForwardDiff\" or \"Enzyme\", got \"$ad_backend\"")
    # NUTS tree-depth cap (each MCMC step uses at most 2^hmc_max_depth gradients)
    global hmc_max_depth = Int(get(config, "hmc_max_depth", 4))
    1 <= hmc_max_depth <= 12 || error("hmc_max_depth must be between 1 and 12, got $hmc_max_depth")
    # Optional MCMC RNG seed (integer) for reproducible chains; unset = random.
    global mcmc_seed = get(config, "mcmc_seed", nothing)

    # Uncertainties?
    global earthUncertainty = config["earth_potential_uncertainties"]

    # Previous file?
    global prevFile = haskey(config, "prevFile") ? config["prevFile"] : nothing
    # Warm start: outFile of an earlier run whose posterior sets the mass matrix and the chain
    # starting points of a new run (src/warmStart.jl). Ignored when resuming (prevFile).
    global warm_start_chain = get(config, "warm_start_chain", nothing)

    # Fixed CC cross-section scale factor (1.0 = the cc_xsec_file table as given). DEAP-3600's 2.4
    # (arXiv:2605.12769) is relative to MARLEY 1.2 / BGG2009, ≈ 2.66 relative to MARLEY 2.0.0.
    global CC_xsec_scale = Float64(get(config, "CC_xsec_scale", 1.0))

    allowed_modes = ["LLH", "MCMC", "derived", "PROFILE", "SETUP"]
    global run_mode = config["RunMode"]

    if run_mode ∉ allowed_modes
        error("Invalid RunMode: '$run_mode'. Must be one of: $(join(allowed_modes, ", "))")
    end
    
    # Determine which script to run based on RunMode
    script_to_run = Dict(
        "LLH" => "llhScan.jl",
        "MCMC" => "mcmc.jl",
        "derived" => "derive_variables_from_chain.jl",
        "PROFILE" => "profiling.jl",
        "SETUP" => "setupOnly.jl"
    )[run_mode]

    script_path = joinpath(@__DIR__, script_to_run)  # Reference script in the same directory

    # A new MCMC run refuses an existing chain file (runMCMCbatch); check it before the run record
    # is written, so an aborted run does not displace the record of the chain already there.
    run_mode == "MCMC" && prevFile === nothing && isfile(outFile * ".jld2") &&
        error("Output file $(outFile).jld2 already exists. Delete it or choose a different outFile before starting a new run.")

    # The warm-start chain is read after the (long) setup: check that it exists before.
    if run_mode == "MCMC" && prevFile === nothing && warm_start_chain !== nothing
        wf = endswith(warm_start_chain, ".jld2") ? warm_start_chain : warm_start_chain * ".jld2"
        isfile(wf) || error("warm_start_chain: $wf not found (give the earlier run's outFile)")
        warn_if_xsec_basis_differs(wf, "warm_start_chain")
    end
    run_mode == "MCMC" && prevFile !== nothing && isfile(prevFile * ".jld2") &&
        warn_if_xsec_basis_differs(prevFile * ".jld2", "resumed chain")

    # Derived mode reads and writes prevFile.jld2: fail before setup (and the run record) without it.
    if run_mode == "derived"
        prevFile === nothing && error("RunMode: derived needs prevFile (the chain prefix, without .jld2)")
        isfile(prevFile * ".jld2") || error("RunMode: derived: chain file $(prevFile).jld2 not found " *
                                            "(prevFile is the prefix, without .jld2)")
        warn_if_xsec_basis_differs(prevFile * ".jld2", "derived-mode chain")
    end

    # Resolved config + provenance, written before the run so a crashed or killed run keeps it.
    # Derived mode writes into prevFile.jld2 (outFile is unused), so its record goes next to it.
    record_prefix = run_mode == "derived" ? prevFile * "_derived" : outFile
    @logmsg Output ("Run record: " * write_run_record(record_prefix, yaml_file, config))

    # include the corresponding script
    include(script_path)

    @logmsg Output ("$(outFile)_configSettings.txt")
    save_settings_to_file("$(outFile)_configSettings.txt")

    plotMode && write_input_report(outFile, collect(keys(detector_configs)))
end



# --- Begin Memory Tracking & Main Execution ---
# 
# @printf("Starting memory tracking...\n")
# mem_usages, times, stop_tracking = track_memory(0.1)

# @printf("Executing main()...\n")
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# @printf("Stopping memory tracking...\n")
# Stop the memory tracker and wait for the task to finish.
# stop_tracking()

