#=
derive_variables_from_chain.jl

Post-processing script for deriving additional quantities from MCMC chains.
This module calculates derived observables (like day-night asymmetries) from
the posterior samples of oscillation parameters.

Key Features:
- Loading and parsing of MCMC chain data from binary files
- Calculation of day-night asymmetries for ES and CC channels
- Progress tracking for large chain processing
- Serialization of derived quantities with original chain data
- Integration with the main analysis pipeline

The derived quantities are calculated by propagating each parameter sample
through the full detector simulation to obtain observable asymmetries.

Author: [Author name]
=#

using Serialization   # For binary data I/O
using ElasticArrays    # For dynamic array handling
using ProgressMeter    # For progress tracking
using JLD2            # For data file operations

include("../src/setup.jl")


function parse_info_file(infoFile::String)
    """
    Parse the info file that describes the structure of the binary MCMC data.
    
    The info file contains metadata about parameter names and data structure
    that allows proper reconstruction of the parameter samples.
    
    Returns:
    - entries: Ordered list of (index, variable_name) pairs
    - param_names: Vector of parameter names as symbols
    """
    index_map = Dict{Int,String}()
    param_names = Symbol[]

    for (line_idx, line) in enumerate(eachline(infoFile))
        line_str = strip(line)

        # Skip comments and blank lines
        if isempty(line_str) || startswith(line_str, "#")
            continue
        end

        # Match index and variable name patterns
        m = match(r"^(\d+):\s*(\S+)", line_str)
        if m !== nothing
            idx = parse(Int, m.captures[1])
            varName = m.captures[2]
            index_map[idx] = varName
            continue
        end

        # Extract parameter names if present
        m_param = match(r"Parameter names:\s*(.+)", line_str)
        if m_param !== nothing
            raw_names = split(m_param.captures[1], ',')
            param_names = Symbol.(strip.(raw_names))
        end
    end

    # Return sorted entries and parameter names
    sorted_keys = sort(collect(keys(index_map)))
    entries = [(i, index_map[i]) for i in sorted_keys]
    return (entries=entries, param_names=param_names)
end


function loadAllBatches(binFile::String, infoFile::String)
    # Parse info file to get parameter structure
    parsed_info = parse_info_file(infoFile)
    param_names = parsed_info.param_names

    # Initialize accumulators
    param_data_accum = Dict{Symbol,Vector}()
    for pname in param_names
        param_data_accum[pname] = Vector()
    end

    weights_accum = Int[]
    stepno_accum = Int[]
    chainid_accum = Int[]

    # Read and accumulate batches
    open(binFile, "r") do io
        while !eof(io)
            batch = deserialize(io)  # (param_data, weights, stepno, chainid)
            param_data, weights, stepno, chainid = batch

            # Accumulate each parameter's values
            for (pname, vals) in param_data
                if !haskey(param_data_accum, pname)
                    param_data_accum[pname] = Any[]
                end
                append!(param_data_accum[pname], vals)
            end

            # Accumulate metadata
            append!(weights_accum, weights)
            append!(stepno_accum, stepno)
            append!(chainid_accum, chainid)
        end
    end

    return param_data_accum, weights_accum, stepno_accum, chainid_accum
end


function saveDerivedChain(samples, derived, weights, stepno, chainid)
    # write file
    fileName = outFile * "_mcmc.bin"
    try
        open(fileName, isfile(fileName) ? "a" : "w") do io
            serialize(io, (samples, derived, weights, stepno, chainid))
        end

    catch err
        @error "Failed to write MCMC data to $fileName.\nError: $err"
        return
    end

    # write info file
    infoFileName = outFile * "_info.txt"
    open(infoFileName, "w") do io
        println(io, "# Saved fields in binary (in order):")
        println(io, "1: param_data (Dict{Symbol,Any})")
        println(io, "   Parameter names: ", join(string.(keys(samples)), ", "))
        println(io, "2: derived_quantities (Dict{Symbol,Any})")
        println(io, "   Derived quantity names: ", join(string.(keys(derived)), ", "))
        println(io, "3: weights (Vector)")
        println(io, "4: stepno  (Vector)")
        println(io, "5: chainid (Vector)")
    end
end


######## BEGIN EXECUTION ########

println(" ")

if prevFile === nothing
    error("Please specify an input chain")
else
    @logmsg Setup ("Calculating DN-asymmetry for chain *$(prevFile)*")
end

param_data, weights, stepno, chainid = loadAllBatches(prevFile * "_mcmc.bin", prevFile * "_info.txt")

param_fields = collect(keys(param_data))
n_total = length(param_data[param_fields[1]])

@logmsg Setup ("Processing $(n_total) parameter values")
println(" ")

ES_asymmetries = Float64[]
CC_asymmetries = Float64[]

# --- parameter running sums (scalar) ---
N = 0.0
sum_sin2 = 0.0; sum_sin2_2 = 0.0
sum_dm2  = 0.0; sum_dm2_2  = 0.0

# --- per-bin running sums for CC ---
S1_CCnight = nothing
S2_CCnight = nothing
Ssin2_CCnight = nothing
Sdm2_CCnight  = nothing

S1_CCday = nothing
S2_CCday = nothing
Ssin2_CCday = nothing
Sdm2_CCday  = nothing

# --- per-bin running sums for ES ---
S1_ESday = nothing
S2_ESday = nothing
Ssin2_ESday = nothing
Sdm2_ESday  = nothing

S1_ESnight = nothing
S2_ESnight = nothing
Ssin2_ESnight = nothing
Sdm2_ESnight  = nothing

# --- per-sample TOTAL-llh running sums (scalars) ---
S1_CCnight_tot = 0.0; S2_CCnight_tot = 0.0
Ssin2_CCnight_tot = 0.0; Sdm2_CCnight_tot = 0.0

S1_CCday_tot = 0.0; S2_CCday_tot = 0.0
Ssin2_CCday_tot = 0.0; Sdm2_CCday_tot = 0.0

S1_ESnight_tot = 0.0; S2_ESnight_tot = 0.0
Ssin2_ESnight_tot = 0.0; Sdm2_ESnight_tot = 0.0

S1_ESday_tot = 0.0; S2_ESday_tot = 0.0
Ssin2_ESday_tot = 0.0; Sdm2_ESday_tot = 0.0

# --- posterior predictives (accumulate weighted moments) ---
CCday_mean = nothing
CCnight_mean = nothing
ESday_mean = nothing
ESnight_mean = nothing

CCday_m2 = nothing
CCnight_m2 = nothing
ESday_m2 = nothing
ESnight_m2 = nothing

# Initialize progress bar
p = Progress(n_total,
             dt = 0.1,              # Update time
             showspeed=true,        # Show iterations per second
             color=:yellow)         # set color

for i in 1:n_total
    global N, sum_sin2, sum_sin2_2, sum_dm2, sum_dm2_2
    global S1_CCnight, S2_CCnight, Ssin2_CCnight, Sdm2_CCnight
    global S1_CCday,   S2_CCday,   Ssin2_CCday,   Sdm2_CCday
    global S1_ESnight, S2_ESnight, Ssin2_ESnight, Sdm2_ESnight
    global S1_ESday,   S2_ESday,   Ssin2_ESday,   Sdm2_ESday

    global S1_CCnight_tot, S2_CCnight_tot, Ssin2_CCnight_tot, Sdm2_CCnight_tot
    global S1_CCday_tot,   S2_CCday_tot,   Ssin2_CCday_tot,   Sdm2_CCday_tot
    global S1_ESnight_tot, S2_ESnight_tot, Ssin2_ESnight_tot, Sdm2_ESnight_tot
    global S1_ESday_tot,   S2_ESday_tot,   Ssin2_ESday_tot,   Sdm2_ESday_tot

    global CCday_mean, CCnight_mean, ESday_mean, ESnight_mean
    global CCday_m2,   CCnight_m2,   ESday_m2,   ESnight_m2

    temp_pars = NamedTuple{Tuple(param_fields)}((param_data[field][i] for field in param_fields))
    expectedRate_ES_day, expectedRate_CC_day, expectedRate_ES_night, expectedRate_CC_night, BG_ES_tot, BG_CC_tot =
        propagateSamples(unoscillatedSample, responseMatrices, temp_pars, solarModel, bin_edges, backgrounds)

    #-- Get asymmetry --#
    BG_CC_expected = sum(BG_CC_tot[index_CC:end])
    BG_ES_expected = sum(BG_ES_tot[index_ES:end])

    temp_CC_Ntot = sum(@view expectedRate_CC_night[:, index_CC:end]) - 0.5 * BG_CC_expected
    temp_CC_Dtot = sum(expectedRate_CC_day[index_CC:end]) - 0.5 * BG_CC_expected

    temp_ES_Ntot = sum(@view expectedRate_ES_night[:, index_ES:end]) - 0.5 * BG_ES_expected
    temp_ES_Dtot = sum(expectedRate_ES_day[index_ES:end]) - 0.5 * BG_ES_expected  # (kept consistent with index_ES)

    expected_asymm_CC = 2 * (temp_CC_Dtot - temp_CC_Ntot) / (temp_CC_Dtot + temp_CC_Ntot)
    expected_asymm_ES = 2 * (temp_ES_Dtot - temp_ES_Ntot) / (temp_ES_Dtot + temp_ES_Ntot)

    # Save the asymmetries
    push!(ES_asymmetries, expected_asymm_ES)
    push!(CC_asymmetries, expected_asymm_CC)

    (i % thinning == 0) || continue

    # Run diagnostics
    w = float(weights[i])

    # Pull the two parameters (assumes they exist in temp_pars)
    θ1 = temp_pars.sin2_th12
    θ2 = temp_pars.dm2_21

    # Lazy allocation on first diagnostic iteration
    if S1_CCnight === nothing
        # Per-bin llh accumulators
        # (allocate after we compute per-bin llh shapes below; but we also want PP shapes now)
        CCday_mean   = zeros(Float64, size(expectedRate_CC_day))
        CCnight_mean = zeros(Float64, size(expectedRate_CC_night))
        ESday_mean   = zeros(Float64, size(expectedRate_ES_day))
        ESnight_mean = zeros(Float64, size(expectedRate_ES_night))

        CCday_m2   = zeros(Float64, size(expectedRate_CC_day))
        CCnight_m2 = zeros(Float64, size(expectedRate_CC_night))
        ESday_m2   = zeros(Float64, size(expectedRate_ES_day))
        ESnight_m2 = zeros(Float64, size(expectedRate_ES_night))
    end

    # --- posterior predictive accumulation (weighted 1st/2nd moments) ---
    CCday_mean   .+= w .* expectedRate_CC_day
    CCnight_mean .+= w .* expectedRate_CC_night
    ESday_mean   .+= w .* expectedRate_ES_day
    ESnight_mean .+= w .* expectedRate_ES_night

    CCday_m2   .+= w .* (expectedRate_CC_day   .^ 2)
    CCnight_m2 .+= w .* (expectedRate_CC_night .^ 2)
    ESday_m2   .+= w .* (expectedRate_ES_day   .^ 2)
    ESnight_m2 .+= w .* (expectedRate_ES_night .^ 2)

    # Update scalar sums (parameters)
    N += w
    sum_sin2   += w * θ1
    sum_sin2_2 += w * θ1^2
    sum_dm2    += w * θ2
    sum_dm2_2  += w * θ2^2

    # --- Get per-bin loglik contributions ---
    rates = (
        ES_day    = expectedRate_ES_day,
        CC_day    = expectedRate_CC_day,
        ES_night  = expectedRate_ES_night,
        CC_night  = expectedRate_CC_night,
        BG_ES_tot = BG_ES_tot,
        BG_CC_tot = BG_CC_tot,
    )

    perbin = per_bin_llh(temp_pars, rates=rates)

    ℓ_CCnight = perbin.CC_night  # Matrix (Ereco × cosz)
    ℓ_CCday   = perbin.CC_day    # Vector (Ereco)
    ℓ_ESnight = perbin.ES_night
    ℓ_ESday   = perbin.ES_day

    # Allocate per-bin accumulators on first time we see ℓ shapes
    if S1_CCnight === nothing
        S1_CCnight = zeros(Float64, size(ℓ_CCnight))
        S2_CCnight = zeros(Float64, size(ℓ_CCnight))
        Ssin2_CCnight = zeros(Float64, size(ℓ_CCnight))
        Sdm2_CCnight  = zeros(Float64, size(ℓ_CCnight))

        S1_CCday = zeros(Float64, length(ℓ_CCday))
        S2_CCday = zeros(Float64, length(ℓ_CCday))
        Ssin2_CCday = zeros(Float64, length(ℓ_CCday))
        Sdm2_CCday  = zeros(Float64, length(ℓ_CCday))

        S1_ESnight = zeros(Float64, size(ℓ_ESnight))
        S2_ESnight = zeros(Float64, size(ℓ_ESnight))
        Ssin2_ESnight = zeros(Float64, size(ℓ_ESnight))
        Sdm2_ESnight  = zeros(Float64, size(ℓ_ESnight))

        S1_ESday = zeros(Float64, size(ℓ_ESday))
        S2_ESday = zeros(Float64, size(ℓ_ESday))
        Ssin2_ESday = zeros(Float64, size(ℓ_ESday))
        Sdm2_ESday  = zeros(Float64, size(ℓ_ESday))
    end

    # --- per-sample total llh for this draw ---
    ℓtot_CCnight = sum(ℓ_CCnight)
    ℓtot_CCday   = sum(ℓ_CCday)
    ℓtot_ESnight = sum(ℓ_ESnight)
    ℓtot_ESday   = sum(ℓ_ESday)

    # Accumulate per-sample total llh moments
    S1_CCnight_tot += w * ℓtot_CCnight
    S2_CCnight_tot += w * ℓtot_CCnight^2
    Ssin2_CCnight_tot += w * (ℓtot_CCnight * θ1)
    Sdm2_CCnight_tot  += w * (ℓtot_CCnight * θ2)

    S1_CCday_tot += w * ℓtot_CCday
    S2_CCday_tot += w * ℓtot_CCday^2
    Ssin2_CCday_tot += w * (ℓtot_CCday * θ1)
    Sdm2_CCday_tot  += w * (ℓtot_CCday * θ2)

    S1_ESnight_tot += w * ℓtot_ESnight
    S2_ESnight_tot += w * ℓtot_ESnight^2
    Ssin2_ESnight_tot += w * (ℓtot_ESnight * θ1)
    Sdm2_ESnight_tot  += w * (ℓtot_ESnight * θ2)

    S1_ESday_tot += w * ℓtot_ESday
    S2_ESday_tot += w * ℓtot_ESday^2
    Ssin2_ESday_tot += w * (ℓtot_ESday * θ1)
    Sdm2_ESday_tot  += w * (ℓtot_ESday * θ2)

    # --- per-bin accumulation (broadcasted) ---
    S1_CCnight .+= w .* ℓ_CCnight
    S2_CCnight .+= w .* (ℓ_CCnight .^ 2)
    Ssin2_CCnight .+= w .* (ℓ_CCnight .* θ1)
    Sdm2_CCnight  .+= w .* (ℓ_CCnight .* θ2)

    S1_CCday .+= w .* ℓ_CCday
    S2_CCday .+= w .* (ℓ_CCday .^ 2)
    Ssin2_CCday .+= w .* (ℓ_CCday .* θ1)
    Sdm2_CCday  .+= w .* (ℓ_CCday .* θ2)

    S1_ESnight .+= w .* ℓ_ESnight
    S2_ESnight .+= w .* (ℓ_ESnight .^ 2)
    Ssin2_ESnight .+= w .* (ℓ_ESnight .* θ1)
    Sdm2_ESnight  .+= w .* (ℓ_ESnight .* θ2)

    S1_ESday .+= w .* ℓ_ESday
    S2_ESday .+= w .* (ℓ_ESday .^ 2)
    Ssin2_ESday .+= w .* (ℓ_ESday .* θ1)
    Sdm2_ESday  .+= w .* (ℓ_ESday .* θ2)

    ProgressMeter.update!(p, i, showvalues = [(:Progress, @sprintf("%.2f%%", 100*i/n_total))])
end

# --- scalar parameter moments ---
mean_sin2 = sum_sin2 / N
var_sin2  = sum_sin2_2 / N - mean_sin2^2

mean_dm2 = sum_dm2 / N
var_dm2  = sum_dm2_2 / N - mean_dm2^2

function finalize_bin_stats(S1, S2, Sθ; N, meanθ, varθ)
    Eℓ   = S1 ./ N
    Varℓ = S2 ./ N .- (Eℓ .^ 2)

    Varℓ = max.(Varℓ, 0.0) # Clamp for statistics
    varθ = max.(varθ, 0.0) # Clamp for statistics

    Cov  = Sθ ./ N .- Eℓ .* meanθ
    Corr = Cov ./ sqrt.(Varℓ .* varθ)
    return (mean=Eℓ, var=Varℓ, cov=Cov, corr=Corr)
end

# --- scalar version for sample totals (same structure as finalize_bin_stats) ---
function finalize_scalar_stats(S1, S2, Sθ; N, meanθ, varθ)
    Eℓ   = S1 / N
    Varℓ = S2 / N - Eℓ^2

    Varℓ = max(Varℓ, 0.0)
    varθ = max(varθ, 0.0)

    Cov  = Sθ / N - Eℓ * meanθ
    Corr = Cov / sqrt(Varℓ * varθ)
    return (mean=Eℓ, var=Varℓ, cov=Cov, corr=Corr)
end

# CC night/day per-bin
ccnight_sin2 = finalize_bin_stats(S1_CCnight, S2_CCnight, Ssin2_CCnight; N=N, meanθ=mean_sin2, varθ=var_sin2)
ccnight_dm2  = finalize_bin_stats(S1_CCnight, S2_CCnight, Sdm2_CCnight;  N=N, meanθ=mean_dm2,  varθ=var_dm2)

ccday_sin2 = finalize_bin_stats(S1_CCday, S2_CCday, Ssin2_CCday; N=N, meanθ=mean_sin2, varθ=var_sin2)
ccday_dm2  = finalize_bin_stats(S1_CCday, S2_CCday, Sdm2_CCday;  N=N, meanθ=mean_dm2,  varθ=var_dm2)

# ES night/day per-bin
esnight_sin2 = finalize_bin_stats(S1_ESnight, S2_ESnight, Ssin2_ESnight; N=N, meanθ=mean_sin2, varθ=var_sin2)
esnight_dm2  = finalize_bin_stats(S1_ESnight, S2_ESnight, Sdm2_ESnight;  N=N, meanθ=mean_dm2,  varθ=var_dm2)

esday_sin2 = finalize_bin_stats(S1_ESday, S2_ESday, Ssin2_ESday; N=N, meanθ=mean_sin2, varθ=var_sin2)
esday_dm2  = finalize_bin_stats(S1_ESday, S2_ESday, Sdm2_ESday;  N=N, meanθ=mean_dm2,  varθ=var_dm2)

# --- CC/ES day/night TOTAL sample llh stats (scalar) ---
ccnight_tot_sin2 = finalize_scalar_stats(S1_CCnight_tot, S2_CCnight_tot, Ssin2_CCnight_tot; N=N, meanθ=mean_sin2, varθ=var_sin2)
ccnight_tot_dm2  = finalize_scalar_stats(S1_CCnight_tot, S2_CCnight_tot, Sdm2_CCnight_tot;  N=N, meanθ=mean_dm2,  varθ=var_dm2)

ccday_tot_sin2 = finalize_scalar_stats(S1_CCday_tot, S2_CCday_tot, Ssin2_CCday_tot; N=N, meanθ=mean_sin2, varθ=var_sin2)
ccday_tot_dm2  = finalize_scalar_stats(S1_CCday_tot, S2_CCday_tot, Sdm2_CCday_tot;  N=N, meanθ=mean_dm2,  varθ=var_dm2)

esnight_tot_sin2 = finalize_scalar_stats(S1_ESnight_tot, S2_ESnight_tot, Ssin2_ESnight_tot; N=N, meanθ=mean_sin2, varθ=var_sin2)
esnight_tot_dm2  = finalize_scalar_stats(S1_ESnight_tot, S2_ESnight_tot, Sdm2_ESnight_tot;  N=N, meanθ=mean_dm2,  varθ=var_dm2)

esday_tot_sin2 = finalize_scalar_stats(S1_ESday_tot, S2_ESday_tot, Ssin2_ESday_tot; N=N, meanθ=mean_sin2, varθ=var_sin2)
esday_tot_dm2  = finalize_scalar_stats(S1_ESday_tot, S2_ESday_tot, Sdm2_ESday_tot;  N=N, meanθ=mean_dm2,  varθ=var_dm2)

# --- posterior predictive finalize ---
CCday_pp_mean   = CCday_mean   ./ N
CCnight_pp_mean = CCnight_mean ./ N
ESday_pp_mean   = ESday_mean   ./ N
ESnight_pp_mean = ESnight_mean ./ N

CCday_pp_var   = max.(CCday_m2   ./ N .- CCday_pp_mean.^2,   0.0)
CCnight_pp_var = max.(CCnight_m2 ./ N .- CCnight_pp_mean.^2, 0.0)
ESday_pp_var   = max.(ESday_m2   ./ N .- ESday_pp_mean.^2,   0.0)
ESnight_pp_var = max.(ESnight_m2 ./ N .- ESnight_pp_mean.^2, 0.0)

# Save output
derived = Dict(
    :ES_asymmetry => ES_asymmetries,
    :CC_asymmetry => CC_asymmetries,

    # --- per-bin llh stats (existing) ---
    :CCnight_mean_llh => ccnight_sin2.mean,
    :CCnight_var_llh  => ccnight_sin2.var,
    :CCnight_corr_llh_sin2_th12 => ccnight_sin2.corr,
    :CCnight_corr_llh_dm2_21    => ccnight_dm2.corr,

    :CCday_mean_llh => ccday_sin2.mean,
    :CCday_var_llh  => ccday_sin2.var,
    :CCday_corr_llh_sin2_th12 => ccday_sin2.corr,
    :CCday_corr_llh_dm2_21    => ccday_dm2.corr,

    :ESnight_mean_llh => esnight_sin2.mean,
    :ESnight_var_llh  => esnight_sin2.var,
    :ESnight_corr_llh_sin2_th12 => esnight_sin2.corr,
    :ESnight_corr_llh_dm2_21    => esnight_dm2.corr,

    :ESday_mean_llh => esday_sin2.mean,
    :ESday_var_llh  => esday_sin2.var,
    :ESday_corr_llh_sin2_th12 => esday_sin2.corr,
    :ESday_corr_llh_dm2_21    => esday_dm2.corr,

    # --- per-sample TOTAL llh stats (scalars) ---
    :CCnight_tot_mean_llh => [ccnight_tot_sin2.mean],
    :CCnight_tot_var_llh  => [ccnight_tot_sin2.var],
    :CCnight_tot_corr_llh_sin2_th12 => [ccnight_tot_sin2.corr],
    :CCnight_tot_corr_llh_dm2_21    => [ccnight_tot_dm2.corr],

    :CCday_tot_mean_llh => [ccday_tot_sin2.mean],
    :CCday_tot_var_llh  => [ccday_tot_sin2.var],
    :CCday_tot_corr_llh_sin2_th12 => [ccday_tot_sin2.corr],
    :CCday_tot_corr_llh_dm2_21    => [ccday_tot_dm2.corr],

    :ESnight_tot_mean_llh => [esnight_tot_sin2.mean],
    :ESnight_tot_var_llh  => [esnight_tot_sin2.var],
    :ESnight_tot_corr_llh_sin2_th12 => [esnight_tot_sin2.corr],
    :ESnight_tot_corr_llh_dm2_21    => [esnight_tot_dm2.corr],

    :ESday_tot_mean_llh => [esday_tot_sin2.mean],
    :ESday_tot_var_llh  => [esday_tot_sin2.var],
    :ESday_tot_corr_llh_sin2_th12 => [esday_tot_sin2.corr],
    :ESday_tot_corr_llh_dm2_21    => [esday_tot_dm2.corr],

    # --- posterior predictive moments for rates ---
    :CCday_pp_mean   => CCday_pp_mean,
    :CCday_pp_var    => CCday_pp_var,
    :CCnight_pp_mean => CCnight_pp_mean,
    :CCnight_pp_var  => CCnight_pp_var,

    :ESday_pp_mean   => ESday_pp_mean,
    :ESday_pp_var    => ESday_pp_var,
    :ESnight_pp_mean => ESnight_pp_mean,
    :ESnight_pp_var  => ESnight_pp_var,

    ## SAVE SHAPES FOR DE-SERIALISATION
    :CCnight_shape => collect(size(ccnight_sin2.mean)),
    :ESnight_shape => collect(size(esnight_sin2.mean)),
    :CCday_shape   => collect(size(ccday_sin2.mean)),
    :ESday_shape   => collect(size(esday_sin2.mean)),

    :CCday_pp_shape   => collect(size(CCday_pp_mean)),
    :CCnight_pp_shape => collect(size(CCnight_pp_mean)),
    :ESday_pp_shape   => collect(size(ESday_pp_mean)),
    :ESnight_pp_shape => collect(size(ESnight_pp_mean)),
)


println(" ")
@logmsg Output ("True derived values: $(true_parameters[:ES_asymmetry]), $(true_parameters[:CC_asymmetry])")
println(" ")

saveDerivedChain(param_data, derived, weights, stepno, chainid)

@logmsg Output ("Saved derived quantities")