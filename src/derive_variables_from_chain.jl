using Serialization
using ElasticArrays
using ProgressMeter
using JLD2

include("../src/setup.jl")

"""
    parse_info_file(infoFile::String)

Parses an info file describing the contents of a serialized MCMC binary file.

Returns a named tuple:
- `entries`: Ordered list of (Int, String) representing top-level fields.
- `param_names`: Vector{Symbol} of parameter names inside `param_data`, if available.
"""
function parse_info_file(infoFile::String)
    index_map = Dict{Int,String}()
    param_names = Symbol[]

    for (line_idx, line) in enumerate(eachline(infoFile))
        line_str = strip(line)

        # Skip comments and blank lines
        if isempty(line_str) || startswith(line_str, "#")
            continue
        end

        # Match index and variable name
        m = match(r"^(\d+):\s*(\S+)", line_str)
        if m !== nothing
            idx = parse(Int, m.captures[1])
            varName = m.captures[2]
            index_map[idx] = varName
            continue
        end

        # Match parameter names if present
        m_param = match(r"Parameter names:\s*(.+)", line_str)
        if m_param !== nothing
            raw_names = split(m_param.captures[1], ',')
            param_names = Symbol.(strip.(raw_names))
        end
    end

    sorted_keys = sort(collect(keys(index_map)))
    entries = [(i, index_map[i]) for i in sorted_keys]
    return (entries=entries, param_names=param_names)
end



"""
    loadAllBatches(binFile::String, infoFile::String)

Reads all MCMC batches from the binary `binFile`, using the structure
described in the `infoFile`. Each batch is expected to be a tuple of:

  1. `param_data` (Dict{Symbol, Any}) — parameter name => list of values
  2. `weights`    (Vector{<:Number})
  3. `stepno`     (Vector{<:Integer})
  4. `chainid`    (Vector{<:Integer})

Returns:
  - param_data_accum :: Dict{Symbol, Vector}
  - weights_accum    :: Vector{<:Number}
  - stepno_accum     :: Vector{<:Integer}
  - chainid_accum    :: Vector{<:Integer}
"""
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

ES_asymmetries = []
CC_asymmetries = []

# Initialize progress bar
p = Progress(n_total,
             dt = 0.1,              # Update time
             showspeed=true,        # Show iterations per second
             color=:yellow)         # set color

for i in 1:n_total
    temp_pars = NamedTuple{Tuple(param_fields)}((param_data[field][i] for field in param_fields))
    expectedRate_ES_nue_day, expectedRate_ES_nuother_day, expectedRate_CC_day, expectedRate_ES_nue_night, expectedRate_ES_nuother_night, expectedRate_CC_night, BG_ES_tot, BG_CC_tot = propagateSamples(unoscillatedSample, responseMatrices, temp_pars, solarModel, bin_edges, backgrounds)

    #-- Get asymmetry --#
    BG_CC_expected = sum(BG_CC_tot[index_CC:end])
    BG_ES_expected = sum(BG_ES_tot[index_ES:end])

    temp_CC_Ntot = sum(@view expectedRate_CC_night[:, index_CC:end]) - 0.5 * BG_CC_expected
    temp_CC_Dtot = sum(expectedRate_CC_day[index_CC:end]) - 0.5 * BG_CC_expected

    temp_ES_Ntot = sum(@view expectedRate_ES_nue_night[:, index_ES:end]) + sum(@view expectedRate_ES_nuother_night[:, index_ES:end]) - 0.5 * BG_ES_expected
    temp_ES_Dtot = sum(expectedRate_ES_nue_day[index_CC:end]) + sum(expectedRate_ES_nuother_day[index_CC:end]) - 0.5 * BG_ES_expected

    expected_asymm_CC = 2 * (temp_CC_Dtot - temp_CC_Ntot) / (temp_CC_Dtot + temp_CC_Ntot)
    expected_asymm_ES = 2 * (temp_ES_Dtot - temp_ES_Ntot) / (temp_ES_Dtot + temp_ES_Ntot)

    # Save the asymmetries
    push!(ES_asymmetries, expected_asymm_ES)
    push!(CC_asymmetries, expected_asymm_CC)

    # Update progress bar with decimal percentage
    ProgressMeter.update!(p, i, showvalues = [(:Progress, @sprintf("%.2f%%", 100*i/n_total))])
end


# Save output
derived = Dict(
    :ES_asymmetry => ES_asymmetries,
    :CC_asymmetry => CC_asymmetries
)

println(" ")
@logmsg Output ("True derived values: $(true_parameters[:ES_asymmetry]), $(true_parameters[:CC_asymmetry])")
println(" ")

saveDerivedChain(param_data, derived, weights, stepno, chainid)

@logmsg Output ("Saved derived quantities")