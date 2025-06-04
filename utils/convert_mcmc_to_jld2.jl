using Serialization
using ElasticArrays
using JLD2


function parse_info_file(infoFile::String)
    index_map = Dict{Int, String}()
    param_names = Symbol[]
    derived_names = Symbol[]
    found_derived = false

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

        # Match derived quantity names if present
        m_derived = match(r"Derived quantity names:\s*(.+)", line_str)
        if m_derived !== nothing
            raw_names = split(m_derived.captures[1], ',')
            derived_names = Symbol.(strip.(raw_names))
            found_derived = true
        end
    end

    sorted_keys = sort(collect(keys(index_map)))
    entries = [(i, index_map[i]) for i in sorted_keys]
    
    # Return different structures based on whether we found derived quantities
    if found_derived
        return (entries=entries, param_names=param_names, derived_names=derived_names)
    else
        return (entries=entries, param_names=param_names)
    end
end



function loadAllBatches(binFile::String, infoFile::String)
    # Parse info file to get parameter structure
    parsed_info = parse_info_file(infoFile)
    param_names = parsed_info.param_names
    has_derived = hasproperty(parsed_info, :derived_names)
    derived_names = has_derived ? parsed_info.derived_names : Symbol[]

    # Prefix derived names
    prefixed_derived_names = [Symbol("derived_" * String(d)) for d in derived_names]

    # Initialize accumulators
    sample_data_accum = Dict{Symbol, Vector}()
    for pname in param_names
        sample_data_accum[pname] = Vector()
    end

    derived_data_accum = has_derived ? Dict{Symbol, Vector}() : nothing
    if has_derived
        for pname in prefixed_derived_names
            derived_data_accum[pname] = Vector()
        end
    end

    weights_accum = Int[]
    stepno_accum  = Int[]
    chainid_accum = Int[]

    # Read and accumulate batches
    open(binFile, "r") do io
        while !eof(io)
            if has_derived
                samples, derived, weights, stepno, chainid = deserialize(io)
            else
                samples, weights, stepno, chainid = deserialize(io)
                derived = nothing
            end

            # Accumulate samples
            for (pname, vals) in samples
                if !haskey(sample_data_accum, pname)
                    sample_data_accum[pname] = Any[]
                end
                append!(sample_data_accum[pname], vals)
            end

            # Accumulate renamed derived
            if has_derived && derived !== nothing
                for (pname, vals) in derived
                    prefixed_name = Symbol("derived_", pname)
                    if !haskey(derived_data_accum, prefixed_name)
                        derived_data_accum[prefixed_name] = Any[]
                    end
                    append!(derived_data_accum[prefixed_name], vals)
                end
            end

            # Accumulate metadata
            append!(weights_accum, weights)
            append!(stepno_accum, stepno)
            append!(chainid_accum, chainid)
        end
    end

    if has_derived
        sample_data = merge(sample_data_accum, derived_data_accum)
    else
        sample_data = sample_data_accum
    end

    return sample_data, weights_accum, stepno_accum, chainid_accum
end



"""
    batchesToJLD2(binFile::String, outJLD2::String)

Loads all batches from `binFile` and writes the merged results
to `outJLD2`, with each parameter in `param_data` saved as a 
separate top-level variable, along with:
  • weights  (Vector{Int})
  • stepno   (Vector{Int})
  • chainid  (Vector{Int})
"""
function batchesToJLD2(binFile::String, infoFile::String, outJLD2::String)
    param_data, weights, stepno, chainid = loadAllBatches(binFile, infoFile)
    
    jldopen(outJLD2, "w") do f
        for (name, values) in param_data
            try
                # Convert to Vector of Float64 vectors (force materialization and copy if needed)
                value_array = [Float64.(collect(v)) for v in values]
    
                # Now convert into a Matrix (samples × dimensions)
                mat = reduce(vcat, [v' for v in value_array])  # v' makes it a row vector
    
                if size(mat, 2) == 1
                    f[string(name)] = vec(mat)  # Save as 1D array if scalar
                else
                    for i in 1:size(mat, 2)
                        f["$(name)_$i"] = mat[:, i]
                    end
                end
            catch err
                @warn "Failed to process parameter $name" exception=err typeof=typeof(values)
            end
        end

        # Optionally save weights, stepno, etc.
        f["weights"] = weights
        f["stepno"] = stepno
        f["chainid"] = chainid
    end

    println("Successfully wrote JLD2 to ", outJLD2)
end

#################################
##########  EXECUTION  ##########
#################################

binFile  = ARGS[1]
infoFile = ARGS[2]
outFile  = ARGS[3]

batchesToJLD2(binFile, infoFile, outFile)
