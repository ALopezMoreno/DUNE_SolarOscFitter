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

# --- helper: coerce any per-sample value into a 1D Vector{Float64} ---
function as_f64vec(v)
    if v isa Number
        return [Float64(v)]
    elseif v isa Tuple
        return Float64.(collect(v))              # tuple -> vector
    elseif v isa AbstractArray
        return Float64.(vec(collect(v)))         # array -> flattened vector
    else
        # fallback: try to iterate/collect
        return Float64.(collect(v))
    end
end

# --- helper: stack Vector{Vector{Float64}} into a matrix (nsamp x ndim) ---
function stack_rows(vs::Vector{Vector{Float64}})
    if isempty(vs)
        return zeros(Float64, 0, 0)
    end
    d = length(vs[1])
    # sanity check (avoid silent ragged stacking)
    for (i, v) in enumerate(vs)
        if length(v) != d
            throw(ArgumentError("Ragged parameter: entry $i has length $(length(v)) != $d"))
        end
    end
    mat = Matrix{Float64}(undef, length(vs), d)
    for i in 1:length(vs)
        @inbounds mat[i, :] = vs[i]
    end
    return mat
end

function batchesToJLD2(binFile::String, infoFile::String, outJLD2::String)
    param_data, weights, stepno, chainid = loadAllBatches(binFile, infoFile)

    jldopen(outJLD2, "w") do f
        for (name, values) in param_data
            try
                # values is a Vector of per-sample things (numbers, vectors, arrays, etc.)
                rows = Vector{Vector{Float64}}(undef, length(values))
                for i in eachindex(values)
                    rows[i] = as_f64vec(values[i])
                end

                mat = stack_rows(rows)  # (nsamp x ndim)

                if size(mat, 2) == 1
                    # store as 1D array of length nsamp
                    f[string(name)] = vec(mat)
                else
                    # store as separate columns name_1, name_2, ...
                    for j in 1:size(mat, 2)
                        f["$(name)_$j"] = mat[:, j]
                    end
                end
            catch err
                @warn "Failed to process parameter $name" exception=err typeof=typeof(values)
            end
        end

        f["weights"] = weights
        f["stepno"]  = stepno
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
