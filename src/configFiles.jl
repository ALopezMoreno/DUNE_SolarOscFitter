#=
configFiles.jl — config inheritance through `base:` (one path or a list, relative to the file).

Bases are merged in order, the file on top: mappings merge at every depth, the later value wins,
anything else (scalar, list, null) is replaced. A file reached twice is applied once.
=#
using YAML

"""
    resolve_config(path) -> Dict{Any,Any}

Load the config at `path` with its `base:` chain resolved (see the file header). The `base`
key itself is not in the result. Errors on a missing file, a cycle, or a malformed `base:`.
"""
resolve_config(path::AbstractString) = first(_resolve_config(abspath(path), String[], Set{String}()))

"""
    config_files(path) -> Vector{String}

Absolute paths of every file `resolve_config(path)` reads, in merge order (bases first, `path`
last); a file reached twice is listed once, where it is applied.
"""
config_files(path::AbstractString) = last(_resolve_config(abspath(path), String[], Set{String}()))

# `stack`: the files being resolved (cycle check); `done`: files already applied, skipped if reached again
function _resolve_config(path::String, stack::Vector{String}, done::Set{String})
    path in stack && error("config base cycle: " * join(vcat(stack, path), " → "))
    path in done && return Dict{Any,Any}(), String[]
    isfile(path) || error(isempty(stack) ? "config file not found: $path" :
                          "config base not found: $path (named in $(stack[end]))")
    raw = YAML.load_file(path)
    raw === nothing && (raw = Dict{Any,Any}())          # an empty file
    raw isa AbstractDict || error("config $path: the top level must be a mapping of keys")
    own = Dict{Any,Any}(raw)
    bases = pop!(own, "base", String[])
    bases isa AbstractString && (bases = [bases])
    (bases isa AbstractVector && all(b -> b isa AbstractString, bases)) ||
        error("config $path: base must be a file path or a list of file paths (got $(repr(bases)))")
    merged, files = Dict{Any,Any}(), String[]
    for b in bases
        cfg, fs = _resolve_config(normpath(joinpath(dirname(path), b)), vcat(stack, path), done)
        merged = deep_merge(merged, cfg)
        append!(files, fs)
    end
    push!(files, path); push!(done, path)
    return deep_merge(merged, own), files
end

"""
    deep_merge(a, b) -> Dict{Any,Any}

`b` on top of `a`: keys present in both whose values are both mappings are merged recursively;
otherwise `b`'s value replaces `a`'s. Neither argument is modified.
"""
function deep_merge(a::AbstractDict, b::AbstractDict)
    out = Dict{Any,Any}(a)
    for (k, v) in b
        out[k] = haskey(out, k) && out[k] isa AbstractDict && v isa AbstractDict ? deep_merge(out[k], v) : v
    end
    return out
end
