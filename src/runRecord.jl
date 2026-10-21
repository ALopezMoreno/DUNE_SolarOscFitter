#=
runRecord.jl — the run record: the resolved configuration and its provenance, saved with every run.

`write_run_record(outFile, yaml_file, config)` is called by readConfig.jl's main() once the config
has been parsed (globals set) and before the RunMode script runs, so a run that crashes or is
killed still leaves its record. It writes `<outFile>_config.yaml` with three sections:

  provenance:        config file path and SHA-256 (and of its `base:` files), git commit / branch /
                     dirty state, Julia version, host, threads, timestamp, RunMode
  defaults_applied:  the keys absent from the YAML, with the values readConfig.jl used for them
  config:            the resolved configuration = the YAML as loaded + defaults_applied.
                     `YAML.load_file(record)["config"]` is a complete config file for a re-run.

An existing record is never overwritten: it is renamed to `<outFile>_config.old<k>.yaml`.
Derived mode writes into `<prevFile>.jld2`, so main() passes `<prevFile>_derived` as the prefix.
Input files are pinned by SHA-256 (`provenance: input_sha256`): most of inputs/ is gitignored, so
the git commit alone does not identify them.
MCMC runs also store the same text in the chain file (key `run_config`, see store_run_record!).

`defaults_applied` is built from the globals readConfig.jl set, so its key lists below must
follow the `get(config, key, default)` calls there; tests/test_run_record.jl checks that every
such key is covered.
=#
using YAML, Dates, SHA, JLD2
using DataStructures: OrderedDict
include(joinpath(@__DIR__, "configFiles.jl"))   # resolve_config, config_files (`base:` inheritance)

# Top-level keys with a default in readConfig.jl main() → value used (in config-file units),
# as thunks: only the defaults actually applied are evaluated. `nothing` (mcmc_seed) means
# "not set: random seed".
function _toplevel_defaults_used()
    return OrderedDict{String,Function}(
        "TPC_threshold"      => () -> tpc_threshold * 1e3,              # MeV in the config, GeV internally
        "nBins_cosz_fine"    => () -> nBins_cosz_fine,
        "cosz_fine_edge"     => () -> cosz_fine_edge,
        "fix_cc_xsec"        => () -> fix_cc_xsec,
        "prior_cc_xsec_norm" => () -> "truncated(Normal(1.0, 0.1), 0.1, 2.0)",
        "prior_cc_xsec_tilt" => () -> "Normal(0.0, 0.1)",
        "prior_cc_xsec_curv" => () -> "Normal(0.0, 0.05)",
        "true_8B_flux"       => () -> integrated_8B_flux_true,
        "true_cc_xsec_norm"  => () -> cc_xsec_norm_true,
        "Asimov"             => () -> true,                              # the only data mode
        "nightOsc"           => () -> nightOsc,
        "ad_backend"         => () -> ad_backend,
        "hmc_max_depth"      => () -> hmc_max_depth,
        "mcmc_seed"          => () -> mcmc_seed,
        "warm_start_chain"   => () -> warm_start_chain,                  # nothing = cold start
        "CC_xsec_scale"      => () -> CC_xsec_scale,
        "cc_xsec_file"       => () -> cc_xsec_file,
    )
end

# Per-detector keys with a default in parse_detector_block → value used.
function _detector_defaults_used(det)
    return OrderedDict{String,Function}(
        "singleChannel"          => () -> det.singleChannel,
        "inclusiveMode"          => () -> det.inclusive_analysis,
        "semiInclusiveMode"      => () -> det.semi_inclusive_analysis,
        "ES_cos_cut"             => () -> det.angular_cos_cut,
        "module_mass_kt"         => () -> det.module_mass_kt,
        "ES_bg_generated_events" => () -> det.ES_bg_generated_events,
        "ES_bg_face_areas_m2"    => () -> collect(det.ES_bg_face_areas_m2),
        "CC_bg_reference_flux"   => () -> det.CC_bg_reference_flux,
        "mc_stat_likelihood"     => () -> det.mc_stat_likelihood,
        "mc_stat_day_night"      => () -> det.mc_stat_day_night,
    )
end

# Keys that have a default but are deliberately not recorded: absent means "off/none" and the
# value has no effect on the run (prevFile: no resume; nuFast/binnedNight: removed/renamed keys;
# proposal_matrix: ignored).
const RUN_RECORD_UNRECORDED_KEYS = ("prevFile", "nuFast", "binnedNight", "proposal_matrix")

"""
    defaults_applied(config, detector_configs) -> OrderedDict

The keys absent from `config` with the values readConfig.jl used for them. Per-detector
defaults go under `detectors: <name>:` (or at the top level for a flat, detector-less config).
"""
function defaults_applied(config::AbstractDict, dets::AbstractDict)
    out = OrderedDict{String,Any}()
    for (k, v) in _toplevel_defaults_used()
        # an old config naming nightOsc binnedNight keeps that key (both together are an error)
        (haskey(config, k) || (k == "nightOsc" && haskey(config, "binnedNight"))) && continue
        out[k] = v()
    end
    if haskey(config, "detectors")
        haskey(config, "active_detectors") || (out["active_detectors"] = collect(keys(dets)))
        per = OrderedDict{String,Any}()
        for (name, det) in dets
            block = config["detectors"][name]
            d = OrderedDict{String,Any}(k => v() for (k, v) in _detector_defaults_used(det) if !haskey(block, k))
            isempty(d) || (per[name] = d)
        end
        isempty(per) || (out["detectors"] = per)
    else
        for det in values(dets), (k, v) in _detector_defaults_used(det)   # the single "default" detector
            haskey(config, k) || (out[k] = v())
        end
    end
    return out
end

# Recursive merge (defaults into a copy of the loaded config); keys sorted for readability.
_sorted(x) = x
_sorted(d::AbstractDict) = OrderedDict{String,Any}(string(k) => _sorted(d[k]) for k in sort(collect(keys(d)); by=string))
_sorted(v::AbstractVector) = Any[_sorted(x) for x in v]
function _merge_defaults(config::AbstractDict, defaults::AbstractDict)
    out = Dict{String,Any}(string(k) => v for (k, v) in config)
    for (k, v) in defaults
        out[k] = (v isa AbstractDict && haskey(out, k) && out[k] isa AbstractDict) ? _merge_defaults(out[k], v) : v
    end
    return out
end

"""
    resolved_config(config, detector_configs) -> OrderedDict

The loaded YAML with every default readConfig.jl applied filled in (keys sorted).
"""
resolved_config(config, dets) = _sorted(_merge_defaults(config, defaults_applied(config, dets)))

function _git(args...; dir)
    try
        return strip(read(pipeline(`git -C $dir $args`; stderr=devnull), String))
    catch
        return nothing
    end
end

"""
    run_provenance(yaml_file; run_mode, repo) -> OrderedDict

Where the run came from: config file (absolute path, SHA-256), git commit, branch and whether
tracked files differ from that commit (untracked files are ignored), Julia version, host,
threads, RunMode and a local timestamp. Git fields are "unknown" outside a git checkout.
"""
function run_provenance(yaml_file::AbstractString; run_mode = "unknown", repo::AbstractString = normpath(joinpath(@__DIR__, "..")))
    commit = _git("rev-parse", "HEAD"; dir=repo)
    status = commit === nothing ? nothing : _git("status", "--porcelain", "--untracked-files=no"; dir=repo)
    branch = commit === nothing ? nothing : _git("rev-parse", "--abbrev-ref", "HEAD"; dir=repo)
    p = OrderedDict{String,Any}(
        "config_file"     => abspath(yaml_file),
        "config_sha256"   => isfile(yaml_file) ? bytes2hex(open(sha256, yaml_file)) : "unknown",
        "git_commit"      => something(commit, "unknown"),
        "git_branch"      => something(branch, "unknown"),
        "git_dirty"       => status === nothing ? "unknown" : !isempty(status),
    )
    # Files the config inherits from (`base:`), each with its SHA-256; absent for a flat config.
    bases = try config_files(yaml_file)[1:end-1] catch; String[] end
    isempty(bases) || (p["config_base_files"] = OrderedDict{String,Any}(f => bytes2hex(open(sha256, f)) for f in bases))
    status !== nothing && !isempty(status) && (p["git_dirty_files"] = [String(strip(l)) for l in split(status, '\n')])
    merge!(p, OrderedDict{String,Any}(
        "julia_version"   => string(VERSION),
        "hostname"        => gethostname(),
        "threads"         => Threads.nthreads(),
        "timestamp"       => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "run_mode"        => run_mode,
        "working_dir"     => pwd(),
        "cc_xsec_shape_basis" => CC_XSEC_SHAPE_BASIS,
    ))
    return p
end

# Meaning of the CC xsec tilt/curv parameters (SPEC 3.3). Change it whenever their meaning changes,
# so that a chain from before is flagged when it is resumed, warm-started from or derived.
const CC_XSEC_SHAPE_BASIS = "orthogonal-v1: norm·exp(tilt·p1 + curv·p2), SPEC 3.3"

"""
    chain_xsec_basis(chainfile) -> String or nothing

The `cc_xsec_shape_basis` recorded in a chain file's run record (`run_config`), or `nothing` if the
file has no run record or the record predates the key.
"""
function chain_xsec_basis(chainfile::AbstractString)
    text = JLD2.jldopen(f -> haskey(f, "run_config") ? f["run_config"] : nothing, chainfile, "r")
    text === nothing && return nothing
    return get(get(YAML.load(text), "provenance", Dict()), "cc_xsec_shape_basis", nothing)
end

"""
    warn_if_xsec_basis_differs(chainfile, role)

Warn when a chain that is reused (resume, warm start, derived) was made with another meaning of
the CC xsec tilt/curv parameters: its tilt/curv samples would be read in the current basis.
"""
@noinline function warn_if_xsec_basis_differs(chainfile::AbstractString, role::AbstractString)
    basis = try chain_xsec_basis(chainfile) catch; nothing end
    basis == CC_XSEC_SHAPE_BASIS && return nothing
    @warn "$role $chainfile was made with CC xsec tilt/curv basis $(something(basis, "unknown (older than the orthogonal basis)")), " *
          "not $(CC_XSEC_SHAPE_BASIS). If tilt/curv were free in that chain, their values mean something " *
          "else now; only chains with tilt = curv fixed at 0 carry over."
    return nothing
end

"""
    input_hashes(config) -> OrderedDict

SHA-256 of every existing file named by a string value of `config` (nested dicts and lists
included), keyed by the path as written in the config. A relative path that does not exist from
the working directory is looked up from the repository root, as `cc_xsec_file` is
(resolve_cc_xsec_path in xsec.jl); it is then keyed by that absolute path.
"""
function input_hashes(config; repo::AbstractString = normpath(joinpath(@__DIR__, "..")))
    paths = String[]
    collect_paths(x) = nothing
    function collect_paths(x::AbstractString)
        if isfile(x)
            push!(paths, x)
        elseif !isabspath(x) && isfile(joinpath(repo, x))
            push!(paths, normpath(joinpath(repo, x)))
        end
        return nothing
    end
    collect_paths(d::AbstractDict) = foreach(collect_paths, values(d))
    collect_paths(v::AbstractVector) = foreach(collect_paths, v)
    collect_paths(config)
    return OrderedDict{String,Any}(p => bytes2hex(open(sha256, p)) for p in sort(unique(paths)))
end

# Text of the last run record written in this process (also stored in MCMC chain files).
const RUN_RECORD_TEXT = Ref("")
const RUN_RECORD_STORED = Ref(false)

"""
    run_record_path(outFile) = "<outFile>_config.yaml"
"""
run_record_path(outFile::AbstractString) = outFile * "_config.yaml"

"""
    write_run_record(outFile, yaml_file, config; detectors = detector_configs) -> path

Write `<outFile>_config.yaml` (see the file header). An existing record is moved aside to
`<outFile>_config.old<k>.yaml` (first free k), never overwritten.
"""
function write_run_record(outFile::AbstractString, yaml_file::AbstractString, config::AbstractDict;
                          detectors::AbstractDict = detector_configs)
    path = run_record_path(outFile)
    isempty(dirname(path)) || mkpath(dirname(path))
    if isfile(path)
        k = 1
        while isfile("$(outFile)_config.old$k.yaml"); k += 1; end
        old = "$(outFile)_config.old$k.yaml"
        mv(path, old)
        @warn "Run record $path already existed; moved it to $old"
    end
    resolved = resolved_config(config, detectors)
    prov = run_provenance(yaml_file; run_mode = string(get(config, "RunMode", "unknown")))
    prov["input_sha256"] = input_hashes(resolved)
    record = OrderedDict{String,Any}(
        "provenance"       => prov,
        "defaults_applied" => _sorted(defaults_applied(config, detectors)),
        "config"           => resolved,
    )
    text = """
    # SolarOscFitter run record (src/runRecord.jl).
    # provenance: where this run came from. defaults_applied: keys absent from the config file and
    # the values used for them. config: the configuration actually used (file + defaults);
    # YAML.load_file(this)["config"] is a complete config for a re-run.
    """ * YAML.write(record)
    write(path, text)
    RUN_RECORD_TEXT[] = text
    RUN_RECORD_STORED[] = false
    return path
end

"""
    store_run_record!(f)

Store this process's run record in an open JLD2 file, once per process: as `run_config`, or
`run_config_<k>` if the file already holds one (a resumed run writing into the same file).
Readers only look at `batch_*` and `derived_*` keys, so this does not disturb them.
"""
function store_run_record!(f)
    (RUN_RECORD_STORED[] || isempty(RUN_RECORD_TEXT[])) && return nothing
    name, k = "run_config", 1
    while haskey(f, name); k += 1; name = "run_config_$k"; end
    f[name] = RUN_RECORD_TEXT[]
    RUN_RECORD_STORED[] = true
    return nothing
end
