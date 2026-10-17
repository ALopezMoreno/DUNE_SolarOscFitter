#=
Run record (src/runRecord.jl): every run writes <outFile>_config.yaml with the resolved
configuration (YAML + the defaults readConfig.jl applied) and its provenance. Unit checks here;
tests/derived_mode.jl checks the record of a real run against the globals it used.
=#
using Test, YAML, JLD2
using DataStructures: OrderedDict

isdefined(Main, :write_run_record) || include(joinpath(@__DIR__, "..", "src", "readConfig.jl"))

@testset "run record" begin
    # Every key readConfig.jl reads with a default must be recorded when absent (or be listed as
    # deliberately unrecorded): a new `get(config, "key", default)` needs an entry in runRecord.jl.
    src = read(joinpath(@__DIR__, "..", "src", "readConfig.jl"), String)
    defaulted = Set(m.captures[1] for m in eachmatch(r"get\(\s*(?:config|block|defaults)\s*,\s*\"(\w+)\"", src))
    covered = union(Set(keys(_toplevel_defaults_used())),
                    Set(keys(_detector_defaults_used((;)))),       # thunks, not evaluated
                    Set(RUN_RECORD_UNRECORDED_KEYS), Set(["active_detectors"]))
    @test !isempty(defaulted)
    @test isempty(setdiff(defaulted, covered))

    tmp = mktempdir()
    block = Dict{String,Any}(
        "ES_exposure" => 1.0, "CC_exposure" => 2.0,
        "reconstruction_sample_ES_nue" => "a.csv", "reconstruction_sample_ES_nuother" => "b.csv",
        "reconstruction_sample_ES_angle" => "c.csv", "reconstruction_sample_CC" => "d.csv",
        "ES_background_files" => ["e.csv"], "CC_background_files" => String[],
        "ES_background_normalisations" => [1e-6], "CC_background_normalisations" => Float64[],
        "ES_background_systematics" => [0.1], "CC_background_systematics" => Float64[],
        "Ereco_min_ES" => 5.0, "range_Ereco_ES" => [0, 20], "nBins_Ereco_ES" => 10,
        "Ereco_min_CC" => 4.0, "range_Ereco_CC" => [0, 20], "nBins_Ereco_CC" => 8,
        "nBins_cos_scatter" => 5, "use_scattering_info" => false, "module_mass_kt" => 17.0)
    # all top-level defaulted keys set, so only the detector defaults are applied here
    config = Dict{String,Any}("outFile" => joinpath(tmp, "run"), "RunMode" => "SETUP",
                              "detectors" => Dict{String,Any}("D" => block), "prior_sin2_th12" => "Uniform(0.1, 0.5)")
    for k in keys(_toplevel_defaults_used())
        config[k] = k == "mcmc_seed" ? 42 : k == "ad_backend" ? "ForwardDiff" : k == "nightOsc" ? true :
                    startswith(k, "prior_") ? "Normal(0.0, 0.2)" : k == "fix_cc_xsec" ? false : 1.5
    end
    cfg_file = joinpath(tmp, "in.yaml"); YAML.write_file(cfg_file, config)
    config = YAML.load_file(cfg_file)
    det = parse_detector_block(config["detectors"]["D"], "D"; defaults = config)
    dets = OrderedDict{String,Any}("D" => det)

    path = write_run_record(config["outFile"], cfg_file, config; detectors = dets)
    @test path == joinpath(tmp, "run_config.yaml") && isfile(path)
    rec = YAML.load_file(path)
    @test Set(keys(rec)) == Set(["provenance", "defaults_applied", "config"])

    # provenance
    p = rec["provenance"]
    @test p["config_file"] == abspath(cfg_file)
    @test p["config_sha256"] == bytes2hex(open(sha256, cfg_file))
    @test p["julia_version"] == string(VERSION)
    @test p["run_mode"] == "SETUP"
    @test p["git_dirty"] in (true, false, "unknown")
    head = try strip(read(`git -C $(joinpath(@__DIR__, "..")) rev-parse HEAD`, String)) catch; "unknown" end
    @test p["git_commit"] == head
    @test !isempty(p["timestamp"])

    # defaults: only absent keys, with the values used; the given ones untouched
    dflt = rec["defaults_applied"]
    @test dflt["active_detectors"] == ["D"]
    @test dflt["detectors"]["D"]["mc_stat_likelihood"] == "Poisson"
    @test dflt["detectors"]["D"]["ES_bg_face_areas_m2"] == [2 * 12.0 * 64.0, 2 * 12.0 * 12.0]
    @test !haskey(dflt["detectors"]["D"], "module_mass_kt")
    @test !any(k -> haskey(dflt, k), keys(_toplevel_defaults_used()))

    # resolved config: the loaded values round-trip, and re-parsing it gives the detector used
    res = rec["config"]
    for (k, v) in config
        k == "detectors" && continue
        @test res[k] == v
    end
    for (k, v) in block
        @test res["detectors"]["D"][k] == v
    end
    @test parse_detector_block(res["detectors"]["D"], "D"; defaults = res) == det

    # an existing record is moved aside, not overwritten
    first_text = read(path, String)
    write_run_record(config["outFile"], cfg_file, config; detectors = dets)
    @test read(joinpath(tmp, "run_config.old1.yaml"), String) == first_text
    write_run_record(config["outFile"], cfg_file, config; detectors = dets)
    @test isfile(joinpath(tmp, "run_config.old2.yaml")) && isfile(path)

    # the same text in a chain file: once per process, never replacing an earlier run's
    chain = joinpath(tmp, "chain.jld2")
    jldopen(f -> (f["run_config"] = "earlier run"; store_run_record!(f); store_run_record!(f)), chain, "w")
    jldopen(chain, "r") do f
        @test f["run_config"] == "earlier run"
        @test f["run_config_2"] == read(path, String)
        @test !haskey(f, "run_config_3")
    end
end
