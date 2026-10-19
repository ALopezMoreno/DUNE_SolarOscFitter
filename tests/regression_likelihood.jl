#=
regression_likelihood.jl

End-to-end regression of the full likelihood: builds tests/fixtures/regression_config.yaml
(RunMode SETUP, in a temporary output directory) and compares total_llh at a few fixed
parameter points with tests/regression_reference.toml. The config uses frozen synthetic
detector samples (tests/fixtures/), so changes to the real inputs do not move it; it covers
ES and CC, two detectors, fixed and free backgrounds, and semi-inclusive angular ES.

    julia --project tests/regression_likelihood.jl [variant] [--update]

variant: "night"  (default night calculation, NightOsc)
         "legacy" (3-flavour per-path NumOsc night calculation)
         "night_da", "night_bb", "night_da_indep" (night with the MC statistical uncertainty:
          Dembinski–Abdelmotteleb or Conway's Barlow–Beeston, shared or independent day/night)
         "night_sym" (night with tests/fixtures/solarExposure_symmetric.csv: the legacy night
          exposure mirrored onto the day side, both weight columns equal, so signal and background
          splits are 0.5 exactly, SPEC 4.6. Its
          reference was set to the "night" values of the hard-coded 0.5/0.5 split (bf9ffec) and
          reproduced to rtol 1e-12 when the split moved to the exposure file; since the night
          weights are normalised by their own sum it differs from them by ≤ 6e-9 relative.)
--update rewrites the reference for this variant instead of checking it.
--grad   also checks the Enzyme reverse-mode gradient of the likelihood against ForwardDiff
         at the "near" point with the CC xsec nuisances moved (variant "night" only; the first Enzyme call compiles for minutes).

Each variant runs in its own process (the setup defines constants). runtests.jl runs both
when JULIA_REGRESSION=1.
=#
using YAML, TOML, Printf

const ROOT      = normpath(joinpath(@__DIR__, ".."))
const REF_FILE  = joinpath(@__DIR__, "regression_reference.toml")
const VARIANTS  = Dict(
    "night"  => Dict{String,Any}("nightOsc" => true),
    "legacy" => Dict{String,Any}("nightOsc" => false, "fastFit" => true),
    # MC statistical uncertainty in the likelihood (src/likelihoods/mcstat.jl), all detectors
    "night_da"       => Dict{String,Any}("nightOsc" => true, "mc_stat_likelihood" => "DembinskiAbdelmotteleb"),
    "night_bb"       => Dict{String,Any}("nightOsc" => true, "mc_stat_likelihood" => "BarlowBeeston"),
    "night_da_indep" => Dict{String,Any}("nightOsc" => true, "mc_stat_likelihood" => "DembinskiAbdelmotteleb",
                                         "mc_stat_day_night" => "independent"),
    # day/night split of 0.5 from a symmetric exposure file (SPEC 4.6)
    "night_sym"      => Dict{String,Any}("nightOsc" => true,
                                         "solar_exposure_file" => "tests/fixtures/solarExposure_symmetric.csv"),
)
const RTOL = 1e-9

positional = filter(a -> !startswith(a, "--"), ARGS)
variant = isempty(positional) ? "night" : positional[1]
update  = "--update" in ARGS
check_grad = "--grad" in ARGS
haskey(VARIANTS, variant) || error("unknown variant $variant; use one of $(collect(keys(VARIANTS)))")

# Config: the frozen fixture config with the run control overridden.
cfg = YAML.load_file(joinpath(ROOT, "tests", "fixtures", "regression_config.yaml"))
tmp = mktempdir()
merge!(cfg, Dict{String,Any}("RunMode" => "SETUP", "Asimov" => true, "outFile" => joinpath(tmp, "regression"),
                             "earth_potential_uncertainties" => true,
                             "ad_backend" => "ForwardDiff"),
       VARIANTS[variant])
cfg_file = joinpath(tmp, "regression.yaml")
YAML.write_file(cfg_file, cfg)

cd(ROOT)
empty!(ARGS); push!(ARGS, cfg_file)
include(joinpath(ROOT, "src", "readConfig.jl"))
main()

# Parameter points: the Asimov truth and two displaced points (oscillation parameters and
# Earth-layer normalisations moved together).
function displaced(p, f_dm2, d_s12, d_s13, d_earth)
    d = Dict{Symbol,Any}(pairs(p)...)
    d[:dm2_21]    = p.dm2_21 * f_dm2
    d[:sin2_th12] = p.sin2_th12 + d_s12
    d[:sin2_th13] = p.sin2_th13 + d_s13
    for k in keys(d)
        s = String(k)
        if startswith(s, "earth_norm_")
            i = parse(Int, s[length("earth_norm_")+1:end])
            d[k] = p[k] * (1 + d_earth * (-1)^i)
        end
    end
    return (; d...)
end
# The CC cross-section shape nuisances (SPEC 3.3), moved by about half a prior sigma; every other
# point has norm = 1, tilt = curv = 0, where the shape is exactly 1 and so cannot be tested.
function with_xsec(p; norm = 1.02, tilt = 0.05, curv = -0.03)
    haskey(p, :cc_xsec_tilt) || error("regression config must leave the CC xsec nuisances free")
    return merge(p, (cc_xsec_norm = norm, cc_xsec_tilt = tilt, cc_xsec_curv = curv))
end
p0 = Base.invokelatest(() -> true_params)
points = Dict(
    "truth"  => p0,
    "near"   => displaced(p0, 1.3, 0.02, 0.001, 0.05),
    "slow"   => displaced(p0, 25e-6 / p0.dm2_21, -0.03, -0.001, -0.08),
    "xsec"   => with_xsec(p0),
)
llh_values = Dict(name => Float64(Base.invokelatest(total_llh, p)) for (name, p) in points)

# MC-statistics variants: the per-bin diagnostic (derived mode) at the displaced point. Every piece
# is a finite −Q/2 ≤ 0 (Baker–Cousins form), and per detector the pieces add up to the detector's
# total minus its overflow E_reco bin, which the diagnostic leaves out (computed here on its own).
# (The plain Poisson diagnostic is −(μ − n ln μ), not Baker–Cousins, so it is not checked.)
for dname in (haskey(VARIANTS[variant], "mc_stat_likelihood") ? Base.invokelatest(() -> collect(keys(detector_perbin_fns))) : String[])
    p   = points["near"]
    pb  = Base.invokelatest(detector_perbin_fns[dname], p)
    tot = Base.invokelatest(detector_llh_fns[dname], p)
    out = Base.invokelatest(() -> detector_outputs[dname]); det = Base.invokelatest(() -> detector_configs[dname])
    li  = out.likelihood_inputs
    r   = Base.invokelatest(expected_rates, li, p)
    m, sh = Base.invokelatest(mcstat_method, det.mc_stat_likelihood), det.mc_stat_day_night == "shared"
    overflow = 0.0
    for (ch, on) in ((:ES, det.ES_mode), (:CC, det.CC_mode && (!det.inclusive_analysis || det.semi_inclusive_analysis)))
        on || continue
        a = out.mc_counts[ch].a
        overflow += Base.invokelatest(mcstat_loglikelihood, m, sh, getfield(r, Symbol(ch, "_day")), getfield(r, Symbol(ch, "_night")),
                                      getfield(li.nObserved, Symbol(ch, "_day")), getfield(li.nObserved, Symbol(ch, "_night")), a, length(a))
    end
    vals = reduce(vcat, [vec(v) for v in values(pb)])
    (all(isfinite, vals) && all(<=(1e-9), vals) && isapprox(sum(vals), tot - overflow; rtol = 1e-9, atol = 1e-9)) ||
        error("per-bin diagnostic inconsistent for $dname in '$variant' (sum $(sum(vals)), total − overflow $(tot - overflow))")
end

ref = isfile(REF_FILE) ? TOML.parsefile(REF_FILE) : Dict{String,Any}()
if update
    ref[variant] = llh_values
    open(io -> TOML.print(io, ref; sorted=true), REF_FILE, "w")
    println("regression reference for '$variant' written: ", llh_values)
    exit(0)
end

haskey(ref, variant) || error("no reference for '$variant' in $REF_FILE; run with --update first")
failed = false
for (name, v) in sort(collect(llh_values))
    r = ref[variant][name]
    ok = isapprox(v, r; rtol=RTOL, atol=1e-9)
    global failed |= !ok
    @printf("%-7s %-6s  llh = %.16g  reference = %.16g  rel. diff %.1e  %s\n", variant, name, v, r,
            abs(v - r) / max(abs(r), 1e-300), ok ? "ok" : "MISMATCH")
end
if check_grad
    startswith(variant, "night") || error("--grad needs a night variant (legacy is not Enzyme-differentiable)")
    @eval using ForwardDiff, Enzyme
    p = with_xsec(points["near"])   # tilt/curv ≠ 0, so their adjoints are not taken at the nominal point
    names = keys(p)
    N = length(names)
    f = Base.invokelatest(() -> x -> total_llh(NamedTuple{names}(ntuple(i -> x[i], Val(N)))))
    x = Float64[p[k] for k in names]
    g_fd = Base.invokelatest(ForwardDiff.gradient, f, x)
    g_ez = Base.invokelatest(() -> Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(f), x)[1])
    rel = maximum(abs.(g_ez .- g_fd)) / maximum(abs.(g_fd))
    @printf("gradient: max |Enzyme - ForwardDiff| / max |ForwardDiff| = %.2e\n", rel)
    if rel > 1e-8
        global failed = true
        println("gradient MISMATCH")
    end
end
failed && error("likelihood regression failed for '$variant'")
println("likelihood regression '$variant': all points match")
