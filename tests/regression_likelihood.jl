#=
regression_likelihood.jl — end-to-end regression of total_llh at fixed points on the frozen fixture
config (tests/fixtures/regression_config.yaml: two detectors, ES and CC, fixed and free backgrounds,
semi-inclusive angular ES) against tests/regression_reference.toml.

    julia --project tests/regression_likelihood.jl [variant] [--update] [--grad]

variants: night (NightOsc), legacy (NumOsc), night_da / night_bb / night_da_indep (MC statistics:
Dembinski–Abdelmotteleb or Barlow–Beeston, shared or independent day/night), night_sym (symmetric
exposure, splits 0.5 exactly; matches the old hard-coded 0.5/0.5 split to 6e-9), night_prem (PREM
with the correlated 7-layer prior of utils/make_earth_prior.jl, frozen in tests/fixtures). Every
variant checks the BAT prior (the Earth prior's MvNormal through whitened normals, SPEC 2.6). --update rewrites the
reference; --grad checks Enzyme against ForwardDiff (night only), on total_llh and on the transformed
posterior HMC differentiates. One process per variant.
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
    # analytic PREM (8 layers, the inner core not crossed) with the correlated prior on layers 1–7
    "night_prem"     => Dict{String,Any}("nightOsc" => true, "earth_model_file" => "PREM",
                                         "earth_normalisation_prior_file" => "tests/fixtures/earth_prior_parchita.jld2"),
    # a prior on 3 layers when paths cross 5: setup must refuse it (runtests.jl expects these to fail)
    "refuse_night_K3"  => Dict{String,Any}("nightOsc" => true,
                                           "earth_normalisation_prior_file" => "tests/fixtures/earth_prior_diag3.jld2"),
    "refuse_legacy_K3" => Dict{String,Any}("nightOsc" => false, "fastFit" => true,
                                           "earth_normalisation_prior_file" => "tests/fixtures/earth_prior_diag3.jld2"),
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

# MC-statistics variants: per-bin pieces at the displaced point are finite, ≤ 0, and add up to the
# detector's total minus its overflow bin.
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

# BAT prior (SPEC 2.6): the Earth layers' MvNormal via whitened normals; posterior = total_llh + log prior
@eval using BAT, DensityInterface, Distributions
Base.invokelatest(include, joinpath(ROOT, "src", "priors.jl"))
prior, post = Base.invokelatest(build_posterior, total_llh)
# a point as BAT holds it: the prior's parameters only, the Earth layers as earth_z
bat_point(p) = (pf = Base.invokelatest(bat_params, p); ks = Tuple(keys(BAT.varshape(prior))); NamedTuple{ks}(Tuple(pf[k] for k in ks)))
let pr = Base.invokelatest(() -> bat_priors(build_priors())), x1 = bat_point(points["near"]), x2 = bat_point(points["slow"])
    flat_ok = all(Base.invokelatest(flat_params, x1)[k] ≈ points["near"][k] for k in keys(Base.invokelatest(flat_params, x1)))
    lp(x) = sum(logpdf(d, x[k]) for (k, d) in pr)
    # differences of the log prior are those of the Earth prior's MvNormal in the layer scales
    mv = Base.invokelatest(() -> earth_normalisation_prior)
    earth(p) = [p[k] for k in Base.invokelatest(() -> EARTH_NORM_NAMES)]
    rest(x) = sum(logpdf(d, x[k]) for (k, d) in pr if k !== :earth_z)
    mv_ok = isapprox(lp(x1) - lp(x2), rest(x1) - rest(x2) + logpdf(mv, earth(points["near"])) - logpdf(mv, earth(points["slow"]));
                     rtol = 1e-10)
    prior_ok = isapprox(logdensityof(prior, x1), lp(x1); rtol = 1e-12) &&
               isapprox(logdensityof(post, x1), llh_values["near"] + lp(x1); rtol = 1e-12)
    @printf("BAT prior: Earth layers as %d whitened normals; log density %s, MvNormal of the prior file %s, round trip %s\n",
            length(x1.earth_z), prior_ok ? "ok" : "MISMATCH", mv_ok ? "ok" : "MISMATCH", flat_ok ? "ok" : "MISMATCH")
    global failed |= !(prior_ok && mv_ok && flat_ok)
end

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
    @printf("gradient of total_llh: max |Enzyme - ForwardDiff| / max |ForwardDiff| = %.2e\n", rel)

    # what HMC differentiates: the posterior in BAT's PriorToNormal space, through valgrad_func
    @eval import AutoDiffOperators: ADSelector, valgrad_func
    tm, fpt = Base.invokelatest(BAT.transform_and_unshape, PriorToNormal(), post, BATContext())
    z = Base.invokelatest(fpt, bat_point(p))
    fz = BAT.checked_logdensityof(tm)
    (v_fd, gz_fd), (v_ez, gz_ez) = (Base.invokelatest(valgrad_func(fz, ADSelector(b)), z) for b in (:ForwardDiff, :Enzyme))
    rel_z = maximum(abs.(gz_ez .- gz_fd)) / maximum(abs.(gz_fd))
    @printf("gradient of the transformed posterior (%d dims): max |Enzyme - ForwardDiff| / max |ForwardDiff| = %.2e\n",
            length(z), rel_z)
    if max(rel, rel_z) > 1e-8 || !(v_fd ≈ v_ez)
        global failed = true
        println("gradient MISMATCH")
    end
end
failed && error("likelihood regression failed for '$variant'")
println("likelihood regression '$variant': all points match")
