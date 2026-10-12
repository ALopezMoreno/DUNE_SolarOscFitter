#=
regression_likelihood.jl

End-to-end regression of the full likelihood: builds config.yaml (RunMode SETUP, in a
temporary output directory) and compares total_llh at a few fixed parameter points with
tests/regression_reference.toml.

    julia --project tests/regression_likelihood.jl [variant] [--update]

variant: "night"  (default night calculation, NightOsc)
         "legacy" (3-flavour per-path NumOsc night calculation)
--update rewrites the reference for this variant instead of checking it.
--grad   also checks the Enzyme reverse-mode gradient of the likelihood against ForwardDiff
         at the "near" point (variant "night" only; the first Enzyme call compiles for minutes).

Each variant runs in its own process (the setup defines constants). runtests.jl runs both
when JULIA_REGRESSION=1.
=#
using YAML, TOML, Printf

const ROOT      = normpath(joinpath(@__DIR__, ".."))
const REF_FILE  = joinpath(@__DIR__, "regression_reference.toml")
const VARIANTS  = Dict(
    "night"  => Dict{String,Any}("nightOsc" => true),
    "legacy" => Dict{String,Any}("nightOsc" => false, "fastFit" => true),
)
const RTOL = 1e-9

positional = filter(a -> !startswith(a, "--"), ARGS)
variant = isempty(positional) ? "night" : positional[1]
update  = "--update" in ARGS
check_grad = "--grad" in ARGS
haskey(VARIANTS, variant) || error("unknown variant $variant; use one of $(collect(keys(VARIANTS)))")

# Config: config.yaml with the run control overridden.
cfg = YAML.load_file(joinpath(ROOT, "config.yaml"))
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
p0 = Base.invokelatest(() -> true_params)
points = Dict(
    "truth"  => p0,
    "near"   => displaced(p0, 1.3, 0.02, 0.001, 0.05),
    "slow"   => displaced(p0, 25e-6 / p0.dm2_21, -0.03, -0.001, -0.08),
)
llh_values = Dict(name => Float64(Base.invokelatest(total_llh, p)) for (name, p) in points)

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
    @printf("%-7s %-6s  llh = %.12g  reference = %.12g  %s\n", variant, name, v, r, ok ? "ok" : "MISMATCH")
end
if check_grad
    variant == "night" || error("--grad needs the night variant (legacy is not Enzyme-differentiable)")
    @eval using ForwardDiff, Enzyme
    p = points["near"]
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
