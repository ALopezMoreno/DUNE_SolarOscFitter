#=
make_inputs.jl — idealised event-level inputs for the tutorial (see tutorial_example/README.md).

    julia --project tutorial_example/make_inputs.jl [--no-calibrate]

ES ν_e, ES ν_μτ and CC samples with E_reco = T (1 + 0.1 g), and flat backgrounds; energies in GeV.
The calibration runs SETUP twice: to find each channel's signal endpoint, then to set the
background norms in configs/detectors/VD.yaml so that S/B = 0.5 above threshold.
=#
using Random, CSV, DataFrames, YAML, Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const TUT  = @__DIR__
const OUT  = joinpath(TUT, "inputs")
const DETECTOR_CONFIG = joinpath(TUT, "configs", "detectors", "VD.yaml")

const N_SIGNAL     = 500_000   # events per signal sample
const N_BACKGROUND = 500_000   # events per background file (= ES_bg_generated_events in VD.yaml)
const RESOLUTION   = 0.10      # σ(E_reco)/E
const S_OVER_B     = 0.5
const SIGNAL_CONTAINED = 0.999   # fraction of the in-window signal below the background endpoint
const Q_GS_MEV     = 1.50443   # ⁴⁰Ar → ⁴⁰K ground state: T_e = E_ν − Q − E_x (MARLEY)
const Z_FINAL      = 19
const ALPHA        = 1 / 137.035999

# The fitter's ES couplings and constants (m_e, GeV). xsec.jl reads these two globals.
cd(ROOT)
CC_xsec_scale = 1.0
include(joinpath(ROOT, "src", "configFiles.jl"))
const CFG = resolve_config(joinpath(TUT, "configs", "tutorial.yaml"))
tpc_threshold = CFG["TPC_threshold"] * 1e-3
include(joinpath(ROOT, "src", "xsec.jl"))

const ETRUE_GEV = (CFG["range_Etrue"][1] * 1e-3, CFG["range_Etrue"][2] * 1e-3)
const VD = CFG["detectors"]["VD"]

smear(T, rng) = T * (1 + RESOLUTION * randn(rng))

# ── ES: T from dσ/dT by rejection, envelope = 1.1 × the maximum on a grid ──
es_dsigma_dT(T, enu, kappa, cc) = ((g1, g2) = es_couplings(T, kappa, cc);
                                   g1^2 + g2^2 * (1 - T / enu)^2 - g1 * g2 * m_e * T / enu^2)

function sample_es(rng, n, kappa, cc)
    Etrue, T = Vector{Float64}(undef, n), Vector{Float64}(undef, n)
    i = 0
    while i < n
        enu = ETRUE_GEV[1] + (ETRUE_GEV[2] - ETRUE_GEV[1]) * rand(rng)
        tmax = 2 * enu^2 / (m_e + 2 * enu)
        tmax > tpc_threshold || continue
        fmax = 1.1 * maximum(es_dsigma_dT(t, enu, kappa, cc) for t in range(tpc_threshold, tmax, length = 32))
        while true
            t = tpc_threshold + (tmax - tpc_threshold) * rand(rng)
            if rand(rng) * fmax <= es_dsigma_dT(t, enu, kappa, cc)
                i += 1; Etrue[i] = enu; T[i] = t
                break
            end
        end
    end
    return Etrue, T
end

# cos of the electron angle to the neutrino direction, ν e → ν e kinematics
es_cos(enu, T) = (enu + m_e) / enu * sqrt(T / (T + 2 * m_e))

# ── CC: ⁴⁰K* level ∝ B · p_e E_e F(Z, E_e) ──
function read_levels(path)
    rows = [split(l) for l in eachline(path) if !startswith(l, "#") && !isempty(strip(l))]
    return [parse(Float64, r[1]) * 1e-3 for r in rows], [parse(Float64, r[2]) for r in rows]
end

# Non-relativistic Fermi function for an electron leaving a nucleus of charge Z
function fermi_function(Z, E, p)
    x = 2π * Z * ALPHA * E / p
    return x / (1 - exp(-x))
end

function sample_cc(rng, n, Ex, B)
    Etrue, T = Vector{Float64}(undef, n), Vector{Float64}(undef, n)
    w = similar(B)
    i = 0
    while i < n
        enu = ETRUE_GEV[1] + (ETRUE_GEV[2] - ETRUE_GEV[1]) * rand(rng)
        for k in eachindex(Ex)
            te = enu - Q_GS_MEV * 1e-3 - Ex[k]
            Ee = te + m_e
            pe = sqrt(max(Ee^2 - m_e^2, 0.0))
            w[k] = te > 0 ? B[k] * pe * Ee * fermi_function(Z_FINAL, Ee, pe) : 0.0
        end
        s = sum(w)
        s > 0 || continue                     # below the lowest level: no CC interaction
        k = searchsortedfirst(cumsum(w), rand(rng) * s)
        i += 1; Etrue[i] = enu; T[i] = enu - Q_GS_MEV * 1e-3 - Ex[k]
    end
    return Etrue, T
end

write_sample(name, df) = (CSV.write(joinpath(OUT, name), df); println("  wrote ", joinpath("tutorial_example", "inputs", name), " (", nrow(df), " events)"))

function make_signal()
    mkpath(OUT)
    rng = Xoshiro(20261020)
    println("Signal samples (", N_SIGNAL, " events each)")
    for (name, kappa, cc) in (("ES_nue_ideal.csv", kappa_nue, 1.0), ("ES_numutau_ideal.csv", kappa_nuother, 0.0))
        Etrue, T = sample_es(rng, N_SIGNAL, kappa, cc)
        write_sample(name, DataFrame(Etrue = Etrue, Ereco = smear.(T, Ref(rng)), weights = 1.0, mask = true))
        if cc == 1.0
            write_sample("ES_angular_ideal.csv", DataFrame(Ereco = smear.(T, Ref(rng)), cos_scatter = es_cos.(Etrue, T)))
        end
    end
    Ex, B = read_levels(joinpath(TUT, "marley_levels_Bhattacharya2009.dat"))
    Etrue, T = sample_cc(rng, N_SIGNAL, Ex, B)
    write_sample("CC_nue_ideal.csv", DataFrame(Etrue = Etrue, Ereco = smear.(T, Ref(rng)), weights = 1.0, mask = true))
end

# Backgrounds flat in E_reco on [0, endpoint_MeV[ch]]
function make_backgrounds(endpoint_MeV)
    mkpath(OUT)
    rng = Xoshiro(20261021)
    println("Backgrounds (flat, ", N_BACKGROUND, " events each)")
    for (name, ch) in (("ES_background_flat.csv", "ES"), ("CC_background_flat.csv", "CC"))
        @printf("  %s: flat from 0 to %.2f MeV\n", ch, endpoint_MeV[ch])
        write_sample(name, DataFrame(Ereco = endpoint_MeV[ch] * 1e-3 .* rand(rng, N_BACKGROUND), weights = 1.0, mask = true))
    end
    Int(VD["ES_bg_generated_events"]) == N_BACKGROUND ||
        error("ES_bg_generated_events in $DETECTOR_CONFIG must equal N_BACKGROUND = $N_BACKGROUND")
end

range_top_MeV() = Dict(ch => Float64(VD["range_Ereco_$ch"][2]) for ch in ("ES", "CC"))

const ENDPOINT_LINE = r"^(\s*)# Flat in E_reco from 0 to the signal endpoint:.*$"m
const ENDPOINT_VALUES = r"signal endpoint: ES ([0-9.eE+-]+) MeV, CC ([0-9.eE+-]+) MeV"

# The endpoints written into VD.yaml by the last calibration
function recorded_endpoint_MeV()
    m = match(ENDPOINT_VALUES, read(DETECTOR_CONFIG, String))
    m === nothing && error("no recorded signal endpoints in $DETECTOR_CONFIG; run without --no-calibrate")
    return Dict("ES" => parse(Float64, m[1]), "CC" => parse(Float64, m[2]))
end

# ── Background calibration from the fitter's own SETUP ──
# SETUP on 01_setup.yaml with the current inputs; returns each channel's per-bin breakdown at the truth.
function run_setup(label)
    tmp = mktempdir()
    cfg = resolve_config(joinpath(TUT, "01_setup.yaml"))
    cfg["outFile"] = joinpath(tmp, "calibration")
    cfg_file = joinpath(tmp, "calibration.yaml")
    YAML.write_file(cfg_file, cfg)
    println("\nRunning the fitter's SETUP ($label; log in $tmp/setup.log)")
    cmd = `$(Base.julia_cmd()) -t auto --project=$ROOT $(joinpath(ROOT, "src", "readConfig.jl")) $cfg_file`
    ok = success(pipeline(cmd; stdout = joinpath(tmp, "setup.log"), stderr = joinpath(tmp, "setup.log")))
    ok || error("SETUP failed; see $tmp/setup.log")
    return Dict(ch => CSV.read(joinpath(tmp, "calibration_VD_$(ch)_breakdown.csv"), DataFrame) for ch in ("ES", "CC"))
end

# Upper edge of the bin where the cumulative in-window signal reaches SIGNAL_CONTAINED
function signal_endpoint_MeV(df, top_MeV)
    fit = findall(df.in_fit)
    c = cumsum(df.signal[fit]) ./ sum(df.signal[fit])
    hi = df.Ereco_hi_MeV[fit[searchsortedfirst(c, SIGNAL_CONTAINED)]]
    return isfinite(hi) ? min(hi, top_MeV) : top_MeV
end

function calibrate()
    top = range_top_MeV()
    first_pass = run_setup("signal endpoints")
    endpoint = Dict(ch => signal_endpoint_MeV(first_pass[ch], top[ch]) for ch in ("ES", "CC"))
    make_backgrounds(endpoint)
    breakdown = run_setup("background normalisations")

    text = read(DETECTOR_CONFIG, String)
    println("\nBackground calibration (fit window, day + night, at the Asimov truth)")
    for ch in ("ES", "CC")
        df = breakdown[ch]
        fit = df.in_fit
        S, B = sum(df.signal[fit]), sum(df.background[fit])
        key = "$(ch)_background_normalisations"
        old = Float64(only(VD[key]))
        new = old * (S / S_OVER_B) / B
        @printf("  %s: signal %.1f, background %.1f at norm %.4g → norm %.4g (background %.1f, S/B %.2f)\n",
                ch, S, B, old, new, S / S_OVER_B, S_OVER_B)
        pat = Regex("^(\\s*$key:\\s*)\\[[^\\]]*\\]", "m")
        occursin(pat, text) || error("no '$key: [...]' line in $DETECTOR_CONFIG")
        text = replace(text, pat => SubstitutionString("\\g<1>[" * @sprintf("%.6g", new) * "]"))
    end
    occursin(ENDPOINT_LINE, text) || error("no '# Flat in E_reco from 0 to the signal endpoint:' line in $DETECTOR_CONFIG")
    line = @sprintf("# Flat in E_reco from 0 to the signal endpoint: ES %.10g MeV, CC %.10g MeV (%.4g%% of the signal above threshold lies below it; set by make_inputs.jl)",
                    endpoint["ES"], endpoint["CC"], 100 * SIGNAL_CONTAINED)
    text = replace(text, ENDPOINT_LINE => SubstitutionString("\\g<1>" * line))
    write(DETECTOR_CONFIG, text)
    println("  updated ", relpath(DETECTOR_CONFIG, ROOT))
end

make_signal()
if "--no-calibrate" in ARGS
    make_backgrounds(recorded_endpoint_MeV())
else
    make_backgrounds(range_top_MeV())
    calibrate()
end
