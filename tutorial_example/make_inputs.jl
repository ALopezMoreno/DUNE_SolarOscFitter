#=
make_inputs.jl — idealised detector inputs for the tutorial.

    julia --project tutorial_example/make_inputs.jl            # samples + background calibration
    julia --project tutorial_example/make_inputs.jl --no-calibrate

Run from the repository root. Writes event-level CSVs into tutorial_example/inputs/ (gitignored):

  ES_nue_ideal.csv, ES_numutau_ideal.csv   ν–e scattering, E_true = E_ν uniform over range_Etrue,
                                           recoil T drawn from the fitter's own dσ/dT (BKS95,
                                           src/xsec.jl) on [TPC_threshold, T_max]
  CC_nue_ideal.csv                         ν_e ⁴⁰Ar → e⁻ ⁴⁰K*: T_e = E_ν − Q − E_x, with the ⁴⁰K* level
                                           drawn ∝ B · p_e E_e F(Z = 19, E_e) (allowed transitions,
                                           marley_levels_Bhattacharya2009.dat)
  ES_angular_ideal.csv                     cos of the ν_e ES electron angle, exact kinematics
  ES_background_flat.csv, CC_background_flat.csv
                                           flat in E_reco from 0 to the upper edge of range_Ereco

In every signal sample E_reco = T (1 + 0.1 g), g ~ N(0, 1), and every event is selected
(mask = true, weight 1): an ideal detector. Energies are in GeV, as the fitter expects.

Calibration (default): runs the fitter's SETUP on tutorial_example/01_setup.yaml and rescales the
two background normalisations in configs/detectors/VD.yaml so that, per channel, the background
above Ereco_min is twice the signal at the Asimov truth (S/B = 0.5).
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

function make_samples()
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

    println("Backgrounds (flat, ", N_BACKGROUND, " events each)")
    for (name, ch) in (("ES_background_flat.csv", "ES"), ("CC_background_flat.csv", "CC"))
        emax = VD["range_Ereco_$ch"][2] * 1e-3
        write_sample(name, DataFrame(Ereco = emax .* rand(rng, N_BACKGROUND), weights = 1.0, mask = true))
    end
    Int(VD["ES_bg_generated_events"]) == N_BACKGROUND ||
        error("ES_bg_generated_events in $DETECTOR_CONFIG must equal N_BACKGROUND = $N_BACKGROUND")
end

# ── Background calibration: S and B above threshold from the fitter's own SETUP ──
function calibrate()
    tmp = mktempdir()
    cfg = resolve_config(joinpath(TUT, "01_setup.yaml"))
    cfg["outFile"] = joinpath(tmp, "calibration")
    cfg_file = joinpath(tmp, "calibration.yaml")
    YAML.write_file(cfg_file, cfg)
    println("\nRunning the fitter's SETUP for the calibration (log in $tmp/setup.log)")
    cmd = `$(Base.julia_cmd()) -t auto --project=$ROOT $(joinpath(ROOT, "src", "readConfig.jl")) $cfg_file`
    ok = success(pipeline(cmd; stdout = joinpath(tmp, "setup.log"), stderr = joinpath(tmp, "setup.log")))
    ok || error("SETUP failed; see $tmp/setup.log")

    text = read(DETECTOR_CONFIG, String)
    println("\nBackground calibration (fit window, day + night, at the Asimov truth)")
    for ch in ("ES", "CC")
        df = CSV.read(joinpath(tmp, "calibration_VD_$(ch)_breakdown.csv"), DataFrame)
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
    write(DETECTOR_CONFIG, text)
    println("  updated ", relpath(DETECTOR_CONFIG, ROOT))
end

make_samples()
"--no-calibrate" in ARGS || calibrate()
