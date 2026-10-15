#=
inputChecks.jl — sanity checks on a detector's input samples, and a per-bin rate breakdown.

check_input_files(det) runs before the samples are used and logs a per-file table (rows,
selected rows, weight range, selected events at or above the threshold). It warns about:
  - samples that look pre-binned (Ereco on a regular grid): the fitter needs event-level
    samples and re-bins them, so a pre-binned input aliases;
  - signal and ES background samples in which every event passes the selection (their
    efficiency needs the unselected events; CC background files may hold only selected ones);
  - byte-identical input files (e.g. a "smeared" sample that was never smeared).

save_rate_breakdown writes, per channel, the Asimov data, signal and each background component
per Ereco bin at the true parameters (CSV next to the other outputs), and logs the in-window totals.
=#
using CSV, DataFrames, Printf

"""
    looks_prebinned(E; min_step = 1e-5) -> Bool

True when `E` looks like a histogram saved as bin centres: (at least 5) distinct values with
about one row per value, all on one regular grid whose step is at least `min_step` (GeV;
10 keV). Event samples stored with limited precision (keV rounding, Float32) also sit on a
grid, but a much finer one, and repeat values; neither is flagged. The grid tolerance accepts
bin centres that went through Float32. Log-spaced histograms are not detected.
"""
function looks_prebinned(E::AbstractVector; min_step = 1e-5)
    v = collect(skipmissing(E))
    u = sort!(unique(v))
    length(u) >= 5 || return false
    length(u) >= 0.9 * length(v) || return false       # histograms have one row per bin
    d = diff(u)
    step = minimum(d)
    step >= min_step || return false
    return all(x -> abs(x - round(x)) < 1e-3, d ./ step)
end

"""
    identical_file_groups(paths) -> Vector{Vector{String}}

Groups of distinct paths whose files have byte-identical contents.
"""
function identical_file_groups(paths)
    files = unique(filter(isfile, paths))
    groups = Vector{Vector{String}}()
    seen = Set{String}()
    for (i, a) in enumerate(files)
        a in seen && continue
        g = [a]
        for b in files[i+1:end]
            (b in seen || filesize(a) != filesize(b)) && continue
            read(a) == read(b) && (push!(g, b); push!(seen, b))
        end
        length(g) > 1 && push!(groups, g)
    end
    return groups
end

# One row of the per-file table. `window` = (lo, hi) in GeV, Ereco in the file in GeV.
function input_file_summary(path, role, window)
    df = CSV.read(path, DataFrame)
    hasE = "Ereco" in names(df)
    sel  = "mask" in names(df) ? Bool.(coalesce.(df.mask, false)) : trues(nrow(df))
    w    = "weights" in names(df) ? Float64.(coalesce.(df.weights, 0.0)) : ones(nrow(df))
    inwin = hasE ? (window[1] .<= coalesce.(df.Ereco, -Inf) .<= window[2]) : falses(nrow(df))
    return (file = basename(path), role = role, rows = nrow(df), selected = count(sel),
            has_mask = "mask" in names(df), wmin = minimum(w; init=Inf), wmax = maximum(w; init=-Inf),
            in_window = sum(w[sel .& inwin]; init=0.0),
            prebinned = hasE && looks_prebinned(df.Ereco))
end

function check_input_files(det)
    # the fit uses everything reconstructed at or above the threshold (the last bin holds the overflow)
    win_ES = (det.E_threshold.ES, Inf)
    win_CC = (det.E_threshold.CC, Inf)
    n_ES = min(length(det.ES_filepaths_BG), length(det.ES_bg_norms), length(det.ES_bg_sys))
    n_CC = min(length(det.CC_filepaths_BG), length(det.CC_bg_norms), length(det.CC_bg_sys))
    files = Tuple{String,String,Tuple{Float64,Float64}}[]
    det.ES_mode && append!(files, [(det.nue_filepath, "ES νe signal", win_ES),
                                   (det.other_filepath, "ES νμτ signal", win_ES)])
    det.ES_mode && det.angular_reco && push!(files, (det.angular_filepath, "ES angular", win_ES))
    det.CC_mode && push!(files, (det.CC_filepath, "CC signal", win_CC))
    det.ES_mode && append!(files, [(p, "ES background", win_ES) for p in det.ES_filepaths_BG[1:n_ES]])
    use_CC_bg = det.CC_mode && (!det.inclusive_analysis || det.semi_inclusive_analysis)
    use_CC_bg && append!(files, [(p, "CC background", win_CC) for p in det.CC_filepaths_BG[1:n_CC]])

    rows = [input_file_summary(p, role, win) for (p, role, win) in files]
    @logmsg Setup "  Input samples (\"in window\": selected events, weighted, reconstructed at or above Ereco_min):"
    @logmsg Setup @sprintf("    %-14s %9s %9s  %-21s %12s  %s", "role", "rows", "selected", "weights", "in window", "file")
    for r in rows
        @logmsg Setup @sprintf("    %-14s %9d %9d  [%8.3g, %8.3g] %12.4g  %s",
                               r.role, r.rows, r.selected, r.wmin, r.wmax, r.in_window, r.file)
    end
    for r in rows
        r.prebinned && @warn "[$(det.name)] $(r.file) ($(r.role)) looks pre-binned: its Ereco values lie " *
                             "on a regular grid. The fitter needs event-level samples; a histogram " *
                             "re-binned into the analysis bins aliases."
        # Signal samples and ES backgrounds need the unselected events too (efficiency =
        # selected / total). CC background files may legitimately hold only the selected events.
        if r.role in ("ES νe signal", "ES νμτ signal", "CC signal", "ES background") &&
           r.has_mask && r.selected == r.rows && r.rows > 0
            @warn "[$(det.name)] every event of $(r.file) ($(r.role)) passes the selection (mask = true " *
                  "for all $(r.rows) rows), so its selection efficiency is 1. Was the selection applied?"
        end
    end
    for g in identical_file_groups(first.(files))
        @warn "[$(det.name)] these input files have identical contents: $(join(basename.(g), ", "))"
    end
    return rows
end

"""
    save_rate_breakdown(path_prefix, det_name, channel, edges_GeV, index, data_total, BG_raw,
                        conv, par_counts, norms_true, names, night_factor)

Write `<path_prefix>_<det>_<channel>_breakdown.csv`: per Ereco bin the Asimov data (day + all
night bins), each background component at its true normalisation, their sum and the signal
(data − backgrounds). Logs the in-window totals. `night_factor` = 0.5·(1 + Σ exposure_weights)
turns a background spectrum into its day + night total.
"""
function save_rate_breakdown(path_prefix, det_name, channel, edges_GeV, index, data_total, BG_raw,
                             conv, par_counts, norms_true, names, night_factor)
    comps = Vector{Vector{Float64}}()
    ni = 1
    for (i, beh) in enumerate(par_counts)
        if beh != 0
            push!(comps, BG_raw[i] .* (conv[i] * norms_true[ni]) .* night_factor); ni += 1
        else
            push!(comps, BG_raw[i] .* night_factor)
        end
    end
    n = length(data_total)
    bg_tot = isempty(comps) ? zeros(n) : reduce(+, comps)
    df = DataFrame(bin = 1:n, Ereco_lo_MeV = edges_GeV[1:end-1] .* 1e3, Ereco_hi_MeV = edges_GeV[2:end] .* 1e3,
                   in_fit = (1:n) .>= index, data = data_total, signal = data_total .- bg_tot,
                   background = bg_tot)
    for (nm, c) in zip(names, comps)
        df[!, "bg_" * nm] = c
    end
    path = "$(path_prefix)_$(det_name)_$(channel)_breakdown.csv"
    CSV.write(path, df)

    fit = index:n
    S, B = sum(df.signal[fit]), sum(bg_tot[fit])
    @logmsg Setup @sprintf("  %s in the fit window: data %.4g = signal %.4g + background %.4g (S/B %.3g)",
                           channel, sum(data_total[fit]), S, B, B > 0 ? S / B : Inf)
    for (nm, c) in zip(names, comps)
        @logmsg Setup @sprintf("      background %-24s %.4g", nm, sum(c[fit]))
    end
    @logmsg Setup "    per-bin breakdown: $path"
    return df
end

bg_component_names(paths, n) = [replace(basename(p), ".csv" => "") for p in paths[1:n]]
