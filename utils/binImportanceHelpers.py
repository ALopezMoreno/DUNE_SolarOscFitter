import numpy as np
import warnings

def _A(x):
    """Numpy array + squeeze degenerate axes only (keeps 1D/2D structure intact)."""
    a = np.array(x)
    if a.ndim > 1 and 1 in a.shape:
        a = np.squeeze(a)
    return a

def _Znight(Z, transpose_night=False):
    """Optionally transpose 2D (night) maps if stored as (E, cosz) vs (cosz, E)."""
    return Z.T if (transpose_night and Z is not None and Z.ndim == 2) else Z

def _reshape_julia_matrix(arr, shape, name=""):
    """
    Robust reshape for Julia-saved arrays.
    - Accepts either already-shaped arrays or flattened arrays.
    - Uses column-major reshape (order='F') to match Julia.
    """
    a = _A(arr)
    shape = tuple(shape)

    if a.shape == shape:
        return a

    if a.size == int(np.prod(shape)):
        return a.reshape(shape, order="F")  # <-- key change

    raise ValueError(f"{name}: cannot reshape array of shape {a.shape} to {shape}")

def compute_bin_score(var_llh, corr_llh, var_eps=1e-12, use_abs=True):
    """
    Per-bin proxy used in your script:
        score = sqrt(Var(llh_bin)) * |Corr(llh_bin, theta)|

    Returns:
      score (same shape as var_llh), mask used, plus signed_corr (for debugging).
    """
    var_llh = np.maximum(_A(var_llh), 0.0)
    corr_llh = _A(corr_llh)

    mask = (var_llh > var_eps) & np.isfinite(corr_llh)
    score = np.zeros_like(var_llh, dtype=float)
    if use_abs:
        score[mask] = np.sqrt(var_llh[mask]) * np.abs(corr_llh[mask])
    else:
        score[mask] = np.sqrt(var_llh[mask]) * corr_llh[mask]
    return score, mask, corr_llh

def normalize_within_sample(score):
    """
    Normalize a score map so it sums to 1 within its own sample.
    (Use this for 'interesting maps' that show *where* sensitivity lives.)
    """
    s = np.array(score, dtype=float, copy=True)
    tot = np.nansum(s)
    if not np.isfinite(tot):
        warnings.warn(f"normalize_within_sample: non-finite total ({tot}); returning un-normalized array")
    elif tot > 0:
        s /= tot
    return s

def build_llh_bin_maps(diagnostics, var_eps=1e-12, transpose_night=False):
    """
    Build per-bin LLH mean/var/corr and score maps for:
      - ESnight (2D), ESday (1D)
      - CCnight (2D), CCday (1D)  [omitted when inclusive_mode=True]

    Returns a dict with:
      maps[sample][field] where field in:
        mean_llh, var_llh,
        corr_sin2, corr_dm2,
        score_sin2, score_dm2,
        score_sin2_norm, score_dm2_norm,
        covlike_sin2 (signed version), covlike_dm2 (signed version)  [optional helpful]
    """
    inclusive = diagnostics.get("inclusive_mode", False)

    # Shapes saved in diagnostics
    esnight_shape = tuple(_A(diagnostics["derived_ESnight_shape"]).astype(int).tolist())

    out = {"ESnight": {}, "ESday": {}}
    if not inclusive:
        out["CCnight"] = {}
        out["CCday"] = {}
        ccnight_shape = tuple(_A(diagnostics["derived_CCnight_shape"]).astype(int).tolist())

    # -------------------------
    # CC NIGHT (2D) — non-inclusive only
    # -------------------------
    if not inclusive:
        cc_mean = _reshape_julia_matrix(diagnostics["derived_CCnight_mean_llh"], ccnight_shape, "CCnight_mean_llh")
        cc_var  = _reshape_julia_matrix(diagnostics["derived_CCnight_var_llh"],  ccnight_shape, "CCnight_var_llh")
        cc_c_sin2 = _reshape_julia_matrix(diagnostics["derived_CCnight_corr_llh_sin2_th12"], ccnight_shape, "CCnight_corr_sin2")
        cc_c_dm2  = _reshape_julia_matrix(diagnostics["derived_CCnight_corr_llh_dm2_21"],    ccnight_shape, "CCnight_corr_dm2")

        cc_mean = _Znight(cc_mean, transpose_night)
        cc_var  = _Znight(cc_var,  transpose_night)
        cc_c_sin2 = _Znight(cc_c_sin2, transpose_night)
        cc_c_dm2  = _Znight(cc_c_dm2,  transpose_night)

        cc_score_sin2, _, _ = compute_bin_score(cc_var, cc_c_sin2, var_eps=var_eps, use_abs=True)
        cc_score_dm2,  _, _ = compute_bin_score(cc_var, cc_c_dm2,  var_eps=var_eps, use_abs=True)

        cc_covlike_sin2, _, _ = compute_bin_score(cc_var, cc_c_sin2, var_eps=var_eps, use_abs=False)
        cc_covlike_dm2,  _, _ = compute_bin_score(cc_var, cc_c_dm2,  var_eps=var_eps, use_abs=False)

        out["CCnight"].update(dict(
            mean_llh=cc_mean,
            var_llh=np.maximum(cc_var, 0.0),
            corr_sin2=cc_c_sin2,
            corr_dm2=cc_c_dm2,
            score_sin2=cc_score_sin2,
            score_dm2=cc_score_dm2,
            score_sin2_norm=normalize_within_sample(cc_score_sin2),
            score_dm2_norm=normalize_within_sample(cc_score_dm2),
            covlike_sin2=cc_covlike_sin2,
            covlike_dm2=cc_covlike_dm2,
        ))

    # -------------------------
    # CC DAY (1D) — non-inclusive only
    # -------------------------
    if not inclusive:
        ccday_mean = _A(diagnostics["derived_CCday_mean_llh"])
        ccday_var  = np.maximum(_A(diagnostics["derived_CCday_var_llh"]), 0.0)
        ccday_c_sin2 = _A(diagnostics["derived_CCday_corr_llh_sin2_th12"])
        ccday_c_dm2  = _A(diagnostics["derived_CCday_corr_llh_dm2_21"])

        ccday_score_sin2, _, _ = compute_bin_score(ccday_var, ccday_c_sin2, var_eps=var_eps, use_abs=True)
        ccday_score_dm2,  _, _ = compute_bin_score(ccday_var, ccday_c_dm2,  var_eps=var_eps, use_abs=True)

        ccday_covlike_sin2, _, _ = compute_bin_score(ccday_var, ccday_c_sin2, var_eps=var_eps, use_abs=False)
        ccday_covlike_dm2,  _, _ = compute_bin_score(ccday_var, ccday_c_dm2,  var_eps=var_eps, use_abs=False)

        out["CCday"].update(dict(
            mean_llh=ccday_mean,
            var_llh=ccday_var,
            corr_sin2=ccday_c_sin2,
            corr_dm2=ccday_c_dm2,
            score_sin2=ccday_score_sin2,
            score_dm2=ccday_score_dm2,
            score_sin2_norm=normalize_within_sample(ccday_score_sin2),
            score_dm2_norm=normalize_within_sample(ccday_score_dm2),
            covlike_sin2=ccday_covlike_sin2,
            covlike_dm2=ccday_covlike_dm2,
        ))

    # -------------------------
    # ES NIGHT (2D, or 3D when angular_reco=true)
    # -------------------------
    es_mean = _reshape_julia_matrix(diagnostics["derived_ESnight_mean_llh"], esnight_shape, "ESnight_mean_llh")
    es_var  = _reshape_julia_matrix(diagnostics["derived_ESnight_var_llh"],  esnight_shape, "ESnight_var_llh")
    es_c_sin2 = _reshape_julia_matrix(diagnostics["derived_ESnight_corr_llh_sin2_th12"], esnight_shape, "ESnight_corr_sin2")
    es_c_dm2  = _reshape_julia_matrix(diagnostics["derived_ESnight_corr_llh_dm2_21"],    esnight_shape, "ESnight_corr_dm2")

    es_mean = _Znight(es_mean, transpose_night)
    es_var  = _Znight(es_var,  transpose_night)
    es_c_sin2 = _Znight(es_c_sin2, transpose_night)
    es_c_dm2  = _Znight(es_c_dm2,  transpose_night)

    # When angular_reco=true the array is (n_cos, n_Ereco, n_night_bins).
    # Collapse the night-time axis to (n_cos, n_Ereco) for 2D plotting:
    #   intensive quantities (correlations) → mean; extensive (mean/var) → sum.
    if es_mean.ndim == 3:
        es_mean   = es_mean.sum(axis=-1)
        es_var    = es_var.sum(axis=-1)
        es_c_sin2 = np.nanmean(es_c_sin2, axis=-1)
        es_c_dm2  = np.nanmean(es_c_dm2,  axis=-1)

    es_score_sin2, _, _ = compute_bin_score(es_var, es_c_sin2, var_eps=var_eps, use_abs=True)
    es_score_dm2,  _, _ = compute_bin_score(es_var, es_c_dm2,  var_eps=var_eps, use_abs=True)

    es_covlike_sin2, _, _ = compute_bin_score(es_var, es_c_sin2, var_eps=var_eps, use_abs=False)
    es_covlike_dm2,  _, _ = compute_bin_score(es_var, es_c_dm2,  var_eps=var_eps, use_abs=False)

    out["ESnight"].update(dict(
        mean_llh=es_mean,
        var_llh=np.maximum(es_var, 0.0),
        corr_sin2=es_c_sin2,
        corr_dm2=es_c_dm2,
        score_sin2=es_score_sin2,
        score_dm2=es_score_dm2,
        score_sin2_norm=normalize_within_sample(es_score_sin2),
        score_dm2_norm=normalize_within_sample(es_score_dm2),
        covlike_sin2=es_covlike_sin2,
        covlike_dm2=es_covlike_dm2,
    ))

    # -------------------------
    # ES DAY (1D, or 2D when angular_reco=true: (n_cos_scatter, n_Ereco))
    # -------------------------
    esday_mean = _A(diagnostics["derived_ESday_mean_llh"])
    esday_var  = np.maximum(_A(diagnostics["derived_ESday_var_llh"]), 0.0)
    esday_c_sin2 = _A(diagnostics["derived_ESday_corr_llh_sin2_th12"])
    esday_c_dm2  = _A(diagnostics["derived_ESday_corr_llh_dm2_21"])

    # Collapse scatter-angle axis when angular_reco=true (2D → 1D energy spectrum).
    # Julia stores (n_cos, n_E) column-major; h5py reverses dims → Python sees (n_E, n_cos).
    # axis=0 = n_E (energy), axis=1 = n_cos (scatter angle). Sum over axis=1 to collapse angles.
    if esday_mean.ndim == 2:
        esday_mean = esday_mean.sum(axis=1)
        esday_var  = esday_var.sum(axis=1)
        with warnings.catch_warnings():  # below-threshold rows are all-NaN; NaN result is correct
            warnings.simplefilter('ignore', RuntimeWarning)
            esday_c_sin2 = np.nanmean(esday_c_sin2, axis=1)
            esday_c_dm2  = np.nanmean(esday_c_dm2,  axis=1)

    esday_score_sin2, _, _ = compute_bin_score(esday_var, esday_c_sin2, var_eps=var_eps, use_abs=True)
    esday_score_dm2,  _, _ = compute_bin_score(esday_var, esday_c_dm2,  var_eps=var_eps, use_abs=True)

    esday_covlike_sin2, _, _ = compute_bin_score(esday_var, esday_c_sin2, var_eps=var_eps, use_abs=False)
    esday_covlike_dm2,  _, _ = compute_bin_score(esday_var, esday_c_dm2,  var_eps=var_eps, use_abs=False)

    out["ESday"].update(dict(
        mean_llh=esday_mean,
        var_llh=esday_var,
        corr_sin2=esday_c_sin2,
        corr_dm2=esday_c_dm2,
        score_sin2=esday_score_sin2,
        score_dm2=esday_score_dm2,
        score_sin2_norm=normalize_within_sample(esday_score_sin2),
        score_dm2_norm=normalize_within_sample(esday_score_dm2),
        covlike_sin2=esday_covlike_sin2,
        covlike_dm2=esday_covlike_dm2,
    ))

    return out

def build_posterior_predictive_maps(diagnostics, transpose_night=False):
    """
    Build posterior predictive mean/var maps for rates:
      ES day/night always; CC day/night only when not inclusive_mode.

    Returns dict maps[sample] = {pp_mean, pp_var}
    """
    inclusive = diagnostics.get("inclusive_mode", False)
    out = {"ESnight": {}, "ESday": {}}

    # CC (non-inclusive only)
    if not inclusive:
        out["CCnight"] = {}
        out["CCday"] = {}

        ccday_mean = _A(diagnostics["derived_CCday_pp_mean"])
        ccday_var  = np.maximum(_A(diagnostics["derived_CCday_pp_var"]), 0.0)

        ccnight_mean = _A(diagnostics["derived_CCnight_pp_mean"])
        ccnight_var  = np.maximum(_A(diagnostics["derived_CCnight_pp_var"]), 0.0)

        bg_ccday_mean = _A(diagnostics["derived_BGCCday_pp_mean"]) if "derived_BGCCday_pp_mean" in diagnostics else None
        bg_ccday_var  = _A(diagnostics["derived_BGCCday_pp_var"])  if "derived_BGCCday_pp_var"  in diagnostics else None
        out["CCday"].update(dict(
            pp_mean=ccday_mean, pp_var=ccday_var,
            signal_pp_mean=(ccday_mean - bg_ccday_mean) if bg_ccday_mean is not None else None,
            bg_mean=bg_ccday_mean, bg_var=bg_ccday_var,
            pp_lo=_A(diagnostics["derived_CCday_pp_lo"]) if "derived_CCday_pp_lo" in diagnostics else None,
            pp_hi=_A(diagnostics["derived_CCday_pp_hi"]) if "derived_CCday_pp_hi" in diagnostics else None,
            pp_lo2=_A(diagnostics["derived_CCday_pp_lo2"]) if "derived_CCday_pp_lo2" in diagnostics else None,
            pp_hi2=_A(diagnostics["derived_CCday_pp_hi2"]) if "derived_CCday_pp_hi2" in diagnostics else None,
            pp_lo3=_A(diagnostics["derived_CCday_pp_lo3"]) if "derived_CCday_pp_lo3" in diagnostics else None,
            pp_hi3=_A(diagnostics["derived_CCday_pp_hi3"]) if "derived_CCday_pp_hi3" in diagnostics else None,
        ))
        out["CCnight"].update(dict(
            pp_mean=_Znight(ccnight_mean, transpose_night),
            pp_var=_Znight(ccnight_var, transpose_night),
        ))

    # ES (always)
    esday_mean = _A(diagnostics["derived_ESday_pp_mean"])
    esday_var  = np.maximum(_A(diagnostics["derived_ESday_pp_var"]), 0.0)

    # Collapse scatter-angle axis when angular_reco=true (2D → 1D).
    # Julia (n_cos, n_E) → h5py reverses → Python (n_E, n_cos): sum over axis=1 for energy spectrum.
    if esday_mean.ndim == 2:
        esday_mean = esday_mean.sum(axis=1)
        esday_var  = esday_var.sum(axis=1)   # used only for page-6 variance display

    esnight_mean = _A(diagnostics["derived_ESnight_pp_mean"])
    esnight_var  = np.maximum(_A(diagnostics["derived_ESnight_pp_var"]), 0.0)

    esnight_mean = _Znight(esnight_mean, transpose_night)
    esnight_var  = _Znight(esnight_var,  transpose_night)

    # Collapse scatter-angle axis for night (3D → 2D)
    if esnight_mean.ndim == 3:
        esnight_mean = esnight_mean.sum(axis=-1)
        esnight_var  = esnight_var.sum(axis=-1)

    # pp_lo/pp_hi are stored as 1D by Julia (scatter already collapsed, signal-only); absent in old JLD2 files
    bg_esday_mean = _A(diagnostics["derived_BGESday_pp_mean"]) if "derived_BGESday_pp_mean" in diagnostics else None
    bg_esday_var  = _A(diagnostics["derived_BGESday_pp_var"])  if "derived_BGESday_pp_var"  in diagnostics else None
    out["ESday"].update(dict(
        pp_mean=esday_mean, pp_var=esday_var,
        signal_pp_mean=(esday_mean - bg_esday_mean) if bg_esday_mean is not None else None,
        bg_mean=bg_esday_mean, bg_var=bg_esday_var,
        pp_lo=_A(diagnostics["derived_ESday_pp_lo"]) if "derived_ESday_pp_lo" in diagnostics else None,
        pp_hi=_A(diagnostics["derived_ESday_pp_hi"]) if "derived_ESday_pp_hi" in diagnostics else None,
        pp_lo2=_A(diagnostics["derived_ESday_pp_lo2"]) if "derived_ESday_pp_lo2" in diagnostics else None,
        pp_hi2=_A(diagnostics["derived_ESday_pp_hi2"]) if "derived_ESday_pp_hi2" in diagnostics else None,
        pp_lo3=_A(diagnostics["derived_ESday_pp_lo3"]) if "derived_ESday_pp_lo3" in diagnostics else None,
        pp_hi3=_A(diagnostics["derived_ESday_pp_hi3"]) if "derived_ESday_pp_hi3" in diagnostics else None,
    ))
    out["ESnight"].update(dict(pp_mean=esnight_mean, pp_var=esnight_var))

    # --- Asimov data (if stored by derive mode) ---
    idx_es = int(_A(diagnostics["derived_index_ES"])) - 1  # Julia 1-based → Python 0-based
    if "derived_data_ESday" in diagnostics:
        data_ESday = _A(diagnostics["derived_data_ESday"])
        if data_ESday.ndim == 2:               # angular: (n_E, n_cos) after h5py reversal
            data_ESday = data_ESday.sum(axis=1)
        data_ESday[:idx_es] = 0.0              # zero below threshold (matches pp_mean masking)
        out["ESday"]["data"] = data_ESday

    if "derived_data_ESnight" in diagnostics:
        data_ESnight = _A(diagnostics["derived_data_ESnight"])
        data_ESnight = _Znight(data_ESnight, transpose_night)
        if data_ESnight.ndim == 3:             # angular: collapse scatter-angle axis
            data_ESnight = data_ESnight.sum(axis=-1)
        out["ESnight"]["data"] = data_ESnight

    if not inclusive:
        idx_cc = int(_A(diagnostics["derived_index_CC"])) - 1
        if "derived_data_CCday" in diagnostics:
            data_CCday = _A(diagnostics["derived_data_CCday"])
            data_CCday[:idx_cc] = 0.0
            out["CCday"]["data"] = data_CCday
        if "derived_data_CCnight" in diagnostics:
            data_CCnight = _A(diagnostics["derived_data_CCnight"])
            data_CCnight = _Znight(data_CCnight, transpose_night)
            out["CCnight"]["data"] = data_CCnight

    return out

def build_sample_total_scores(diagnostics, var_eps=1e-12):
    """
    Sample-level dominance proxy using the saved total-llh stats:

      score_sample(theta) = sqrt(Var(llh_sample)) * |Corr(llh_sample, theta)|

    Returns:
      dict with keys like out["ESnight"]["dm2"], out["ESnight"]["sin2"].
      CC keys omitted when inclusive_mode=True.
    """
    inclusive = diagnostics.get("inclusive_mode", False)
    samples = ["ESnight", "ESday"] if inclusive else ["CCnight", "CCday", "ESnight", "ESday"]
    out = {s: {} for s in samples}

    for s in samples:
        v = float(_A(diagnostics[f"derived_{s}_tot_var_llh"]))
        v = max(v, 0.0)

        c_sin2 = float(_A(diagnostics[f"derived_{s}_tot_corr_llh_sin2_th12"]))
        c_dm2  = float(_A(diagnostics[f"derived_{s}_tot_corr_llh_dm2_21"]))

        out[s]["sin2"] = (np.sqrt(v) * abs(c_sin2)) if (v > var_eps and np.isfinite(c_sin2)) else 0.0
        out[s]["dm2"]  = (np.sqrt(v) * abs(c_dm2))  if (v > var_eps and np.isfinite(c_dm2))  else 0.0

        # also keep signed versions handy
        out[s]["sin2_signed"] = (np.sqrt(v) * c_sin2) if (v > var_eps and np.isfinite(c_sin2)) else 0.0
        out[s]["dm2_signed"]  = (np.sqrt(v) * c_dm2)  if (v > var_eps and np.isfinite(c_dm2))  else 0.0

    return out

def compute_shared_color_limits(bin_maps):
    """
    Convenience: compute global vmin/vmax for var, corr, and score_norm across samples.
    Returns dict with: vmax_var, corr_lim, vmax_score_norm (per param), etc.
    Automatically skips CC samples when not present in bin_maps (inclusive mode).
    """
    samples = list(bin_maps.keys())

    # Var (>=0)
    var_list = []
    for s in samples:
        v = bin_maps[s].get("var_llh", None)
        if v is not None:
            var_list.append(np.nanmax(v))
    vmax_var = np.nanmax(var_list) if var_list else 1.0

    # Corr (symmetric)
    corr_list = []
    for s in samples:
        for k in ["corr_sin2", "corr_dm2"]:
            c = bin_maps[s].get(k, None)
            if c is not None:
                corr_list.append(np.nanmax(np.abs(c)))
    corr_lim = np.nanmax(corr_list) if corr_list else 1.0

    # Within-sample normalized scores (>=0, each sums to 1, so vmax is meaningful across samples)
    score_sin2 = []
    score_dm2 = []
    for s in samples:
        ss = bin_maps[s].get("score_sin2_norm", None)
        sd = bin_maps[s].get("score_dm2_norm", None)
        if ss is not None:
            score_sin2.append(np.nanmax(ss))
        if sd is not None:
            score_dm2.append(np.nanmax(sd))

    vmax_score_sin2_norm = np.nanmax(score_sin2) if score_sin2 else 1.0
    vmax_score_dm2_norm  = np.nanmax(score_dm2)  if score_dm2 else 1.0

    return dict(
        vmin_var=0.0, vmax_var=float(vmax_var),
        corr_lim=float(corr_lim),
        vmin_score=0.0,
        vmax_score_sin2_norm=float(vmax_score_sin2_norm),
        vmax_score_dm2_norm=float(vmax_score_dm2_norm),
    )
