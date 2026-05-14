function CC_conversions(norm_value)
    # The MC is normalised to 2.2 x 10^-6 neutrons / cm2 / s
    return norm_value / 2.2e-6
end

function ES_conversions(side, norm_value)
    # The ES background MC is not flux-normalised; convert by multiplying by the
    # relevant detector face area (in cm²). Detector cross-section: 12 m × 64 m long
    # faces (side=0) and 12 m × 12 m end caps (side=1); side=-1 (or other) uses total.
    if side == 0
        area = 2 * 12e2*64e2   # two 12 m × 64 m long faces [cm²]
    elseif side == 1
        area = 2 * 12e2*12e2   # two 12 m × 12 m end caps [cm²]
    else
        area = 2 * 12e2*64e2 + 2 * 12e2*12e2  # total exposed area [cm²]
    end
    return area * norm_value
end

"""
    normalize_backgrounds(raw_backgrounds, params, det_name::String)

Compute the summed ES and CC background vectors by applying nuisance-parameter
scale factors on-the-fly, without copying the raw arrays.

`det_name` is used to construct the parameter-name prefix, e.g. `"VD"` →
looks up `:VD_ES_bg_norm_1`, `:VD_ES_bg_norm_2`, … in `params`.

`raw_backgrounds` must carry `ES_par_counts` and `CC_par_counts` fields
(set by `build_backgrounds`).

Returns:
- BG_ES :: Vector{Float64}
- BG_CC :: Vector{Float64}
"""
function normalize_backgrounds(raw_backgrounds, params, det_name::String)
    es_prefix = Val(Symbol(det_name, "_ES_bg_norm_"))
    cc_prefix = Val(Symbol(det_name, "_CC_bg_norm_"))
    es_norms = _params_by_prefix(params, es_prefix)
    cc_norms = _params_by_prefix(params, cc_prefix)

    ES_bg_par_counts = raw_backgrounds.ES_par_counts
    CC_bg_par_counts = raw_backgrounds.CC_par_counts

    # ES backgrounds
    if isempty(raw_backgrounds.ES)
        BG_ES = Float64[]
    else
        T_es = isempty(es_norms) ? Float64 : typeof(first(es_norms) * one(Float64))
        BG_ES = zeros(T_es, length(raw_backgrounds.ES[1]))
        norm_index = 1
        for (i, behaviour) in enumerate(ES_bg_par_counts)
            if behaviour != 0
                factor = ES_conversions(raw_backgrounds.sides[i], es_norms[norm_index])
                @. BG_ES += raw_backgrounds.ES[i] * factor
                norm_index += 1
            else
                BG_ES .+= raw_backgrounds.ES[i]
            end
        end
    end

    # CC backgrounds
    if isempty(raw_backgrounds.CC)
        BG_CC = Float64[]
    else
        T_cc = isempty(cc_norms) ? Float64 : typeof(first(cc_norms) * one(Float64))
        BG_CC = zeros(T_cc, length(raw_backgrounds.CC[1]))
        norm_index = 1
        for (i, behaviour) in enumerate(CC_bg_par_counts)
            if behaviour != 0
                factor = CC_conversions(cc_norms[norm_index])
                @. BG_CC += raw_backgrounds.CC[i] * factor
                norm_index += 1
            else
                BG_CC .+= raw_backgrounds.CC[i]
            end
        end
    end

    return BG_ES, BG_CC
end