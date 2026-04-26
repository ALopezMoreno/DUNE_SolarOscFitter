function CC_conversions(norm_value)
    # The MC is normalised to 2.2 x 10^-6 neutrons / cm2 / s
    return norm_value / 2.2e-6
end

function ES_conversions(side, norm_value)
    # The MC is not normalised. We need to load metadata
    if side == 0
        area = 2 * 12e2*64e2
    elseif side == 1
        area = 2 * 12e2*12e2
    else
        area = 2 * 12e2*64e2 + 2 * 12e2*12e2
    end
    return area * norm_value
end

"""
    normalize_backgrounds(raw_backgrounds, params)

Compute the summed ES and CC background vectors by applying nuisance-parameter
scale factors on-the-fly, without copying the raw arrays.

Uses globals:
- `ES_bg_par_counts`
- `CC_bg_par_counts`

Returns:
- BG_ES :: Vector{Float64}
- BG_CC :: Vector{Float64}
"""
function normalize_backgrounds(raw_backgrounds, params)
    es_norms = _params_by_prefix(params, Val(:ES_bg_norm_))
    cc_norms = _params_by_prefix(params, Val(:CC_bg_norm_))

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