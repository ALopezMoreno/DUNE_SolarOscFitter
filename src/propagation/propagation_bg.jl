function CC_conversions(params, norm_index)
    # The MC is normalised to 2.2 x 10^-6 neutrons / cm2 / s
    factor = getfield(params, Symbol("CC_bg_norm_", norm_index)) / 2.2e-6
    return factor
end

function ES_conversions(side, params, norm_index)
    # The MC is not normalised. We need to load metadata
    if side == 0
        # This is the long side of the far detector module
        area = 2 * 12e2*64e2
    elseif side == 1
        # This is the short side of the far detector module
        area = 2 * 12e2*12e2
    else
        # No side information. Assume 4side production region
        area = 2 * 12e2*64e2 + 2 * 12e2*12e2
    end
    
    factor = area * getfield(params, Symbol("ES_bg_norm_", norm_index))
    return factor
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
    # ES backgrounds
    if isempty(raw_backgrounds.ES)
        BG_ES = Float64[]
    else
        BG_ES = zeros(length(raw_backgrounds.ES[1]))
        norm_index = 1
        for (i, behaviour) in enumerate(ES_bg_par_counts)
            if behaviour != 0
                factor = ES_conversions(raw_backgrounds.sides[i], params, norm_index)
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
        BG_CC = zeros(length(raw_backgrounds.CC[1]))
        norm_index = 1
        for (i, behaviour) in enumerate(CC_bg_par_counts)
            if behaviour != 0
                factor = CC_conversions(params, norm_index)
                @. BG_CC += raw_backgrounds.CC[i] * factor
                norm_index += 1
            else
                BG_CC .+= raw_backgrounds.CC[i]
            end
        end
    end

    return BG_ES, BG_CC
end