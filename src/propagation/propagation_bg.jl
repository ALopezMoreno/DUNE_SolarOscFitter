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

Deep-copy and normalize ES/CC backgrounds according to the nuisance parameters
in `params` and the behaviour arrays `ES_bg_par_counts` and `CC_bg_par_counts`.

Uses globals:
- `ES_bg_par_counts`
- `CC_bg_par_counts`

Returns:
- backgrounds (normalized)
- BG_ES :: Vector
- BG_CC :: Vector
"""
function normalize_backgrounds(raw_backgrounds, params)
    backgrounds = deepcopy(raw_backgrounds)

    # ES backgrounds
    norm_index = 1
    for (i, behaviour) in enumerate(ES_bg_par_counts)
        if behaviour != 0
            # The MC used 50M events, but only those that interacted are saved

            backgrounds.ES[i] .*= ES_conversions(backgrounds.sides[i], params, norm_index)
        end
    end

    # CC backgrounds
    norm_index = 1
    for (i, behaviour) in enumerate(CC_bg_par_counts)
        if behaviour != 0
            backgrounds.CC[i] .*= CC_conversions(params, norm_index)
            norm_index += 1
        end
    end

    BG_ES = reduce(+, backgrounds.ES)
    BG_CC = reduce(+, backgrounds.CC)

    return backgrounds, BG_ES, BG_CC
end