

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
            backgrounds.ES[i] .*= getfield(params, Symbol("ES_bg_norm_", norm_index))
            norm_index += 1
        end
    end

    # CC backgrounds
    norm_index = 1
    for (i, behaviour) in enumerate(CC_bg_par_counts)
        if behaviour != 0
            backgrounds.CC[i] .*= getfield(params, Symbol("CC_bg_norm_", norm_index))
            norm_index += 1
        end
    end

    BG_ES = reduce(+, backgrounds.ES)
    BG_CC = reduce(+, backgrounds.CC)

    return backgrounds, BG_ES, BG_CC
end