# out .+= x .* factor as a plain loop: a fused broadcast would hold the constant template
# `x` and the differentiable `factor` in one object, which Enzyme's activity analysis rejects.
function _add_scaled!(out::AbstractVector, x::AbstractVector, factor)
    @inbounds for k in eachindex(out, x)
        out[k] += x[k] * factor
    end
    return out
end

"""
    normalize_backgrounds(raw_backgrounds, params, det_name::String)

Compute the summed ES and CC background vectors by applying nuisance-parameter
scale factors on-the-fly, without copying the raw arrays.

`det_name` is used to construct the parameter-name prefix, e.g. `"VD"` →
looks up `:VD_ES_bg_norm_1`, `:VD_ES_bg_norm_2`, … in `params`.

`raw_backgrounds` must carry `ES_par_counts`, `CC_par_counts` and the per-background
conversion factors `ES_conv` (face area, cm²) and `CC_conv` (1 / reference flux), all set
by `build_backgrounds`. The i-th background with a free normalisation uses the i-th
`*_bg_norm_` parameter.

Returns:
- BG_ES :: Vector{Float64}
- BG_CC :: Vector{Float64}
"""
normalize_backgrounds(raw_backgrounds, params, det_name::String) =
    normalize_backgrounds(raw_backgrounds, params, Val(Symbol(det_name, "_ES_bg_norm_")),
                          Val(Symbol(det_name, "_CC_bg_norm_")))

# Type-stable form: the prefixes are known at compile time (see bg_norm_prefixes).
normalize_backgrounds(raw_backgrounds, params, det_flags::DetFlags) =
    normalize_backgrounds(raw_backgrounds, params, bg_norm_prefixes(det_flags)...)

function normalize_backgrounds(raw_backgrounds, params, es_prefix::Val, cc_prefix::Val)
    es_norms = _params_by_prefix(params, es_prefix)
    cc_norms = _params_by_prefix(params, cc_prefix)

    ES_bg_par_counts = raw_backgrounds.ES_par_counts
    CC_bg_par_counts = raw_backgrounds.CC_par_counts

    # ES backgrounds
    if isempty(raw_backgrounds.ES)
        BG_ES = Float64[]
    else
        T_es = promote_type(Float64, map(typeof, es_norms)...)   # inferable, also for ()
        BG_ES = zeros(T_es, length(raw_backgrounds.ES[1]))
        norm_index = 1
        for (i, behaviour) in enumerate(ES_bg_par_counts)
            if behaviour != 0
                factor = raw_backgrounds.ES_conv[i] * es_norms[norm_index]   # face area × flux
                _add_scaled!(BG_ES, raw_backgrounds.ES[i], factor)
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
        T_cc = promote_type(Float64, map(typeof, cc_norms)...)   # inferable, also for ()
        BG_CC = zeros(T_cc, length(raw_backgrounds.CC[1]))
        norm_index = 1
        for (i, behaviour) in enumerate(CC_bg_par_counts)
            if behaviour != 0
                factor = raw_backgrounds.CC_conv[i] * cc_norms[norm_index]   # norm / reference flux
                _add_scaled!(BG_CC, raw_backgrounds.CC[i], factor)
                norm_index += 1
            else
                BG_CC .+= raw_backgrounds.CC[i]
            end
        end
    end

    return BG_ES, BG_CC
end