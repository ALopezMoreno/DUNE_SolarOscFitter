# out .+= x .* factor as a loop: Enzyme rejects a broadcast holding the constant `x` and the active
# `factor` together.
function _add_scaled!(out::AbstractVector, x::AbstractVector, factor)
    @inbounds for k in eachindex(out, x)
        out[k] += x[k] * factor
    end
    return out
end

"""
    normalize_backgrounds(raw_backgrounds, params, det_name::String)

Summed ES and CC background spectra (BG_ES, BG_CC), each free background scaled by its norm
parameter (`:<det>_ES_bg_norm_i`, …) and conversion factor (build_backgrounds).
"""
normalize_backgrounds(raw_backgrounds, params, det_name::String) =
    normalize_backgrounds(raw_backgrounds, params, Val(Symbol(det_name, "_ES_bg_norm_")),
                          Val(Symbol(det_name, "_CC_bg_norm_")))

# Type-stable form: prefixes known at compile time (bg_norm_prefixes)
normalize_backgrounds(raw_backgrounds, params, det_flags::DetFlags) =
    normalize_backgrounds(raw_backgrounds, params, bg_norm_prefixes(det_flags)...)

function normalize_backgrounds(raw_backgrounds, params, es_prefix::Val, cc_prefix::Val)
    es_norms = _params_by_prefix(params, es_prefix)
    cc_norms = _params_by_prefix(params, cc_prefix)

    ES_bg_par_counts = raw_backgrounds.ES_par_counts
    CC_bg_par_counts = raw_backgrounds.CC_par_counts

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
                _add_scaled!(BG_ES, raw_backgrounds.ES[i], 1.0)
            end
        end
    end

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
                _add_scaled!(BG_CC, raw_backgrounds.CC[i], 1.0)
            end
        end
    end

    return BG_ES, BG_CC
end