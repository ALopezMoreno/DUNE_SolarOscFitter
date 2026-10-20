
using LinearAlgebra

"""
    DetFlags(; ES_mode, CC_mode, angular_reco, inclusive_analysis, semi_inclusive_analysis,
               angular_cos_cut, det_name)

Per-detector mode flags. The Bool modes are type parameters, so `if det_flags.ES_mode` is resolved
at compile time and propagateSamples stays type-stable (required by Enzyme).
"""
struct DetFlags{ES,CC,ANG,INCL,SEMI,NAME}
    angular_cos_cut::Float64
    det_name::String
end

DetFlags(; ES_mode::Bool, CC_mode::Bool, angular_reco::Bool, inclusive_analysis::Bool,
         semi_inclusive_analysis::Bool, angular_cos_cut::Real, det_name::AbstractString) =
    DetFlags{ES_mode,CC_mode,angular_reco,inclusive_analysis,semi_inclusive_analysis,Symbol(det_name)}(
        Float64(angular_cos_cut), String(det_name))

# Parameter-name prefixes of this detector's background norms, as compile-time Val types.
@generated function bg_norm_prefixes(::DetFlags{ES,CC,ANG,INCL,SEMI,NAME}) where {ES,CC,ANG,INCL,SEMI,NAME}
    es, cc = QuoteNode(Symbol(NAME, "_ES_bg_norm_")), QuoteNode(Symbol(NAME, "_CC_bg_norm_"))
    return :((Val($es), Val($cc)))
end

@inline function Base.getproperty(f::DetFlags{ES,CC,ANG,INCL,SEMI,NAME}, s::Symbol) where {ES,CC,ANG,INCL,SEMI,NAME}
    s === :ES_mode                 && return ES
    s === :CC_mode                 && return CC
    s === :angular_reco            && return ANG
    s === :inclusive_analysis      && return INCL
    s === :semi_inclusive_analysis && return SEMI
    return getfield(f, s)
end

Base.propertynames(::DetFlags) = (:ES_mode, :CC_mode, :angular_reco, :inclusive_analysis,
                                  :semi_inclusive_analysis, :angular_cos_cut, :det_name)

# Averages over rectangular blocks (fine calculation grid → analysis bins)
function block_average(mat::AbstractMatrix, block_dims::Tuple{Int,Int}=(5, 3))
    block_rows, block_cols = block_dims
    n_rows, n_cols = size(mat)
    
    if n_rows % block_rows != 0 || n_cols % block_cols != 0
        error("Matrix dimensions must be multiples of block dimensions: got $(n_rows)x$(n_cols) for blocks of size $(block_rows)x$(block_cols)")
    end

    out_n = n_rows ÷ block_rows
    out_m = n_cols ÷ block_cols
    result = Array{eltype(mat)}(undef, out_n, out_m)

    for i in 1:out_n
        for j in 1:out_m
            rows_range = ((i - 1) * block_rows + 1):(i * block_rows)
            cols_range = ((j - 1) * block_cols + 1):(j * block_cols)
            block = view(mat, rows_range, cols_range)
            result[i, j] = sum(block) / (block_rows * block_cols)
        end
    end

    return result
end

function block_average(vec::AbstractVector, block_size::Int=5)
    n = length(vec)
    
    if n % block_size != 0
        error("Vector length must be a multiple of block size: got length $(n) for blocks of size $(block_size)")
    end
    
    out_n = n ÷ block_size
    result = Array{eltype(vec)}(undef, out_n)
    
    for i in 1:out_n
        range = ((i - 1) * block_size + 1):(i * block_size)
        block = view(vec, range)
        result[i] = sum(block) / block_size
    end
    
    return result
end


# Fold oscillated spectra with the response and efficiency: day scale·((R'·osc) .* eff), night the
# same per cos z row. `scale` is the signal's day or night exposure share (SPEC 4.6), never defaulted.
function apply_day_response(osc::AbstractVector, response::AbstractMatrix, eff::AbstractVector; scale::Real)
    # loops: Enzyme rejects the Adjoint of the constant response next to the active `osc`
    n_true, n_reco = size(response)
    length(osc) == n_true || throw(DimensionMismatch("osc has $(length(osc)) true bins, response has $n_true"))
    T = promote_type(eltype(osc), eltype(response), eltype(eff), typeof(scale))
    out = Vector{T}(undef, n_reco)
    @inbounds for j in 1:n_reco
        acc = zero(T)
        for i in 1:n_true
            acc += response[i, j] * osc[i]
        end
        out[j] = scale * (acc * eff[j])
    end
    return out
end

function apply_night_response(osc_rows::AbstractMatrix, response::AbstractMatrix, eff::AbstractVector; scale::Real)
    # loops: a vcat splat would be a dynamic call Enzyme cannot differentiate
    n_rows, n_true = size(osc_rows)
    n_reco = size(response, 2)
    size(response, 1) == n_true || throw(DimensionMismatch("osc rows have $n_true true bins, response has $(size(response, 1))"))
    T = promote_type(eltype(osc_rows), eltype(response), eltype(eff), typeof(scale))
    out = Matrix{T}(undef, n_rows, n_reco)
    @inbounds for j in 1:n_reco, k in 1:n_rows
        acc = zero(T)
        for i in 1:n_true
            acc += osc_rows[k, i] * response[i, j]
        end
        out[k, j] = scale * (acc * eff[j])
    end
    return out
end



# Fraction of the CC background the CC prediction carries: f_below in semi-inclusive mode, else 1.
cc_background_fraction(semi::Bool, R) = semi ? R.CC_split.below : 1.0

"""
    dn_asymmetry(D, N, f_day, f_night)

Day-night asymmetry 2(d − n)/(d + n) per unit exposure, d = D/f_day and n = N/f_night with the
signal's split (SPEC 4.6); 0 when d + n = 0.
"""
function dn_asymmetry(D::Real, N::Real, f_day::Real, f_night::Real)
    d, n = D / f_day, N / f_night
    return d + n == 0 ? 0.0 : 2 * (d - n) / (d + n)
end

"""
    dn_asymmetry(day, night, BG, idx, signal_fractions, bg_fractions)

Signal day-night asymmetry over E_reco bins idx:end of predicted arrays that include the background
BG. The background is removed with the factors the predictions carry (`bg_fractions`, i.e.
BG_TOTAL_FACTORS), the asymmetry taken with the signal split. Energy is the last axis of `day` and
axis 2 of `night`.
"""
function dn_asymmetry(day::AbstractArray, night::AbstractArray, BG::AbstractVector, idx::Integer,
                      signal_fractions, bg_fractions)
    bg = sum(BG[idx:end])
    N = sum(selectdim(night, 2, idx:size(night, 2))) - bg_fractions.night * bg
    D = sum(selectdim(day, ndims(day), idx:size(day, ndims(day)))) - bg_fractions.day * bg
    return dn_asymmetry(D, N, signal_fractions.day, signal_fractions.night)
end
