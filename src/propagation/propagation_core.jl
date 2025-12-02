
using LinearAlgebra

# Block averaging utilities for energy binning
# These functions reduce high-resolution calculations to analysis binning

# Block averaging for 2D matrices (e.g., zenith vs energy oscillation probabilities)
function block_average(mat::AbstractMatrix, block_dims::Tuple{Int,Int}=(5, 3))
    """Average matrix elements over rectangular blocks"""
    block_rows, block_cols = block_dims
    n_rows, n_cols = size(mat)
    
    # Ensure that matrix dimensions are multiples of block dimensions
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
            # Calculate average over the block
            result[i, j] = sum(block) / (block_rows * block_cols)
        end
    end

    return result
end

# Block averaging for 1D vectors (e.g., energy-dependent oscillation probabilities)
function block_average(vec::AbstractVector, block_size::Int=5)
    """Average vector elements over consecutive blocks"""
    n = length(vec)
    
    # Ensure that vector length is a multiple of block size
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

# Generic wrapper function for block averaging
function block_average(arr::AbstractArray, block_dims...)
    """Dispatch to appropriate block averaging function based on array dimension"""
    if ndims(arr) == 1
        return block_average(arr, block_dims...)
    elseif ndims(arr) == 2
        return block_average(arr, block_dims...)
    else
        error("Unsupported array dimension: $(ndims(arr)). Only 1D and 2D arrays are supported.")
    end
end


# Helper functions to fold oscillated spectra with response matrices and efficiencies
# - Day:   scale * ((R' * osc_vec) .* eff)
# - Night: stack over cosZ rows: scale * ((row' * R) .* eff')
function apply_day_response(osc::AbstractVector, response::AbstractMatrix, eff::AbstractVector; scale::Real=0.5)
    return scale .* ((response' * osc) .* eff)
end

function apply_night_response(osc_rows::AbstractMatrix, response::AbstractMatrix, eff::AbstractVector; scale::Real=0.5)
    return vcat([scale .* ((row' * response) .* eff') for row in eachrow(osc_rows)]...)
end

