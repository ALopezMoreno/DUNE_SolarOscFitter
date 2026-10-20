#=
warmStart.jl — start the HMC sampler from a previous run's posterior (config key warm_start_chain).

BAT's RAM tuning of the affine map z ↦ A z + b (the mass matrix, on top of PriorToNormal) does not
learn the posterior shape under HMC, so NUTS needs far more gradients per step than it should. The
warm start sets b and A = chol(Σ) from the previous run's samples in the pretransformed space,
keeps A fixed (the step size is still adapted) and starts the chains from its draws. The previous
run needs the same parameters (extra ones are ignored) and priors.
=#

using JLD2, Statistics, LinearAlgebra
using BAT: DensitySampleVector

# Samples (name ⇒ values over all batches) and weights of the run with outFile `prefix`
function load_chain_samples(prefix::AbstractString, names)
    file = endswith(prefix, ".jld2") ? prefix : prefix * ".jld2"
    isfile(file) || error("warm_start_chain: $file not found")
    cols = Dict{Symbol,Vector{Any}}(n => Any[] for n in names)
    w = Float64[]
    jldopen(file, "r") do f
        batches = sort([k for k in keys(f) if startswith(k, "batch_") && tryparse(Int, k[7:end]) !== nothing],
                       by = k -> parse(Int, k[7:end]))
        isempty(batches) && error("warm_start_chain: $file has no batch_* groups")
        for b in batches
            g = f[b]
            missing_names = [n for n in names if !haskey(g, string(n))]
            isempty(missing_names) ||
                error("warm_start_chain: $file ($b) has no samples for $(join(missing_names, ", ")); " *
                      "the previous run must have the same fitted parameters")
            for n in names
                append!(cols[n], g[string(n)])
            end
            append!(w, g["weights"])
        end
    end
    return cols, w
end

# Weighted mean and covariance of the columns of Z (weights are repeat counts)
function weighted_mean_cov(Z::AbstractMatrix, w::AbstractVector)
    W = sum(w)
    μ = Z * w ./ W
    D = Z .- μ
    Σ = (D .* reshape(w, 1, :)) * D' ./ (W - 1)
    return μ, Symmetric(Σ)
end

# (transform, init, nsamples, scales) for BAT from the run `prefix`
function build_warm_start(prefix::AbstractString, posterior, prior, pretransform;
                          max_samples::Integer = 20_000, init = nothing)
    names = collect(keys(BAT.varshape(prior)))
    cols, w = load_chain_samples(prefix, names)
    n = length(w)
    n > 2 * length(names) || error("warm_start_chain: only $n samples in $prefix for $(length(names)) parameters")
    keep = n <= max_samples ? (1:n) : round.(Int, range(1, n, length = max_samples))   # thin evenly
    nts = [NamedTuple{Tuple(names)}(Tuple(cols[k][i] for k in names)) for i in keep]
    wk = w[keep]

    _, f_pretransform = BAT.transform_and_unshape(pretransform, posterior, BATContext())
    Z = reduce(hcat, [f_pretransform(x) for x in nts])
    # drop samples outside the prior (NaN) or clamped on its boundary (|z| ≈ 8), which inflate Σ
    ok = vec(all(z -> isfinite(z) && abs(z) < 7.5, Z; dims = 1))
    count(!, ok) <= 0.01 * length(ok) ||
        error("warm_start_chain: $(count(!, ok)) of $(length(ok)) samples in $prefix lie on or outside the " *
              "current prior boundaries (were the priors changed?)")
    count(!, ok) > 0 && @warn "warm_start_chain: dropped $(count(!, ok)) samples on or outside a prior boundary"
    Z, nts, wk = Z[:, ok], nts[ok], wk[ok]
    μ, Σ = weighted_mean_cov(Z, wk)
    C = cholesky(Σ; check = false)
    issuccess(C) || error("warm_start_chain: covariance of the samples in $prefix is not positive definite " *
                          "(a parameter that never moved?)")
    transform = BAT.CustomTransform(BAT.MulAdd(LowerTriangular(Matrix(C.L)), μ))

    vs = BAT.varshape(prior)
    flat = BAT.ArrayOfSimilarArrays([Vector{Float64}(BAT.unshaped(x, vs)) for x in nts])
    smp = vs.(DensitySampleVector(flat, zeros(length(nts)); weight = round.(Int, wk)))
    init_alg = init === nothing ? MCMCChainPoolInit(initval_alg = InitFromSamples(smp)) :
        MCMCChainPoolInit(init.init_tries_per_chain, init.nsteps_init, InitFromSamples(smp), init.strict)
    return (transform = transform, init = init_alg, nsamples = length(nts),
            scales = sqrt.(diag(Σ)), names = names)
end
