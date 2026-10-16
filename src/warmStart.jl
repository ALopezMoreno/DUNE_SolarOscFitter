#=
warmStart.jl — start the HMC sampler from a previous run's posterior (config key warm_start_chain).

BAT samples in a pretransformed space (PriorToNormal: each parameter mapped so its prior is a
standard normal) and, on top of it, an affine map z ↦ A z + b that plays the role of the mass
matrix. With HMC, BAT's default tuning of that map (RAMTuning, a random-walk Metropolis rule)
does not learn the posterior shape: on a toy posterior with a ×100 scale spread it returned
A ≈ identity from a cold start, and it erased a correct A it was started from, within the
burn-in. NUTS then needs ~100 gradients per step where 8 suffice.

The warm start therefore
  - maps the previous run's samples into the pretransformed space,
  - sets b = their mean and A = chol(their covariance), and keeps A fixed (no RAM tuning; the
    step size is still adapted),
  - starts the chains from previous posterior draws.
The previous run must have the same parameters (it may have more: extra ones are ignored) and
should have the same priors, since the pretransform depends on them; a changed likelihood is
fine, as long as the posterior has not moved by much more than its width.
=#

using JLD2, Statistics, LinearAlgebra
using BAT: DensitySampleVector

# Saved samples of the run with output prefix `prefix` (the outFile of that run, with or without
# ".jld2"): name ⇒ values over all batches, and the weights. Only `names` are returned; each must
# be present.
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

# Weighted mean and covariance of the columns of Z (one sample per column).
function weighted_mean_cov(Z::AbstractMatrix, w::AbstractVector)
    W = sum(w)
    μ = Z * w ./ W
    D = Z .- μ
    Σ = (D .* reshape(w, 1, :)) * D' ./ (W - sum(abs2, w) / W)     # unbiased for frequency weights
    return μ, Symmetric(Σ)
end

# Warm start from the run `prefix` for the posterior `posterior` (prior `prior`), sampled with
# pretransform `pretransform`. Returns (transform = BAT adaptive transform, init = BAT init
# algorithm, nsamples, scales = sqrt(diag Σ) in the pretransformed space).
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
    ok = vec(all(isfinite, Z; dims = 1))        # samples at (or past) a prior boundary map to ±Inf
    count(!, ok) <= 0.01 * length(ok) ||
        error("warm_start_chain: $(count(!, ok)) of $(length(ok)) samples in $prefix fall outside the " *
              "current prior support (were the priors changed?)")
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
