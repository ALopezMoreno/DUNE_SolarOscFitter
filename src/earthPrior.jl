#=
earthPrior.jl — covariance of the fast-mode Earth layer scales from full-model Earth draws (SPEC 2.9).
Used by utils/make_earth_prior.jl; no fitter globals.
=#
using LinearAlgebra, Statistics

"""
    nnls(A, b) -> x

Non-negative least squares, min ‖A x − b‖ subject to x ≥ 0 (Lawson & Hanson 1974, active set).
"""
function nnls(A::AbstractMatrix, b::AbstractVector; tol=1e-12 * opnorm(A, 1) * norm(b, Inf), maxiter=30size(A, 2))
    n = size(A, 2)
    x, P = zeros(n), falses(n)
    for _ in 1:maxiter
        g = A' * (b .- A * x)
        (all(P) || maximum(g[.!P]) <= tol) && return x
        P[argmax(ifelse.(P, -Inf, g))] = true
        while true
            z = zeros(n)
            z[P] = A[:, P] \ b
            all(>(0), z[P]) && (x = z; break)
            neg = P .& (z .<= 0)
            α = minimum(x[neg] ./ (x[neg] .- z[neg]))
            x .+= α .* (z .- x)
            P .&= x .> tol
            x[.!P] .= 0
        end
    end
    error("nnls: no convergence in $maxiter iterations")
end

"""
    bin_variance(J, Σ) -> Vector

Per-bin variance J Σ Jᵀ (diagonal only) of a linear response `J` (bins × parameters).
"""
bin_variance(J, Σ) = vec(sum((J * Σ) .* J; dims=2))

"""
    earth_prior_covariance(J, D, w) -> (; mean, cov_projection, extra_variance, cov)

Prior covariance of the layer scales from draws of the full Earth model.

`J` is the response of the binned night P₁ₑ to the layer scales at the nominal Earth (bins × layers),
`D` each draw's deviation from the nominal (bins × draws), `w` the bin weights. Each draw is projected
onto the layer scales by weighted least squares; their covariance (`cov_projection`) misses the part of
the draws' per-bin variance the scales cannot represent. Independent per-layer variances
(`extra_variance` ≥ 0) are added so that J Σ Jᵀ matches the draws' per-bin variance in the weighted
least-squares sense. `mean` is the mean projected deviation from the nominal.
"""
function earth_prior_covariance(J::AbstractMatrix, D::AbstractMatrix, w::AbstractVector)
    sw = sqrt.(w)
    θ = (sw .* J) \ (sw .* D)
    Σp = Symmetric(cov(θ; dims=2))
    s = nnls(sw .* J .^ 2, sw .* (vec(var(D; dims=2)) .- bin_variance(J, Σp)))
    return (; mean=vec(mean(θ; dims=2)), cov_projection=Matrix(Σp), extra_variance=s,
            cov=Matrix(Σp + Diagonal(s)))
end
