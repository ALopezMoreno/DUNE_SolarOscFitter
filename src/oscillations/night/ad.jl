#################################################################################
# ForwardDiff: one Jacobian per parameter point
#################################################################################

struct _P1eTag end

# Value and Jacobian of the bin averages at Float64 inputs, in one dual pass
# (layers crossed by no path get zero columns)
function _value_jacobian(b::BinnedP1e, x::Vector{Float64}, has_norms::Bool, ::Val{N}) where {N}
    active = has_norms ? vcat(1:3, 3 .+ b.used_layers) : collect(1:3)
    length(active) == N || error("BinnedP1e: expected $N active inputs, got $(length(active))")
    d = ntuple(i -> ForwardDiff.Dual{_P1eTag}(x[active[i]], ntuple(j -> Float64(i == j), Val(N))), Val(N))
    D = eltype(d)
    norms = has_norms ? D[D(v) for v in x[4:end]] : nothing
    if has_norms
        for (i, l) in enumerate(b.used_layers)
            norms[l] = d[3+i]
        end
    end
    out = b(d[1], d[2], d[3], norms)
    val = ForwardDiff.value.(out)
    J = zeros(length(out), length(x))
    for (k, o) in enumerate(out), i in 1:N
        J[k, active[i]] = ForwardDiff.partials(o, i)
    end
    return val, J
end

# Cache of the last point: ForwardDiff calls once per chunk at the same point
function _cached_value_jacobian(b::BinnedP1e, x::Vector{Float64}, has_norms::Bool)
    store = _task_store(b)
    key = (:jacobian, has_norms)
    hit = get(store, key, nothing)
    if hit !== nothing && hit[1] == x
        return hit[2], hit[3]
    end
    val, J = has_norms ? _value_jacobian(b, x, true, Val(3 + length(b.used_layers))) :
                         _value_jacobian(b, x, false, Val(3))
    store[key] = (copy(x), val, J)
    return val, J
end

"""
    binned_p1e(b, dm2, th12, th13, layer_norms=nothing)

Same as `b(dm2, th12, th13, layer_norms)`, with fast derivatives: for ForwardDiff duals the
Jacobian is computed once per point and cached; Float64 calls carry an Enzyme reverse rule.
"""
function binned_p1e(b::BinnedP1e, dm2, th12, th13, layer_norms=nothing)
    args = layer_norms === nothing ? (dm2, th12, th13) : (dm2, th12, th13, layer_norms...)
    D = promote_type(map(typeof, args)...)
    D === Float64 && return _p1e_vec(b, Float64[a for a in args])
    (D <: ForwardDiff.Dual && ForwardDiff.valtype(D) === Float64) || return b(dm2, th12, th13, layer_norms)
    x = Float64[ForwardDiff.value(a) for a in args]
    val, J = _cached_value_jacobian(b, x, layer_norms !== nothing)
    N = ForwardDiff.npartials(D)
    P = zeros(length(args), N)
    for (i, a) in enumerate(args)
        a isa ForwardDiff.Dual && (P[i, :] .= ForwardDiff.partials(a).values)
    end
    return _assemble(D, val, J * P)
end

function _assemble(::Type{D}, val::Matrix{Float64}, JP::Matrix{Float64}) where {D<:ForwardDiff.Dual{<:Any,Float64,N}} where {N}
    out = Matrix{D}(undef, size(val))
    for k in eachindex(val)
        out[k] = D(val[k], ForwardDiff.Partials(ntuple(n -> JP[k, n], Val(N))))
    end
    return out
end

#################################################################################
# Reverse pass: hand-written adjoints of stages 2-3, Enzyme on stage 1
#################################################################################

# Complex cotangents: z̄ = ∂L/∂Re z + i ∂L/∂Im z


# Adjoint scratch for one interval
struct _AdjWorkspace
    kap_all::Array{ComplexF64,3}   # stage-1 outputs of every path
    psi_all::Array{Float64,3}
    Sxbar::Array{ComplexF64,3}
    Pxbar::Array{Float64,3}
    Rxbar::Matrix{Float64}
    Sbar::Array{ComplexF64,3}
    Pbar::Array{Float64,3}
    Rbar::Matrix{Float64}
    kbar::Matrix{ComplexF64}
    pbar::Matrix{Float64}
    Ckbar::Matrix{ComplexF64}
    Pkbar::Matrix{Float64}
    Cubar::Vector{ComplexF64}
    Pubar::Vector{Float64}
    accbar::Vector{ComplexF64}
    dkap::Vector{ComplexF64}       # Enzyme shadows of _node!'s buffers
    dpsi::Vector{Float64}
    deph::Vector{ComplexF64}
    kapn::Vector{ComplexF64}       # _node!'s buffers
    psin::Vector{Float64}
    ephn::Vector{ComplexF64}
end

function _AdjWorkspace(iv::_Interval)
    n_paths, n_steps = size(iv.L)
    nb, n_nodes, n_u, n_bins, n_xi = n_steps + 1, length(iv.nodes), length(iv.u), length(iv.width), length(iv.xi)
    C, R = ComplexF64, Float64
    return _AdjWorkspace(zeros(C, nb, n_nodes, n_paths), zeros(R, nb, n_nodes, n_paths),
                         zeros(C, n_bins, nb, n_xi), zeros(R, n_bins, nb, n_xi), zeros(R, n_bins, n_xi),
                         zeros(C, n_bins, nb, n_paths), zeros(R, n_bins, nb, n_paths), zeros(R, n_bins, n_paths),
                         zeros(C, nb, n_nodes), zeros(R, nb, n_nodes), zeros(C, nb, n_u), zeros(R, nb, n_u),
                         zeros(C, n_u), zeros(R, n_u), zeros(C, n_bins),
                         zeros(C, nb), zeros(R, nb), zeros(C, nb), zeros(C, nb), zeros(R, nb), zeros(C, nb))
end

function _adj_workspace(b::BinnedP1e, k::Int)
    return get!(() -> _AdjWorkspace(b.intervals[k]), _task_store(b), (k, :adjoint))::_AdjWorkspace
end

# Exposure normalisation per zenith bin
function _den(b::BinnedP1e)
    den = zeros(b.shape[1])
    for iv in b.intervals, q in axes(iv.pieces, 1)
        t0, tm, t1 = iv.pieces[q, 1], iv.pieces[q, 2], iv.pieces[q, 3]
        den[iv.piece_bin[q]] += (iv.xi[t1] - iv.xi[t0]) * (iv.w_xi[t0] + 4 * iv.w_xi[tm] + iv.w_xi[t1]) / 6
    end
    return den
end

# Adjoint of _contract_last!
function _contract_last_adj!(inbar::Array{E}, outbar::Array{E}, B::Matrix{Float64}) where {E}
    m_in, m_out = length(inbar) ÷ size(inbar, ndims(inbar)), length(outbar) ÷ size(outbar, ndims(outbar))
    Ib, Ob = reshape(inbar, m_in, :), reshape(outbar, m_out, :)
    fill!(Ib, zero(E))
    @inbounds for t in axes(B, 1), p in axes(B, 2)
        w = B[t, p]
        w == 0 && continue
        @simd for r in axes(Ib, 1)
            Ib[r, p] += w * Ob[r, t]
        end
    end
    return inbar
end

# Adjoint of stage 3
function _zenith_adj!(aw::_AdjWorkspace, iv::_Interval, ws::_Workspace{Float64}, numbar::Matrix{Float64})
    n_bins, nb, _ = size(ws.S_bin)
    xi, w_xi = iv.xi, iv.w_xi
    Sx, Px = ws.Sx, ws.Px
    Sxbar, Pxbar, Rxbar = aw.Sxbar, aw.Pxbar, aw.Rxbar
    fill!(Sxbar, 0); fill!(Pxbar, 0); fill!(Rxbar, 0)
    @inbounds for bb in 1:nb, q in axes(iv.pieces, 1)
        t0, tm, t1 = iv.pieces[q, 1], iv.pieces[q, 2], iv.pieces[q, 3]
        hq = xi[t1] - xi[t0]
        i = iv.piece_bin[q]
        for j in 1:n_bins
            g = hq * numbar[i, j]
            g == 0 && continue
            p0, pm, p1 = Px[j, bb, t0], Px[j, bb, tm], Px[j, bb, t1]
            f0b, fmb, f1b, p0b, pmb, p1b = _filon_adj(complex(g), w_xi[t0] * Sx[j, bb, t0], w_xi[tm] * Sx[j, bb, tm],
                                                      w_xi[t1] * Sx[j, bb, t1], p0, pm, p1, cis(-p0), cis(-p1))
            Sxbar[j, bb, t0] += w_xi[t0] * f0b
            Sxbar[j, bb, tm] += w_xi[tm] * fmb
            Sxbar[j, bb, t1] += w_xi[t1] * f1b
            Pxbar[j, bb, t0] += p0b
            Pxbar[j, bb, tm] += pmb
            Pxbar[j, bb, t1] += p1b
        end
    end
    @inbounds for q in axes(iv.pieces, 1)
        t0, tm, t1 = iv.pieces[q, 1], iv.pieces[q, 2], iv.pieces[q, 3]
        hq = xi[t1] - xi[t0]
        i = iv.piece_bin[q]
        for j in 1:n_bins
            g = hq * numbar[i, j] / 6
            Rxbar[j, t0] += g * w_xi[t0]
            Rxbar[j, tm] += 4 * g * w_xi[tm]
            Rxbar[j, t1] += g * w_xi[t1]
        end
    end
    _contract_last_adj!(aw.Sbar, Sxbar, iv.B_xi)
    _contract_last_adj!(aw.Pbar, Pxbar, iv.B_xi)
    _contract_last_adj!(aw.Rbar, Rxbar, iv.B_xi)
    return aw
end

# Adjoint of stage 2 for path ip (needs this path's forward values in ws)
function _energy_path_adj!(aw::_AdjWorkspace, ws::_Workspace{Float64}, iv::_Interval, ip::Int)
    u, width = iv.u, iv.width
    nb = size(ws.kap, 1)
    Ck = _interp_nodes!(ws.Ck, ws.kap, iv.lag_first, iv.lag_w)
    Pk = _interp_nodes!(ws.Pk, ws.psi, iv.lag_first, iv.lag_w)
    Cu, Pu, Eu = ws.Cu, ws.Pu, ws.Eu
    Cubar, Pubar, accbar, Ckbar, Pkbar = aw.Cubar, aw.Pubar, aw.accbar, aw.Ckbar, aw.Pkbar
    @inbounds for bb in 1:nb
        for q in eachindex(u)
            Cu[q] = Ck[bb, q, 1] * iv.inv_u2[q]
            Pu[q] = Pk[bb, q, 1]
            Eu[q] = cis(-Pu[q])
        end
        fill!(Cubar, 0); fill!(Pubar, 0)
        for j in eachindex(width)
            st = iv.u_strip[j]
            Sb = aw.Sbar[j, bb, ip]
            Pubar[st] += real(conj(Sb) * (im * ws.S_bin[j, bb, ip])) + aw.Pbar[j, bb, ip]
            accbar[j] = Eu[st] * Sb / width[j]
        end
        for q in axes(iv.u_pieces, 1)
            t0, tm, t1 = iv.u_pieces[q, 1], iv.u_pieces[q, 2], iv.u_pieces[q, 3]
            yb = (u[t0] - u[t1]) * accbar[iv.u_piece_bin[q]]
            f0b, fmb, f1b, p0b, pmb, p1b = _filon_adj(yb, Cu[t0], Cu[tm], Cu[t1], Pu[t0], Pu[tm], Pu[t1], Eu[t0], Eu[t1])
            Cubar[t0] += f0b; Cubar[tm] += fmb; Cubar[t1] += f1b
            Pubar[t0] += p0b; Pubar[tm] += pmb; Pubar[t1] += p1b
        end
        for q in eachindex(u)
            Ckbar[bb, q] = Cubar[q] * iv.inv_u2[q]
            Pkbar[bb, q] = Pubar[q]
        end
    end
    # adjoint of the energy interpolation
    fill!(aw.kbar, 0); fill!(aw.pbar, 0)
    @inbounds for q in eachindex(u)
        i0 = iv.lag_first[q]
        for a in 1:E_ORDER
            wgt = iv.lag_w[q, a]
            for bb in 1:nb
                aw.kbar[bb, i0+a-1] += wgt * Ckbar[bb, q]
                aw.pbar[bb, i0+a-1] += wgt * Pkbar[bb, q]
            end
        end
    end
    return aw
end

# Objective whose Enzyme gradient is the stage-1 adjoint at one (path, node)
function _node_objective(θ::Vector{Float64}, iv::_Interval, ip::Int, n::Int, kbar::Matrix{ComplexF64},
                         pbar::Matrix{Float64}, rbar::Float64, kap::Vector{ComplexF64}, psi::Vector{Float64},
                         eph::Vector{ComplexF64}, phase_const::Float64)
    r = _node!(kap, psi, eph, iv, ip, n, θ[1], θ[2], θ[3], view(θ, 4:length(θ)), phase_const)
    acc = rbar * r
    @inbounds for bb in eachindex(kap)
        acc += real(conj(kbar[bb, n]) * kap[bb]) + pbar[bb, n] * psi[bb]
    end
    return acc
end

# wᵀ ∂P/∂x for x = [Δm²₂₁, θ₁₂, θ₁₃, (layer_norms...)]
function _p1e_vjp_manual(b::BinnedP1e, x::Vector{Float64}, w::Matrix{Float64})
    θ = length(x) == 3 ? vcat(x, ones(b.n_layers)) : copy(x)
    θbar = zeros(length(θ))
    norms = θ[4:end]
    den = _den(b)
    numbar = [den[i] > 0 ? w[i, j] / den[i] : 0.0 for i in axes(w, 1), j in axes(w, 2)]
    pc = b.phase_const
    for (k, iv) in enumerate(b.intervals)
        ws = _workspace(b, k, Float64)
        aw = _adj_workspace(b, k)
        # forward pass, keeping the stage-1 outputs
        fill!(ws.R_bin, 0.0)
        for ip in axes(iv.L, 1)
            _nodes_path!(ws, iv, ip, θ[1], θ[2], θ[3], norms, pc)
            aw.kap_all[:, :, ip] .= view(ws.kap, :, :, 1)
            aw.psi_all[:, :, ip] .= view(ws.psi, :, :, 1)
            _energy_path!(ws, iv, ip)
        end
        _contract_last!(ws.Sx, ws.S_bin, iv.B_xi)
        _contract_last!(ws.Px, ws.P_bin, iv.B_xi)
        _zenith_adj!(aw, iv, ws, numbar)
        for ip in axes(iv.L, 1)
            view(ws.kap, :, :, 1) .= view(aw.kap_all, :, :, ip)
            view(ws.psi, :, :, 1) .= view(aw.psi_all, :, :, ip)
            _energy_path_adj!(aw, ws, iv, ip)
            for n in eachindex(iv.nodes)
                rbar = 0.0
                for j in axes(aw.Rbar, 1)
                    rbar += aw.Rbar[j, ip] * iv.W_r[n, j]
                end
                fill!(aw.dkap, 0); fill!(aw.dpsi, 0); fill!(aw.deph, 0)
                Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(_node_objective), Enzyme.Active,
                                Enzyme.Duplicated(θ, θbar), Enzyme.Const(iv), Enzyme.Const(ip), Enzyme.Const(n),
                                Enzyme.Const(aw.kbar), Enzyme.Const(aw.pbar), Enzyme.Const(rbar),
                                Enzyme.Duplicated(aw.kapn, aw.dkap), Enzyme.Duplicated(aw.psin, aw.dpsi),
                                Enzyme.Duplicated(aw.ephn, aw.deph), Enzyme.Const(pc))
            end
        end
    end
    return θbar[1:length(x)]
end

#################################################################################
# Enzyme rule
#################################################################################

# Entry point with the custom rule; x = [Δm²₂₁, θ₁₂, θ₁₃, (layer_norms...)]
function _p1e_vec(b::BinnedP1e, x::Vector{Float64})
    return length(x) == 3 ? b(x[1], x[2], x[3]) : b(x[1], x[2], x[3], x[4:end])
end

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1}, ::Const{typeof(_p1e_vec)},
                                      ::Type{RT}, b::Const{BinnedP1e}, x::Annotation{Vector{Float64}}) where {RT}
    P = _p1e_vec(b.val, x.val)
    dP = zero(P)
    primal = EnzymeRules.needs_primal(config) ? P : nothing
    shadow = EnzymeRules.needs_shadow(config) ? dP : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (copy(x.val), dP))
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1}, ::Const{typeof(_p1e_vec)},
                             ::Type{RT}, tape, b::Const{BinnedP1e}, x::Annotation{Vector{Float64}}) where {RT}
    xv, dP = tape
    if !(x isa Const) && any(!iszero, dP)
        # invokelatest: Enzyme 0.13 fails to compile the nested autodiff call otherwise
        x.dval .+= Base.invokelatest(_p1e_vjp_manual, b.val, xv, dP)::Vector{Float64}
    end
    return (nothing, nothing)
end
