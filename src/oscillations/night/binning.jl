#################################################################################
# Interpolation and bin-average maps
#################################################################################

# Lagrange interpolation: first node index and weights of the `order` nodes around each x
function _lagrange(xn::AbstractVector, x::AbstractVector, order::Int)
    n = length(xn)
    first = [clamp(searchsortedfirst(xn, xi) - 1 - order ÷ 2, 0, n - order) + 1 for xi in x]
    w = ones(length(x), order)
    for (q, xi) in enumerate(x)
        t = @view xn[first[q]:first[q]+order-1]
        for a in 1:order, b in 1:order
            b != a && (w[q, a] *= (xi - t[b]) / (t[a] - t[b]))
        end
    end
    return first, w
end

# Linear map (n_nodes, n_bins) from node values to bin averages of their not-a-knot cubic spline
function _spline_bin_average_map(nodes::AbstractVector, edges::AbstractVector)
    n = length(nodes)
    h = diff(nodes)
    # Second derivatives M from continuity and the not-a-knot conditions
    A = zeros(n, n); B = zeros(n, n)
    for i in 2:n-1
        A[i, i-1] = h[i-1] / 6; A[i, i] = (h[i-1] + h[i]) / 3; A[i, i+1] = h[i] / 6
        B[i, i-1] = 1 / h[i-1]; B[i, i] = -1 / h[i-1] - 1 / h[i]; B[i, i+1] = 1 / h[i]
    end
    A[1, 1] = -1 / h[1]; A[1, 2] = 1 / h[1] + 1 / h[2]; A[1, 3] = -1 / h[2]
    A[n, n-2] = -1 / h[n-2]; A[n, n-1] = 1 / h[n-2] + 1 / h[n-1]; A[n, n] = -1 / h[n-1]
    Mmap = A \ B                                  # M = Mmap * y
    # Antiderivative from nodes[1] to x, as a row acting on the node values
    function anti(x)
        row = zeros(n)
        j = clamp(searchsortedlast(nodes, x), 1, n - 1)
        for i in 1:j-1          # full segments
            hi = h[i]
            row[i] += hi / 2; row[i+1] += hi / 2
            row .-= hi^3 / 24 .* (Mmap[i, :] .+ Mmap[i+1, :])
        end
        # partial segment from nodes[j] to x
        hj = h[j]; t = x - nodes[j]
        iy_j  = t - t^2 / (2hj)
        iy_j1 = t^2 / (2hj)
        iM_j  = (hj^4 - (hj - t)^4) / (24hj) - hj * (hj * t - t^2 / 2) / 6
        iM_j1 = t^4 / (24hj) - hj * t^2 / 12
        row[j] += iy_j; row[j+1] += iy_j1
        row .+= iM_j .* Mmap[j, :] .+ iM_j1 .* Mmap[j+1, :]
        return row
    end
    rows = [anti(x) for x in edges]
    W = zeros(n, length(edges) - 1)
    for j in 1:length(edges)-1
        W[:, j] = (rows[j+1] .- rows[j]) ./ (edges[j+1] - edges[j])
    end
    return W
end

#################################################################################
# BinnedP1e
#################################################################################

# Precomputed grid for the cos z range between two layer tangencies
struct _Interval
    nodes::Vector{Float64}          # energy nodes [GeV]
    L::Matrix{Float64}              # step lengths (n_paths, n_steps)
    V::Matrix{Float64}              # step potentials
    S::Matrix{Float64}              # step slopes
    layer::Matrix{Int}
    u::Vector{Float64}              # 1/E at the energy-piece points
    inv_u2::Vector{Float64}
    lag_first::Vector{Int}
    lag_w::Matrix{Float64}
    u_pieces::Matrix{Int}
    u_piece_bin::Vector{Int}
    u_strip::Vector{Int}
    W_r::Matrix{Float64}
    width::Vector{Float64}          # energy-bin widths [GeV]
    B_xi::Matrix{Float64}
    xi::Vector{Float64}
    w_xi::Vector{Float64}           # exposure per unit ξ
    pieces::Matrix{Int}
    piece_bin::Vector{Int}
end

"""
    BinnedP1e(cosz_edges, energy_edges, exposure, earth; depth_km=DEFAULT_DEPTH_KM, phase_const)

Precompute the bin-averaged P(ν₁→νₑ) on `cosz_edges` (within [-1, 0]) × `energy_edges`
[GeV], weighting cos z by `exposure(cosz)`. Evaluate with `b(dm2, th12, th13, layer_norms)`.
"""
# Mutable so it can be a weak key of the per-task stores; all fields are const
mutable struct BinnedP1e
    const shape::Tuple{Int,Int}
    const n_layers::Int
    const used_layers::Vector{Int}        # layers crossed by some path
    const phase_const::Float64
    const intervals::Vector{_Interval}
end

function BinnedP1e(cosz_edges::AbstractVector, energy_edges::AbstractVector, exposure, earth::EarthModel;
                   depth_km::Real=DEFAULT_DEPTH_KM, phase_const::Real, e_piece::Real=E_PIECE_GEV, xi_piece::Real=XI_PIECE,
                   xi_nodes=XI_NODES, nodes_per_mev=NODES_PER_MEV, step_km::Real=STEP_KM)
    cosz_edges = Float64.(cosz_edges)
    energy_edges = Float64.(energy_edges)
    (cosz_edges[1] >= -1 && cosz_edges[end] <= 0 && all(diff(cosz_edges) .> 0)) ||
        error("BinnedP1e: cosz_edges must be ascending within [-1, 0]")
    all(diff(energy_edges) .> 0) && energy_edges[1] > 0 || error("BinnedP1e: energy_edges must be ascending and positive")
    shape = (length(cosz_edges) - 1, length(energy_edges) - 1)

    r_det = EARTH_RADIUS_KM - depth_km
    width = diff(energy_edges)
    # Energy pieces in u = 1/E
    u = [1 / energy_edges[1]]
    u_pieces = NTuple{3,Int}[]; u_piece_bin = Int[]; u_strip = Int[]
    for j in 1:shape[2]
        lo, hi = energy_edges[j], energy_edges[j+1]
        m = ceil(Int, (hi - lo) / e_piece - 1e-9)
        ends = 1 ./ range(lo, hi; length=m + 1)
        for (u0, u1) in zip(ends[1:end-1], ends[2:end])
            push!(u, (u0 + u1) / 2, u1)
            push!(u_pieces, (length(u) - 2, length(u) - 1, length(u)))
            push!(u_piece_bin, j)
        end
        centre = length(u_pieces) - m + 1 + m ÷ 2
        push!(u_strip, isodd(m) ? u_pieces[centre][2] : u_pieces[centre][1])
    end
    u_pieces_m = reduce(vcat, [collect(p)' for p in u_pieces])

    radii = vcat(r_det, sort(filter(<(r_det), earth.layer_boundaries); rev=true), 0.0)
    intervals = _Interval[]
    for k in 0:length(radii)-2
        r_top, r_bot = radii[k+1], radii[k+2]
        c_hi = -sqrt(max(0.0, 1 - (r_top / r_det)^2))
        c_lo = -sqrt(1 - (r_bot / r_det)^2)
        a, b = max(c_lo, cosz_edges[1]), min(c_hi, cosz_edges[end])
        b <= a && continue
        D = r_top^2 - r_bot^2
        xi_of(c) = sqrt(max((r_top^2 - r_det^2 * (1 - c^2)) / D, 0.0))
        c_of(x) = -sqrt(max(1 - (r_top^2 - x^2 * D) / r_det^2, 0.0))

        # Paths at the ξ nodes; all get the step count of the longest crossing of each layer
        x_lo, x_hi = xi_of(b), xi_of(a)
        g, _ = gauss(xi_nodes isa Tuple ? xi_nodes[1] + xi_nodes[2] * k : xi_nodes(k))
        x_nodes = 0.5 * (x_hi - x_lo) .* g .+ 0.5 * (x_hi + x_lo)
        steps = Dict(i => 1 for i in 1:n_layers(earth))
        for c in range(a, b; length=STEP_SCAN_POINTS)[2:end-1]
            for seg in build_earth_path(c, earth; depth_km=depth_km)
                steps[seg.layer] = max(steps[seg.layer], ceil(Int, seg.length / step_km))
            end
        end
        paths = [build_earth_path(c_of(x), earth; depth_km=depth_km, steps_per_layer=steps) for x in x_nodes]
        nseg = unique(length.(paths))
        length(nseg) == 1 || error("BinnedP1e: paths in tangency interval $k have different step counts $nseg")
        L, V, S, layer = _stack_steps(paths)
        all(layer .== layer[1:1, :]) || error("BinnedP1e: layer sequence differs across paths in interval $k")

        # Energy nodes and the interpolation / bin-average maps
        n_E = max(MIN_E_NODES, round(Int, (energy_edges[end] - energy_edges[1]) * 1e3 * nodes_per_mev(k))) + 1
        nodes = collect(range(energy_edges[1], energy_edges[end]; length=n_E))
        first_r, w_r = _lagrange(reverse(1 ./ nodes), u, E_ORDER)
        lag_first = n_E - 2 .- first_r         # lowest node index (1-based, ascending E)
        lag_w = w_r[:, end:-1:1]
        W_r = _spline_bin_average_map(nodes, energy_edges)

        # Zenith pieces in ξ
        xi = Float64[]; pieces = NTuple{3,Int}[]; piece_bin = Int[]
        for i in 1:shape[1]
            lo, hi = max(cosz_edges[i], a), min(cosz_edges[i+1], b)
            hi <= lo && continue
            x0, x1 = xi_of(hi), xi_of(lo)
            ends = range(x0, x1; length=max(2, ceil(Int, (x1 - x0) / xi_piece)) + 1)
            for (y0, y1) in zip(ends[1:end-1], ends[2:end])
                if isempty(xi) || abs(xi[end] - y0) > 1e-15
                    push!(xi, y0)
                end
                t0 = length(xi)
                push!(xi, (y0 + y1) / 2, y1)
                push!(pieces, (t0, t0 + 1, t0 + 2))
                push!(piece_bin, i)
            end
        end
        c = abs.(c_of.(xi))
        dc_dxi = [ci > 1e-12 ? x * D / (r_det^2 * max(ci, 1e-300)) : sqrt(D) / r_det for (x, ci) in zip(xi, c)]
        first_x, w_x = _lagrange(x_nodes, xi, min(XI_ORDER, length(x_nodes)))
        B_xi = zeros(length(xi), length(x_nodes))
        for t in eachindex(xi), a_ in axes(w_x, 2)
            B_xi[t, first_x[t]+a_-1] = w_x[t, a_]
        end
        w_xi = Float64[exposure(-ci) for ci in c] .* dc_dxi

        push!(intervals, _Interval(nodes, L, V, S, layer, u, 1 ./ u .^ 2, lag_first, lag_w, u_pieces_m, u_piece_bin,
                                   u_strip, W_r, width, B_xi, xi, w_xi,
                                   reduce(vcat, [collect(p)' for p in pieces]), piece_bin))
    end
    used = sort(unique(reduce(vcat, [vec(iv.layer) for iv in intervals]; init=Int[])))
    return BinnedP1e(shape, n_layers(earth), used, Float64(phase_const), intervals)
end

"""
    (b::BinnedP1e)(dm2, th12, th13, layer_norms=nothing; missing_value=0)

Bin-averaged P(ν₁→νₑ), shape (n_cosz_bins, n_energy_bins). `layer_norms[i]` scales the
density of layer i (1 = outermost). Bins with no exposure return `missing_value`.
"""
function (b::BinnedP1e)(dm2, th12, th13, layer_norms=nothing; missing_value=0)
    if layer_norms !== nothing
        length(layer_norms) == b.n_layers ||
            error("BinnedP1e: got $(length(layer_norms)) layer normalisations, Earth model has $(b.n_layers) layers")
    end
    T = promote_type(typeof(dm2), typeof(th12), typeof(th13),
                     layer_norms === nothing ? Float64 : eltype(layer_norms))
    norms = layer_norms === nothing ? ones(T, b.n_layers) : T[x for x in layer_norms]
    num = zeros(T, b.shape)
    den = zeros(b.shape[1])
    for (k, iv) in enumerate(b.intervals)
        _interval!(num, den, iv, _workspace(b, k, T), T(dm2), T(th12), T(th13), norms, b.phase_const)
    end
    out = similar(num)
    for i in axes(num, 1), j in axes(num, 2)
        out[i, j] = den[i] > 0 ? num[i, j] / den[i] : T(missing_value)
    end
    return out
end

# Scratch arrays for one interval and number type
struct _Workspace{T,CT}
    kap::Array{CT,3}     # boundary amplitudes of one path
    psi::Array{T,3}      # boundary phases to the detector
    eph::Vector{CT}
    Ck::Array{CT,3}
    Pk::Array{T,3}
    Cu::Vector{CT}
    Pu::Vector{T}
    Eu::Vector{CT}
    acc::Vector{CT}
    S_bin::Array{CT,3}
    P_bin::Array{T,3}
    R_bin::Matrix{T}
    Sx::Array{CT,3}
    Px::Array{T,3}
    Rx::Matrix{T}
end

function _Workspace{T}(iv::_Interval) where {T}
    CT = Complex{T}
    n_paths, n_steps = size(iv.L)
    nb, n_nodes, n_u, n_bins, n_xi = n_steps + 1, length(iv.nodes), length(iv.u), length(iv.width), length(iv.xi)
    return _Workspace{T,CT}(Array{CT}(undef, nb, n_nodes, 1), Array{T}(undef, nb, n_nodes, 1), Vector{CT}(undef, nb),
                            Array{CT}(undef, nb, n_u, 1), Array{T}(undef, nb, n_u, 1),
                            Vector{CT}(undef, n_u), Vector{T}(undef, n_u), Vector{CT}(undef, n_u), Vector{CT}(undef, n_bins),
                            Array{CT}(undef, n_bins, nb, n_paths), Array{T}(undef, n_bins, nb, n_paths), Matrix{T}(undef, n_bins, n_paths),
                            Array{CT}(undef, n_bins, nb, n_xi), Array{T}(undef, n_bins, nb, n_xi), Matrix{T}(undef, n_bins, n_xi))
end

# Workspaces and caches of an engine, per task (BAT runs each chain in its own task).
# Weak keys, so a discarded engine is freed with its arrays.
const _StoreDict = WeakKeyDict{BinnedP1e,Dict{Any,Any}}

function _task_store(b::BinnedP1e)
    stores = get!(_StoreDict, task_local_storage(), :night_osc_store)::_StoreDict
    return get!(() -> Dict{Any,Any}(), stores, b)::Dict{Any,Any}
end

function _workspace(b::BinnedP1e, k::Int, ::Type{T}) where {T}
    return get!(() -> _Workspace{T}(b.intervals[k]), _task_store(b), (k, T))::_Workspace{T,Complex{T}}
end

function _interval!(num::AbstractMatrix{T}, den, iv::_Interval, ws::_Workspace{T}, dm2::T, th12::T, th13::T,
                    norms::Vector{T}, phase_const) where {T}
    fill!(ws.R_bin, zero(T))
    for ip in axes(iv.L, 1)
        _nodes_path!(ws, iv, ip, dm2, th12, th13, norms, phase_const)
        _energy_path!(ws, iv, ip)
    end
    _stage_zenith!(num, den, iv, ws)
    return num
end

#################################################################################
# Evaluation: stage 1 (propagation), stage 2 (energy), stage 3 (zenith)
#################################################################################

# Stage 1 for one path, at every energy node
function _nodes_path!(ws::_Workspace{T}, iv::_Interval, ip::Int, dm2::T, th12::T, th13::T,
                      norms::Vector{T}, phase_const) where {T}
    R_bin = ws.R_bin
    @inbounds for n in eachindex(iv.nodes)
        r = _node!(view(ws.kap, :, n, 1), view(ws.psi, :, n, 1), ws.eph, iv, ip, n,
                   dm2, th12, th13, norms, phase_const)
        for j in axes(R_bin, 1)
            R_bin[j, ip] += r * iv.W_r[n, j]
        end
    end
    return ws
end

# Stage 1 at one path and energy node: fills the boundary amplitudes `kap` and phases `psi`,
# returns the smooth part of P₁ₑ. All parameter dependence enters here.
function _node!(kap::AbstractVector, psi::AbstractVector, eph::AbstractVector, iv::_Interval,
                ip::Int, n::Int, dm2::T, th12::T, th13::T, norms::AbstractVector, phase_const) where {T}
    L, V, S, layer = iv.L, iv.V, iv.S, iv.layer
    n_steps = size(L, 2)
    nb = n_steps + 1
    CT = Complex{T}
    c13sq = cos(th13)^2
    c2, s2 = cos(2th12), sin(2th12)
    hx = dm2 * s2 / 2
    @inbounds begin
        e = iv.nodes[n]
        k = phase_const / e
        psi1 = complex(cos(th12))
        psi2 = complex(-sin(th12))
        phi = zero(T)
        ephi = one(CT)
        th_prev = th12
        end_prev = zero(CT)
        for s in 1:n_steps
            Ls = L[ip, s]
            nrm = norms[layer[ip, s]]
            G = S[ip, s] * nrm * c13sq * e
            hz = (V[ip, s] * nrm * c13sq * e - dm2 * c2) / 2
            w2 = hz * hz + hx * hx
            w = sqrt(w2)
            # Matter angle at the step centre and its change across the step
            th_mid = atan(hx, -hz) / 2
            dth = hx * G * Ls / (4 * w2)
            z = 2 * k * w * Ls
            sn, cs = sincos(z / 2)
            ez = complex(cs * cs - sn * sn, 2 * sn * cs)
            # Split dth (exp(iz) - 1)/(iz) into a start and an end term, both finite as z → 0
            if z < SERIES_NODE_Z
                gg = complex(one(T), 1.5 * z)
                hh = complex(zero(T), -z)
            elseif z < BLEND_NODE_Z
                ex = exp(-z * z)
                iz = 1 / z
                gg = (ez * ex - 1) * complex(zero(T), -iz)
                hh = complex(zero(T), -(1 - ex) * iz)
            else
                iz = 1 / z
                gg = complex(zero(T), iz)
                hh = complex(zero(T), -iz)
            end
            kap[s] = (th_mid - dth / 2 - th_prev) + end_prev + dth * gg
            psi[s] = phi
            eph[s] = ephi
            phi += z
            ephi *= ez
            th_prev = th_mid + dth / 2
            end_prev = dth * hh
            psi1, psi2 = _step(psi1, psi2, k, hx, hz, w, Ls, G / 2, cs, sn, conj(ez))
        end
        kap[nb] = end_prev
        psi[nb] = phi
        eph[nb] = ephi
        sc = 2 * c13sq * sin(th_prev) * cos(th_prev)
        r = c13sq * abs2(psi1)
        cphi = conj(ephi)
        for bb in 1:nb
            psi[bb] = phi - psi[bb]
            kap[bb] *= sc
            r -= real(kap[bb] * cphi * eph[bb])
        end
    end
    return r
end

# Element types made only of Float64 words (Float64, ComplexF64, Float64 duals, ...)
_f64(::Type{Float64}) = true
_f64(::Type{E}) where {E} = isbitstype(E) && fieldcount(E) > 0 && all(_f64, fieldtypes(E))

# Float64 view of A's words. Unsafe: keep A alive with GC.@preserve while it is used.
@inline _words(A::Array{E}, dims::Tuple) where {E} =
    unsafe_wrap(Array, Ptr{Float64}(pointer(A)), ((sizeof(E) ÷ 8) * dims[1], Base.tail(dims)...))

# Interpolate in energy: out[:, q, p] = Σₐ lag_w[q, a] inp[:, lag_first[q] + a - 1, p]
function _interp_nodes!(out::Array{E,3}, inp::Array{E,3}, lag_first, lag_w) where {E}
    if _f64(E)
        GC.@preserve out inp _interp_loop!(_words(out, size(out)), _words(inp, size(inp)), lag_first, lag_w)
    else
        _interp_loop!(out, inp, lag_first, lag_w)
    end
    return out
end

function _interp_loop!(O, I, lag_first, lag_w)
    @inbounds for p in axes(I, 3), q in axes(O, 2)
        i0 = lag_first[q]
        w1, w2, w3, w4 = lag_w[q, 1], lag_w[q, 2], lag_w[q, 3], lag_w[q, 4]
        @simd for r in axes(O, 1)
            O[r, q, p] = w1 * I[r, i0, p] + w2 * I[r, i0+1, p] + w3 * I[r, i0+2, p] + w4 * I[r, i0+3, p]
        end
    end
    return O
end

# Interpolate across paths: out[.., t] = Σₚ B[t, p] inp[.., p]. B is sparse; a plain loop
# vectorises and avoids BLAS threads.
function _contract_last!(out::Array{E}, inp::Array{E}, B::Matrix{Float64}) where {E}
    m_out, m_in = length(out) ÷ size(out, ndims(out)), length(inp) ÷ size(inp, ndims(inp))
    if _f64(E)
        GC.@preserve out inp _contract_loop!(_words(out, (m_out, size(out, ndims(out)))),
                                             _words(inp, (m_in, size(inp, ndims(inp)))), B)
    else
        _contract_loop!(reshape(out, m_out, :), reshape(inp, m_in, :), B)
    end
    return out
end

function _contract_loop!(O::AbstractMatrix, I::AbstractMatrix, B::Matrix{Float64})
    @inbounds for t in axes(B, 1)
        fill!(view(O, :, t), zero(eltype(O)))
        for p in axes(B, 2)
            w = B[t, p]
            w == 0 && continue
            @simd for r in axes(O, 1)
                O[r, t] += w * I[r, p]
            end
        end
    end
    return O
end

# Stage 2 for one path: integrate each boundary term over the energy bins (Filon)
function _energy_path!(ws::_Workspace{T}, iv::_Interval, ip::Int) where {T}
    u, width = iv.u, iv.width
    nb = size(ws.kap, 1)
    CT = Complex{T}
    Ck = _interp_nodes!(ws.Ck, ws.kap, iv.lag_first, iv.lag_w)
    Pk = _interp_nodes!(ws.Pk, ws.psi, iv.lag_first, iv.lag_w)
    Cu, Pu, Eu, acc = ws.Cu, ws.Pu, ws.Eu, ws.acc
    S_bin, P_bin = ws.S_bin, ws.P_bin
    @inbounds for bb in 1:nb
        for q in eachindex(u)
            Cu[q] = Ck[bb, q, 1] * iv.inv_u2[q]
            Pu[q] = Pk[bb, q, 1]
            Eu[q] = cis(-Pu[q])
        end
        fill!(acc, zero(CT))
        for q in axes(iv.u_pieces, 1)
            t0, tm, t1 = iv.u_pieces[q, 1], iv.u_pieces[q, 2], iv.u_pieces[q, 3]
            acc[iv.u_piece_bin[q]] += (u[t0] - u[t1]) * _filon(Cu[t0], Cu[tm], Cu[t1], Pu[t0], Pu[tm], Pu[t1], Eu[t0], Eu[t1])
        end
        for j in eachindex(width)
            S_bin[j, bb, ip] = acc[j] * (conj(Eu[iv.u_strip[j]]) / width[j])
            P_bin[j, bb, ip] = Pu[iv.u_strip[j]]
        end
    end
    return ws
end

# Stage 3: interpolate across paths, integrate over the zenith bins (Filon)
function _stage_zenith!(num::AbstractMatrix{T}, den, iv::_Interval, ws::_Workspace{T}) where {T}
    n_bins, nb, _ = size(ws.S_bin)
    xi, w_xi = iv.xi, iv.w_xi
    Sx = _contract_last!(ws.Sx, ws.S_bin, iv.B_xi)
    Px = _contract_last!(ws.Px, ws.P_bin, iv.B_xi)
    Rx = _contract_last!(ws.Rx, ws.R_bin, iv.B_xi)
    @inbounds for bb in 1:nb
        for q in axes(iv.pieces, 1)
            t0, tm, t1 = iv.pieces[q, 1], iv.pieces[q, 2], iv.pieces[q, 3]
            hq = xi[t1] - xi[t0]
            i = iv.piece_bin[q]
            for j in 1:n_bins
                p0, pm, p1 = Px[j, bb, t0], Px[j, bb, tm], Px[j, bb, t1]
                val = _filon(w_xi[t0] * Sx[j, bb, t0], w_xi[tm] * Sx[j, bb, tm], w_xi[t1] * Sx[j, bb, t1],
                             p0, pm, p1, cis(-p0), cis(-p1))
                num[i, j] += hq * real(val)
            end
        end
    end
    @inbounds for q in axes(iv.pieces, 1)
        t0, tm, t1 = iv.pieces[q, 1], iv.pieces[q, 2], iv.pieces[q, 3]
        hq = xi[t1] - xi[t0]
        i = iv.piece_bin[q]
        den[i] += hq * (w_xi[t0] + 4 * w_xi[tm] + w_xi[t1]) / 6
        for j in 1:n_bins
            num[i, j] += hq * (w_xi[t0] * Rx[j, t0] + 4 * w_xi[tm] * Rx[j, tm] + w_xi[t1] * Rx[j, t1]) / 6
        end
    end
    return num
end
