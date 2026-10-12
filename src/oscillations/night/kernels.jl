#################################################################################
# Propagation kernels
#################################################################################

# ∫₀ᴸ (x − L/2) exp(-iκx) dx, with e = exp(-iκL)
@inline function _linear_moment(kappa, L, e)
    z = kappa * L
    if abs(z) < SERIES_LINEAR_MOMENT
        return L * L * complex(-z * z / 24, -z / 12)
    end
    ik = 1 / kappa
    i0 = -im * (1 - e) * ik
    i1 = (e * (1 + im * z) - 1) * (ik * ik)
    return i1 - 0.5 * L * i0
end

# Propagate (ψ₁, ψ₂) through one step: exact at the average potential, plus the
# potential slope to first order (interaction picture)
@inline function _step(psi1, psi2, k, hx, hz, w, L, G, cs, sn, ez_conj)
    iw = 1 / w
    nx, nz = hx * iw, hz * iw
    J = G * _linear_moment(2 * k * w, L, ez_conj)
    wx = -nz * nx * real(J)
    wy = -nx * imag(J)
    wz = (1 - nz * nz) * real(J)
    x2 = k * k * (wx * wx + wy * wy + wz * wz)
    if x2 < SERIES_ROTATION
        cw = 1 - x2 / 2 + x2 * x2 / 24
        f = k * (1 - x2 / 6 + x2 * x2 / 120)
    else
        x = sqrt(x2)
        cw = cos(x)
        f = k * sin(x) / x
    end
    a1 = wz * psi1 + complex(wx, -wy) * psi2
    a2 = complex(wx, wy) * psi1 - wz * psi2
    psi1 = cw * psi1 - im * f * a1
    psi2 = cw * psi2 - im * f * a2
    a1 = nz * psi1 + nx * psi2
    a2 = nx * psi1 - nz * psi2
    psi1 = cs * psi1 - im * sn * a1
    psi2 = cs * psi2 - im * sn * a2
    return psi1, psi2
end

"""
    p1e_paths(energy, dm2, th12, th13, paths; layer_norms=nothing, phase_const)

P(ν₁→νₑ) per path and energy [GeV], shape (n_paths, n_E), without bin averaging.
"""
function p1e_paths(energy::AbstractVector, dm2, th12, th13, paths::Vector{Vector{PathSegment}};
                   layer_norms=nothing, phase_const::Real)
    L, V, S, layer = _stack_steps(paths)
    T = promote_type(typeof(dm2), typeof(th12), typeof(th13),
                     layer_norms === nothing ? Float64 : eltype(layer_norms), eltype(energy))
    out = zeros(T, length(paths), length(energy))
    c13sq = cos(th13)^2
    c2, s2 = cos(2th12), sin(2th12)
    hx = dm2 * s2 / 2
    for (ie, e) in enumerate(energy)
        k = phase_const / e
        for ip in axes(L, 1)
            psi1 = complex(cos(th12) + zero(T))
            psi2 = complex(-sin(th12) + zero(T))
            for s in axes(L, 2)
                Ls = L[ip, s]
                Ls == 0 && continue
                nrm = layer_norms === nothing ? one(T) : layer_norms[layer[ip, s]]
                hz = (V[ip, s] * nrm * c13sq * e - dm2 * c2) / 2
                w = sqrt(hz * hz + hx * hx)
                sn, cs = sincos(k * w * Ls)
                G = S[ip, s] * nrm * c13sq * e / 2
                psi1, psi2 = _step(psi1, psi2, k, hx, hz, w, Ls, G, cs, sn, complex(cs * cs - sn * sn, -2 * sn * cs))
            end
            out[ip, ie] = c13sq * abs2(psi1)
        end
    end
    return out
end

# Filon quadrature: ∫₀¹ f(t) exp(-iψ(t)) dt with f quadratic through (f0, fm, f1) and
# ψ through (p0, pm, p1); e0, e1 = exp(-i p0), exp(-i p1)
@inline function _filon(f0, fm, f1, p0, pm, p1, e0, e1)
    d = p1 - p0
    x = pm - (p0 + p1) / 2
    if abs(x) < FILON_CURVATURE_SERIES
        fm = fm * complex(1 - x * x / 2, -x + x * x * x / 6)
    else
        fm = fm * cis(-x)
    end
    a1 = -3 * f0 + 4 * fm - f1
    a2 = 2 * f0 - 4 * fm + 2 * f1
    if abs(d) < FILON_STEP_SERIES
        i0 = complex(1 - d * d / 6, -d / 2)
        i1 = complex(0.5 - d * d / 8, -d / 3)
        i2 = complex(1 / 3 - d * d / 10, -d / 4 + d * d * d / 36)
    else
        ed = e1 * conj(e0)
        ia = im / d
        ia2 = ia * ia
        i0 = (ed - 1) * ia
        i1 = ed * ia - (ed - 1) * ia2
        i2 = ed * (ia - 2 * ia2 + 2 * ia2 * ia) - 2 * ia2 * ia
    end
    return (f0 * i0 + a1 * i1 + a2 * i2) * e0
end

# Adjoint of _filon: cotangents of (f0, fm, f1, p0, pm, p1) for the output cotangent ybar
@inline function _filon_adj(ybar::Complex{Float64}, f0, fm, f1, p0, pm, p1, e0, e1)
    d = p1 - p0
    x = pm - (p0 + p1) / 2
    if abs(x) < FILON_CURVATURE_SERIES
        c  = complex(1 - x * x / 2, -x + x * x * x / 6)
        dc = complex(-x, -1 + x * x / 2)
    else
        c  = cis(-x)
        dc = -im * c
    end
    if abs(d) < FILON_STEP_SERIES
        i0 = complex(1 - d * d / 6, -d / 2)
        i1 = complex(0.5 - d * d / 8, -d / 3)
        i2 = complex(1 / 3 - d * d / 10, -d / 4 + d * d * d / 36)
        di0 = complex(-d / 3, -0.5)
        di1 = complex(-d / 4, -1 / 3)
        di2 = complex(-d / 5, -0.25 + d * d / 12)
    else
        ed = e1 * conj(e0)
        ia = im / d
        ia2 = ia * ia
        i0 = (ed - 1) * ia
        i1 = ed * ia - (ed - 1) * ia2
        i2 = ed * (ia - 2 * ia2 + 2 * ia2 * ia) - 2 * ia2 * ia
        i3 = ia * (ed - 3 * i2)
        di0 = -im * i1
        di1 = -im * i2
        di2 = -im * i3
    end
    A0, Am, A1 = i0 - 3 * i1 + 2 * i2, 4 * (i1 - i2), 2 * i2 - i1
    dA0, dAm, dA1 = di0 - 3 * di1 + 2 * di2, 4 * (di1 - di2), 2 * di2 - di1
    F  = A0 * f0 + Am * c * fm + A1 * f1
    Fd = dA0 * f0 + dAm * c * fm + dA1 * f1
    Fx = Am * dc * fm
    dy_p0 = e0 * (-im * F - Fd - Fx / 2)
    dy_p1 = e0 * (Fd - Fx / 2)
    dy_pm = e0 * Fx
    cy = conj(ybar)
    return (conj(A0 * e0) * ybar, conj(Am * c * e0) * ybar, conj(A1 * e0) * ybar,
            real(cy * dy_p0), real(cy * dy_pm), real(cy * dy_p1))
end
