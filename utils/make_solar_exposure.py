#!/usr/bin/env python3
"""Solar exposure table in cos z: time-weighted and flux-weighted columns (SPEC 4.6).

    python3 utils/make_solar_exposure.py [--latitude 44.35] [--nodes 702] [-o inputs/solarExposure_v2.csv]

Earth on a Kepler orbit, sampled uniformly in time; on each day the hour angle is uniform, so
cos z = a + b cos H has an exact CDF. `time_weight` (backgrounds) is the fraction of time per unit
cos z; `flux_weight` (signal) weights it by (1 AU / r)², uniform in solar longitude by Kepler's
second law (night fraction 0.5). Node values are cell-mean densities; both columns integrate to 1.
Ignored: refraction, the solar disc, the equation of time, precession.
"""
import argparse

import numpy as np

OBLIQUITY_DEG = 23.44
ECCENTRICITY = 0.0167
SUN_LONGITUDE_AT_PERIHELION_DEG = 282.94   # ~3 January


def orbit(n):
    """Declination [rad] and Earth-Sun distance [AU] at n times uniform over a year."""
    M = 2 * np.pi * (np.arange(n) + 0.5) / n
    E = M.copy()
    for _ in range(50):
        E = M + ECCENTRICITY * np.sin(E)
    nu = 2 * np.arctan2(np.sqrt(1 + ECCENTRICITY) * np.sin(E / 2), np.sqrt(1 - ECCENTRICITY) * np.cos(E / 2))
    r = 1 - ECCENTRICITY * np.cos(E)
    lam = nu + np.radians(SUN_LONGITUDE_AT_PERIHELION_DEG)
    return np.arcsin(np.sin(np.radians(OBLIQUITY_DEG)) * np.sin(lam)), r


def exposure(latitude_deg, nodes, n_times):
    phi = np.radians(latitude_deg)
    dec, r = orbit(n_times)
    a, b = np.sin(phi) * np.sin(dec), np.cos(phi) * np.cos(dec)
    hi = np.cos(phi - np.radians(OBLIQUITY_DEG))       # noon at the summer solstice; midnight at the
    x = np.linspace(-hi, hi, nodes)                    # winter one gives -hi (latitude above eps)
    edges = np.concatenate([[x[0]], (x[:-1] + x[1:]) / 2, [x[-1]]])
    widths = np.diff(edges)

    def cell_fractions(weight):
        acc = np.zeros(nodes)
        for chunk in np.array_split(np.arange(n_times), max(1, n_times // 2000)):
            u = np.clip((edges[None, :] - a[chunk, None]) / b[chunk, None], -1, 1)
            F = 1 - np.arccos(u) / np.pi
            acc += (weight[chunk, None] * np.diff(F, axis=1)).sum(axis=0)
        return acc / weight.sum()

    out = {}
    for name, w in (("time_weight", np.ones_like(r)), ("flux_weight", r ** -2)):
        frac = cell_fractions(w)
        density = frac / widths
        density /= np.trapezoid(density, x)
        out[name] = density
    return x, out


def night_fraction(x, y):
    xs = np.concatenate([x[x < 0], [0.0], x[x > 0]])
    ys = np.interp(xs, x, y)
    n = np.trapezoid(ys[xs <= 0], xs[xs <= 0])
    return n / np.trapezoid(ys, xs)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--latitude", type=float, default=44.35, help="site latitude [deg N] (SURF: 44.35)")
    p.add_argument("--nodes", type=int, default=702)
    p.add_argument("--times", type=int, default=200_000, help="time samples over the year")
    p.add_argument("-o", "--output", default="inputs/solarExposure_v2.csv")
    args = p.parse_args()
    x, cols = exposure(args.latitude, args.nodes, args.times)
    fn_t, fn_f = night_fraction(x, cols["time_weight"]), night_fraction(x, cols["flux_weight"])
    header = (f"# Solar exposure in cos z at latitude {args.latitude} deg N, Kepler orbit (e = {ECCENTRICITY},\n"
              f"# obliquity {OBLIQUITY_DEG} deg, Sun at perihelion at ecliptic longitude {SUN_LONGITUDE_AT_PERIHELION_DEG} deg),\n"
              f"# {args.times} time samples. Made by utils/make_solar_exposure.py. cos z < 0 is night.\n"
              f"# time_weight: fraction of time per unit cos z (backgrounds); night fraction {fn_t:.6f}.\n"
              f"# flux_weight: weighted by (1 AU / r)^2 (signal); night fraction {fn_f:.6f}.\n"
              "cosz,time_weight,flux_weight")
    np.savetxt(args.output, np.column_stack([x, cols["time_weight"], cols["flux_weight"]]),
               delimiter=",", header=header, comments="", fmt="%.9e")
    print(f"{args.output}: {len(x)} nodes on [{x[0]:.6f}, {x[-1]:.6f}]; night fraction time {fn_t:.6f}, flux {fn_f:.6f}")


if __name__ == "__main__":
    main()
