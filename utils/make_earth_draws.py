"""Draw Earth density profiles from the EarthModels prior (parchita) for utils/make_earth_prior.jl.

Spherical radial profiles only (lateral fields off). For each draw: the 9 radii where n_e can jump and
n_e tabulated inside each shell, on the draw's own (shifted) radii.

Usage::

    python3 utils/make_earth_draws.py [--draws 500] [--seed 1] [--step-km 2] [--out inputs/earth_draws_parchita.h5]

Needs parchita (~/Desktop/parchita) and h5py.
"""
import argparse

import jax

jax.config.update("jax_enable_x64", True)
import h5py  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from parchita.osc import earth_model as em  # noqa: E402

EPS_KM = 1e-6   # table ends just inside each shell, so no point sits on the far side of a jump


def tables(n_draws, seed, step_km):
    tops = np.concatenate([[em.EARTH_RADIUS_KM], em.BOUNDARIES])
    bots = np.concatenate([em.BOUNDARIES, [0.0]])
    counts = [max(2, int(np.ceil((t - b) / step_km)) + 1) for t, b in zip(tops, bots)]
    shell = np.concatenate([np.full(n, i) for i, n in enumerate(counts)])
    unit = np.concatenate([np.linspace(0.0, 1.0, n) for n in counts])

    def one(earth):
        theta, _ = em.radial(earth)
        b = em.boundary_radii(theta)
        top = jnp.concatenate([jnp.array([em.EARTH_RADIUS_KM]), b])[shell] - EPS_KM
        bot = jnp.concatenate([b, jnp.array([0.0])])[shell] + EPS_KM
        r = bot + unit * (top - bot)
        return b, r, em.electron_density(theta, r, shell)

    draws = em.sample_prior(jax.random.PRNGKey(seed), n_draws)
    b, r, n_e = jax.jit(jax.vmap(one))(draws)
    amp = np.exp(em.AMP_SPREAD * np.asarray(draws.log_amp))
    f220 = np.asarray(jax.nn.sigmoid(draws.f220))
    return np.asarray(b), np.asarray(r), np.asarray(n_e), amp, f220


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--step-km", type=float, default=2.0)
    ap.add_argument("--out", default="inputs/earth_draws_parchita.h5")
    a = ap.parse_args()
    b, r, n_e, amp, f220 = tables(a.draws, a.seed, a.step_km)
    with h5py.File(a.out, "w") as f:
        f["boundaries_km"] = b          # (draws, 9), outermost first
        f["radius_km"] = r              # (draws, points), shells outermost first, each ascending
        f["n_e"] = n_e                  # N_A/cm^3
        f["amplitude"] = amp
        f["f220"] = f220
        f.attrs["source"] = "parchita.osc.earth_model.sample_prior, lateral fields off"
        f.attrs["seed"] = a.seed
    print(f"wrote {a.out}: {a.draws} draws, {r.shape[1]} points each")


if __name__ == "__main__":
    main()
