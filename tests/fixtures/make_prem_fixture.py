"""Write prem_parchita.csv: n_e of parchita's nominal Earth (theta = 0) at random radii.

Reference for tests/test_prem.jl. Needs parchita (~/Desktop/parchita) and JAX in float64.

Usage::

    python3 tests/fixtures/make_prem_fixture.py
"""
import os

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from parchita.osc import earth_model as em  # noqa: E402

rng = np.random.default_rng(20261022)
r = np.sort(rng.uniform(0.0, em.EARTH_RADIUS_KM, 2000))
# points within 1 m of a jump are ambiguous: drop them
r = r[np.min(np.abs(r[:, None] - em.BOUNDARIES[None, :]), axis=1) > 1e-3]
shell = np.sum(r[:, None] < em.BOUNDARIES[None, :], axis=1)
theta = np.zeros(em.N_THETA)
n_e = np.asarray(em.electron_density(theta, r, shell))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prem_parchita.csv")
np.savetxt(out, np.column_stack([r, n_e]), delimiter=",", header="radius_km,n_e", comments="", fmt="%.17g")
print(f"wrote {out}: {len(r)} radii")
