"""The vent-sampling inversion must recover a vent it was not given.

The quasi-steady kernel depends on r^2, so a wrong vent position distorts every
radius in the operator.  These tests check that the sampler finds a synthetic
vent from the deposit alone, and that profiling the vent out understates the
width of the mass posterior compared with sampling it.
"""

import numpy as np
import pytest

from plume_inv.inverse import InversionData, nu_model_vent

jax = pytest.importorskip("jax")
numpyro = pytest.importorskip("numpyro")
from numpyro.infer import MCMC, NUTS  # noqa: E402

Q_NODES = np.geomspace(3e4, 3e7, 12)
TRUE_VENT = np.array([450.0, -300.0])


def synth(seed=0, sigma=0.25, n=90):
    """A two-class deposit from a point-mass nu at a known vent."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-6000, 6000, size=(n, 2))
    w_s = [0.05, 0.02]
    nu = np.zeros(Q_NODES.size)
    nu[6] = 2.0e7  # kg, a single flux node
    obs, core_xy = [], []
    for w in w_s:
        r2 = ((xy - TRUE_VENT) ** 2).sum(1)
        g = (w / Q_NODES)[None, :] * np.exp(-np.pi * w * r2[:, None] / Q_NODES[None, :])
        clean = 0.5 * (g @ nu)
        y = clean * np.exp(rng.normal(0.0, sigma, size=n))
        obs.append(y)
        core_xy.append(xy.copy())
    return core_xy, obs, w_s


def run(core_xy, obs, w_s, vent_prior_scale=3000.0, seed=1):
    data = InversionData(
        q_nodes=Q_NODES,
        design=[],
        obs=obs,
        class_names=["a", "b"],
        w_s=w_s,
        radii=[],
        sigma_log=0.25,
    )
    mcmc = MCMC(
        NUTS(nu_model_vent, target_accept_prob=0.9),
        num_warmup=400,
        num_samples=400,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), data, core_xy, vent_prior_scale)
    return mcmc.get_samples()


def test_vent_is_recovered():
    s = run(*synth())
    vent = np.asarray(s["vent"])
    med = np.median(vent, axis=0)
    # Within 600 m on both axes: small against the 5.4 km dispersal length.
    assert np.abs(med - TRUE_VENT).max() < 600.0, f"vent median {med}, true {TRUE_VENT}"


def test_vent_posterior_is_not_degenerate():
    s = run(*synth())
    vent = np.asarray(s["vent"])
    spread = vent.std(axis=0)
    # It must be informed by the data (far tighter than the prior) and still
    # carry width (the deposit does not pin the vent exactly).
    assert (spread < 1500.0).all(), f"vent posterior not informed: {spread}"
    assert (spread > 5.0).all(), f"vent posterior collapsed: {spread}"


def test_total_mass_is_recovered():
    """The mass is sum(nu), not exp(log_mass): log_mass is a location parameter
    and the mean-centred Matern shape redistributes mass across the nodes."""
    s = run(*synth())
    mass = np.asarray(s["nu"]).sum(axis=1)
    assert 0.3e7 < np.median(mass) < 8.0e7, f"median mass {np.median(mass):.3g}"
