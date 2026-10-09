r"""Bayesian inversion for the measure nu(Q).

The quasi-steady forward map is linear in nu (``kernel.py`` (K3)), so with the
design matrix precomputed on a fixed log-Q grid every likelihood evaluation is
one matrix-vector product.  That is what makes NUTS cheap here.

Model
-----
For core j of size class i, with Omega the deposit in kg m^-2,

    Omega_ij = p_i * (A_i @ nu)_j ,      A_i[j,k] = (w_i/Q_k) exp[-pi w_i r_j^2 / Q_k]

    log Omega_ij^obs ~ Normal( log Omega_ij , sigma )

a multiplicative error, which is what a visually estimated glass percentage
produces (see ``data.py``).

The unknown is the vector nu >= 0 of released particle mass per log-Q node.
The deposit resolves only 2-6 modes of nu (``results/identifiability.json``),
so the prior has to do
visible work between them and must be stated rather than hidden:

    log nu_k = log(M_tot) - log(K) + tau * z_k,     z ~ GP(0, Matern-5/2 on log Q)

with ``tau`` a smoothness amplitude and the GP correlated across log Q.  A
correlated prior is not a convenience: with a white prior the posterior in the
unresolved directions is the prior, and a white prior puts mass on wildly rough
nu that no physics supports.

Reported summaries are defined in :func:`summarise_nu`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "InversionData",
    "matern52_chol",
    "nu_model",
    "steady_model",
    "summarise_nu",
    "fit_steady_lsq",
]


@dataclass(frozen=True)
class InversionData:
    """Everything the likelihood needs, precomputed.

    ``design[i]`` is the (n_cores_i, n_nodes) matrix for class i, already
    including that class's settling speed.  ``obs[i]`` is the deposit in
    kg m^-2 and ``mask[i]`` selects the cores with a finite measurement.
    """

    q_nodes: np.ndarray
    design: list[np.ndarray]
    obs: list[np.ndarray]
    class_names: list[str]
    w_s: list[float]
    radii: list[np.ndarray]
    sigma_log: float = 0.8
    meta: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.q_nodes.size)

    @property
    def n_obs(self) -> int:
        return int(sum(o.size for o in self.obs))


def matern52_chol(log_q: np.ndarray, length_scale: float, jitter: float = 1e-8) -> np.ndarray:
    """Cholesky factor of a Matern-5/2 correlation matrix on the log-Q grid.

    Matern-5/2 gives twice-differentiable sample paths: smooth enough that the
    prior does not put mass on nu that oscillates between adjacent nodes, rough
    enough not to force a Gaussian-process-flavoured shape onto the answer.
    """
    x = np.asarray(log_q, dtype=float)[:, None]
    d = np.abs(x - x.T) / length_scale
    s5 = np.sqrt(5.0)
    k = (1.0 + s5 * d + 5.0 * d**2 / 3.0) * np.exp(-s5 * d)
    return np.linalg.cholesky(k + jitter * np.eye(k.shape[0]))


def summarise_nu(
    q_nodes: np.ndarray,
    nu: np.ndarray,
    n_buoy: float,
    epsilon: float,
    k_heat: float,
    c_f: float,
    m_exponent: float = 4.0 / 3.0,
) -> dict:
    """Physical summaries of a nu draw.

    ``E`` is the energy written in terms of nu (paper, Methods):
    E = k c_f (N^5/eps^2)^(1/3) c^-1 sum nu_k Q_k^(4/3 - m).  The constant ``c``
    of the source link is not identified by the deposit, so ``E`` is returned
    per unit ``c``; the source-discrimination step supplies a prior on it.
    """
    q = np.asarray(q_nodes, dtype=float)
    nu = np.asarray(nu, dtype=float)
    pref = k_heat * c_f * (n_buoy**5 / epsilon**2) ** (1.0 / 3.0)
    total = float(nu.sum())
    if total <= 0:
        raise ValueError("nu must have positive total mass")
    weights = nu / total
    return {
        "mass_total_kg": total,
        "q_mass_weighted_mean": float(np.sum(weights * q)),
        "q_peak": float(q[int(np.argmax(nu))]),
        "log_q_sd": float(
            np.sqrt(np.sum(weights * (np.log(q) - np.sum(weights * np.log(q))) ** 2))
        ),
        "phi_peak_W": float(pref * q.max() ** (4.0 / 3.0)),
        "phi_at_mass_weighted_mean_W": float(pref * np.sum(weights * q) ** (4.0 / 3.0)),
        "energy_per_unit_c_J": float(pref * np.sum(nu * q ** (4.0 / 3.0 - m_exponent))),
        "volume_per_unit_c_m3": float(np.sum(nu * q ** (1.0 - m_exponent))),
    }


def fit_steady_lsq(radii: list[np.ndarray], obs: list[np.ndarray], w_s: list[float]) -> dict:
    """Steady polydisperse fit by linear least squares in log space.

    For a fixed vent the steady model is linear in (log Omega_0i, 1/Q):

        log Omega_i(r) = log Omega_{0,i} - pi w_i r^2 / Q.

    Returns Q, the per-class intercepts, the residual scatter, and L for each
    class.  This is the steady reference the unsteady model is compared against,
    and the basis of the benchmark reproduction check.
    """
    rows, rhs = [], []
    n_cls = len(radii)
    for i, (r, o, _w) in enumerate(zip(radii, obs, w_s, strict=True)):
        if r.size == 0:
            continue
        design = np.zeros((r.size, n_cls + 1))
        design[:, i] = 1.0
        design[:, -1] = -np.pi * w_s[i] * r**2
        rows.append(design)
        rhs.append(np.log(o))
    a = np.vstack(rows)
    b = np.concatenate(rhs)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    resid = b - a @ sol
    inv_q = float(sol[-1])
    q_umb = 1.0 / inv_q if inv_q > 0 else float("nan")
    return {
        "q_umb": q_umb,
        "log_omega0": [float(v) for v in sol[:-1]],
        "L_by_class": [float(np.sqrt(q_umb / w)) if np.isfinite(q_umb) else None for w in w_s],
        "residual_sd_log": float(np.std(resid, ddof=len(sol))),
        "n_points": int(b.size),
        "mass_by_class_kg": [
            float(np.exp(sol[i]) * q_umb / w_s[i]) if np.isfinite(q_umb) else None
            for i in range(n_cls)
        ],
        "mass_total_kg": (
            float(np.sum([np.exp(sol[i]) * q_umb / w_s[i] for i in range(n_cls)]))
            if np.isfinite(q_umb)
            else None
        ),
        "_mass_note": (
            "int Omega_0i exp[-pi w_i r^2/Q] 2 pi r dr = Omega_0i Q / w_i is the "
            "mass of class i.  Summing over classes gives the total ONLY if the "
            "classes partition the erupted mass, i.e. if sum p_i = 1.  They do "
            "for the NESCA sieve fractions."
        ),
    }


# --------------------------------------------------------------------------
# NumPyro models.  Imported lazily so that the rest of the package, and the
# tests that do not sample, work without JAX present.
# --------------------------------------------------------------------------
def nu_model(
    data: InversionData,
    length_scale: float = 1.5,
    tau_scale: float = 1.0,
    log_mass_loc: float = None,
    log_mass_scale: float = 2.0,
    sigma_known: bool = False,
    dirichlet_conc: float = 1.0,
):
    """NumPyro model for the nonparametric nu(Q).

    Parameters
    ----------
    length_scale : float
        Matern correlation length in units of ln Q.  1.5 means nu is smooth over
        a factor e^1.5 ~ 4.5 in flux -- comparable to the resolution the
        identifiability analysis found, so the prior is regularising where the
        data are silent rather than fighting them where they are not.
    dirichlet_conc : float
        Concentration of the Dirichlet prior on the size-fraction weights p_i.
        1.0 is uniform on the simplex.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    chol = jnp.asarray(matern52_chol(np.log(data.q_nodes), length_scale))
    k = data.n_nodes
    loc = (
        log_mass_loc
        if log_mass_loc is not None
        else float(np.log(sum(o.sum() for o in data.obs) + 1e-12))
    )

    log_mass = numpyro.sample("log_mass", dist.Normal(loc, log_mass_scale))
    # Size-fraction weights.  Omega_i = p_i * int g_i dnu, so without p the model
    # asserts that every sieve fraction carries the WHOLE erupted mass, which
    # over-counts the total by the number of classes.
    p_frac = numpyro.sample("p_frac", dist.Dirichlet(jnp.ones(len(data.design)) * dirichlet_conc))
    tau = numpyro.sample("tau", dist.HalfNormal(tau_scale))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([k]).to_event(1))
    shape = chol @ z
    log_nu = log_mass - jnp.log(k) + tau * (shape - shape.mean())
    nu = numpyro.deterministic("nu", jnp.exp(log_nu))

    if sigma_known:
        sigma = data.sigma_log
    else:
        sigma = numpyro.sample("sigma", dist.HalfNormal(data.sigma_log * 2.0))

    for i, (a, o) in enumerate(zip(data.design, data.obs, strict=True)):
        pred = p_frac[i] * (jnp.asarray(a) @ nu)
        numpyro.sample(
            f"obs_{i}", dist.Normal(jnp.log(pred + 1e-30), sigma), obs=jnp.log(jnp.asarray(o))
        )


def nu_model_vent(
    data: InversionData,
    core_xy: list,
    vent_prior_scale: float,
    log_mass_loc: float = None,
    log_mass_scale: float = 2.0,
    tau_scale: float = 1.0,
    length_scale: float = 1.5,
    dirichlet_conc: float = 1.0,
    sigma_known: bool = False,
):
    """``nu_model`` with the vent position sampled rather than profiled out.

    The quasi-steady kernel depends on the core radius squared, so the vent
    position enters the operator itself and cannot be concentrated out without
    understating the width of every posterior that depends on it.  Here the
    vent is a parameter: the design matrix is rebuilt from the sampled position
    at each likelihood evaluation.

    Parameters
    ----------
    core_xy : list of (n_cores_i, 2) arrays
        Core coordinates per size class, in the local metric frame whose origin
        is the survey centroid.
    vent_prior_scale : float
        Standard deviation, in metres, of the independent Gaussian priors on
        the two vent coordinates, centred on the survey centroid.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    chol = jnp.asarray(matern52_chol(np.log(data.q_nodes), length_scale))
    q = jnp.asarray(data.q_nodes)
    k = data.n_nodes
    loc = (
        log_mass_loc
        if log_mass_loc is not None
        else float(np.log(sum(o.sum() for o in data.obs) + 1e-12))
    )

    vent = numpyro.sample("vent", dist.Normal(0.0, vent_prior_scale).expand([2]).to_event(1))
    log_mass = numpyro.sample("log_mass", dist.Normal(loc, log_mass_scale))
    p_frac = numpyro.sample("p_frac", dist.Dirichlet(jnp.ones(len(data.obs)) * dirichlet_conc))
    tau = numpyro.sample("tau", dist.HalfNormal(tau_scale))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([k]).to_event(1))
    shape = chol @ z
    log_nu = log_mass - jnp.log(k) + tau * (shape - shape.mean())
    nu = numpyro.deterministic("nu", jnp.exp(log_nu))

    if sigma_known:
        sigma = data.sigma_log
    else:
        sigma = numpyro.sample("sigma", dist.HalfNormal(data.sigma_log * 2.0))

    for i, (xy, o, w) in enumerate(zip(core_xy, data.obs, data.w_s, strict=True)):
        xy = jnp.asarray(xy)
        r2 = (xy[:, 0] - vent[0]) ** 2 + (xy[:, 1] - vent[1]) ** 2
        # g(r, Q) = (w / Q) exp[-pi w r^2 / Q], the quasi-steady kernel.
        g = (w / q)[None, :] * jnp.exp(-np.pi * w * r2[:, None] / q[None, :])
        pred = p_frac[i] * (g @ nu)
        numpyro.sample(
            f"obs_{i}", dist.Normal(jnp.log(pred + 1e-30), sigma), obs=jnp.log(jnp.asarray(o))
        )


def steady_model(
    data: InversionData,
    log_q_loc: float = None,
    log_q_scale: float = 1.5,
    sigma_known: bool = False,
):
    """The steady model: nu is a single point mass, so one flux and one mass.

    Nested inside :func:`nu_model` (a point mass is an admissible nu), which is
    what makes the steady-versus-unsteady comparison between them meaningful.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    loc = log_q_loc if log_q_loc is not None else float(np.log(np.median(data.q_nodes)))
    log_q = numpyro.sample("log_q", dist.Normal(loc, log_q_scale))
    log_mass = numpyro.sample(
        "log_mass",
        dist.Normal(float(np.log(sum(o.sum() for o in data.obs) + 1e-12)), 2.0),
    )
    p_frac = numpyro.sample("p_frac", dist.Dirichlet(jnp.ones(len(data.radii))))
    q = numpyro.deterministic("q_umb", jnp.exp(log_q))
    mass = jnp.exp(log_mass)

    if sigma_known:
        sigma = data.sigma_log
    else:
        sigma = numpyro.sample("sigma", dist.HalfNormal(data.sigma_log * 2.0))

    for i, (r, o) in enumerate(zip(data.radii, data.obs, strict=True)):
        w = data.w_s[i]
        pred = p_frac[i] * mass * (w / q) * jnp.exp(-jnp.pi * w * jnp.asarray(r) ** 2 / q)
        numpyro.sample(
            f"obs_{i}", dist.Normal(jnp.log(pred + 1e-30), sigma), obs=jnp.log(jnp.asarray(o))
        )
