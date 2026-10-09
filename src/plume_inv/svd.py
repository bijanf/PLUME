r"""Identifiability: how many modes of the source a deposit can actually resolve.

The quasi-steady forward map is linear in the measure nu (``kernel.py`` (K3)),
so the whole question is the singular-value spectrum of one matrix.  Two
criteria are reported, because a single number here is easy to get wrong.

**Relative-spectrum count.**  ``N_rel(delta) = #{n : sigma_n / sigma_1 > delta}``
with ``delta`` the relative noise level of the data.  This is the standard
truncated-SVD criterion.  It is invariant under rescaling the operator or the
unknown, which matters: a criterion of the form ``sigma_n > delta`` with
``sigma_n`` carrying the operator's units and ``delta`` the data's is
dimensionally meaningless and changes answer with the choice of units.

**Signal-to-noise count.**  Whiten by the observation errors,
``A~ = diag(1/sigma_j) A``, and give the unknown a prior scale ``nu_0``.  Mode
``n`` is resolvable when ``sigma~_n nu_0 > 1``: its contribution to the data
exceeds one standard deviation of noise.  This one has a physical meaning but
requires ``nu_0`` to be stated.

**Discrete Picard condition.**  A measure of whether a solution exists at all:
the coefficients ``|u_n^T d~|`` must decay faster than ``sigma~_n`` for the
truncated inverse to converge.  Where they flatten out is where noise takes over.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kernel import design_matrix

__all__ = [
    "ObservationSet",
    "stacked_operator",
    "SpectrumAnalysis",
    "analyse_spectrum",
    "jacobian_log_q",
]


@dataclass(frozen=True)
class ObservationSet:
    """Radii and noise for one size class."""

    name: str
    w_s: float
    radii: np.ndarray  # m, one per core
    sigma: np.ndarray  # observation sd, kg m^-2, one per core
    p_i: float = 1.0  # mass fraction of this class


def stacked_operator(
    obs: list[ObservationSet], q_nodes: np.ndarray, whiten: bool = True
) -> np.ndarray:
    """Stack the per-class design matrices into one operator.

    With ``whiten=True`` each row is divided by its observation sd, so the
    residual it produces is in units of standard deviations.
    """
    blocks = []
    for o in obs:
        a = design_matrix(o.radii, q_nodes, o.w_s, o.p_i)
        if whiten:
            a = a / np.asarray(o.sigma, dtype=float)[:, None]
        blocks.append(a)
    return np.vstack(blocks)


@dataclass
class SpectrumAnalysis:
    """Singular spectrum of a forward operator and the mode counts it implies."""

    singular_values: np.ndarray
    u: np.ndarray = field(repr=False)
    vt: np.ndarray = field(repr=False)
    decay_rate: float = 0.0
    """c in sigma_n ~ sigma_1 exp(-c n), fitted over the leading decade."""

    def n_resolvable_relative(self, delta: float) -> int:
        """#{n : sigma_n/sigma_1 > delta}."""
        s = self.singular_values
        return int(np.sum(s / s[0] > delta))

    def n_resolvable_snr(self, nu_scale: float) -> int:
        """#{n : sigma_n * nu_scale > 1}, for a whitened operator."""
        return int(np.sum(self.singular_values * nu_scale > 1.0))

    def picard(self, data_whitened: np.ndarray) -> np.ndarray:
        """|u_n^T d| for the whitened data -- compare with the singular values."""
        return np.abs(self.u.T @ np.asarray(data_whitened, dtype=float))

    def picard_ratio(self, data_whitened: np.ndarray) -> np.ndarray:
        """|u_n^T d| / sigma_n: the coefficients of the truncated-SVD solution."""
        return self.picard(data_whitened) / self.singular_values

    def picard_truncation(self, data_whitened: np.ndarray) -> int:
        """Optimal truncation index from the discrete Picard condition.

        The solution coefficients |u_n^T d| / sigma_n fall while the data still
        carry signal in mode n, then rise once sigma_n has decayed past the noise
        and the division amplifies it.  The minimum is where to cut.  Returns the
        number of modes to keep (1-based).
        """
        c = self.picard_ratio(data_whitened)
        if c.size < 3:
            return int(c.size)
        # Smooth over three modes so a single noisy coefficient cannot set the cut.
        sm = np.convolve(np.log(np.maximum(c, 1e-300)), np.ones(3) / 3.0, mode="valid")
        return int(np.argmin(sm) + 2)


def analyse_spectrum(a: np.ndarray) -> SpectrumAnalysis:
    """SVD of ``a``, with an exponential decay rate fitted to the leading modes."""
    u, s, vt = np.linalg.svd(np.asarray(a, dtype=float), full_matrices=False)
    keep = s > 0
    s = s[keep]
    u = u[:, keep]
    vt = vt[keep]
    # Fit sigma_n ~ sigma_1 exp(-c n) over the modes spanning the first decade,
    # which is where the decay is cleanly geometric.
    within = np.where(s / s[0] > 1e-1)[0]
    if within.size >= 3:
        n = np.arange(within.size)
        c = -np.polyfit(n, np.log(s[within] / s[0]), 1)[0]
    else:
        c = float("nan")
    return SpectrumAnalysis(singular_values=s, u=u, vt=vt, decay_rate=float(c))


def jacobian_log_q(forward, log_q0: np.ndarray, step: float = 1e-4) -> np.ndarray:
    """Central-difference Jacobian of a forward map with respect to log Q nodes.

    The unsteady, front-limited kernel is NOT linear in the source, so its
    identifiability cannot be read off a design matrix.  Linearising about a
    reference history and taking the singular values of the Jacobian is the
    matching construction: it answers "how many independent directions of the
    source does the data see, near this history".

    Parameters
    ----------
    forward : callable
        ``log_q -> whitened data vector``.
    log_q0 : array
        Reference log-flux nodes to linearise about.
    """
    log_q0 = np.asarray(log_q0, dtype=float)
    base = np.asarray(forward(log_q0), dtype=float)
    jac = np.empty((base.size, log_q0.size))
    for k in range(log_q0.size):
        up, dn = log_q0.copy(), log_q0.copy()
        up[k] += step
        dn[k] -= step
        jac[:, k] = (np.asarray(forward(up)) - np.asarray(forward(dn))) / (2 * step)
    return jac


# --------------------------------------------------------------------------
# Mode structure: resolution, averaging kernels and data-space patterns
# --------------------------------------------------------------------------
def orient_modes(vt: np.ndarray) -> np.ndarray:
    """Fix the arbitrary sign of each singular vector.

    Each row of ``vt`` is flipped so that its entry of largest magnitude is
    positive.  The singular spectrum is unchanged; only the display sign is
    fixed, so that plots of successive modes are comparable.
    """
    vt = np.array(vt, dtype=float, copy=True)
    idx = np.argmax(np.abs(vt), axis=1)
    sign = np.sign(vt[np.arange(vt.shape[0]), idx])
    sign[sign == 0] = 1.0
    return vt * sign[:, None]


def resolution_matrix(vt: np.ndarray, k: int) -> np.ndarray:
    r"""Model resolution matrix of a truncated SVD, R = V_k V_k^T.

    Row j of R is the averaging kernel for node j: the truncated-SVD estimate of
    nu at node j is the inner product of that row with the true nu.  R is a
    projector, so it is symmetric, idempotent and has trace k.
    """
    v = np.asarray(vt, dtype=float)[:k].T
    return v @ v.T


def averaging_kernel_row(r: np.ndarray, x_nodes: np.ndarray, x_target: float) -> np.ndarray:
    """Averaging kernel at an arbitrary target abscissa.

    Linear interpolation between the two rows of ``r`` whose nodes bracket
    ``x_target`` (``x_nodes`` increasing, for instance log10 Q).
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    j = int(np.clip(np.searchsorted(x_nodes, x_target) - 1, 0, x_nodes.size - 2))
    t = (x_target - x_nodes[j]) / (x_nodes[j + 1] - x_nodes[j])
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * np.asarray(r)[j] + t * np.asarray(r)[j + 1]


def main_lobe_fwhm(x: np.ndarray, y: np.ndarray) -> dict:
    """Full width at half maximum of the lobe containing the maximum of ``y``.

    Walks outward from the maximum to the first crossing of half the maximum on
    each side, interpolating linearly in ``x``.  If the lobe reaches the end of
    the grid on a side before falling to half, that side is reported as
    censored and the width is a lower bound.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    i0 = int(np.argmax(y))
    half = 0.5 * y[i0]
    out = {
        "x_peak": float(x[i0]),
        "y_peak": float(y[i0]),
        "censored_low": False,
        "censored_high": False,
    }
    j = i0
    while j > 0 and y[j - 1] > half:
        j -= 1
    if j == 0:
        x_lo = float(x[0])
        out["censored_low"] = True
    else:
        x_lo = float(x[j - 1] + (half - y[j - 1]) * (x[j] - x[j - 1]) / (y[j] - y[j - 1]))
    j = i0
    while j < x.size - 1 and y[j + 1] > half:
        j += 1
    if j == x.size - 1:
        x_hi = float(x[-1])
        out["censored_high"] = True
    else:
        x_hi = float(x[j] + (y[j] - half) * (x[j + 1] - x[j]) / (y[j] - y[j + 1]))
    out.update({"x_low": x_lo, "x_high": x_hi, "fwhm": x_hi - x_lo})
    return out


def captured_fraction(u: np.ndarray, data: np.ndarray, k: int) -> float:
    """Share of the squared norm of ``data`` in the span of the first k left
    singular vectors: sum_{n<=k} (u_n^T d)^2 / |d|^2."""
    d = np.asarray(data, dtype=float)
    c = np.asarray(u, dtype=float)[:, :k].T @ d
    return float(np.sum(c**2) / np.sum(d**2))


def laplace_response(q_nodes: np.ndarray, v: np.ndarray, lam) -> np.ndarray:
    r"""Deposit response per unit settling speed of a source perturbation v.

    For a perturbation ``v`` of the node masses of nu, the quasi-steady deposit
    of a class with settling speed w at radius r changes by
    ``dOmega = sum_k (w/Q_k) exp(-pi w r^2/Q_k) v_k``.  Dividing by w leaves a
    function of the Laplace abscissa lambda = pi w r^2 alone,

        F(lambda) = sum_k v_k exp(-lambda/Q_k) / Q_k ,

    which is why every size class samples one and the same transform.
    """
    q = np.asarray(q_nodes, dtype=float)
    lam = np.asarray(lam, dtype=float)
    return (np.exp(-lam[..., None] / q) / q) @ np.asarray(v, dtype=float)


def sign_changes(y: np.ndarray, rel_floor: float = 0.05) -> int:
    """Number of sign changes of ``y`` among entries above ``rel_floor`` of its
    largest magnitude (entries below the floor are skipped, so numerical noise
    in the tails does not count)."""
    y = np.asarray(y, dtype=float)
    keep = np.abs(y) > rel_floor * np.max(np.abs(y))
    s = np.sign(y[keep])
    return int(np.sum(s[1:] != s[:-1]))


__all__ += [
    "orient_modes",
    "resolution_matrix",
    "averaging_kernel_row",
    "main_lobe_fwhm",
    "captured_fraction",
    "laplace_response",
    "sign_changes",
]
