"""Source-mechanism evidence under each measured clast shape.

Writes ``results/model_comparison_by_shape.json``.

``experiments/source_discrimination/run.py`` computes the Bayesian evidence of
each candidate megaplume mechanism with the settling speeds of one clast shape,
blocky.  The deposit itself prefers sheet clasts (shape weight 0.83), and the
inferred flux scales as w_s^(4/3), so a mechanism whose heat flux is too small
under blocky settling may fit under sheet settling.  This script repeats the
evidence calculation, with the same priors, likelihood, sampler settings and
seed, for the blocky, long and sheet settling laws, and runs the prior-width
sweep for every mechanism.  The blocky run reproduces the original evidences
and serves as the check.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import pathlib
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from plume_inv import kernel as K  # noqa: E402
from plume_inv import settling, sources  # noqa: E402
from plume_inv.data import FRACTIONS, load_nesca  # noqa: E402

OUT = ROOT / "results" / "model_comparison_by_shape.json"
SHAPES = ("blocky", "long", "sheet")
SWEEPS = ((1.0, "nominal"), (0.5, "half_width"), (2.0, "double_width"))


def _base():
    """The original source-discrimination module, imported by path."""
    path = ROOT / "experiments" / "source_discrimination" / "run.py"
    spec = importlib.util.spec_from_file_location("source_discrimination_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup(shape: str):
    """As ``load_setup`` in the base script, with the clast shape as an argument."""
    base = _base()
    n_buoy = json.loads((ROOT / "results" / "stratification.json").read_text())["N_primary"][
        "value_s^-1"
    ]
    u = json.loads((ROOT / "results" / "nesca_unsteady.json").read_text())
    centre = u["vent_xy_m"]
    df = load_nesca()
    keep = ~df["Location"].astype(str).str.contains("on flow", case=False, na=False).values
    r_all = np.hypot(df["x_m"].values - centre[0], df["y_m"].values - centre[1])
    radii, obs, w_s = [], [], []
    for name, d_lo, d_hi, _m, _p in FRACTIONS:
        v = df[f"glass_g_per_m2[{name}]"].values / 1000.0
        good = keep & np.isfinite(v) & (v > 0)
        radii.append(r_all[good])
        obs.append(v[good])
        w_s.append(
            float(settling.settling_velocity(settling.sieve_midpoint(d_lo, d_hi), shape=shape))
        )
    design = [K.design_matrix(r, base.Q_NODES, w, 1.0) for r, w in zip(radii, w_s, strict=True)]
    return base, design, obs, n_buoy, w_s


def run_shape(shape: str) -> dict:
    base, design, obs, n_buoy, w_s = setup(shape)
    t_grid = np.linspace(0.0, 3 * 24 * 3600.0, base.N_T)
    ev = {}
    for m in sources.MECHANISMS:
        ev[m] = {}
        for ws, label in SWEEPS:
            try:
                logz, logzerr, _r, keys = base.run_model(
                    m, design, obs, n_buoy, t_grid, width_scale=ws, seed=7
                )
                ev[m][label] = {"logz": logz, "logz_err": logzerr}
                print(
                    f"[{shape}] {m:24s} {label:13s} logZ = {logz:10.2f} +/- {logzerr:.2f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                ev[m][label] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"[{shape}] {m:24s} {label:13s} FAILED: {exc}", flush=True)
    ref = "dyke_heating"
    ln10 = np.log(10.0)
    bf = {}
    for m in ev:
        vals = {
            lab: (ev[m][lab]["logz"] - ev[ref][lab]["logz"]) / ln10
            for _w, lab in SWEEPS
            if "logz" in ev[m].get(lab, {}) and "logz" in ev[ref].get(lab, {})
        }
        bf[m] = {
            "log10B_vs_dyke": vals.get("nominal"),
            "log10B_range_over_prior_widths": [min(vals.values()), max(vals.values())]
            if vals
            else None,
            "own_prior_swing_log10": (
                max(ev[m][lab]["logz"] for _w, lab in SWEEPS if "logz" in ev[m].get(lab, {}))
                - min(ev[m][lab]["logz"] for _w, lab in SWEEPS if "logz" in ev[m].get(lab, {}))
            )
            / ln10,
        }
    return {"w_s_m s^-1": w_s, "evidence": ev, "bayes_factors": bf, "reference": ref}


def main() -> int:
    git = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    with Pool(len(SHAPES)) as pool:
        rows = pool.map(run_shape, SHAPES)
    res = {
        "_description": (
            "Bayesian evidence of the four megaplume source mechanisms "
            "under the blocky, long and sheet settling laws, with the "
            "prior-width sweep (0.5x, 1x, 2x) for every mechanism. Same "
            "priors, likelihood, sampler and seed as "
            "experiments/source_discrimination/run.py, whose blocky "
            "evidences the blocky row reproduces."
        ),
        "_generated": datetime.date.today().isoformat(),
        "_script": "experiments/source_evidence_shape/run.py",
        "_git_commit": git,
        "by_shape": dict(zip(SHAPES, rows, strict=True)),
    }
    orig = json.loads((ROOT / "results" / "model_comparison.json").read_text())
    check = {}
    for m, row in orig.get("evidence", {}).items():
        if "logz" in row.get("nominal", {}):
            check[m] = {
                "original_logz": row["nominal"]["logz"],
                "rerun_logz": res["by_shape"]["blocky"]["evidence"][m]
                .get("nominal", {})
                .get("logz"),
            }
    res["blocky_reproduction_check"] = check
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
