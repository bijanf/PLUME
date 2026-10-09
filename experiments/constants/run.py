"""Export the model constants quoted in the paper.

Writes ``results/constants.json``, so that constants set in ``src/plume_inv``
reach the results manifest by the same route as computed numbers.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from plume_inv import constants as C  # noqa: E402
from plume_inv import stem  # noqa: E402

OUT = ROOT / "results" / "constants.json"


def main() -> int:
    strat = json.loads((ROOT / "results" / "stratification.json").read_text())
    # Barreyre et al. (2011) quote 12, 8 and 4 cm/s at D = 1 mm for blocky, long
    # and sheet clasts.  Evaluate their tabulated coefficients at a basaltic-glass
    # density in fresh water and in deep seawater, and find the clast density
    # that would reproduce the quoted speeds in fresh water.
    from scipy.optimize import brentq

    from plume_inv import settling

    fresh = {"rho_w": 1000.0, "nu": 1.05e-6}
    quoted = {"blocky": 0.12, "long": 0.08, "sheet": 0.04}
    bar = {"quoted_m_s_at_1mm": quoted, "fresh_water": fresh, "rho_s_test_kgm3": 2700.0}
    bar["w_fresh_2700_m_s"] = {
        sh: float(
            settling.settling_velocity(
                1e-3, rho_s=2700.0, rho_w=fresh["rho_w"], nu=fresh["nu"], shape=sh
            )
        )
        for sh in quoted
    }
    bar["w_seawater_2700_m_s"] = {
        sh: float(settling.settling_velocity(1e-3, rho_s=2700.0, shape=sh)) for sh in quoted
    }
    bar["rho_s_to_reproduce_quoted_fresh_kgm3"] = {
        sh: float(
            brentq(
                lambda rs, sh=sh: (
                    settling.settling_velocity(
                        1e-3, rho_s=rs, rho_w=fresh["rho_w"], nu=fresh["nu"], shape=sh
                    )
                    - quoted[sh]
                ),
                1100.0,
                9000.0,
            )
        )
        for sh in quoted
    }
    res = {
        "_description": "Model constants from src/plume_inv, exported for the manifest.",
        "barreyre_check": bar,
        "_generated": "2026-09-23",
        "_script": "experiments/constants/run.py",
        "_git_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "rho_tephra_kgm3": C.RHO_TEPHRA,
        "nu_seawater_m2s": C.NU_SEAWATER,
        "rho_seawater_kgm3": C.RHO_W,
        "c_p_seawater_J_kg_K": C.C_P,
        "alpha_thermal_expansion_K": C.ALPHA_T,
        "g_m_s2": C.G,
        "epsilon_entrainment": C.EPSILON_ENTRAIN,
        "lambda_front": C.LAMBDA_FRONT,
        "c_q_point_closure": C.C_Q_POINT,
        "c_f_point_closure": C.C_F_POINT,
        "c_f_planar_closure": C.C_F_PLANAR,
        "megaplume_heat_J": list(C.MEGAPLUME_HEAT_J),
        "mtt_point_zmax_coefficient": stem.C_ZMAX_POINT,
        "mtt_point_zneutral_coefficient": stem.C_ZNEUTRAL_POINT,
        "mtt_planar_zmax_coefficient": stem.C_ZMAX_PLANAR,
        "rise_height_buoyancy_flux_m4s3": strat["rise_height_consistency"]["f0_benchmark_m^4s^-3"],
        "_sources": {
            "rho_tephra_kgm3": "src/plume_inv/constants.py RHO_TEPHRA",
            "nu_seawater_m2s": "src/plume_inv/constants.py NU_SEAWATER",
            "mtt_*_coefficient": (
                "src/plume_inv/stem.py C_ZMAX_POINT, C_ZNEUTRAL_POINT, "
                "C_ZMAX_PLANAR, verified in tests/test_stem.py"
            ),
            "rise_height_buoyancy_flux_m4s3": (
                "results/stratification.json#rise_height_consistency.f0_benchmark_m^4s^-3"
            ),
        },
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("_")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
