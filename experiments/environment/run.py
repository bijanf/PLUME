"""Freeze the environment that produced results/ and figures/.

Writes ``results/environment.json``.  Re-run after any change to the env.
"""

from __future__ import annotations

import datetime
import importlib
import json
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "environment.json"

MODULES = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "xarray",
    "netCDF4",
    "jax",
    "numpyro",
    "arviz",
    "dynesty",
    "pytest",
    "gsw",
    "yaml",
    "xlrd",
]


def sh(cmd: str) -> str:
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    versions = {}
    for m in MODULES:
        try:
            versions[m] = getattr(importlib.import_module(m), "__version__", "unknown")
        except Exception as exc:
            versions[m] = f"MISSING ({exc.__class__.__name__})"

    commit = sh("git rev-parse HEAD")
    dirty = bool(sh("git status --porcelain"))
    payload = {
        "_description": "Frozen record of the environment that produced results/ and figures/.",
        "_generated": datetime.date.today().isoformat(),
        "_script": "experiments/environment/run.py",
        "_git_commit": commit or "not-a-git-repository",
        "_git_dirty": dirty,
        "env_name": "plume",
        "env_file": "env/environment.yml",
        "python": sys.version.split()[0],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": versions,
        "conda": sh("mamba --version | head -1"),
        "notes": (
            "gsw (TEOS-10) and xlrd were added after the first env build and "
            "are in env/environment.yml.  jax runs on CPU; the problem is "
            "small enough that no accelerator is needed."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}  (commit {payload['_git_commit'][:10]}, dirty={dirty})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
