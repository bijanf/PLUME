"""Shared pytest configuration.

Tests marked ``nesca_data`` read the Clague, Paduan & Davis (2009) Appendix A
workbooks, which are not redistributed with this repository.  When the
workbooks are absent those tests are skipped with a pointer to
``scripts/fetch_data.py``.  Set ``PLUME_REQUIRE_DATA=1`` (as continuous
integration does) to turn a missing workbook into a test failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plume_inv.data import RAW_DIR  # noqa: E402

NESCA_WORKBOOKS = ("mmc1.xls", "mmc3.xls")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if all((RAW_DIR / name).is_file() for name in NESCA_WORKBOOKS):
        return
    if os.environ.get("PLUME_REQUIRE_DATA") == "1":
        return
    skip = pytest.mark.skip(
        reason="NESCA workbooks not found; run `python scripts/fetch_data.py` to fetch them"
    )
    for item in items:
        if "nesca_data" in item.keywords:
            item.add_marker(skip)
