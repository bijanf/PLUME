"""Download the raw input data and verify their SHA-256 checksums.

The data are not redistributed with this repository.  This script fetches them
from their original publishers into ``data/raw/`` and refuses to keep any file
whose checksum differs from the one recorded below (and in the README.rst of
each data directory).

    python scripts/fetch_data.py          # NESCA workbooks only (about 0.4 MB)
    python scripts/fetch_data.py --woa    # also the WOA23 files (about 162 MB)

Only the Python standard library is used.  A file that is already present with
the right checksum is not downloaded again.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# The publisher's CDN rejects requests without a browser-like User-Agent.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_ELS = "https://ars.els-cdn.com/content/image/1-s2.0-S0377027308004563-{}"
_WOA = "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/{}/netcdf/decav/1.00/{}"

# (directory, file name, URL, size in bytes, SHA-256)
NESCA = [
    (
        "nesca_clague2009",
        "mmc1.xls",
        _ELS.format("mmc1.xls"),
        135168,
        "8583f42bba0160ce5e8c48303b4c63476768bc8ae63d9a704074b3ca74d6ee6e",
    ),
    (
        "nesca_clague2009",
        "mmc2.xls",
        _ELS.format("mmc2.xls"),
        216576,
        "4194e8768dbad9a3185e23fd13282e2e3a172348b073e0db8e100d416a25a4bd",
    ),
    (
        "nesca_clague2009",
        "mmc3.xls",
        _ELS.format("mmc3.xls"),
        71680,
        "5dd7e52fdecf8d70f860403d7cd472e20dde162ef5a1aee8899c5f05425ed606",
    ),
]
WOA = [
    (
        "woa",
        "woa23_decav_t00_01.nc",
        _WOA.format("temperature", "woa23_decav_t00_01.nc"),
        85168080,
        "0bd98d3c9c20254e8fcf714aefdd23b4a191eacdfdc3ce5d019205e4a8bc41b5",
    ),
    (
        "woa",
        "woa23_decav_s00_01.nc",
        _WOA.format("salinity", "woa23_decav_s00_01.nc"),
        77439270,
        "3ab18000cea789a2b349cf6b08531f1ebf144f601d90a83f4c3508456003f019",
    ),
]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(directory: str, name: str, url: str, size: int, digest: str, force: bool = False) -> None:
    dest = RAW / directory / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if sha256(dest) == digest:
            print(f"ok       {dest.relative_to(ROOT)} (already present, checksum verified)")
            return
        print(f"refetch  {dest.relative_to(ROOT)} (checksum mismatch on the local copy)")

    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"download {url}")
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)

    got_size, got_digest = part.stat().st_size, sha256(part)
    if got_size != size or got_digest != digest:
        part.unlink()
        raise SystemExit(
            f"CHECKSUM MISMATCH for {name}\n"
            f"  expected {size} bytes, sha256 {digest}\n"
            f"  received {got_size} bytes, sha256 {got_digest}\n"
            "The file was deleted.  The publisher may have changed or moved it; "
            f"see {(RAW / directory / 'README.rst').relative_to(ROOT)}."
        )
    part.replace(dest)
    print(f"ok       {dest.relative_to(ROOT)} ({got_size} bytes, checksum verified)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--woa",
        action="store_true",
        help="also fetch the WOA23 temperature and salinity files "
        "(about 162 MB; needed only by experiments/stratification)",
    )
    parser.add_argument(
        "--force", action="store_true", help="download again even if a verified copy exists"
    )
    args = parser.parse_args(argv)

    for entry in NESCA + (WOA if args.woa else []):
        fetch(*entry, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
