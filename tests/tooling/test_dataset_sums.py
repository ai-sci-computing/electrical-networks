"""The datasets that `scripts/fetch_data.py` cannot fetch (ACTIVSg2000 behind a form, the IEEE 39-bus files of the
benchmark report, the PyPSA-Eur outputs of the project's own run) carry a table "| File | SHA-256 |" in their
SOURCE.md. Every file of such a table that is present is checked against its sum, so a run on changed data shows
up as a failing test; the fetched datasets are checked by the fetcher itself (conftest.py)."""

import hashlib
import re
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[2] / "data"
SOURCES = sorted(p for p in DATA.rglob("SOURCE.md") if "| File | SHA-256 |" in p.read_text())


def _rows(source):
    text = source.read_text()
    table = text[text.index("| File | SHA-256 |"):]
    for line in table.splitlines()[2:]:
        m = re.match(r"^\| (.+?) \| ([0-9a-f]{64}) \|$", line)
        if not m:
            break
        yield m.group(1), m.group(2)


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.parent.name)
def test_present_files_match_their_recorded_sums(source):
    rows = list(_rows(source))
    assert rows, source
    present = [(name, digest) for name, digest in rows if (source.parent / name).exists()]
    if not present:
        pytest.skip(f"no file of {source.parent.name} is present")
    for name, digest in present:
        h = hashlib.sha256((source.parent / name).read_bytes()).hexdigest()
        assert h == digest, f"{source.parent.name}/{name}"
