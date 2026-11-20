"""Tests that read third-party data carry the marker `data("<dataset>", ...)`.

A marked test skips when one of its datasets is not on disk, with the command that fetches it
(or, for the local datasets in LOCAL, noting that they are not distributed);
`pytest --fetch` fetches the missing datasets first and checks the SHA-256 of the present ones,
once per session (scripts/fetch_data.py, pinned versions and checksums), and `pytest -m "not data"` runs only the tests that need no data.
"""

import importlib.util
from pathlib import Path

import pytest

_FETCH_DATA = []                                         # scripts/fetch_data.py, loaded when a test first needs it


def _fetch_data():
    """The fetcher module, loaded on first use: an error in it then fails the tests that need a
    fetched dataset, not the collection of the whole suite (`-m "not data"` included)."""
    if not _FETCH_DATA:
        spec = importlib.util.spec_from_file_location(
            "fetch_data", Path(__file__).resolve().parents[1] / "scripts" / "fetch_data.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FETCH_DATA.append(module)
    return _FETCH_DATA[0]


def pytest_addoption(parser):
    parser.addoption("--fetch", action="store_true",
                     help="fetch the datasets that marked tests need and that are missing, and check the sums of the present ones")


# datasets the fetcher does not provide (transcriptions of printed tables and a download), kept
# locally; a test that needs one skips where the folder is missing
LOCAL = {"ieee39": "data/ieee39", "chow1982": "data/chow1982", "ACTIVSg2000": "data/ACTIVSg2000"}
ROOT = Path(__file__).resolve().parents[1]
_CHECKED = set()                                         # datasets fetched or checked in this session


def pytest_runtest_setup(item):
    for mark in item.iter_markers("data"):
        for name in mark.args:
            if name in LOCAL:
                if not (ROOT / LOCAL[name]).is_dir():
                    pytest.skip(f"dataset {name!r} ({LOCAL[name]}) is not distributed with the code")
                continue
            if item.config.getoption("--fetch"):
                if name not in _CHECKED:
                    _fetch_data().fetch(name)              # stops on a present file whose sum differs
                    _CHECKED.add(name)
                continue
            if not _fetch_data().present(name):
                pytest.skip(f"dataset {name!r} missing; run python scripts/fetch_data.py --only {name}")
