"""Smoke test: the package imports and reports its version."""

import gridnet


def test_version_is_string():
    assert isinstance(gridnet.__version__, str)
    assert gridnet.__version__ == "0.0.1"
