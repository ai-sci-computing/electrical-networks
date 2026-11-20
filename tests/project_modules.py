"""The `common.py` of a study folder, loaded by path for the tests that share a builder with the scripts.

Several study folders have a module named `common` (studies/dynamics/coherency, studies/dynamics/cascades), so
each is loaded under its own name, `<part>_<name>_common`, instead of through sys.path, where the first one
imported would stand for all."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def common(folder: str):
    """studies/<folder>/common.py as a module of its own name (loaded once); `folder` is `<part>/<name>`."""
    name = f"{folder.replace('/', '_')}_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "studies" / folder / "common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
