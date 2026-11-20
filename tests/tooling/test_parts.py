"""The layering of the package by the parts of the series (parts.toml): every module belongs to
one part, a module imports only modules of its own part and of earlier parts, and no module
names, in its text, a module of a later release, which a reader of the earlier release would
not find. A public release holds the parts up to its own; the tests here read the parts that are
present (those whose modules are on disk) and ask that these are the first parts, with nothing
of a later one beside them."""

import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARTS = tomllib.loads((ROOT / "parts.toml").read_text())
ORDER = list(PARTS)
PART_OF = {m: p for p in ORDER for m in PARTS[p]["modules"]}
RELEASES = list(dict.fromkeys(PARTS[p]["release"] for p in ORDER))          # in the order of the parts
ON_DISK = sorted(p.stem for p in (ROOT / "src" / "gridnet").glob("*.py") if p.stem != "__init__")
PRESENT = [p for p in ORDER if all(m in ON_DISK for m in PARTS[p]["modules"])]   # the parts of this checkout


def gridnet_imports(path: Path) -> set[str]:
    """The modules of gridnet that the file imports, anywhere in it (also inside functions)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "gridnet":
                found |= {a.name for a in node.names}
            elif node.module.startswith("gridnet."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            found |= {a.name.split(".")[1] for a in node.names if a.name.startswith("gridnet.")}
    return found & set(PART_OF)


def test_every_module_belongs_to_one_part():
    listed = [m for p in ORDER for m in PARTS[p]["modules"]]
    assert len(set(listed)) == len(listed)
    assert PRESENT == ORDER[:len(PRESENT)] and PRESENT                 # the first parts, without a gap
    assert sorted(m for p in PRESENT for m in PARTS[p]["modules"]) == ON_DISK     # and no module of a later part
    assert [PARTS[p]["release"] for p in ORDER] == sorted((PARTS[p]["release"] for p in ORDER), key=RELEASES.index)


def test_no_module_imports_a_module_of_a_later_part():
    later = {}
    for module, part in PART_OF.items():
        if part not in PRESENT:
            continue
        used = gridnet_imports(ROOT / "src" / "gridnet" / f"{module}.py")
        ahead = sorted(u for u in used if ORDER.index(PART_OF[u]) > ORDER.index(part))
        if ahead:
            later[module] = ahead
    assert later == {}


def test_no_module_names_a_module_of_a_later_release():
    named = {}
    for module, part in PART_OF.items():
        if part not in PRESENT:
            continue
        text = (ROOT / "src" / "gridnet" / f"{module}.py").read_text()
        own = RELEASES.index(PARTS[part]["release"])
        ahead = sorted({m for m in re.findall(r"gridnet\.(\w+)", text)
                        if m in PART_OF and RELEASES.index(PARTS[PART_OF[m]]["release"]) > own})
        # nor a study of a later release, by its path
        ahead += sorted({f"studies/{q}" for q in re.findall(r"studies/(\w+)/", text)
                         if q in PARTS and RELEASES.index(PARTS[q]["release"]) > own})
        if ahead:
            named[module] = ahead
    assert named == {}


def test_the_tests_of_a_part_import_nothing_of_a_later_part():
    """The tests lie in one folder per part (tests/<part>), so that the tests of a part pass with
    the code of that part and the earlier ones alone."""
    later = {}
    assert sorted(p.name for p in (ROOT / "tests").iterdir() if p.is_dir() and p.name in ORDER) == sorted(PRESENT)
    for part in PRESENT:
        for path in sorted((ROOT / "tests" / part).glob("test_*.py")):
            ahead = sorted(u for u in gridnet_imports(path) if ORDER.index(PART_OF[u]) > ORDER.index(part))
            if ahead:
                later[f"{part}/{path.name}"] = ahead
        assert (ROOT / "tests" / part).is_dir()
    assert later == {}


def test_the_studies_of_a_part_import_nothing_of_a_later_part():
    """The scripts of a part lie in studies/<part>/<study>, the studies of each part listed
    in parts.toml; a script uses the package up to its own part. (Foreign code under a
    `third_party` folder is not read.)"""
    later = {}
    for part in PRESENT:
        folders = sorted(p.name for p in (ROOT / "studies" / part).iterdir() if p.is_dir())
        assert folders == sorted(PARTS[part]["studies"])
        for path in sorted((ROOT / "studies" / part).rglob("*.py")):
            if "third_party" in path.parts:
                continue
            ahead = sorted(u for u in gridnet_imports(path) if ORDER.index(PART_OF[u]) > ORDER.index(part))
            if ahead:
                later[str(path.relative_to(ROOT / "studies"))] = ahead
    assert later == {}


def test_the_datasets_of_the_parts_are_those_of_the_fetcher():
    """Every dataset that parts.toml gives a part is one that scripts/fetch_data.py fetches, each is
    given to one part, and none of the fetcher's is left out."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("fetch_data_for_parts", ROOT / "scripts" / "fetch_data.py")
    fetch_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_data)
    named = [d for p in ORDER for d in PARTS[p]["data"]]
    assert sorted(named) == sorted(fetch_data.DATASETS) and len(set(named)) == len(named)
    assert fetch_data.release_datasets(RELEASES[-1]) == named

