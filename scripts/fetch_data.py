"""Download the third-party input data the code reads into data/<dataset>/.

None of these files is redistributed with the repository. Each is fetched from its original
source at a pinned version and checked against its SHA-256 sum; a file that is already present
with the right sum is left alone, one with a wrong sum stops the run.

    python scripts/fetch_data.py                 # everything
    python scripts/fetch_data.py --only matpower psat
    python scripts/fetch_data.py --list          # datasets, sources, whether present

Datasets
--------
matpower
    Test cases of MATPOWER release 8.1 (https://github.com/MATPOWER/matpower, tag 8.1,
    directory data/) and MATPOWER's LICENSE. The license states that the case files are not
    covered by MATPOWER's BSD license. Each case carries its own provenance header (mostly the
    University of Washington power systems test case archive). Format: MATPOWER case format
    version 2 (Zimmerman & Murillo-Sanchez, MATPOWER User's Manual 8.0, 2024, Appendix B).
bukhsh
    Test cases and published local solutions of Bukhsh, Grothey, McKinnon & Trodden, "Local
    Solutions of the Optimal Power Flow Problem", IEEE Trans. Power Systems 28(4), 2013, from
    their archive https://webhomes.maths.ed.ac.uk/OptEnergy/LocalOpt/ (no version on the page;
    the checksums pin the files as mirrored on 2026-09-16). Only the case files (testcases/)
    and the solution tables (data/*.csv) are fetched, never the paper or the figures.
psat
    The six-bus system of Avalos, Canizares, Milano & Conejo (2009), file tests/d_006_mdl.m of
    PSAT 2.1.11 (F. Milano, http://faraday1.ucd.ie/psat.html), extracted from the release
    archive psat-2.1.11-mat.zip. PSAT is licensed under the GNU GPL; the file is downloaded and
    read as input, not copied into or distributed with this repository.
osm
    The OSM-based European transmission network of the PyPSA-Eur project, version 0.7
    (https://data.pypsa.org/workflows/eur/osm/0.7/), five CSV tables. Derived from
    OpenStreetMap data, (c) OpenStreetMap contributors, ODbL.
powerplants
    The matched European power plant list released with powerplantmatching v0.8.0 (Gotzens,
    Heinrichs, Hörsch & Hofmann 2019), one CSV, the source of PyPSA-Eur's generators; the dispatch
    of the continental grid in studies/dynamics/cascades/europe_dispatch.py is built from it. The list is
    released under CC BY 4.0 (the README of powerplantmatching v0.8.0, section Licence).
matpower_code
    The code of MATPOWER release 8.1 that runpf and runcpf need under Octave (the library
    directories lib, mips/lib, mp-opt-model/lib and mptest/lib without their self-tests, and the
    LICENSE, 3-clause BSD), extracted from the release archive matpower8.1.zip; each of the 598
    files pinned in scripts/matpower_code.sha256. The tests use it as an independent reference for
    the power flow and the continuation with reactive limits (tests/matpower_octave.py), and skip
    where Octave is not installed.
naturalearth
    Natural Earth 1:50m admin-0 countries, release v5.1.1
    (https://github.com/nvkelso/natural-earth-vector, tag v5.1.1, directory 50m_cultural/),
    public domain; used only for the borders on the map figures.
"""

import argparse
import hashlib
import http.client
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

MATPOWER = "https://raw.githubusercontent.com/MATPOWER/matpower/8.1/"
BUKHSH = "https://webhomes.maths.ed.ac.uk/OptEnergy/LocalOpt/"
PSAT_ZIP = ("http://faraday1.ucd.ie/archive/psat/psat-2.1.11-mat.zip",
            "65528a910c51f59a4f90bf333ae460436a883655cc31599f61b52835e710c8b4")
OSM = "https://data.pypsa.org/workflows/eur/osm/0.7/"
MATPOWER_ZIP = ("https://github.com/MATPOWER/matpower/releases/download/8.1/matpower8.1.zip",
                "7f13b1441669a64e312d14a60e564cd91977ff1676ff77d25538e94ff313dd56")


def _manifest(name: str) -> list[tuple[str, str, str]]:
    """(file, "zip:<member>", SHA-256) for every line "<sha256>  <member>" of scripts/<name>."""
    entries = []
    for line in (Path(__file__).parent / name).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            sha, member = line.split(maxsplit=1)
            entries.append((member, "zip:" + member, sha))
    return entries
NATURALEARTH = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.1/50m_cultural/"

# dataset -> (target directory, [(file relative to it, URL, SHA-256)]); a URL of the form
# "zip:<member>" names a member of the dataset's archive instead of a download of its own.
DATASETS = {
    "matpower": ("matpower", [
        ("LICENSE-MATPOWER", MATPOWER + "LICENSE",
         "5d14c09b3e4f2adf62c0373e6320163697aa4603186f8925c07c6b84201e1750"),
        *((f"{name}.m", MATPOWER + f"data/{name}.m", sha) for name, sha in [
            ("case4gs", "a752c05867eace569e7d67876fa215fd9c0836287b587ae8d312dd4613acf757"),
            ("case5", "ed9072089d080debde2aa029ed8a95491c311b08ff4f45ad7bf06882d86fe279"),
            ("case6ww", "2ddf9154a81e4a5a7bce287b116df328fabf31d9d8ddb55552fff9c97291c65a"),
            ("case9", "ee50fc7bf9f6019c0f3a3bc94d20978cc667b08f695dc725d00dbd998b358623"),
            ("case14", "2ffc4e1b734ae6c5e92dbe68b4e36010ed695a4bbcc4d065c74c4fbc39fcf3c1"),
            ("case_ieee30", "b3bcc6165577e9e70a39af98f808405233a184289056cab8ce35fbc5a3ad953e"),
            ("case30", "3d9030311259b553be85d02336b7e1bcb24ec04775bee6671bdb62d18e4e2137"),
            ("case33bw", "b40831eeb444669ae876e2996f0dda9f05cd83e81b314b8dfca51e4890cca95d"),
            ("case39", "440833f998d1d8768c620477dacb6afd4e03f25f394bb0d200524513ad3fdc3c"),
            ("case57", "2218325a6e8fe6c7b8b28202f523670459268075a6fd41b4959d66f17d47d28b"),
            ("case69", "7bbdc8c39394eb6c924a64cad49d528bc8e99061823eb0f4a42b22da5c146a2c"),
            ("case118", "bc2e6f22b4b9e776572885ee4b50e4f4ab2ee0c5577e9126e86d906f14c4b5f7"),
            ("case300", "69a90280e999ef533d94656e0fbc08311f1347c962dd2753ff2005ff5e3f9ac5"),
            ("case1354pegase", "1b08b25a2f6c1d540d090009dfaff41ff2b05784a2d8d302a7ad695821557b89"),
            ("case2383wp", "cffde7da790c36a864e7998ae5ff97367227c6960be7ae8ec0eb50c1bb809bf3"),
            ("case9241pegase", "593a58ecddb5af509ff94410a6630f81021b48fa31da0694ff516acfa9ea5f3b"),
        ]),
    ]),
    "bukhsh": ("bukhsh_local_optima", [
        (path, BUKHSH + path, sha) for path, sha in [
            ("data/LMBM3.csv", "69e6f7797ef6b990f6b8fec35c9fe0397a32d4a776c002b2af2586090bf546f8"),
            ("data/LMBM3_50.csv", "7d6e4d9072a53a72b4ea37a04e3060bd8127e1ca674a5362774115bbb9405128"),
            ("data/WB2data_2solns.csv", "5eba8523b28f21b0279791198438369df40a11527e53e6b2826d8afcedc508e3"),
            ("data/WB3data_2solns.csv", "dd62eb002e0bcb65868abe730ab65403f44c77d89e5cabd0c45a2a938380b17e"),
            ("data/WB5data_2solns.csv", "f6c7073228db0d89cb922105eb7c685cc63f8fe22a45a2dea47bda1168319de0"),
            ("data/case118data_3solns.csv", "e0f02c591e9cc7013cc5d98e876a7589e66e868dbd703fe242b4c2509f145ab9"),
            ("data/case22loopdata_2solns.csv", "8ce1461da4fd1eab970ebe7adae2d084c0809574d09bcb5f00ddffa96b64e84d"),
            ("data/case300data_7solns.csv", "a43b26fc0861d4e695151dbff19d95697ebea18e30c00ff556c54ca435885178"),
            ("data/case30loopdata_2solns.csv", "7a529e737208f2575a404099c041eaa841dd3c20d24a59e7c8023e45c574be42"),
            ("data/case39data_16solns_linear.csv", "bc9549a1f31865efc5f9ec8758962486b5b229dfbe20ea0988d27fa0ebf253d8"),
            ("data/case39data_2solns_quadratic.csv", "dc0c808b98ba6be997df3e0db7a35af7bafff3e77dc27b6e5b4820b3e1ced7ed"),
            ("data/case9data_4solns.csv", "66eaace02da4dbfc6b5650586db9cfc7e51fec048f4b0b6ba7e63866ab6ffb2c"),
            ("data/other_solutions/case118mod1data_2solns.csv", "0e6b37e08fe7774ccf5034e89404f6e26863a2fbdc61dc34eaaf7031416611fc"),
            ("data/other_solutions/case300mod1data_2solns.csv", "48340d8c8772970f19d1d2c7b97d256480e0bd256df585341ba757b94c73bad2"),
            ("data/other_solutions/case30loopmoddata_3solns.csv", "7a505979ce7ff34a52f483b93f5b2bdfadb15fa3cbb6b098c90e008c81af99b4"),
            ("data/other_solutions/case39mod3data_5solns.csv", "6a8a3b4bbb279ec172057ed6bf08e71adce207dc127da669ad73d66c2467d91b"),
            ("data/other_solutions/case39mod4data_3solns.csv", "bfe0520551fb9d5b14e0ad18a634662eed149bc29d2691a6464678f4b9ab8db3"),
            ("data/other_solutions/case5moddata_3solns.csv", "59bfede310f576f0aed115a3a8892bae9303c5eaf6f1d85cd67e63849692de66"),
            ("testcases/LMBM3.m", "a394e491adf6d8e567fe3769fe9aeb62e1226f473d4670fb561122562c4b6067"),
            ("testcases/LMBM3_50.m", "994a3dcee30772539179012ae95d2108c1640af11d6e8ac34d74a920f6cc044b"),
            ("testcases/WB2.m", "cc74c0c6e7f0d0da39d75872e567b8f4dc0f7fe96f17e659e2863954b047d15c"),
            ("testcases/WB3.m", "c1700dd28270f1ba30da1a0114e0e73286fd5a7032a5fb9be5944f1229400b9e"),
            ("testcases/WB5.m", "e0096f5c67684d1dcea1a65f87bc5e2c5cf5266b32dfe7ae69f08d63882aa85b"),
            ("testcases/case118mod.m", "6c2149293a0d981efba24d7241c8d656b47afefcc3f0c17ebeac07c35a09e455"),
            ("testcases/case22loop.m", "1c4e0a80be63b47b838d0df513fdb740f51eb88bdf48854248f8cab3bfb9af19"),
            ("testcases/case300mod.m", "18a94ee754186241ac47494474c8c80f87a27e3cbd968c2f3f35ced60a57f768"),
            ("testcases/case30loop.m", "1a2423b5c18ade225434e6dbd6b5b0a9c69d40355faddaee2c04b289d0d08ebc"),
            ("testcases/case39mod1.m", "5b4f7ed9ddfcc3254dcba731d408e35bc72a2560e5de3f9899aae8433266d1f0"),
            ("testcases/case39mod2.m", "c58b9841f07ef5f5c656013872e9102cbec9c4dc26eceee56ac5706912dffb72"),
            ("testcases/case9mod.m", "bd16e729c267b446a4225fbc7dfb74c85f99e33599203292920ad0bb5e0e20ba"),
            ("testcases/other_cases/WB5mod.m", "913b76285dc95cb5af8d8e38746c2851e7b225d58b3e581db4c058329f111854"),
            ("testcases/other_cases/case118mod1.m", "d80c41fdf0f8d74561e575e91ab732517d4b87805562f8b4e363cef9c696bdc0"),
            ("testcases/other_cases/case300mod1.m", "f578f3b7f29a047332e572706f8bfb04afc1bb5a4ad207e880a051b92b423a6f"),
            ("testcases/other_cases/case30loopmod.m", "dff751c59dddd421ef3e40ffe7deca9245a0c5ae62b4a42c3ce2a560c5eaca8a"),
            ("testcases/other_cases/case39mod3.m", "7c5623b6c306b6fc1cb02f106cdf0ff741fa0bf277218c13f37765dbd46b374d"),
            ("testcases/other_cases/case39mod4.m", "4d3829eabc7040c14c46e2d2e8e231a122139473bd4c800f37621f1a38e841ff"),
        ]
    ]),
    "psat": ("psat", [
        ("d_006_mdl.m", "zip:psat/tests/d_006_mdl.m",
         "1c5a16e34ac5d35aa2993c61eb9582b861df8b6205424482e5ee2477f2d071d0"),
    ]),
    "osm": ("raw/pypsa-eur-osm-0.7", [
        (f"{name}.csv", OSM + f"{name}.csv", sha) for name, sha in [
            ("buses", "f50c75ea0339f5578ae383b47a1c0564b371c5d0d60fd4181ed626c7f3a5a523"),
            ("converters", "7564f11e58604a013b3fae28fa71637f84ff7e0c3555be71fc8c5523790cdc17"),
            ("lines", "d3ee4f45a0d9889eee9e27376cb9cfc877514964f77ba406bea3b6eb336531cb"),
            ("links", "2c7b4769fecc68bb1887f5ea3b3c259a61145fb0a3ac50dec0d3b2ed10b16832"),
            ("transformers", "65ac747f623da832c5d6c1046e106f4b3bba0cc72c443dfbb567a83d4da4a7c0"),
        ]
    ]),
    "powerplants": ("raw/powerplantmatching", [
        ("powerplants.csv", "https://raw.githubusercontent.com/PyPSA/powerplantmatching/v0.8.0/powerplants.csv",
         "eb4fceb8285322ddb34c9f76ddf1bc0311a297628756185d99187fd84db5c2c6"),
    ]),
    "naturalearth": ("raw/naturalearth", [
        (f"ne_50m_admin_0_countries.{ext}", NATURALEARTH + f"ne_50m_admin_0_countries.{ext}", sha)
        for ext, sha in [
            ("shp", "94799b91827fb9f32c0c517d60281572abcf5e4647d19b7f52217071cc4fd30d"),
            ("shx", "1813a4b55c3e9db05b7f5f77adbfe7b8f8ad5bce870dab0f4e5ec937f0e5a791"),
            ("dbf", "4ec70fc63842df719ee1f9f7bb0469b528b1eab8099c44aea6c7339a10cacc4b"),
            ("prj", "98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb"),
            ("cpg", "3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824"),
        ]
    ]),
}
DATASETS["matpower_code"] = ("matpower_code", _manifest("matpower_code.sha256"))
ARCHIVES = {"psat": PSAT_ZIP, "matpower_code": MATPOWER_ZIP}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, attempts: int = 6) -> bytes:
    """The body of `url`; a dropped connection is retried (some of the sources are slow)."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except (OSError, http.client.HTTPException) as err:
            if attempt == attempts:
                raise
            print(f"retry {url} ({type(err).__name__})")
            time.sleep(2 * attempt)


def present(name: str) -> bool:
    """True if every file of the dataset is on disk (the sums are checked by `verify` and `fetch`)."""
    folder, files = DATASETS[name]
    return all((DATA / folder / f).exists() for f, _, _ in files)


def verify(name: str) -> list[str]:
    """The files of the dataset that are on disk and whose SHA-256 differs from the pinned one."""
    folder, files = DATASETS[name]
    return [rel for rel, _, want in files
            if (DATA / folder / rel).exists() and sha256((DATA / folder / rel).read_bytes()) != want]


def fetch(name: str) -> None:
    folder, files = DATASETS[name]
    target = DATA / folder
    archive = None
    for rel, url, want in files:
        path = target / rel
        if path.exists():
            if sha256(path.read_bytes()) != want:
                sys.exit(f"{path} is present but its SHA-256 differs from the pinned {want}")
            print(f"have  {name}: {rel}")
            continue
        if url.startswith("zip:"):
            if archive is None:
                zip_url, zip_sha = ARCHIVES[name]
                blob = download(zip_url)
                if sha256(blob) != zip_sha:
                    sys.exit(f"{zip_url}: SHA-256 differs from the pinned {zip_sha}")
                archive = zipfile.ZipFile(io.BytesIO(blob))
            try:
                data = archive.read(url[4:])
            except KeyError:
                sys.exit(f"{ARCHIVES[name][0]} has no member {url[4:]}")
        else:
            data = download(url)
        if sha256(data) != want:
            sys.exit(f"{url}: SHA-256 {sha256(data)} differs from the pinned {want}")
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")              # written whole, then renamed: no half file under the name
        try:
            part.write_bytes(data)
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)
        print(f"fetch {name}: {rel}")


def release_datasets(release: str) -> list[str]:
    """The datasets that the parts of `release` and of every release before it name in parts.toml."""
    import tomllib

    parts = tomllib.loads((DATA.parent / "parts.toml").read_text())
    releases = list(dict.fromkeys(p["release"] for p in parts.values()))
    if release not in releases:
        raise SystemExit(f"release must be one of {releases}")
    upto = releases[: releases.index(release) + 1]
    return list(dict.fromkeys(d for p in parts.values() if p["release"] in upto for d in p["data"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", choices=sorted(DATASETS), help="datasets to fetch")
    ap.add_argument("--list", action="store_true", help="list the datasets and exit")
    ap.add_argument("--release", help="fetch the datasets of this release of the series and of those before it (parts.toml)")
    args = ap.parse_args()
    if args.list:
        for name, (folder, files) in DATASETS.items():
            print(f"{name:13s} data/{folder:28s} {len(files):3d} files  "
                  f"{'present' if present(name) else 'missing'}")
        return
    names = args.only or (release_datasets(args.release) if args.release else DATASETS)
    for name in names:
        fetch(name)


if __name__ == "__main__":
    main()
