"""MATPOWER 8.1 under Octave as a reference for the tests (power flow and continuation with reactive
limits), through tests/octave/mp_jobs.m. MATPOWER's code is the dataset matpower_code of
scripts/fetch_data.py (pinned); a test that uses it carries @pytest.mark.data("matpower_code") and
calls `require_octave()`, which skips it where octave-cli is not installed. Jobs are run in one Octave
call, since starting Octave and loading MATPOWER take most of the time (some ten seconds)."""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from gridnet.network import read_matpower_dict

ROOT = Path(__file__).resolve().parents[1]
MATPOWER_ROOT = ROOT / "data" / "matpower_code" / "matpower8.1"
CASES = ROOT / "data" / "matpower"


def require_octave() -> str:
    octave = shutil.which("octave-cli")
    if octave is None:
        pytest.skip("octave-cli not on the PATH")
    return octave


@dataclass
class Result:
    success: bool
    lam: float
    gen: np.ndarray       # per generator: bus number, QG, QMIN, QMAX (MVAr), VM at its bus, VG
    vm: np.ndarray
    va: np.ndarray        # degrees


def run(jobs: list[tuple[str, str, float]]) -> list[Result]:
    """jobs: (kind, case name, value) with kind "pf" (value = loading), "cpf" (value = 1 with
    reactive limits, 0 without) or "dcopf" (value ignored); see mp_jobs.m. For "dcopf" the fields
    carry rundcopf's results: `lam` the objective f, `gen[:, 1]` PG (MW), `vm` the prices LAM_P
    ($/MWh), `va` the angles."""
    octave = require_octave()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        lines, outs = [], []
        for k, (kind, case, value) in enumerate(jobs):
            out = tmp / f"job{k}.txt"
            outs.append(out)
            lines.append("\t".join([kind, str(CASES / f"{case}.m"), repr(float(value)), str(out)]))
        (tmp / "jobs.txt").write_text("\n".join(lines) + "\n")
        here = Path(__file__).parent / "octave"
        run = subprocess.run([octave, "--no-gui", "-q", "--eval", f"addpath('{here}'); mp_jobs('{MATPOWER_ROOT}', '{tmp / 'jobs.txt'}')"],
                             capture_output=True, text=True, timeout=1800, check=False)
        if run.returncode != 0:                                 # Octave's own error text, not a bare CalledProcessError
            raise RuntimeError(f"Octave exited with {run.returncode}:\n{run.stderr[-4000:]}{run.stdout[-2000:]}")
        tables = [np.loadtxt(out, delimiter=",", ndmin=2) for out in outs]
    results = []
    for (_, case, _), A in zip(jobs, tables):
        n_gen = len(read_matpower_dict(CASES / f"{case}.m")["gen"])
        results.append(Result(success=bool(A[0, 0]), lam=float(A[0, 1]), gen=A[1:1 + n_gen],
                              vm=A[1 + n_gen:, 0], va=A[1 + n_gen:, 1]))
    return results
