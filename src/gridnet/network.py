"""Network container and readers, following the MATPOWER case format.

Why MATPOWER's tables
---------------------
MATPOWER is the reference implementation for the conventions used throughout this
package (branch pi-model with tap and phase shift, per-unit system, sign of
injections), see Zimmerman, Murillo-Sanchez & Thomas, IEEE Trans. Power Systems 26(1)
(2011) and the MATPOWER User's Manual 8.0 (2024), Appendix B for the file format.
pandapower converts to this format internally, and PyPSA's admittance uses the same branch
model (see `gridnet.pypsa_io`), so a container that mirrors it makes machine-precision
comparisons straightforward.

Glossary for the mathematician
------------------------------
bus        a node of the graph; a point of common voltage (substation busbar).
branch     an edge; a transmission line or a transformer, with the "from" and "to" bus
           of the data taken as its orientation.
per-unit   all impedances and admittances are dimensionless, scaled by a common power
           base `base_mva` (typically 100 MVA) and the bus's nominal voltage; this is what
           makes branches at different voltage levels comparable in one matrix.
r, x       series resistance and reactance of the branch (real and imaginary part of its
           series impedance z = r + jx); x > 0 for most lines and transformers, but series
           capacitors have x < 0 (one branch of case300, 16 in service in case9241pegase).
           The weights 1/x are then negative and the susceptance Laplacian is indefinite: the
           pseudoinverse and the power flows are still defined (`gridnet.graph`), while the
           spectral comparisons of `gridnet.sparsify` and `gridnet.reweight` need positive
           semidefinite Laplacians and refuse such networks.
b          total line-charging susceptance of the pi-model, split half to each end.
tap        magnitude of the transformer's off-nominal turns ratio tau, located at the
           from end; 1 for a line. MATPOWER stores 0 for "no transformer"; normalized to 1.
shift      phase-shift angle of the transformer in degrees; nonzero only for
           phase-shifting transformers (this is the U(1) connection of the magnetic
           Laplacian, see `admittance.magnetic_laplacian`).
gs, bs     shunt conductance and susceptance at a bus, in per-unit (the case file gives
           MW and MVAr at nominal voltage; we divide by base_mva, manual Sec. 3.5).
pd, qd     active and reactive power demand at the bus, in per-unit.
pg         active power generation summed per bus, in per-unit (in-service generators
           of the MATPOWER gen table, manual Sec. 3.3); zero where there is none.
qg_min, qg_max  reactive limits of the generation at the bus, in per-unit, the sums of
           QMIN and QMAX over the in-service generators at the bus (manual Table B-2);
           nan where there is no generator. Used where the reactive limits of the generators are enforced.
vg         voltage magnitude setpoint of the generators at the bus (VG of the gen table),
           in per-unit; nan where there is no generator.
rate_a     branch rating (RATE_A of the case, s_nom of the archive) in per-unit; 0 for an unrated branch.
bus_type   MATPOWER code: 1 = PQ (load), 2 = PV (generator, voltage-controlled),
           3 = reference (slack), 4 = isolated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# MATPOWER column indices (manual, Appendix B, Tables B-1 and B-3), zero-based.
_BUS_I, _BUS_TYPE, _PD, _QD, _GS, _BS = 0, 1, 2, 3, 4, 5
_F_BUS, _T_BUS, _BR_R, _BR_X, _BR_B, _RATE_A, _TAP, _SHIFT, _BR_STATUS = 0, 1, 2, 3, 4, 5, 8, 9, 10
_GEN_BUS, _PG, _QG, _QMAX, _QMIN, _VG, _GEN_STATUS = 0, 1, 2, 3, 4, 5, 7


@dataclass
class Network:
    """A power network in MATPOWER's bus/branch representation, internally indexed.

    Buses are indexed 0 .. n_bus-1 in the order of the bus table; `bus_id` keeps the
    external numbers. Branch endpoints `from_bus`, `to_bus` are internal indices.
    Bus quantities pd, qd, gs, bs and branch quantities r, x, b are in per-unit on
    `base_mva`; `shift` is in degrees as in the file; `tap` is normalized so that 0
    (MATPOWER's "line") has become 1.
    """

    base_mva: float
    # bus table
    bus_id: NDArray[np.int64]
    bus_type: NDArray[np.int64]
    pd: NDArray[np.float64]
    qd: NDArray[np.float64]
    gs: NDArray[np.float64]
    bs: NDArray[np.float64]
    # branch table
    from_bus: NDArray[np.int64]
    to_bus: NDArray[np.int64]
    r: NDArray[np.float64]
    x: NDArray[np.float64]
    b: NDArray[np.float64]
    tap: NDArray[np.float64]
    shift: NDArray[np.float64]
    status: NDArray[np.int64]
    rate_a: NDArray[np.float64]
    # generation per bus (optional; zeros if absent)
    pg: NDArray[np.float64] | None = None
    # reactive limits and voltage setpoint of the generation per bus (optional; nan if absent)
    qg_min: NDArray[np.float64] | None = None
    qg_max: NDArray[np.float64] | None = None
    vg: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.pg is None:
            self.pg = np.zeros(len(self.bus_id))
        for name in ("qg_min", "qg_max", "vg"):
            if getattr(self, name) is None:
                setattr(self, name, np.full(len(self.bus_id), np.nan))
        n, m = len(self.bus_id), len(self.from_bus)
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "base_mva":
                continue
            arr = np.asarray(v)
            expected = n if f.name in ("bus_id", "bus_type", "pd", "qd", "gs", "bs", "pg", "qg_min", "qg_max", "vg") else m
            if arr.shape != (expected,):
                raise ValueError(f"{f.name}: expected shape ({expected},), got {arr.shape}")
            setattr(self, f.name, arr)
        if np.any(self.tap == 0):
            self.tap = np.where(self.tap == 0, 1.0, self.tap)   # manual, footnote 25
        if np.any(self.from_bus == self.to_bus):
            raise ValueError("a branch connects a bus to itself")

    @property
    def n_bus(self) -> int:
        return len(self.bus_id)

    @property
    def n_branch(self) -> int:
        return len(self.from_bus)

    @property
    def edges(self) -> list[tuple[int, int]]:
        """Oriented edge list (from, to) for `gridnet.graph`."""
        return list(zip(self.from_bus.tolist(), self.to_bus.tolist()))

    def lossless(self) -> Network:
        """Copy with r = 0, b = 0, no shunts, tap = 1, shift = 0; reactances unchanged.

        This is the model of the DC power flow assumptions (manual eq. 3.18) and of the
        lossless power flow literature (Bergen & Hill 1981; Dorfler & Bullo 2014).
        """
        stripped = {
            "r": np.zeros_like(self.r), "b": np.zeros_like(self.b),
            "gs": np.zeros_like(self.gs), "bs": np.zeros_like(self.bs),
            "tap": np.ones_like(self.tap), "shift": np.zeros_like(self.shift),
        }
        # `replace` would share every other array with the original; copy them all, so that the copy
        # can be modified (limits, setpoints) without touching the original
        copies = {f.name: getattr(self, f.name).copy() for f in fields(self)
                  if f.name != "base_mva" and f.name not in stripped}
        return replace(self, **stripped, **copies)


def from_ppc(ppc: dict) -> Network:
    """Build a Network from a MATPOWER case dict ('baseMVA', 'bus', 'branch', ...).

    Accepts both external numbering (bus numbers arbitrary) and MATPOWER/pandapower
    internal numbering; the map external -> 0..n-1 is taken from the bus table order.
    Bus shunts GS, BS are converted from MW/MVAr to per-unit (manual Sec. 3.5); PD, QD
    likewise (manual Sec. 3.4).

    The bus types are those MATPOWER's power flow uses, not the raw BUS_TYPE column: a bus is a
    PV or reference bus only if a generator is in service there, the others are PQ buses, and if
    the reference bus has no generator in service the first PV bus in bus order becomes the
    reference (MATPOWER's bustypes.m, Zimmerman et al. 2011; PYPOWER's bustypes.py is the port).
    Without this a PV bus whose generators are out would be held at a voltage by a reactive
    injection that no generator supplies. A generator in service at a PQ bus enters with its active
    output; MATPOWER also holds its reactive output QG there, which the network has no field for,
    so a nonzero one is refused (ValueError).

    A bus typed isolated (4) is out of the network with everything at it: its generators and
    branches are taken out of service whatever their own status, and its load and shunt are set
    to zero, as MATPOWER's ext2int removes the bus with its generators and branches before
    anything is solved (PYPOWER's ext2int.py: generator and branch status and-ed with the bus
    status). The bus keeps its row, without any connection. A branch left in service there would
    tie the network to a bus that no power flow solves for, held at its start voltage.
    """
    base = float(ppc["baseMVA"])
    bus = np.array(ppc["bus"], dtype=float)                   # copies: the network shares no memory with the caller's case
    br = np.array(ppc["branch"], dtype=float)
    ext = bus[:, _BUS_I].astype(np.int64)
    ext2int = {e: i for i, e in enumerate(ext.tolist())}
    isolated = bus[:, _BUS_TYPE].astype(np.int64) == 4
    if len(ext2int) < len(ext):
        twice = sorted({int(e) for e in ext if (ext == e).sum() > 1})
        raise ValueError(f"bus number(s) {twice} appear more than once in the bus table")
    pg = np.zeros(len(ext))
    qg_min = np.full(len(ext), np.nan)
    qg_max = np.full(len(ext), np.nan)
    vg = np.full(len(ext), np.nan)
    if "gen" in ppc and ppc["gen"] is not None and len(ppc["gen"]):
        gen = np.asarray(ppc["gen"], dtype=float)
        stray = sorted({int(b) for b in gen[:, _GEN_BUS] if int(b) not in ext2int})
        if stray:
            raise ValueError(f"generator(s) at bus {', '.join(map(str, stray))}, which the bus table does not have")
        for row in gen:
            if row[_GEN_STATUS] > 0 and not isolated[ext2int[int(row[_GEN_BUS])]]:
                i = ext2int[int(row[_GEN_BUS])]
                pg[i] += row[_PG] / base
                # the limits of several generators at one bus add up; the setpoint is the last in-service generator's
                qg_min[i] = np.nansum([qg_min[i], row[_QMIN] / base])
                qg_max[i] = np.nansum([qg_max[i], row[_QMAX] / base])
                vg[i] = row[_VG]
    bus_type = bus[:, _BUS_TYPE].astype(np.int64)
    has_gen = ~np.isnan(vg)
    ref = (bus_type == 3) & has_gen
    pv = (bus_type == 2) & has_gen
    bus_type = np.where(ref, 3, np.where(pv, 2, np.where(bus_type == 4, 4, 1)))
    if not ref.any() and pv.any():
        bus_type[np.flatnonzero(pv)[0]] = 3
    if "gen" in ppc and ppc["gen"] is not None and len(ppc["gen"]):
        # MATPOWER injects a generator at a PQ bus as a fixed Pg + j Qg (makeSbus); the network carries
        # the active part only, so a reactive one would be dropped silently (no case on disk has one)
        on = gen[(gen[:, _GEN_STATUS] > 0) & ~np.array([isolated[ext2int[int(b)]] for b in gen[:, _GEN_BUS]], dtype=bool)]
        at_pq = np.array([bus_type[ext2int[int(b)]] == 1 for b in on[:, _GEN_BUS]], dtype=bool)
        if np.any(at_pq & (on[:, _QG] != 0)):
            raise ValueError(f"a generator in service at a PQ bus has a reactive output QG, which the network does not carry "
                             f"(buses {sorted(set(on[at_pq & (on[:, _QG] != 0), _GEN_BUS].astype(int).tolist()))})")
    try:
        f = np.array([ext2int[int(e)] for e in br[:, _F_BUS]], dtype=np.int64)
        t = np.array([ext2int[int(e)] for e in br[:, _T_BUS]], dtype=np.int64)
    except KeyError as err:
        raise ValueError(f"branch references unknown bus {err.args[0]}") from None
    status = br[:, _BR_STATUS].astype(np.int64)
    status[isolated[f] | isolated[t]] = 0                    # the branches of an isolated bus go with it
    live = np.where(isolated, 0.0, 1.0)                      # and so do its load and its shunt
    return Network(
        base_mva=base,
        bus_id=ext,
        bus_type=bus_type,
        pd=live * bus[:, _PD] / base, qd=live * bus[:, _QD] / base,
        gs=live * bus[:, _GS] / base, bs=live * bus[:, _BS] / base,
        from_bus=f, to_bus=t,
        r=br[:, _BR_R], x=br[:, _BR_X], b=br[:, _BR_B],
        tap=br[:, _TAP], shift=br[:, _SHIFT],
        status=status,
        rate_a=br[:, _RATE_A] / base,          # RATE_A is in MVA (manual Table B-3); stored in per unit like the archive loader's ratings
        pg=pg, qg_min=qg_min, qg_max=qg_max, vg=vg,
    )


_BLOCK = re.compile(r"mpc\.(\w+)\s*=\s*\[(.*?)\];", re.DOTALL)
_SCALAR = re.compile(r"mpc\.(\w+)\s*=\s*([0-9.eE+-]+)\s*;")
# Statements after the tables, 'mpc.<table>(<rows>, <columns>) = <value>': one per line.
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_STATEMENT = re.compile(r"^\s*mpc\.(\w+)\s*\((.*)\)\s*=\s*([^=].*?);?\s*$", re.MULTILINE)
_INDEX = re.compile(r"^(.*),\s*(\d+|\[[\d\s,]+\])$")
_MASK = re.compile(rf"^mpc\.(\w+)\(:,(\d+)\)(<=|>=|<|>|==|~=)({_NUM})$")
_SCALED = re.compile(rf"^({_NUM})\*mpc\.(\w+)\(:,(\d+)\)$")
_COMPARE = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
            "==": np.equal, "~=": np.not_equal}


def _apply_statement(blocks: dict, table: str, index: str, value: str, path) -> None:
    """Apply one column statement of a case file to the parsed tables, in place.

    A case file is a MATLAB function (manual App. B), so it may modify its tables after
    defining them; Bukhsh's archive states its modifications this way. Interpreted forms
    (columns one-based as in MATLAB, rows ':' or a comparison of a column of the same table):
        mpc.gen(:,5) = -5;                        constant
        mpc.bus(:,3) = 0.60*mpc.bus(:,3);         a column scaled by a constant
        mpc.bus(mpc.bus(:,3)<0,[3,4]) = 0;        constant on the rows selected by a mask
    Anything else raises NotImplementedError, since a dropped statement changes the case.
    """
    statement = f"mpc.{table}({index}) = {value}"
    unsupported = NotImplementedError(f"{path}: cannot interpret the statement '{statement}'")
    split = _INDEX.match(index.replace(" ", ""))
    if table not in blocks or split is None:
        raise unsupported
    rows, cols = split.group(1), [int(c) - 1 for c in re.findall(r"\d+", split.group(2))]
    data = blocks[table]
    if rows == ":":
        select = np.ones(len(data), dtype=bool)
    else:
        mask = _MASK.match(rows)
        if mask is None or mask.group(1) != table:
            raise unsupported
        select = _COMPARE[mask.group(3)](data[:, int(mask.group(2)) - 1], float(mask.group(4)))
    value = value.replace(" ", "")
    scaled = _SCALED.match(value)
    if re.fullmatch(_NUM, value):
        data[np.ix_(select, cols)] = float(value)
    elif scaled and scaled.group(2) == table and [int(scaled.group(3)) - 1] == cols:
        data[np.ix_(select, cols)] *= float(scaled.group(1))
    else:
        raise unsupported


def _cell(entry: str) -> float:
    """A table entry: a number or a product of numbers ('97.6*0.5' in Bukhsh's case39mod4.m)."""
    return float(np.prod([float(v) for v in entry.split("*")]))


def read_matpower_dict(path: str | Path) -> dict:
    """Parse a MATPOWER case file (.m, case format version 2) into a dict of arrays.

    Keys 'baseMVA', 'bus', 'branch', 'gen' (and 'gencost' if present), the structure
    MATPOWER and pypower call a "ppc"; comments after '%' are stripped. Column statements
    after the tables are applied in file order (see `_apply_statement`).

    Not supported: statements that use MATLAB variables or named column constants, such as
    the conversion of MATPOWER's distribution cases case33bw.m and case69.m from ohms and kW
    to per unit and MW, 'mpc.branch(:, [BR_R BR_X]) = mpc.branch(:, [BR_R BR_X]) / (Vbase^2 /
    Sbase);' with Vbase and Sbase computed from mpc.bus(1, BASE_KV) and mpc.baseMVA. Reading
    such a file raises NotImplementedError, since dropping the statement would leave the
    impedances in ohms.
    """
    text = "\n".join(line.split("%", 1)[0] for line in Path(path).read_text().splitlines())
    scalars = {k: float(v) for k, v in _SCALAR.findall(text)}
    blocks = {}
    for name, body in _BLOCK.findall(text):
        rows = []
        for line in body.splitlines():
            line = line.split("%", 1)[0].strip().rstrip(";")
            if line:
                rows.append([_cell(v) for v in line.split()])
        if rows:
            if len({len(r) for r in rows}) > 1:
                raise ValueError(f"{path}: the rows of mpc.{name} have unequal lengths {sorted({len(r) for r in rows})}")
            blocks[name] = np.array(rows, dtype=float)
    if "baseMVA" not in scalars or "bus" not in blocks or "branch" not in blocks:
        raise ValueError(f"{path}: not a MATPOWER case (need baseMVA, bus, branch)")
    for table, index, value in _STATEMENT.findall(text):
        _apply_statement(blocks, table, index, value, path)
    ppc = {"baseMVA": scalars["baseMVA"], "bus": blocks["bus"], "branch": blocks["branch"], "gen": blocks.get("gen")}
    if "gencost" in blocks:
        ppc["gencost"] = blocks["gencost"]
    return ppc


def read_matpower_case(path: str | Path) -> Network:
    """Read a MATPOWER case file into a Network (see `read_matpower_dict`)."""
    return from_ppc(read_matpower_dict(path))
