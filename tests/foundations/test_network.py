"""Tests for gridnet.network: the Network container and the MATPOWER case reader.

Format reference: MATPOWER User's Manual 8.0 (Zimmerman & Murillo-Sanchez 2024),
Appendix B (case file format), Sec. 3.2 (branch columns R, X, B, TAP, SHIFT), Sec. 3.5
(bus columns GS, BS).
"""

from pathlib import Path

import numpy as np
import pytest

from gridnet.network import Network, read_matpower_case, read_matpower_dict

CASES = Path(__file__).resolve().parents[2] / "data" / "matpower"


@pytest.mark.data("matpower")
def test_read_case14_dimensions():
    net = read_matpower_case(CASES / "case14.m")
    assert isinstance(net, Network)
    assert net.n_bus == 14
    assert net.n_branch == 20
    assert net.base_mva == 100.0


@pytest.mark.data("matpower")
def test_read_case14_bus_numbering_is_internal_zero_based():
    net = read_matpower_case(CASES / "case14.m")
    np.testing.assert_array_equal(net.bus_id, np.arange(1, 15))   # external numbers kept
    assert net.from_bus.min() >= 0 and net.to_bus.max() <= 13     # internal indices
    # first branch of case14 is 1 -> 2
    assert (net.from_bus[0], net.to_bus[0]) == (0, 1)


@pytest.mark.data("matpower")
def test_read_case14_transformers_and_shunt():
    net = read_matpower_case(CASES / "case14.m")
    # MATPOWER: TAP == 0 means "a line", equivalent to tau = 1 (manual, footnote 25).
    # case14 has three transformers with off-nominal taps and no phase shifters.
    assert np.sum(net.tap != 1.0) == 3
    assert np.all(net.shift == 0.0)
    # one shunt capacitor (bus 9, 19 MVAr) -> bs in per-unit on base_mva
    assert np.count_nonzero(net.bs) == 1
    assert np.isclose(net.bs[8], 19.0 / 100.0)
    assert np.all(net.gs == 0.0)


@pytest.mark.data("matpower")
def test_read_case300_noncontiguous_bus_numbers():
    net = read_matpower_case(CASES / "case300.m")
    assert net.n_bus == 300
    assert net.bus_id.max() > 300           # external numbers are not 1..300
    assert net.from_bus.max() < 300         # but internal indices are
    assert net.n_branch == 411


@pytest.mark.data("matpower")
def test_edges_property_matches_from_to():
    net = read_matpower_case(CASES / "case9.m")
    assert net.edges == list(zip(net.from_bus.tolist(), net.to_bus.tolist()))


def test_network_validation_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        Network(
            base_mva=100.0,
            bus_id=np.array([1, 2]),
            bus_type=np.array([3, 1]),
            pd=np.zeros(2), qd=np.zeros(2), gs=np.zeros(2), bs=np.zeros(2),
            from_bus=np.array([0]), to_bus=np.array([1]),
            r=np.array([0.0]), x=np.array([0.1, 0.2]),  # wrong length
            b=np.array([0.0]), tap=np.array([1.0]), shift=np.array([0.0]),
            status=np.array([1]), rate_a=np.array([0.0]),
        )


@pytest.mark.data("matpower")
def test_lossless_copy_strips_losses_shunts_taps_and_shifts():
    net = read_matpower_case(CASES / "case14.m")
    ll = net.lossless()
    assert np.all(ll.r == 0) and np.all(ll.b == 0)
    assert np.all(ll.gs == 0) and np.all(ll.bs == 0)
    assert np.all(ll.tap == 1.0) and np.all(ll.shift == 0.0)
    np.testing.assert_array_equal(ll.x, net.x)          # reactances untouched
    assert np.any(net.tap != 1.0)                        # original is untouched


@pytest.mark.data("matpower")
def test_lossless_copy_shares_no_array_with_the_original():
    """The copy is independent: changing any of its arrays (a generator limit in `gridnet.limits`, a setpoint)
    must leave the original network as it was."""
    from dataclasses import fields

    net = read_matpower_case(CASES / "case14.m")
    ll = net.lossless()
    for f in fields(net):
        if f.name != "base_mva":
            assert not np.shares_memory(getattr(ll, f.name), getattr(net, f.name)), f.name


@pytest.mark.data("matpower")
def test_generators_are_read_into_per_bus_injection():
    """case14 has five generators (buses 1, 2, 3, 6, 8); only the slack and bus 2 produce
    active power (232.4 and 40 MW); loads total 259 MW (MATPOWER manual, App. B, Table B-2)."""
    net = read_matpower_case(CASES / "case14.m")
    assert np.isclose(net.pg.sum(), 2.724)                         # per unit on 100 MVA
    assert np.isclose(net.pg[0], 2.324) and np.isclose(net.pg[1], 0.40)
    assert np.count_nonzero(net.pg) == 2
    assert np.isclose(net.pd.sum(), 2.59)
    assert net.pg.shape == (14,)


# --- statements after the tables ---------------------------------------------------------------

_MINIMAL = """function mpc = tiny
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [
1 3 0   0  0 0 1 1 0 345 1 1.1 0.9;
2 1 90  30 0 0 1 1 0 345 1 1.1 0.9;
3 1 -10 5  0 0 1 1 0 345 1 1.1 0.9;
];
mpc.gen = [
1 0 0 300 -300 1 100 1 250 10 0 0 0 0 0 0 0 0 0 0 0;
];
mpc.branch = [
1 2 0.01 0.05 0 250 250 250 0 0 1 -360 360;
2 3 0.01 0.05 0 250 250 250 0 0 1 -360 360;
];
"""


def _write(tmp_path, tail):
    path = tmp_path / "tiny.m"
    path.write_text(_MINIMAL + tail)
    return path


def test_column_statements_after_the_tables_are_applied_in_order(tmp_path):
    """A MATPOWER case file is a MATLAB function (manual App. B), and Bukhsh's archive states
    its modifications as column assignments after the tables (case9mod.m: reactive limits,
    then loads scaled to 60 %, case300mod.m: negative loads deleted). The reader applies
    constant assignments, scalings of a column by itself and the masked assignment, in file
    order, with MATLAB's one-based columns."""
    ppc = read_matpower_dict(_write(tmp_path, """
%Reactive power bounds were tightened
mpc.gen(:,5) = -5;
mpc.bus(:,3) = 0.60*mpc.bus(:,3);
mpc.bus(:,4)=2*mpc.bus(:,4)
mpc.bus(mpc.bus(:,3)<0,[3,4]) = 0;
mpc.bus(:,12) = 1.05;
"""))
    assert ppc["gen"][0, 4] == -5.0
    assert np.allclose(ppc["bus"][:, 2], [0.0, 54.0, 0.0])        # scaled, then the negative load deleted
    assert np.allclose(ppc["bus"][:, 3], [0.0, 60.0, 0.0])        # no trailing semicolon; row 3 zeroed by the mask
    assert np.allclose(ppc["bus"][:, 11], 1.05)


def test_a_commented_statement_is_not_applied(tmp_path):
    ppc = read_matpower_dict(_write(tmp_path, "% mpc.bus(:,3) = 0.60*mpc.bus(:,3);\n"))
    assert ppc["bus"][1, 2] == 90.0


@pytest.mark.data("matpower")
def test_an_unsupported_statement_raises_instead_of_being_dropped(tmp_path):
    """Dropping a statement silently changes the case (found 2026-09-18 on case22loop.m and
    case9mod.m), so any assignment into a table that the reader cannot interpret is an error;
    MATPOWER's case33bw.m converts ohms to per unit this way."""
    with pytest.raises(NotImplementedError, match="mpc.branch"):
        read_matpower_dict(_write(tmp_path, "mpc.branch(:, [BR_R BR_X]) = mpc.branch(:, [BR_R BR_X]) / (Vbase^2 / Sbase);\n"))
    with pytest.raises(NotImplementedError):
        read_matpower_dict(CASES / "case33bw.m")


@pytest.mark.data("bukhsh")
def test_bukhsh_archive_files_carry_their_modifications():
    """case22loop.m scales its loads by 2.15 (95 MW -> 204.25 MW, 20 -> 43 MVAr); case9mod.m sets
    Qmin = -5 MVAr and scales the loads of case9 to 60 % (Bukhsh et al. 2013, Sec. III-B and
    the archive's comments)."""
    bukhsh = CASES.parent / "bukhsh_local_optima" / "testcases"
    loop = read_matpower_dict(bukhsh / "case22loop.m")["bus"]
    assert np.allclose(loop[1::2, 2], 204.25) and np.allclose(loop[1::2, 3], 43.0)
    nine = read_matpower_dict(bukhsh / "case9mod.m")
    assert np.allclose(nine["gen"][:, 4], -5.0)
    assert np.allclose(nine["bus"][[4, 6, 8], 2], [54.0, 60.0, 75.0])


def test_products_inside_table_cells_are_evaluated(tmp_path):
    """The archive's further cases write scaled entries as products in the cell
    (case39mod4.m: '97.6*0.5')."""
    path = tmp_path / "tiny.m"
    path.write_text(_MINIMAL.replace("2 1 90  30", "2 1 90*0.5  30*2*0.1"))
    ppc = read_matpower_dict(path)
    assert np.allclose(ppc["bus"][1, 2:4], [45.0, 6.0])


@pytest.mark.data("matpower")
def test_case9241pegase_against_the_tables_of_josz_et_al_2016():
    """Josz, Fliscounakis, Maeght & Panciatici 2016, Sec. II (pp. 2-3), the tables for case9241pegase: 9,241
    buses, 1,445 generating units, 16,049 branches and 1,319 transformers (counted as branches with a tap
    ratio other than 0 and 1; 1,334 have a tap or a phase shift); the voltage levels 750 to 110 kV; 1,945
    buses at 330 kV or more, 3,185 at 220 kV, 4,111 from 90 to 154 kV; 75 branches with R < 0, 16 with X < 0,
    none with both. The MATPOWER file the contingency and cascade projects read is the one they describe."""
    ppc = read_matpower_dict(CASES / "case9241pegase.m")
    bus, gen, br = ppc["bus"], ppc["gen"], ppc["branch"]
    assert (len(bus), len(gen), len(br)) == (9241, 1445, 16049)
    tap = br[:, 8]
    assert np.count_nonzero((tap != 0) & (tap != 1)) == 1319
    kv = bus[:, 9]
    assert sorted(set(kv.tolist()), reverse=True) == [750, 400, 380, 330, 220, 154, 150, 120, 110]
    assert (np.count_nonzero(kv >= 330), np.count_nonzero(kv == 220), np.count_nonzero((kv >= 90) & (kv <= 154))) == (1945, 3185, 4111)
    r, x = br[:, 2], br[:, 3]
    assert (np.count_nonzero(r < 0), np.count_nonzero(x < 0), np.count_nonzero((r < 0) & (x < 0))) == (75, 16, 0)


_OCTAVE_DUMP = r"""
function dump(listfile, outdir)
  % evaluate every case file as the function it is and write its tables at 17 significant digits
  fid = fopen(listfile); files = {};
  while true, l = fgetl(fid); if ~ischar(l), break; end; files{end+1} = l; end; fclose(fid);
  for k = 1:numel(files)
    [d, name] = fileparts(files{k});
    addpath(d); m = feval(name); rmpath(d);
    for f = {'bus', 'gen', 'branch', 'gencost', 'baseMVA'}
      if isfield(m, f{1}), dlmwrite(fullfile(outdir, [name '.' f{1} '.txt']), m.(f{1}), 'precision', '%.17g'); end
    end
  end
end
"""


@pytest.mark.parametrize("folder", [
    pytest.param("matpower", marks=pytest.mark.data("matpower")),
    pytest.param("bukhsh_local_optima/testcases", marks=pytest.mark.data("bukhsh")),
    pytest.param("ACTIVSg2000", marks=pytest.mark.data("ACTIVSg2000")),
])
def test_reader_agrees_with_octave_on_every_case_file(folder, tmp_path):
    """An independent reading of every case file the reader accepts: Octave evaluates each file as the
    function it is (no MATPOWER needed) and writes bus, gen, branch, gencost and baseMVA at 17 significant
    digits; `read_matpower_dict` gives the same numbers bit for bit, NaN where Octave has NaN. Covers the
    statements after the tables in the Bukhsh archive, Texas and both PEGASE cases (34 files on
    2026-10-24). The two distribution cases, which convert their units through MATPOWER's index
    functions, are refused by the reader (test above) and left out."""
    import shutil
    import subprocess

    octave = shutil.which("octave-cli")
    if octave is None:
        pytest.skip("octave-cli not on the PATH")
    root = Path(__file__).resolve().parents[2] / "data" / folder
    files = sorted(p for p in root.rglob("*.m")
                   if p.stem not in ("case33bw", "case69") and not p.stem.startswith(("scenarios_", "contab_")))
    files = [p for p in files if "function mpc" in p.read_text(errors="ignore")[:4000]]
    assert files
    (tmp_path / "dump.m").write_text(_OCTAVE_DUMP)
    (tmp_path / "list.txt").write_text("\n".join(str(p) for p in files) + "\n")
    subprocess.run([octave, "--no-gui", "-q", "--eval", f"addpath('{tmp_path}'); dump('{tmp_path / 'list.txt'}', '{tmp_path}')"],
                   check=True, capture_output=True, timeout=600)
    for p in files:
        ours = read_matpower_dict(p)
        for key in ("bus", "gen", "branch", "gencost", "baseMVA"):
            out = tmp_path / f"{p.stem}.{key}.txt"
            assert out.exists() == (key in ours), (p.name, key)
            if out.exists():
                theirs = np.loadtxt(out, delimiter=",", ndmin=2)
                mine = np.atleast_2d(np.asarray(ours[key], dtype=float))
                assert mine.shape == theirs.shape, (p.name, key)
                np.testing.assert_array_equal(mine, theirs, err_msg=f"{p.name} {key}")


@pytest.mark.data("matpower")
def test_ratings_are_stored_in_per_unit_of_the_base():
    """RATE_A is in MVA in the case file (MATPOWER manual, Table B-3); the network stores it divided by the base,
    like the archive loader's ratings, and an unrated branch (0) stays 0."""
    p = read_matpower_dict(CASES / "case2383wp.m")
    net = read_matpower_case(CASES / "case2383wp.m")
    np.testing.assert_allclose(net.rate_a * net.base_mva, p["branch"][:, 5], rtol=1e-14)
    assert net.base_mva == 100.0 and net.rate_a.max() < 100


def test_malformed_tables_are_refused_by_name(tmp_path):
    """A repeated bus number, a generator at a bus that does not exist and gencost rows of unequal length are
    refused with a ValueError that says what is wrong, not parsed silently or left to a KeyError or numpy
    (remaining findings, code/02 first #17, #19, #32)."""
    from gridnet.network import from_ppc

    ppc = read_matpower_dict(_write(tmp_path, ""))
    twice = {**ppc, "bus": ppc["bus"].copy()}
    twice["bus"][2, 0] = 2                                              # bus numbers 1, 2, 2
    with pytest.raises(ValueError, match="bus number"):
        from_ppc(twice)
    stray = {**ppc, "gen": ppc["gen"].copy()}
    stray["gen"][0, 0] = 7
    with pytest.raises(ValueError, match="generator.*bus 7"):
        from_ppc(stray)
    with pytest.raises(ValueError, match="gencost"):
        read_matpower_dict(_write(tmp_path, "mpc.gencost = [\n2 0 0 3 0.01 40 0;\n2 0 0 2 40;\n];\n"))


def test_the_network_does_not_share_memory_with_the_case_it_was_read_from():
    """`from_ppc` copies the case's tables: r, x, b, shift and tap used to be views into the caller's
    branch array, so editing the case afterward changed the network."""
    from gridnet.network import from_ppc

    ppc = {"baseMVA": 100.0,
           "bus": np.array([[1, 3, 0, 0, 0, 0, 1, 1, 0, 100, 1, 1.1, 0.9], [2, 1, 50, 10, 0, 0, 1, 1, 0, 100, 1, 1.1, 0.9]], dtype=float),
           "gen": np.array([[1, 0, 0, 100, -100, 1, 100, 1, 200, 0]], dtype=float),
           "branch": np.array([[1, 2, 0.01, 0.1, 0.02, 0, 0, 0, 1.0, 5.0, 1, -360, 360]], dtype=float)}
    net = from_ppc(ppc)
    before = (net.r.copy(), net.x.copy(), net.b.copy(), net.tap.copy(), net.shift.copy())
    ppc["branch"][:, 2:5] *= 7.0; ppc["branch"][:, 8] = 0.9; ppc["branch"][:, 9] = -3.0
    for a, b in zip((net.r, net.x, net.b, net.tap, net.shift), before):
        np.testing.assert_array_equal(a, b)
