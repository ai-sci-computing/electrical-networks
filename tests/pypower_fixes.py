"""PYPOWER 5.1.21's runpf with its reactive-limit branch (ENFORCE_Q_LIMS) corrected.

That branch has four defects corrected here, in rwl/PYPOWER's master as well (checked 2026-10-24),
and a fifth left as it is: the one-at-a-time branch tests `if k > len(mx)` (runpf.py:225, should
be `>=`), unreached with ENFORCE_Q_LIMS = 1, which the tests use:
1. `bus[gen[:, GEN_BUS], BUS_TYPE]` and `bi = gen[limited[i], GEN_BUS]` index with the generator bus
   numbers, which are floats in the data; NumPy refuses float indices since 1.12, so the branch
   raises IndexError (the neighboring lines cast with `.astype(int)`; these four do not).
2. `bus[...] == PV | bus[...] == REF` lacks parentheses: `|` binds tighter than `==`, and Python
   reads the chained comparison `bus[...] == (PV | bus[...]) == REF`, not "PV or REF".
3. The infeasibility test "all remaining generators violate" is `len(infeas) == len(remaining) or
   all(infeas == remaining)`; MATPOWER's runpf.m has `&&` and a third conjunct, all violating the
   same limit, `(isempty(mx) || isempty(mn))` (its 8.1 source, lines 379-380, checked 2026-10-24;
   the third conjunct added 2026-10-29).
4. When a limited reference bus forces a new one, `bus[pv, BUS_TYPE] = pv` writes the bus indices
   into the type column; MATPOWER's runpf.m sets `bus(pv, BUS_TYPE) = PV`.
And pfsoln, which splits a bus's reactive output among its generators in proportion to their ranges,
gives NaN for a generator with infinite limits (inf/inf); in a plain run that is cosmetic, but the
limit loop feeds the outputs back into the injections and the next solve diverges (six generators of
case2383wp). PYPOWER ports MATPOWER 4.1; MATPOWER's later pfsoln.m (8.1, checked 2026-10-24) replaces
infinite limits by a finite proxy M, the sum over the bus's generators of |Q_g| and their finite
limits, before the split. `runpf_fixed` uses a pfsoln with that proxy added and nothing else changed.
Finally, after converting generators at a limit, PYPOWER restarts Newton from the initial voltages,
with the converted buses still at their setpoints, and on case2383wp the second solve diverges from
there; MATPOWER 8.1 restarts from the current solution (`V0 = V; %% start next solve with current
solution`), which is added.
PYPOWER is pinned at 5.1.21 (pyproject.toml, requirements.lock), the version these corrections are
written against. `runpf_fixed()` compiles runpf.py's own source with exactly these corrections, each asserted to
apply once, so that a changed PYPOWER fails here instead of being patched silently; nothing else
of the algorithm is touched and the installed package is not modified.
"""

import types


def _compiled(module_name, original, corrections):
    with open(original.__file__) as f:
        source = f.read()
    for old, new in corrections:
        assert source.count(old) == 1, f"PYPOWER's {original.__name__} changed: {old!r} occurs {source.count(old)} times"
        source = source.replace(old, new)
    module = types.ModuleType(module_name)
    module.__file__ = original.__file__
    exec(compile(source, original.__file__, "exec"), module.__dict__)   # noqa: S102  the patched source of the installed PYPOWER module, compiled on purpose
    return module


def pfsoln_fixed():
    import pypower.pfsoln as original

    proxy = """        ## MATPOWER 8.1, pfsoln.m: a finite proxy M for infinite limits, the sum over all gens at the
        ## bus of abs(Qg) plus their finite limits
        Qmin = gen[on, QMIN].copy()
        Qmax = gen[on, QMAX].copy()
        M = abs(gen[on, QG])
        M[~isinf(Qmax)] = M[~isinf(Qmax)] + abs(Qmax[~isinf(Qmax)])
        M[~isinf(Qmin)] = M[~isinf(Qmin)] + abs(Qmin[~isinf(Qmin)])
        M = Cg * (Cg.T * M)
        Qmin[Qmin == inf] = M[Qmin == inf]
        Qmin[Qmin == -inf] = -M[Qmin == -inf]
        Qmax[Qmax == inf] = M[Qmax == inf]
        Qmax[Qmax == -inf] = -M[Qmax == -inf]

        ## divide proportionally
        Cmin = csr_matrix((Qmin, (range(ngon), gbus)), (ngon, nb))
        Cmax = csr_matrix((Qmax, (range(ngon), gbus)), (ngon, nb))"""
    corrections = [
        ("from numpy import asarray, angle, pi, conj, zeros, ones, finfo, c_, ix_",
         "from numpy import asarray, angle, pi, conj, zeros, ones, finfo, c_, ix_, isinf, inf"),
        ("""        ## divide proportionally
        Cmin = csr_matrix((gen[on, QMIN], (range(ngon), gbus)), (ngon, nb))
        Cmax = csr_matrix((gen[on, QMAX], (range(ngon), gbus)), (ngon, nb))""", proxy),
        ("""        gen[on, QG] = gen[on, QMIN] + \\
            (Cg * ((Qg_tot - Qg_min) / (Qg_max - Qg_min + EPS))) * \\
                (gen[on, QMAX] - gen[on, QMIN])""",
         """        gen[on, QG] = Qmin + \\
            (Cg * ((Qg_tot - Qg_min) / (Qg_max - Qg_min + EPS))) * \\
                (Qmax - Qmin)"""),
    ]
    return _compiled("pypower_pfsoln_fixed", original, corrections).pfsoln


def runpf_fixed():
    import pypower.runpf as original

    corrections = [
        ("bus[gen[:, GEN_BUS], BUS_TYPE] == PV |", "(bus[gen[:, GEN_BUS].astype(int), BUS_TYPE] == PV) |"),
        ("bus[gen[:, GEN_BUS], BUS_TYPE] == REF))", "(bus[gen[:, GEN_BUS].astype(int), BUS_TYPE] == REF)))"),
        ("if len(infeas) == len(remaining) or all(infeas == remaining):",
         "if len(infeas) == len(remaining) and all(infeas == remaining) and (len(mx) == 0 or len(mn) == 0):"),
        ("any(bus[gen[mx, GEN_BUS], BUS_TYPE] == REF)", "any(bus[gen[mx, GEN_BUS].astype(int), BUS_TYPE] == REF)"),
        ("bus[pv, BUS_TYPE] = pv", "bus[pv, BUS_TYPE] = PV"),
        ("bi = gen[limited[i], GEN_BUS]", "bi = gen[limited[i], GEN_BUS].astype(int)"),
        ("                    limited = r_[limited, mx].astype(int)\n",
         ("                    limited = r_[limited, mx].astype(int)\n"
         "                    V0 = V    ## MATPOWER 8.1, runpf.m: start the next solve with the current solution\n")),
    ]
    module = _compiled("pypower_runpf_fixed", original, corrections)
    module.pfsoln = pfsoln_fixed()
    return module.runpf
