"""The synthetic experiment, one method per command. Each method runs on its instance sets, in this order: the grid
(results/), the residual sweep (results/sweep/), the two-structure cell (results/two/),
the scale cell (results/scale/) and the delta cell (results/delta_0.01/, results/delta_0.1/). It appends JSON lines
and skips the instances that already have one, so a stopped method restarts with the same command. run_synthetic.sh
runs every method in this order.

    python -m maxcon_pruning.synthetic.run METHOD [--repeats N]

  method           instance sets                what it does
  generate         grid sweep two scale         write the instance files that are not there yet
  prune            grid sweep two scale delta   LO-RANSAC once per instance (lo.jsonl), then Algorithm 1 from the level
                                                d + 2 with LO-RANSAC's set as M0 (prune.jsonl); the delta cell starts
                                                from the grid's LO-RANSAC lines
  trim             grid sweep two scale         the trimming at k eps of LO-RANSAC's model (trimming.jsonl)
  gore             grid two                     GORE at its published settings (gore.jsonl); a cell ends after STOP
                                                runs in a row in which every test timed out and nothing was removed
  wi-mbf           grid sweep two scale         WI and MBF-MaxCon, the authors' MATLAB code at their demo settings, N
                                                repetitions per instance (default REPEATS), repetition j from 2 on
                                                seeded with j as well (influence.jsonl)
  astar-survivors  grid sweep two scale delta   A* on the survivors of each pruning line (astar.jsonl)
  astar-trimmed    grid sweep two scale         A* on each trimmed set
  astar-gore       grid two                     A* on what each GORE line leaves
  astar-alone      grid sweep two scale         A* on the full instance; on the sweep only at k in SWEEP_ALONE
  all                                           every method above, in this order

The two-structure cell (two_structure.py) has no planted inliers or outliers, so its lines leave outliers_left and
inliers_removed empty. The scale cell is the grid's generator at n = 1,000,
d = 8, t = 200. The delta cell is the grid's 20 instances at d = 8, t = 40, pruned again at each delta of DELTAS with
the grid's seeds; A* runs on its survivors, and its other methods are the grid's.

A* (matlab/astar_jobs.m) runs one MATLAB session per group (an instance set, a cell and an arm, each k of the
trimming an arm of its own), under a watchdog that ends a job running past the limit, and the group ends after
astar.STOP runs in a row at the limit, a rule astar_jobs.m applies. A job whose point set equals that of another job
of the same instance with a line is not run: it gets a copy of that line, with same_as naming the job that ran.
"""
import argparse
import math
import sys
import time
from collections import namedtuple

import numpy as np

from .. import astar, common, lo_ransac, prune, trimming
from ..common import STREAMS
from ..comparators import gore
from ..problems import regression
from . import instances, two_structure
from .instances import EPS, GRID, ROOT, SCALE_NAMES, SWEEP_NAMES, parse

RES = ROOT / "results"                                  # results/ for the grid, a folder in it for every other set
STOP = 3                                                # GORE's per-cell stopping rule
REPEATS = 10                                            # the default N of WI/MBF-MaxCon repetitions
DELTAS = {"delta_0.01": 1e-2, "delta_0.1": 0.1}         # the delta cell: its results folder and delta
SWEEP_ALONE = (5.0, 10.0, 50.0)                         # the sweep's k with A* alone
GORE_M, GORE_CAP, GORE_SHARE = 1e4, 15.0, 0.15          # GORE's published settings, T = ceil(GORE_SHARE n) tests
SETS = ("grid", "sweep", "two", "scale")                # the instance sets in run order, the delta cell last

DELTA_NAMES = [nm for nm in GRID if parse(nm)[:2] == (8, 40)]             # the delta cell's instances, the grid's

Space = namedtuple("Space", "names res cell label mod")


def space(which):
    """The instance set `which`: its names in run order, its results folder, the cell of a name, a cell's label, and
    the module that generates, loads and seeds its instances (instances, or two_structure for the two-structure
    cell)."""
    grid_cell, grid_label = (lambda nm: parse(nm)[:2]), (lambda c: f"d{c[0]}_t{c[1]:02d}")
    if which == "grid":
        return Space(GRID, RES, grid_cell, grid_label, instances)
    if which == "sweep":
        return Space(SWEEP_NAMES, RES / "sweep", lambda nm: (parse(nm)[3],), lambda c: f"o{c[0]:g}", instances)
    if which == "two":
        return Space(two_structure.CELL_NAMES, RES / "two", lambda nm: (), lambda c: "two", two_structure)
    if which == "scale":
        return Space(SCALE_NAMES, RES / "scale", lambda nm: (), lambda c: "scale", instances)
    if which in DELTAS:
        return Space(DELTA_NAMES, RES / which, grid_cell, grid_label, instances)
    raise ValueError(which)


def read(which, fname):
    return common.read(space(which).res / fname)


def append(which, fname, rec):
    common.append(space(which).res / fname, rec)


def head(I):
    """The fields that start every line of an instance: its name and run, and d, t and out, or for the two-structure
    cell d, m, g and a."""
    return {f: I[f] for f in ("name", "d", "t", "run", "out", "m", "g", "a") if f in I}


def labels(I, keep):
    """outliers_left, the planted outliers among the points keep, and inliers_removed, the planted inliers outside
    them; both None for the two-structure cell, which has no planted labels."""
    if "outlier" not in I:
        return dict(outliers_left=None, inliers_removed=None)
    kept = I["outlier"][np.asarray(keep, int)]
    return dict(outliers_left=int(kept.sum()), inliers_removed=int((~I["outlier"]).sum() - (~kept).sum()))


def lo_line(I):
    """LO-RANSAC on one instance: its set, model, hypotheses and time."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(I["seed"] + [STREAMS["lo_ransac"]])
    model, S = lo_ransac.run(regression.Problem(I["A"], I["b"], I["epsilon"]), rng)
    return dict(**head(I), m0=len(S), set=S.tolist(), model=model.tolist(), trials=lo_ransac.TRIALS,
                seconds=time.perf_counter() - t0)


def prune_line(I, lo, delta=prune.DELTA):
    """Algorithm 1 at delta on one instance from LO-RANSAC's line, with the planted labels of its survivors (labels);
    seconds counts LO-RANSAC and Algorithm 1."""
    P = regression.Problem(I["A"], I["b"], I["epsilon"])
    rng = np.random.default_rng(I["seed"] + [STREAMS["prune"]])
    line = dict(**head(I), **prune.run(P, I["d"] + 2, lo["set"], lo["model"], rng, delta))
    line.update(**labels(I, line["keep"]), prune_seconds=line["seconds"], m0_seconds=lo["seconds"],
                seconds=lo["seconds"] + line["seconds"])
    return line


def planted(r):
    """The planted labels of a line as printed, nothing where there are none."""
    if r["outliers_left"] is None:
        return ""
    return f" ({r['outliers_left']} planted outliers left, {r['inliers_removed']} planted inliers removed)"


def do_generate(which):
    sp = space(which)
    sp.mod.generate(sp.names)


def do_prune(which):
    """Algorithm 1 on every instance, after LO-RANSAC where it has no line. The delta cell prunes at its delta from
    the grid's LO-RANSAC lines, and writes a grid line for an instance that has none."""
    sp = space(which)
    delta, lo_space = DELTAS.get(which, prune.DELTA), "grid" if which in DELTAS else which
    los = {r["name"]: r for r in read(lo_space, "lo.jsonl")}
    done = {r["name"] for r in read(which, "prune.jsonl")}
    for nm in sp.names:
        if nm in done:
            continue
        I = sp.mod.load(nm)
        if nm not in los:
            los[nm] = lo_line(I)
            append(lo_space, "lo.jsonl", los[nm])
        r = prune_line(I, los[nm], delta)
        append(which, "prune.jsonl", r)
        print(f"{nm} at delta {delta:g}: m0 {r['m0_initial']} -> {r['m0']}, {r['n_rounds']} rounds ({r['stop']}), "
              f"{r['samples']:,} samples, survivors {r['survivors']}{planted(r)}, bounds [{r['cert_lower']}, "
              f"{r['cert_upper']}] ({r['cert_route']}), {r['seconds']:.1f} s", flush=True)


def do_trim(which):
    """The trimming of every instance with a LO-RANSAC line: one line per k with the kept points and the planted
    outliers among them. trim_seconds is the trimming's own time and seconds adds LO-RANSAC's, as for the pruning."""
    sp = space(which)
    done = {r["name"] for r in read(which, "trimming.jsonl")}
    for lo in read(which, "lo.jsonl"):
        nm = lo["name"]
        if nm in done:
            continue
        I = sp.mod.load(nm)
        t0 = time.perf_counter()
        P = regression.Problem(I["A"], I["b"], I["epsilon"])
        sets = trimming.trim(P, lo["model"])
        secs = time.perf_counter() - t0
        out = [dict(**head(I), k=k, kept=len(S), keep=S.tolist(), outliers_kept=labels(I, S)["outliers_left"],
                    trim_seconds=secs, m0_seconds=lo["seconds"], seconds=lo["seconds"] + secs)
               for k, S in sets.items()]
        for x in out:
            append(which, "trimming.jsonl", x)
        print(f"{nm}: " + ", ".join(f"k={x['k']} kept {x['kept']}" + ("" if x["outliers_kept"] is None else
                                                                      f" ({x['outliers_kept']} planted outliers)")
                                    for x in out), flush=True)


def do_wi_mbf(which, repeats=REPEATS):
    """WI and MBF-MaxCon, the authors' code: the repetitions 1 to `repeats`, repetition by repetition so that the
    first ones are complete before the later ones start, one MATLAB session per repetition over the instances that
    lack a line not in error of either method for it, so that a run in error runs again, as A*'s do. The session's
    watchdog allows 60 s per 200 points summed over its instances, plus the slack, from the session's start."""
    sp = space(which)
    done = {(r["name"], r["method"], r["rep"]) for r in read(which, "influence.jsonl") if "error" not in r}
    for rep in range(1, repeats + 1):
        pending = [nm for nm in sp.names if not {(nm, "MBF", rep), (nm, "WI", rep)} <= done]
        if not pending:
            continue
        jobs = [dict(name=nm, file=str(sp.mod.path(nm)), problem="linear", eps=EPS, rep=rep,
                     seed_mbf=sp.mod.matlab_seed(nm, "mbf", rep), seed_wi=sp.mod.matlab_seed(nm, "wi", rep))
                for nm in pending]
        points = sum(len(sp.mod.load(nm)["b"]) for nm in pending)
        f = sp.res / "jobs" / f"influence_rep{rep:02d}.json"
        common.write_json(f, dict(out=str(sp.res / "influence.jsonl"), jobs=jobs))
        print(f"influence (authors): repetition {rep}, {len(jobs)} instances -> {f.name}", flush=True)
        common.session(f, "influence_jobs", timeout=60.0 * points / 200 + common.SLACK)


def job_sets(which, n):
    """The point set of every A* job of an instance set whose input is there, by job name: surv the survivors of each
    pruning line, k2, k3 and k5 each trimmed set, gore what each GORE line leaves, alone the full instance (n points,
    by name)."""
    sets = {astar.job(nm, "alone"): list(range(k)) for nm, k in n.items()}
    sets.update({astar.job(r["name"], "surv"): r["keep"] for r in read(which, "prune.jsonl")})
    sets.update({astar.job(r["name"], f"k{r['k']}"): r["keep"] for r in read(which, "trimming.jsonl")})
    sets.update({astar.job(r["name"], "gore"): sorted(set(range(r["n"])) - set(r["removed"]))
                 for r in read(which, "gore.jsonl") if r["removed"]})
    return sets


def share(group, sets, out):
    """Give each job of the group that has no line the line of another job of the same instance on the same point
    set, if that one has a line not in error: a copy under the job's name, with same_as naming the job that ran."""
    runs = astar.runs(out)
    for j in group:
        if j["name"] in runs and "error" not in runs[j["name"]]:
            continue
        inst = j["name"].rsplit("_", 1)[0]
        for other, S in sets.items():
            a = runs.get(other)
            if other != j["name"] and other.rsplit("_", 1)[0] == inst and S == j["keep"] and a is not None \
                    and "error" not in a:
                common.append(out, {**a, "name": j["name"], "same_as": a.get("same_as", other)})
                print(f"A*: {j['name']} has the point set of {other}, whose line it shares", flush=True)
                break


def do_astar(arm, which):
    """A* on every job of the arm, arm by arm (each k of the trimming its own) and cell by cell: surv on the survivors
    of each pruning line, alone on the full instance, k on each trimmed set, gore on what each GORE line leaves. One
    session per group under the watchdog and the stopping rule, after the jobs that share another job's line."""
    sp = space(which)
    n = {nm: len(sp.mod.load(nm)["b"]) for nm in sp.names}
    names = [nm for nm in sp.names if not (arm == "alone" and which == "sweep" and parse(nm)[3] not in SWEEP_ALONE)]
    sets, out = job_sets(which, n), sp.res / "astar.jsonl"
    arms = [f"k{k}" for k in trimming.KS] if arm == "k" else [arm]
    for c in sorted({sp.cell(nm) for nm in names}):
        for a in arms:
            group = [dict(name=astar.job(nm, a), file=sp.mod.path(nm), keep=sets[astar.job(nm, a)],
                          problem="linear", eps=EPS)
                     for nm in names if sp.cell(nm) == c and astar.job(nm, a) in sets]
            share(group, sets, out)
            astar.run_jobs(group, out, sp.res / "jobs" / f"{a}_{sp.label(c)}.json", f"{a} {sp.label(c)}")


def gore_line(I, lo):
    """GORE on one instance from LO-RANSAC's line, at most ceil(GORE_SHARE n) tests. survivors counts the points it
    leaves, with their planted labels (labels); seconds counts LO-RANSAC and GORE, as for the pruning."""
    P = regression.Problem(I["A"], I["b"], I["epsilon"])
    line = dict(**head(I), **gore.run(P, lo["model"], lo["m0"], math.ceil(GORE_SHARE * P.n), GORE_M, GORE_CAP))
    left = np.ones(P.n, bool)
    left[line["removed"]] = False
    line.update(survivors=int(left.sum()), **labels(I, np.flatnonzero(left)), gore_seconds=line["seconds"],
                m0_seconds=lo["seconds"], seconds=lo["seconds"] + line["seconds"])
    return line


def do_gore(which):
    """GORE on every instance with a LO-RANSAC line, cell by cell under the stopping rule. A run in which Gurobi
    raised, as when its licence is missing, is printed but not recorded, so it runs again next time."""
    sp = space(which)
    los = {r["name"]: r for r in read(which, "lo.jsonl")}
    lines = read(which, "gore.jsonl")
    done = {r["name"] for r in lines}
    order = {nm: i for i, nm in enumerate(sp.names)}
    for nm in sp.names:
        c = sp.cell(nm)
        cell = [r["at_limit"] for r in sorted(lines, key=lambda r: order[r["name"]]) if sp.cell(r["name"]) == c]
        if nm in done or nm not in los or astar.stopped(cell, STOP):
            continue
        r = gore_line(sp.mod.load(nm), los[nm])
        print(f"{nm}: m0 {r['m0']} -> {r['m0_final']}, {len(r['tests'])} of {r['candidates']} candidates tested, "
              f"removed {r['n_removed']}, time-outs {r['timeouts']}, incumbents {r['incumbents']}, "
              f"{r['seconds']:.1f} s{', AT THE LIMIT' if r['at_limit'] else ''}", flush=True)
        if r["gurobi_errors"]:
            print(f"{nm}: not recorded, Gurobi raised in {r['gurobi_errors']} tests", flush=True)
            continue
        append(which, "gore.jsonl", r)
        lines.append(r)


METHODS = {                                     # method: (what it runs, the instance sets), in the order of `all`
    "generate": (do_generate, SETS),
    "prune": (do_prune, (*SETS, *DELTAS)),
    "trim": (do_trim, SETS),
    "gore": (do_gore, ("grid", "two")),
    "wi-mbf": (do_wi_mbf, SETS),
    "astar-survivors": (lambda which: do_astar("surv", which), (*SETS, *DELTAS)),
    "astar-trimmed": (lambda which: do_astar("k", which), SETS),
    "astar-gore": (lambda which: do_astar("gore", which), ("grid", "two")),
    "astar-alone": (lambda which: do_astar("alone", which), SETS),
}


def main(args):
    """Run one method, or `all` in the order of METHODS, on each instance set the method runs on."""
    ap = argparse.ArgumentParser(prog="python -m maxcon_pruning.synthetic.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", metavar="METHOD", choices=[*METHODS, "all"])
    ap.add_argument("--repeats", metavar="N", type=int, default=REPEATS)
    a = ap.parse_args(args)
    for method in (METHODS if a.method == "all" else [a.method]):
        fn, spaces = METHODS[method]
        for which in spaces:
            print(f"=== {method}, {which}", flush=True)
            fn(which, a.repeats) if method == "wi-mbf" else fn(which)


if __name__ == "__main__":
    main(sys.argv[1:])
