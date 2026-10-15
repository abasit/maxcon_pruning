"""The synthetic experiment, one method per command. Each method runs on the instance sets the protocol gives it, the
grid (results/) and the residual sweep (results/sweep/), appends JSON lines, and skips the instances that already have
one, so a stopped method restarts with the same command.

    python -m maxcon_pruning.synthetic.run METHOD

  generate         grid, sweep  write the instance files that are not there yet (instances/)
  prune            grid, sweep  LO-RANSAC once per instance (lo.jsonl), then Algorithm 1 from the level d + 2 with
                                LO-RANSAC's set as M0 (prune.jsonl)
  trim             grid, sweep  the trimming at k eps of LO-RANSAC's model (trimming.jsonl)
  astar-survivors  grid, sweep  A* on the survivors of each pruning line (astar.jsonl)
  astar-trimmed    grid         A* on each trimmed set
  gore             grid         GORE at its published settings (gore.jsonl); a cell ends after STOP runs in a row in
                                which every test timed out and nothing was removed
  astar-gore       grid         A* on what each GORE line leaves
  wi-mbf           grid, sweep  WI and MBF-MaxCon, the authors' MATLAB code at their demo settings (influence.jsonl)
  astar-alone      grid         A* on the full instance
  all                           every method above, in this order

A* (matlab/astar_jobs.m) runs one MATLAB session per cell. For astar-alone, and on the sweep, it runs one session per
job under the watchdog, and a cell ends after astar.STOP runs in a row at the limit.
"""
import math
import sys
import time

import numpy as np

from .. import astar, common, lo_ransac, prune, trimming
from ..comparators import gore
from ..problems import regression
from .instances import GRID, N, ROOT, STREAMS, SWEEP_NAMES, generate, load, matlab_seed, parse, path

RES = ROOT / "results"                                  # results/ for the grid, results/sweep/ for the sweep
STOP = 3                                                # GORE's per-cell stopping rule
GORE_M, GORE_CAP, GORE_TESTS = 1e4, 15.0, math.ceil(0.15 * N)   # GORE's published settings


def space(which):
    """(instance names, results folder, cell of a name, cell label) of the grid or the sweep."""
    if which == "grid":
        return GRID, RES, lambda nm: parse(nm)[:2], lambda c: f"d{c[0]}_t{c[1]:02d}"
    if which == "sweep":
        return SWEEP_NAMES, RES / "sweep", lambda nm: (parse(nm)[3],), lambda c: f"o{c[0]:g}"
    raise ValueError(which)


def read(which, fname):
    return common.read(space(which)[1] / fname)


def append(which, fname, rec):
    common.append(space(which)[1] / fname, rec)


def head(I):
    return dict(name=I["name"], d=I["d"], t=I["t"], run=I["run"], out=I["out"])


def lo_line(I):
    """LO-RANSAC on one instance: its set, model, trials and time."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(I["seed"] + [STREAMS["lo_ransac"]])
    model, S, trials = lo_ransac.regression(I["A"], I["b"], I["epsilon"], rng)
    return dict(**head(I), m0=len(S), set=S.tolist(), model=model.tolist(), trials=trials,
                seconds=time.perf_counter() - t0)


def prune_line(I, lo):
    """Algorithm 1 on one instance from LO-RANSAC's line. outliers_left counts the planted outliers among the
    survivors and inliers_removed the planted inliers removed; seconds counts LO-RANSAC and Algorithm 1."""
    P = regression.Problem(I["A"], I["b"], I["epsilon"])
    rng = np.random.default_rng(I["seed"] + [STREAMS["prune"]])
    line = dict(**head(I), **prune.run(P, I["d"] + 2, lo["set"], lo["model"], rng))
    kept = I["outlier"][line["keep"]]
    line.update(outliers_left=int(kept.sum()), inliers_removed=int((~I["outlier"]).sum() - (~kept).sum()),
                prune_seconds=line["seconds"], m0_seconds=lo["seconds"], seconds=lo["seconds"] + line["seconds"])
    return line


def do_generate(which):
    generate(space(which)[0])


def do_prune(which):
    names = space(which)[0]
    los = {r["name"]: r for r in read(which, "lo.jsonl")}
    done = {r["name"] for r in read(which, "prune.jsonl")}
    for nm in names:
        if nm in done:
            continue
        I = load(nm)
        if nm not in los:
            los[nm] = lo_line(I)
            append(which, "lo.jsonl", los[nm])
        r = prune_line(I, los[nm])
        append(which, "prune.jsonl", r)
        print(f"{nm}: m0 {r['m0_initial']} -> {r['m0']}, {r['n_rounds']} rounds ({r['stop']}), {r['samples']:,} "
              f"samples, survivors {r['survivors']} ({r['outliers_left']} planted outliers left, "
              f"{r['inliers_removed']} planted inliers removed), bounds [{r['cert_lower']}, {r['cert_upper']}] "
              f"({r['cert_route']}), {r['seconds']:.1f} s", flush=True)


def do_trim(which):
    """The trimming of every instance with a LO-RANSAC line: one line per k with the kept points and the planted
    outliers among them."""
    done = {r["name"] for r in read(which, "trimming.jsonl")}
    for lo in read(which, "lo.jsonl"):
        nm = lo["name"]
        if nm in done:
            continue
        I = load(nm)
        P = regression.Problem(I["A"], I["b"], I["epsilon"])
        out = [dict(**head(I), k=k, kept=len(S), keep=S.tolist(), outliers_kept=int(I["outlier"][S].sum()))
               for k, S in trimming.trim(P, lo["model"]).items()]
        for x in out:
            append(which, "trimming.jsonl", x)
        print(f"{nm}: " + ", ".join(f"k={x['k']} kept {x['kept']} ({x['outliers_kept']} planted outliers)"
                                    for x in out), flush=True)


def do_wi_mbf(which):
    """WI and MBF-MaxCon, the authors' code, one MATLAB session over every instance without both lines."""
    names, res, cell_of, label = space(which)
    done = {(r["name"], r["method"]) for r in read(which, "influence.jsonl")}
    pending = [nm for nm in names if not {(nm, "MBF"), (nm, "WI")} <= done]
    if not pending:
        print("influence (authors): nothing pending", flush=True)
        return
    jobs = [dict(name=nm, file=str(path(nm)), seed_mbf=matlab_seed(nm, "mbf"), seed_wi=matlab_seed(nm, "wi"))
            for nm in pending]
    f = res / "jobs" / "influence_authors.json"
    common.write_json(f, dict(out=str(res / "influence.jsonl"), jobs=jobs))
    print(f"influence (authors): {len(jobs)} instances -> {f.name}", flush=True)
    common.session(f, "influence_jobs", timeout=60.0 * len(jobs) + common.SLACK)


def do_astar(arm, which):
    """A* on every job of the arm, cell by cell: surv on the survivors of each pruning line, alone on the full
    instance, k on each trimmed set, gore on what each GORE line leaves. Alone, and everything on the sweep, runs one
    job per session under the watchdog and the stopping rule; the other arms run one session per cell."""
    names, res, cell_of, label = space(which)
    if arm == "surv":
        jobs = [(r["name"], "surv", r["keep"]) for r in read(which, "prune.jsonl")]
    elif arm == "alone":
        jobs = [(nm, "alone", list(range(N))) for nm in names]
    elif arm == "k":
        jobs = [(r["name"], f"k{r['k']}", r["keep"]) for r in read(which, "trimming.jsonl")]
    elif arm == "gore":
        jobs = [(r["name"], "gore", sorted(set(range(N)) - set(r["removed"]))) for r in read(which, "gore.jsonl")
                if r["removed"]]
    else:
        raise ValueError(arm)
    for c in sorted({cell_of(nm) for nm in names}):
        group = [(astar.job(nm, a), keep, path(nm)) for nm, a, keep in jobs if cell_of(nm) == c]
        astar.run_jobs(group, res / "astar.jsonl", path(names[0]), res / "jobs" / f"{arm}_{label(c)}.json",
                       f"{arm} {label(c)}", one_per_session=arm == "alone" or which == "sweep")


def gore_line(I, lo):
    """GORE on one instance from LO-RANSAC's line. survivors counts the points it leaves, outliers_left the planted
    outliers among them, and inliers_removed the planted inliers it removed; seconds counts LO-RANSAC and GORE, as
    for the pruning."""
    P = regression.Problem(I["A"], I["b"], I["epsilon"])
    line = dict(**head(I), **gore.run(P, lo["model"], lo["m0"], GORE_TESTS, GORE_M, GORE_CAP))
    left = np.ones(N, bool)
    left[line["removed"]] = False
    line.update(survivors=int(left.sum()), outliers_left=int(I["outlier"][left].sum()),
                inliers_removed=int((~I["outlier"][~left]).sum()), gore_seconds=line["seconds"],
                m0_seconds=lo["seconds"], seconds=lo["seconds"] + line["seconds"])
    return line


def do_gore(which):
    names, res, cell_of, label = space(which)
    los = {r["name"]: r for r in read(which, "lo.jsonl")}
    lines = read(which, "gore.jsonl")
    done = {r["name"] for r in lines}
    for nm in names:
        c = cell_of(nm)
        cell = [r["at_limit"] for r in sorted(lines, key=lambda r: r["run"]) if cell_of(r["name"]) == c]
        if nm in done or nm not in los or astar.stopped(cell, STOP):
            continue
        r = gore_line(load(nm), los[nm])
        append(which, "gore.jsonl", r)
        lines.append(r)
        print(f"{nm}: m0 {r['m0']} -> {r['m0_final']}, {len(r['tests'])} of {r['candidates']} candidates tested, "
              f"removed {r['n_removed']} ({r['inliers_removed']} planted inliers), time-outs {r['timeouts']}, "
              f"incumbents {r['incumbents']}, {r['seconds']:.1f} s{', AT THE LIMIT' if r['at_limit'] else ''}",
              flush=True)


METHODS = {                                     # method: (what it runs, the instance sets), in the order of `all`
    "generate": (do_generate, ("grid", "sweep")),
    "prune": (do_prune, ("grid", "sweep")),
    "trim": (do_trim, ("grid", "sweep")),
    "gore": (do_gore, ("grid",)),
    "wi-mbf": (do_wi_mbf, ("grid", "sweep")),
    "astar-survivors": (lambda which: do_astar("surv", which), ("grid", "sweep")),
    "astar-trimmed": (lambda which: do_astar("k", which), ("grid",)),
    "astar-gore": (lambda which: do_astar("gore", which), ("grid",)),
    "astar-alone": (lambda which: do_astar("alone", which), ("grid",)),
}


def main(args):
    """Run one method, or `all` in the order of METHODS, on each instance set the method runs on."""
    if len(args) != 1 or args[0] not in [*METHODS, "all"]:
        sys.exit(__doc__)
    for method in (METHODS if args[0] == "all" else args):
        fn, spaces = METHODS[method]
        for which in spaces:
            print(f"=== {method}, {which}", flush=True)
            fn(which)


if __name__ == "__main__":
    main(sys.argv[1:])
