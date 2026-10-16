"""The real experiment, one method per command. A method runs on every instance of one dataset, or of every dataset
when none is given, appends JSON lines to results/real/<dataset>/, and skips what already has a line, so a stopped
method restarts with the same command.

    python -m maxcon_pruning.real.run METHOD [DATASET] [--repeats N]

  prune            N starts per instance (default REPEATS), each LO-RANSAC (lo.jsonl) and then Algorithm 1 from the
                   level p + 2 with LO-RANSAC's set as M0 (prune.jsonl); start j appends j to both seeds
  trim             the trimming at k eps of the model of every LO-RANSAC line (trimming.jsonl)
  gore             GORE at its published settings for real data (gore.jsonl); a dataset ends after STOP runs in a row
                   in which every test timed out and nothing was removed
  gore-tests       GORE given Algorithm 1's own time t on the instance as more tests: max(10, ceil(t / 15 s)) tests
                   of at most 15 s (gore_tests.jsonl)
  gore-cap         GORE given t as longer tests: 10 tests of at most max(15 s, t / 10) (gore_cap.jsonl)
  wi-mbf           WI and MBF-MaxCon, the authors' MATLAB code at their demo settings, N repetitions per instance
                   (default REPEATS; influence.jsonl), then each method's consensus per instance
  astar-survivors  A* on the survivors of each pruning line (astar.jsonl)
  astar-trimmed    A* on each trimmed set
  astar-gore       A* on what each GORE line leaves
  astar-alone      A* on the full instance
  all              every method above, in this order

The datasets are those of instances.py. Every method but prune and wi-mbf needs prune's lines: the trimming takes
every start, GORE and A* start 1. A method that starts from LO-RANSAC reports LO-RANSAC plus its own time as
seconds, and the two apart as m0_seconds and prune_seconds or gore_seconds. The counts of labelled outliers appear on
labelled instances only.

A* (matlab/astar_jobs.m) starts from LO-RANSAC's model and runs each job in a MATLAB session of its own under the
watchdog. astar-alone takes the instances by their number of LO-RANSAC outliers, fewest first, and a dataset ends
after astar.STOP runs in a row at the limit. WI and MBF-MaxCon (matlab/influence_jobs.m) run one MATLAB session per
instance. On a homography their feasibility test is our L-infinity program, so their returned sets are scored in
the problem's own residual (score).
"""
import argparse
import math
import sys
import time

import numpy as np

from .. import astar, common, lo_ransac, prune, trimming
from ..comparators import gore
from .instances import DATASETS, ROOT, load, outlier, path, problem

RES = ROOT / "results" / "real"                         # one folder per dataset
REPEATS = 20                                            # the default N: starts of prune, repetitions of wi-mbf
STOP = 3                                                # GORE's per-dataset stopping rule
GORE_M, GORE_CAP, GORE_TESTS = 1000.0, 15.0, 10         # GORE's published settings for real data
REP_SECONDS = 600.0                                     # a WI/MBF-MaxCon session's allowance per repetition


def read(dataset, fname):
    return common.read(RES / dataset / fname)


def append(dataset, fname, rec):
    common.append(RES / dataset / fname, rec)


def first(dataset, fname):
    """The lines of start 1 in a results file, by instance name."""
    return {r["name"]: r for r in read(dataset, fname) if r["start"] == 1}


def head(I):
    return dict(name=I["name"], dataset=I["set"])


def counts(I, keep):
    """On a labelled instance, the outliers among the points keep (outliers_left) and the inliers not among them
    (inliers_removed); nothing on an unlabelled one."""
    out = outlier(I)
    if out is None:
        return {}
    kept = out[keep]
    return dict(outliers_left=int(kept.sum()), inliers_removed=int((~out).sum() - (~kept).sum()))


def lo_line(I, P, start):
    """LO-RANSAC on one instance for one start: its set, model, hypotheses and time."""
    t0 = time.perf_counter()
    model, S = lo_ransac.run(P, np.random.default_rng(common.seed(I["name"], "lo_ransac", start)))
    return dict(**head(I), start=start, m0=len(S), set=S.tolist(), model=model.tolist(), trials=lo_ransac.TRIALS,
                seconds=time.perf_counter() - t0)


def prune_line(I, P, lo):
    """Algorithm 1 on one instance from a LO-RANSAC line, seeded by its start; seconds counts LO-RANSAC and
    Algorithm 1."""
    rng = np.random.default_rng(common.seed(I["name"], "prune", lo["start"]))
    line = dict(**head(I), start=lo["start"], **prune.run(P, P.p + 2, lo["set"], lo["model"], rng))
    line.update(**counts(I, line["keep"]), prune_seconds=line["seconds"], m0_seconds=lo["seconds"],
                seconds=lo["seconds"] + line["seconds"])
    return line


def do_prune(dataset, repeats):
    """The starts 1 to `repeats` of every instance, start by start, so that start 1 comes first."""
    los = {(r["name"], r["start"]): r for r in read(dataset, "lo.jsonl")}
    done = {(r["name"], r["start"]) for r in read(dataset, "prune.jsonl")}
    for start in range(1, repeats + 1):
        for nm in DATASETS[dataset]:
            if (nm, start) in done:
                continue
            I = load(dataset, nm)
            P = problem(I)
            if (nm, start) not in los:
                los[nm, start] = lo_line(I, P, start)
                append(dataset, "lo.jsonl", los[nm, start])
            r = prune_line(I, P, los[nm, start])
            append(dataset, "prune.jsonl", r)
            left = f" ({r['outliers_left']} labelled outliers left)" if "outliers_left" in r else ""
            print(f"{nm} start {start}: m0 {r['m0_initial']} -> {r['m0']}, {r['n_rounds']} rounds ({r['stop']}), "
                  f"{r['samples']:,} samples, survivors {r['survivors']} of {r['n']}{left}, bounds "
                  f"[{r['cert_lower']}, {r['cert_upper']}] ({r['cert_route']}), {r['seconds']:.1f} s", flush=True)


def do_trim(dataset):
    """The trimming of every LO-RANSAC line, one line per k."""
    done = {(r["name"], r["start"]) for r in read(dataset, "trimming.jsonl")}
    for lo in read(dataset, "lo.jsonl"):
        nm, start = lo["name"], lo["start"]
        if (nm, start) in done:
            continue
        I = load(dataset, nm)
        out = [dict(**head(I), start=start, k=k, kept=len(S), keep=S.tolist(), **counts(I, S))
               for k, S in trimming.trim(problem(I), lo["model"]).items()]
        for x in out:
            append(dataset, "trimming.jsonl", x)
        print(f"{nm} start {start}: " + ", ".join(f"k={x['k']} kept {x['kept']} of {I['n']}" for x in out),
              flush=True)


def gore_line(I, P, lo, T=GORE_TESTS, cap=GORE_CAP):
    """GORE on one instance from LO-RANSAC's line, at most T tests of at most cap seconds. survivors counts the points
    it leaves; seconds counts LO-RANSAC and GORE, as for the pruning."""
    line = dict(**head(I), **gore.run(P, lo["model"], lo["m0"], T, GORE_M, cap))
    keep = np.setdiff1d(np.arange(I["n"]), line["removed"])
    line.update(survivors=len(keep), **counts(I, keep), gore_seconds=line["seconds"], m0_seconds=lo["seconds"],
                seconds=lo["seconds"] + line["seconds"])
    return line


def do_gore(dataset, variant=None):
    """GORE on every instance with a LO-RANSAC line of start 1, in the dataset's order. At the published settings
    (variant None) the dataset ends after STOP runs in a row at the limit. The variants give GORE Algorithm 1's own
    time t (prune_seconds of start 1), never less than the published settings, and have no stopping rule: 'tests'
    runs max(GORE_TESTS, ceil(t / GORE_CAP)) tests of at most GORE_CAP, 'cap' GORE_TESTS tests of at most
    max(GORE_CAP, t / GORE_TESTS). A run in which Gurobi raised, as for an instance too large for a size-limited
    licence, is printed but not recorded, so it runs again next time; any other error is recorded."""
    fname = "gore.jsonl" if variant is None else f"gore_{variant}.jsonl"
    los, prs = first(dataset, "lo.jsonl"), first(dataset, "prune.jsonl")
    done = {r["name"]: r for r in read(dataset, fname)}
    for nm in DATASETS[dataset]:
        if nm in done or nm not in los or (variant and nm not in prs):
            continue
        if variant is None and astar.stopped([done[x]["at_limit"] for x in DATASETS[dataset] if x in done], STOP):
            print(f"GORE {dataset}: {STOP} runs in a row at the limit, {nm} is not run", flush=True)
            continue
        T, cap = GORE_TESTS, GORE_CAP
        if variant == "tests":
            T = max(GORE_TESTS, math.ceil(prs[nm]["prune_seconds"] / GORE_CAP))
        elif variant == "cap":
            cap = max(GORE_CAP, prs[nm]["prune_seconds"] / GORE_TESTS)
        I = load(dataset, nm)
        r = gore_line(I, problem(I), los[nm], T, cap)
        print(f"{nm}: m0 {r['m0']} -> {r['m0_final']}, {len(r['tests'])} of {r['candidates']} candidates tested "
              f"(T {T}, cap {cap:.1f} s), removed {r['n_removed']}, time-outs {r['timeouts']}, incumbents "
              f"{r['incumbents']}, errors {r['errors']}, {r['seconds']:.1f} s"
              f"{', AT THE LIMIT' if r['at_limit'] else ''}", flush=True)
        if r["gurobi_errors"]:
            print(f"{nm}: not recorded, Gurobi raised in {r['gurobi_errors']} tests", flush=True)
            continue
        append(dataset, fname, r)
        done[nm] = r


def matlab_seed(name, use, rep):
    """A seed for MATLAB's rng: one 32-bit number, drawn by numpy's SeedSequence from the seed of the stream."""
    return int(np.random.SeedSequence(common.seed(name, use, rep)).generate_state(1)[0])


def score(P, S):
    """The consensus size a returned set S stands for in the problem's own residual: its size when our test finds it
    feasible (or it has at most p points), else the largest consensus set grown from the models fitted to it, the
    problem's fit and the model of its feasibility program when that is solved."""
    S = np.asarray(S, int)
    if len(S) <= P.p or P.decide(S) == "feasible":
        return len(S)
    best = 0
    for M in [P.fit(S), *P.test(S[None, :])[1]]:
        if M is not None:
            best = max(best, len(P.grow(np.flatnonzero(P.residual(M) <= P.eps), M, np.arange(P.n))[0]))
    return best


def do_wi_mbf(dataset, repeats):
    """WI and MBF-MaxCon, the authors' code: the repetitions 1 to `repeats` of each instance that lack a line of
    either method, one MATLAB session per instance under the watchdog. Then, per instance and method, the returned
    set's size and its consensus (score) over the repetitions (the first line of each), and the time."""
    out = RES / dataset / "influence.jsonl"
    done = {(r["name"], r["method"], r["rep"]) for r in common.read(out)}
    for nm in DATASETS[dataset]:
        reps = [j for j in range(1, repeats + 1) if not {(nm, "MBF", j), (nm, "WI", j)} <= done]
        if not reps:
            continue
        I = load(dataset, nm)
        jobs = [dict(name=nm, file=str(path(dataset, nm)), problem=I["problem"], eps=I["eps"], rep=j,
                     seed_mbf=matlab_seed(nm, "mbf", j), seed_wi=matlab_seed(nm, "wi", j)) for j in reps]
        f = RES / dataset / "jobs" / f"influence_{nm}.json"
        common.write_json(f, dict(out=str(out), jobs=jobs))
        print(f"WI and MBF-MaxCon: {nm}, {len(jobs)} repetitions -> {f.name}", flush=True)
        common.session(f, "influence_jobs", timeout=REP_SECONDS * len(jobs) + common.SLACK)
    lines = {}
    for r in common.read(out):
        lines.setdefault((r["name"], r["method"], r["rep"]), r)
    for nm in DATASETS[dataset]:
        P = None
        for method in ("WI", "MBF"):
            L = [r for (n, m, j), r in sorted(lines.items()) if n == nm and m == method and j <= repeats]
            if not L:
                continue
            ok = [r for r in L if "error" not in r]
            P = P or problem(load(dataset, nm))
            cs = [score(P, r["returned"]) for r in ok]
            summary = (f"returned {np.mean([r['size'] for r in ok]):.1f}, consensus {np.mean(cs):.1f} ({min(cs)} to "
                       f"{max(cs)}), {np.mean([r['seconds'] for r in ok]):.1f} s") if ok else "no result"
            print(f"{nm} {method}: {len(L)} repetitions, {len(L) - len(ok)} errors, {summary}", flush=True)


def do_astar(dataset, arm):
    """A* on every job of the arm, from start 1: surv on the survivors of each pruning line, k on each trimmed set,
    gore on what each GORE line leaves, alone on the full instance (see the module docstring)."""
    los = first(dataset, "lo.jsonl")
    insts = {nm: load(dataset, nm) for nm in los}
    if arm == "surv":
        jobs = [(nm, "surv", r["keep"]) for nm, r in first(dataset, "prune.jsonl").items()]
    elif arm == "k":
        jobs = [(r["name"], f"k{r['k']}", r["keep"]) for r in read(dataset, "trimming.jsonl") if r["start"] == 1]
    elif arm == "gore":
        jobs = [(r["name"], "gore", np.setdiff1d(np.arange(r["n"]), r["removed"]).tolist())
                for r in read(dataset, "gore.jsonl") if r["removed"]]
    elif arm == "alone":
        jobs = sorted([(nm, "alone", list(range(I["n"]))) for nm, I in insts.items()],
                      key=lambda j: insts[j[0]]["n"] - los[j[0]]["m0"])
    else:
        raise ValueError(arm)
    group = [dict(name=astar.job(nm, a), file=path(dataset, nm), keep=keep, problem=insts[nm]["problem"],
                  eps=insts[nm]["eps"], init=np.ravel(los[nm]["model"]).tolist()) for nm, a, keep in jobs]
    astar.run_jobs(group, RES / dataset / "astar.jsonl", RES / dataset / "jobs" / f"{arm}.json", f"{arm} {dataset}",
                   one_per_session=True, stop=astar.STOP if arm == "alone" else None)


METHODS = {                                             # in the order of `all`; each takes (dataset, repeats)
    "prune": do_prune,
    "trim": lambda dataset, repeats: do_trim(dataset),
    "gore": lambda dataset, repeats: do_gore(dataset),
    "gore-tests": lambda dataset, repeats: do_gore(dataset, "tests"),
    "gore-cap": lambda dataset, repeats: do_gore(dataset, "cap"),
    "wi-mbf": do_wi_mbf,
    "astar-survivors": lambda dataset, repeats: do_astar(dataset, "surv"),
    "astar-trimmed": lambda dataset, repeats: do_astar(dataset, "k"),
    "astar-gore": lambda dataset, repeats: do_astar(dataset, "gore"),
    "astar-alone": lambda dataset, repeats: do_astar(dataset, "alone"),
}


def main(args):
    """Run one method, or `all` in the order of METHODS, on one dataset or on every dataset."""
    ap = argparse.ArgumentParser(prog="python -m maxcon_pruning.real.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", metavar="METHOD", choices=[*METHODS, "all"])
    ap.add_argument("dataset", metavar="DATASET", nargs="?", choices=list(DATASETS))
    ap.add_argument("--repeats", metavar="N", type=int, default=REPEATS)
    a = ap.parse_args(args)
    for method in (METHODS if a.method == "all" else [a.method]):
        for dataset in ([a.dataset] if a.dataset else DATASETS):
            print(f"=== {method}, {dataset}", flush=True)
            METHODS[method](dataset, a.repeats)


if __name__ == "__main__":
    main(sys.argv[1:])
