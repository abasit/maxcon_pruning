"""The real experiment, one method per command. A method runs on every instance of one dataset, or of every dataset
when none is given, appends JSON lines to results/real/<dataset>/, and skips what already has a line, so a stopped
method restarts with the same command.

    python -m maxcon_pruning.real.run METHOD [DATASET] [--repeats N]

  prune            N starts per instance (default REPEATS), each LO-RANSAC (lo.jsonl) and then Algorithm 1 from the
                   subset size p + 2 with LO-RANSAC's set as M0 (prune.jsonl); start j appends j to both seeds; the
                   starts run start by start, every instance's start j before any start j + 1
  trim             the trimming at k eps of the model of every LO-RANSAC line (trimming.jsonl)
  gore             GORE at its published settings for real data (gore.jsonl); a dataset ends after STOP runs in a row
                   in which every test timed out and nothing was removed
  gore-tests       GORE given Algorithm 1's own time t on the instance as more tests: max(10, ceil(t / 15 s)) tests
                   of at most 15 s (gore_tests.jsonl)
  gore-cap         GORE given t as longer tests: 10 tests of at most max(15 s, t / 10) (gore_cap.jsonl)
  wi-mbf           WI and MBF-MaxCon, the authors' MATLAB code at their demo settings, N repetitions per instance
                   (default REPEATS; influence.jsonl), then a summary against the optimal set A* found, which needs
                   A*'s lines: A* comes later in this order, and wi-mbf run again after it runs nothing new
  astar-survivors  A* on the survivors of each pruning line (astar.jsonl)
  astar-trimmed    A* on each trimmed set, one group per k
  astar-gore       A* on what each GORE line leaves
  astar-alone      A* on the full instance, then on the teaser pairs
  all              every method above, in this order

The datasets are those of instances.py. Every method but prune and wi-mbf needs prune's lines: the trimming takes
every start, GORE and A* start 1. A method that starts from LO-RANSAC reports LO-RANSAC plus its own time as
seconds, and the two apart as m0_seconds and prune_seconds, trim_seconds or gore_seconds. The counts of labelled
outliers appear on labelled instances only.

A* (matlab/astar_jobs.m) starts from LO-RANSAC's model on a homography and from the instance's x0 on a linear
instance (astar_jobs.m ignores the job's init there), and runs each job in a MATLAB session of its own under the
watchdog. astar-survivors runs on every instance. The other arms take a group's jobs (a dataset's, per k for the
trimming) by their number of points beyond LO-RANSAC's m0, fewest first (for astar-alone, LO-RANSAC's outliers), and
end the group after astar.STOP runs in a row at the limit. On a dataset with a fixed subset in instances.ALONE,
astar-alone runs on every instance of the subset and on no other, with no stop. On any other dataset it then also
runs, with no stop, on the teaser pairs (teaser): those where start 1's pruning removed at least TEASER_REMOVED
matches and A* on its survivors finished within TEASER_SECONDS. WI and MBF-MaxCon (matlab/influence_jobs.m) run
unmodified, one MATLAB session per instance, with our L-infinity program as their feasibility oracle on a homography,
so a returned set need not be L2-feasible. wi-mbf reports each returned set against the optimal set A* found: its
size, the points of the optimal set it lacks, the points outside that set it keeps, and whether our test confirms it
L2-feasible on a homography (feasible on a linear instance, where the test is the problem's own).
"""
import argparse
import math
import sys
import time
from collections import defaultdict

import numpy as np

from .. import astar, common, lo_ransac, prune, trimming
from ..comparators import gore
from .instances import ALONE, DATASETS, ROOT, load, outlier, path, problem

RES = ROOT / "results" / "real"                         # one folder per dataset
REPEATS = 20                                            # the default N: starts of prune, repetitions of wi-mbf
STOP = 3                                                # GORE's per-dataset stopping rule
GORE_M, GORE_CAP, GORE_TESTS = 1000.0, 15.0, 10         # GORE's published settings for real data
REP_SECONDS = 600.0                                     # a WI/MBF-MaxCon session's allowance per repetition
TEASER_REMOVED, TEASER_SECONDS = 90, 10.0               # the teaser pairs: removed at least, A* on the survivors within


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
    """The trimming of every LO-RANSAC line, one line per k. trim_seconds is the trimming's own time and seconds adds
    LO-RANSAC's, as for the pruning."""
    done = {(r["name"], r["start"]) for r in read(dataset, "trimming.jsonl")}
    for lo in read(dataset, "lo.jsonl"):
        nm, start = lo["name"], lo["start"]
        if (nm, start) in done:
            continue
        I = load(dataset, nm)
        t0 = time.perf_counter()
        sets = trimming.trim(problem(I), lo["model"])
        secs = time.perf_counter() - t0
        out = [dict(**head(I), start=start, k=k, kept=len(S), keep=S.tolist(), **counts(I, S), trim_seconds=secs,
                    m0_seconds=lo["seconds"], seconds=lo["seconds"] + secs) for k, S in sets.items()]
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


def optimal_set(dataset, name):
    """The optimal set of an instance as A* found it (0-based indices): the set of a finished A* line, one not in
    error, not at the limit and with a model of finite numbers, on the full instance if that finished, else on the
    survivors (an optimum of the full instance with probability at least 1 - DELTA); None when neither finished. On
    rows that repeat, linearFitV2 can return a model of NaN (null in the line) with a set that is not feasible;
    astar_jobs.m gives such a run a line in error, and the test of the model here stays as a second guard."""
    runs = astar.runs(RES / dataset / "astar.jsonl")
    for arm in ("alone", "surv"):
        a = runs.get(astar.job(name, arm))
        if (a is not None and "error" not in a and not a["hit_limit"]
                and np.isfinite(np.asarray(a.get("model"), float)).all()):
            return set(a["set"])
    return None


def wi_mbf_summary(dataset, repeats):
    """Per instance and method, over the repetitions 1 to `repeats` (the first line not in error of each, else its
    line in error): the size of the returned set (survivors), the points of the optimal set it lacks (inliers
    removed) and the points outside the optimal set it keeps (outliers left), each as min / median / max, a dash for
    the two counts when no optimal set is known; how many returned sets our test confirms L2-feasible on a
    homography, feasible on a linear instance (decide's 'feasible': the program's own model keeps every returned
    match within eps; undecided counts as not confirmed); the median time. Without A*'s lines the two counts print
    as dashes. Nothing is written: the records hold everything these numbers need."""
    lines = {}
    for r in read(dataset, "influence.jsonl"):
        key = (r["name"], r["method"], r["rep"])
        if key not in lines or ("error" in lines[key] and "error" not in r):
            lines[key] = r
    stats = lambda v: f"{min(v)} / {np.median(v):g} / {max(v)}"                      # noqa: E731
    for nm in DATASETS[dataset]:
        P, opt = None, optimal_set(dataset, nm)
        for method in ("WI", "MBF"):
            L = [r for (n, m, j), r in sorted(lines.items()) if n == nm and m == method and j <= repeats]
            ok = [r for r in L if "error" not in r]
            if not ok:
                if L:
                    print(f"{nm} {method}: {len(L)} repetitions, all in error", flush=True)
                continue
            if P is None:
                I = load(dataset, nm)
                P, kind = problem(I), "L2-feasible" if I["problem"] == "homography" else "feasible"
            ret = [set(r["returned"]) for r in ok]
            removed = stats([len(opt - S) for S in ret]) if opt is not None else "-"
            left = stats([len(S - opt) for S in ret]) if opt is not None else "-"
            confirmed = sum(P.decide(np.asarray(sorted(S))) == "feasible" for S in ret)
            print(f"{nm} {method}: {len(L)} repetitions ({len(L) - len(ok)} in error), survivors "
                  f"{stats([len(S) for S in ret])}, inliers removed {removed}, outliers left {left}, confirmed "
                  f"{kind} {confirmed} of {len(ok)}, median {np.median([r['seconds'] for r in ok]):.1f} s",
                  flush=True)


def do_wi_mbf(dataset, repeats):
    """WI and MBF-MaxCon, the authors' code: the repetitions 1 to `repeats` of each instance that lack a line not in
    error of either method, so that a run in error runs again, as A*'s do; one MATLAB session per instance under the
    watchdog; then the summary against the optimal set (wi_mbf_summary)."""
    out = RES / dataset / "influence.jsonl"
    done = {(r["name"], r["method"], r["rep"]) for r in common.read(out) if "error" not in r}
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
    wi_mbf_summary(dataset, repeats)


def teaser(dataset):
    """The teaser pairs of a dataset: those where start 1's pruning removed at least TEASER_REMOVED matches and A* on
    its survivors finished within TEASER_SECONDS (a line not in error and not at the limit)."""
    runs = astar.runs(RES / dataset / "astar.jsonl")
    out = []
    for nm, r in first(dataset, "prune.jsonl").items():
        a = runs.get(astar.job(nm, "surv"))
        if (r["n"] - r["survivors"] >= TEASER_REMOVED and a is not None and "error" not in a and not a["hit_limit"]
                and a["time"] <= TEASER_SECONDS):
            out.append(nm)
    return out


def do_astar(dataset, arm):
    """A* on every job of the arm, from start 1: surv on the survivors of each pruning line, k on each trimmed set (one
    group per k), gore on what each GORE line leaves, alone on the full instance. surv runs every job; the other arms
    take a group's jobs by their number of points beyond LO-RANSAC's m0, fewest first, and end it after astar.STOP runs
    in a row at the limit. On a dataset with a fixed subset in instances.ALONE, alone runs on every pair of the subset
    and on no other, with no stop; on any other dataset alone then also runs, with no stop, on its teaser pairs."""
    los = first(dataset, "lo.jsonl")
    insts = {nm: load(dataset, nm) for nm in los}
    if arm == "surv":
        groups = {"surv": [(nm, "surv", r["keep"]) for nm, r in first(dataset, "prune.jsonl").items()]}
    elif arm == "k":
        groups = defaultdict(list)
        for r in read(dataset, "trimming.jsonl"):
            if r["start"] == 1:
                groups[f"k{r['k']}"].append((r["name"], f"k{r['k']}", r["keep"]))
    elif arm == "gore":
        groups = {"gore": [(r["name"], "gore", np.setdiff1d(np.arange(r["n"]), r["removed"]).tolist())
                           for r in read(dataset, "gore.jsonl") if r["removed"]]}
    elif arm == "alone":
        groups = {"alone": [(nm, "alone", list(range(insts[nm]["n"]))) for nm in insts
                            if nm in ALONE.get(dataset, insts)]}
    else:
        raise ValueError(arm)
    beyond = lambda j: len(j[2]) - los[j[0]]["m0"]                                     # noqa: E731
    out = RES / dataset / "astar.jsonl"

    def run(name, jobs, stop):
        group = [dict(name=astar.job(nm, a), file=path(dataset, nm), keep=keep, problem=insts[nm]["problem"],
                      eps=insts[nm]["eps"], init=np.ravel(los[nm]["model"]).tolist()) for nm, a, keep in jobs]
        astar.run_jobs(group, out, RES / dataset / "jobs" / f"{name}.json", f"{name} {dataset}", one_per_session=True,
                       stop=stop)

    for name, jobs in groups.items():
        if arm == "surv" or (arm == "alone" and dataset in ALONE):
            run(name, jobs, None)
        else:
            run(name, sorted(jobs, key=beyond), astar.STOP)
    if arm == "alone" and dataset not in ALONE:
        names = teaser(dataset)
        print(f"A* alone {dataset}: teaser pairs {', '.join(names) or 'none'}", flush=True)
        run("alone_teaser", [(nm, "alone", list(range(insts[nm]["n"]))) for nm in names], None)


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
