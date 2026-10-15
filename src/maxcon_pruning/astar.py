"""The exact algorithm, A*-NAPA-DIBP of Cai, Chin and Koltun (ICCV 2019), run by MATLAB on job lists this module
writes for matlab/astar_jobs.m (regression; the runner for the vision problems comes with the real data).

A job is one A* run on the rows `keep` of an instance file with the limit CAP. Its name is '<instance>_<arm>', the arm
without underscores: surv (on the survivors), alone (on the full instance), k2, k3, k5 (on the trimmed sets) and gore
(on what GORE leaves). A run at the limit is not finished and gives no consensus size. A* checks its limit only between
nodes, and a single node has run for hours, so every session has a watchdog that kills it after its allowance. A job
alone in a killed session then gets a line at the limit.
"""
import time
from pathlib import Path

from .common import SLACK, append, read, session, write_json

CAP, STOP = 3500.0, 3                              # the limit of a run; STOP runs in a row at the limit end a group


def job(name, arm):
    """The name of the A* job of an instance and an arm: '<instance>_<arm>'."""
    return f"{name}_{arm}"


def at_limit(a):
    """True for an A* line at its limit or in error: a run with no answer."""
    return a is not None and ("error" in a or a["hit_limit"])


def runs(out):
    """The A* lines of a results file by job name (the last line of a name wins)."""
    return {r["name"]: r for r in read(out)}


def stopped(flags, stop=STOP):
    """True once the booleans (runs at the limit, in run order) hold `stop` consecutive Trues."""
    c = 0
    for x in flags:
        c = c + 1 if x else 0
        if c == stop:
            return True
    return False


def write_list(path, jobs, out, warmup):
    """Write the job list astar_jobs.m reads: the results file, the warm-up instance and the jobs (name, keep, file)."""
    write_json(Path(path), dict(out=str(out), warmup=str(warmup), jobs=[
        dict(name=nm, file=str(f), keep=[int(i) for i in keep], cap=CAP) for nm, keep, f in jobs]))


def killed_line(out, name, n, elapsed):
    """The line of a job whose session the watchdog killed: at the limit, no answer."""
    append(out, dict(name=name, n=n, limit=CAP, hit_limit=True, killed=True, time=elapsed, consensus=None, set=None))


def run_jobs(jobs, out, warmup, listfile, label, one_per_session=False):
    """A* on the jobs (name, keep, instance file) of one group, in order, skipping those with a line in out; warmup is
    the instance of the untimed warm-up call. One session runs every pending job; with one_per_session each job runs
    in a session of its own under the watchdog, and the group ends once STOP jobs in a row are at the limit."""
    done = runs(out)
    pending = [j for j in jobs if j[0] not in done]
    if not pending:
        print(f"A* {label}: nothing pending", flush=True)
        return
    print(f"A* {label}: {len(pending)} jobs pending", flush=True)
    if not one_per_session:
        write_list(listfile, pending, out, warmup)
        session(listfile, "astar_jobs", timeout=CAP * (len(pending) + 1) + SLACK)
        return
    flags = []
    for name, keep, f in jobs:
        if stopped(flags):
            print(f"A* {label}: {STOP} runs in a row at the limit, the rest is not run", flush=True)
            return
        if name not in done:
            write_list(listfile, [(name, keep, f)], out, warmup)
            t0 = time.perf_counter()
            if session(listfile, "astar_jobs", timeout=CAP + SLACK) == "killed":
                killed_line(out, name, len(keep), time.perf_counter() - t0)
            done = runs(out)
        flags.append(at_limit(done.get(name)))
