"""The exact algorithm, A*-NAPA-DIBP of Cai, Chin and Koltun (ICCV 2019), run by MATLAB on job lists this module
writes for matlab/astar_jobs.m.

A job is one A* run on the points `keep` of an instance file with the limit CAP, a dict with the keys name, file,
keep, problem ('linear' or 'homography'), eps and, for a homography, init (the initial H, in pixels, row-major).
Its name is '<instance>_<arm>', the arm without underscores: surv (on the survivors), alone (on the full instance),
k2, k3, k5 (on the trimmed sets) and gore (on what GORE leaves). A run at the limit is not finished and gives no
consensus size. A* checks its limit only between nodes, and a single node has run for hours, so every session has a
watchdog that kills it when no job has finished within the limit plus the slack; the job it was running then gets
a line at the limit, and the session starts again on the jobs left. A session that crashes (MATLAB exits with an
error code) is handled the same way, except that its job gets a line in error, so that the job runs again when the
group is next started; two crashes in a row before any new line end the group. A group of jobs ends once STOP of
them in a row are at the limit: astar_jobs.m applies that rule within a session, from the job list's stop and the
lines the jobs already have (skip, at_limit), and run_jobs before starting one.
"""
import time
from collections import Counter
from pathlib import Path

from .common import SLACK, append, read, session, write_json

CAP, STOP = 3500.0, 3                              # the limit of a run; STOP runs in a row at the limit end a group


def job(name, arm):
    """The name of the A* job of an instance and an arm: '<instance>_<arm>'."""
    return f"{name}_{arm}"


def at_limit(a):
    """True for an A* line at its limit (or killed there by the watchdog): a run with no answer. A line in error is
    not at the limit; its job runs again."""
    return a is not None and "error" not in a and bool(a["hit_limit"])


def runs(out):
    """The A* lines of a results file by job name (the last line of a name wins)."""
    return {r["name"]: r for r in read(out)}


def finished(out):
    """The names of the jobs with a line not in error, so that a job whose line is in error runs again."""
    return {nm for nm, r in runs(out).items() if "error" not in r}


def counts(out):
    """The number of A* lines of each job name in a results file."""
    return Counter(r["name"] for r in read(out))


def stopped(flags, stop=STOP):
    """True once the booleans (runs at the limit, in run order) hold `stop` consecutive Trues."""
    c = 0
    for x in flags:
        c = c + 1 if x else 0
        if c == stop:
            return True
    return False


def write_list(path, jobs, out, stop=None):
    """Write the job list astar_jobs.m reads: the results file, the stop (that many jobs in a row at the limit end
    the list; None for no stop) and the jobs, each with the limit CAP. A job with skip true has a line already: it is
    not run, and counts toward the stop by at_limit, whether that line is at the limit."""
    write_json(Path(path), dict(out=str(out), stop=stop, jobs=[
        {**j, "file": str(j["file"]), "keep": [int(i) for i in j["keep"]], "cap": CAP,
         "skip": bool(j.get("skip", False)), "at_limit": bool(j.get("at_limit", False))} for j in jobs]))


def reachable(jobs, stop):
    """True if a job to run (skip false) comes before `stop` jobs in a row at the limit, as astar_jobs.m walks the
    list."""
    c = 0
    for j in jobs:
        if stop and c >= stop:
            return False
        if not j.get("skip"):
            return True
        c = c + 1 if j["at_limit"] else 0
    return False


def killed_line(out, name, n, elapsed):
    """The line of a job whose session the watchdog killed: at the limit, no answer."""
    append(out, dict(name=name, n=n, limit=CAP, hit_limit=True, killed=True, time=elapsed, consensus=None, set=None))


def crash_line(out, name, n, code):
    """The line of a job whose session crashed with MATLAB's exit code: in error, so the job runs again."""
    append(out, dict(name=name, n=n, limit=CAP, crashed=True, error=f"MATLAB exited with code {code}"))


def run_jobs(jobs, out, listfile, label, one_per_session=False, stop=STOP):
    """A* on the jobs of one group, in order, skipping those with a line in out not in error, until `stop` jobs in a
    row are at the limit (never with stop None). One session runs the group, astar_jobs.m applying the stop and the
    watchdog counting from its last line. When the watchdog kills the session or MATLAB crashes, the job it was
    running (the first to run with no new line) gets a killed line or a crash line, and a new session runs the rest,
    a crashed job counting as not at the limit; two crashes in a row with no new line end the group. With
    one_per_session each job runs in a session of its own, the stop is applied here, and kills and crashes are
    handled in the same way."""
    done = finished(out)
    pending = [j for j in jobs if j["name"] not in done]
    if not pending:
        print(f"A* {label}: nothing pending", flush=True)
        return
    print(f"A* {label}: {len(pending)} jobs pending", flush=True)
    if not one_per_session:
        crashed, blank = set(), 0                     # jobs given a crash line here; crashes in a row with no line
        while True:
            before, done, seen = runs(out), finished(out), counts(out)
            listed = [dict(j, skip=True, at_limit=at_limit(before[j["name"]])) if j["name"] in done
                      else dict(j, skip=True, at_limit=False) if j["name"] in crashed else j for j in jobs]
            todo = [j for j in listed if not j.get("skip")]
            if not todo:
                return
            if not reachable(listed, stop):
                print(f"A* {label}: {stop} runs in a row at the limit, the rest is not run", flush=True)
                return
            write_list(listfile, listed, out, stop)
            code = session(listfile, "astar_jobs", timeout=CAP + SLACK, watch=Path(out))
            if code == 0:
                return
            now = counts(out)                         # the job it was running: the first to run with no new line
            cut = next((j for j in todo if now[j["name"]] == seen[j["name"]]), None)
            if cut is None:
                return
            if code == "killed":
                killed_line(out, cut["name"], len(cut["keep"]), CAP + SLACK)
                blank = 0
                continue
            crash_line(out, cut["name"], len(cut["keep"]), code)
            crashed.add(cut["name"])
            blank = 0 if sum(now.values()) > sum(seen.values()) else blank + 1
            print(f"A* {label}: the session crashed on {cut['name']}, which gets a line in error", flush=True)
            if blank == 2:
                print(f"A* {label}: two sessions in a row crashed before writing a line, the rest is not run",
                      flush=True)
                return
    flags, blank = [], 0                              # blank: crashes in a row with no new line
    for j in jobs:
        if stop and stopped(flags, stop):
            print(f"A* {label}: {stop} runs in a row at the limit, the rest is not run", flush=True)
            return
        if j["name"] not in done:
            write_list(listfile, [j], out)
            t0, seen = time.perf_counter(), counts(out)[j["name"]]
            code = session(listfile, "astar_jobs", timeout=CAP + SLACK)
            if code == "killed":
                killed_line(out, j["name"], len(j["keep"]), time.perf_counter() - t0)
            crash = code not in (0, "killed") and counts(out)[j["name"]] == seen
            if crash:
                crash_line(out, j["name"], len(j["keep"]), code)
                print(f"A* {label}: the session crashed on {j['name']}, which gets a line in error", flush=True)
            blank = blank + 1 if crash else 0
            if blank == 2:
                print(f"A* {label}: two sessions in a row crashed before writing a line, the rest is not run",
                      flush=True)
                return
            done = finished(out)
        flags.append(at_limit(runs(out).get(j["name"])))
