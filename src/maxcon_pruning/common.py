"""Small helpers for the drivers and astar.py: the results files (JSON Lines), the MATLAB job lists (JSON) and the
MATLAB sessions that run them, and the seeds of the random streams."""
import json
import os
import signal
import subprocess
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]         # the project folder, which holds matlab/ and external/
MATLAB = os.environ.get("MATLAB", "matlab")        # the MATLAB executable; set MATLAB to its full path if not on PATH
SLACK = 300.0                                      # a session's allowance beyond the time its jobs may take
POLL = 5.0                                         # seconds between the watchdog's checks
STREAMS = dict(x0=1, lo_ransac=2, prune=3, wi=4, mbf=5)    # appended to a seed; not 0, numpy pads short seeds with 0


def read(path):
    """The records of the JSON Lines file at path, in order; none if the file does not exist."""
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def append(path, rec):
    """Append rec as one line of the JSON Lines file at path, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def seed(name, use, *more):
    """The seed of a real instance's stream for a use: [crc32(name), STREAMS[use], *more]. The name is hashed to a
    32-bit number because it has no numbers of its own; a synthetic instance's seed is the numbers of its name."""
    return [zlib.crc32(name.encode()), STREAMS[use], *more]


def write_json(path, obj):
    """Write obj as the JSON file at path, creating its folder (the job lists MATLAB reads)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def session(jobfile, script, timeout=None, watch=None):
    """One MATLAB session running matlab/<script>.m on a job list, on one thread, in a process group of its own. The
    watchdog kills the group once `timeout` seconds pass without progress: since the start, or, with `watch` (the
    results file the session appends to), since its last new line. Returns MATLAB's exit code, or 'killed'."""
    cmd = [MATLAB, "-singleCompThread", "-batch", f"addpath('{ROOT / 'matlab'}'); {script}('{jobfile}')"]
    proc = subprocess.Popen(cmd, start_new_session=True)

    def size():
        return watch.stat().st_size if watch is not None and watch.exists() else 0

    last, seen = time.monotonic(), size()
    while True:
        try:
            proc.wait(timeout=POLL)                    # returns as soon as MATLAB exits
            break
        except subprocess.TimeoutExpired:
            pass
        if size() != seen:
            last, seen = time.monotonic(), size()
        if timeout is not None and time.monotonic() - last > timeout:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            print(f"watchdog: the session made no progress for {timeout:.0f} s and was killed", flush=True)
            return "killed"
    if proc.returncode:
        print(f"MATLAB exited with code {proc.returncode} ({script} on {jobfile})", flush=True)
    return proc.returncode
