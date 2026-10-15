"""Small helpers for the drivers and astar.py: the results files (JSON Lines), the MATLAB job lists (JSON) and the
MATLAB sessions that run them, and the seeds of the random streams."""
import json
import os
import subprocess
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]         # the project folder, which holds matlab/ and external/
MATLAB = os.environ.get("MATLAB", "matlab")        # the MATLAB executable; set MATLAB to its full path if not on PATH
SLACK = 300.0                                      # a session's allowance beyond the time its jobs may take
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


def session(jobfile, script, timeout=None):
    """One MATLAB session running matlab/<script>.m on a job list, on one thread, killed after `timeout` seconds (the
    watchdog). Returns MATLAB's exit code, or 'killed'."""
    cmd = [MATLAB, "-singleCompThread", "-batch", f"addpath('{ROOT / 'matlab'}'); {script}('{jobfile}')"]
    try:
        return subprocess.run(cmd, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"watchdog: the session outlived its allowance of {timeout:.0f} s and was killed", flush=True)
        return "killed"
