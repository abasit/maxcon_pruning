"""Small helpers for the drivers and astar.py: the results files (JSON Lines), the MATLAB job lists (JSON) and the
MATLAB sessions that run them, and seeds from an instance's name."""
import json
import os
import subprocess
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]         # the project folder, which holds matlab/ and external/
MATLAB = os.environ.get("MATLAB", "matlab")        # the MATLAB executable; set MATLAB to its full path if not on PATH
SLACK = 300.0                                      # a session's allowance beyond the time its jobs may take


def read(path):
    """The records of the JSON Lines file at path, in order; none if the file does not exist."""
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def append(path, rec):
    """Append rec as one line of the JSON Lines file at path, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def seed(nm, k):
    """A seed for np.random.default_rng from an instance's name and a purpose k: [crc32(name), k]. The synthetic
    instances use their own seeds (synthetic/instances.py)."""
    return [zlib.crc32(nm.encode()), k]


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
