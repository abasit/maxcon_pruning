"""The synthetic regression instances, one .mat file each in instances/synthetic/grid/, sweep/ or scale/.

The generator is the one Cai, Chin and Koltun describe (ICCV 2019, Section 6.1): a random model theta, N points that
fit it exactly, and t of them picked at random as outliers. Inliers get noise uniform in [-EPS, EPS), outliers an
offset uniform in (EPS, out] with a random sign, so an outlier can sit just outside the inlier band. Their paper does
not say how the points are drawn, and we draw them as Tennakoon et al. do (CVPR 2021, genRandomLinearData.m): d - 1
coordinates uniform in [-1, 1] and an intercept column.

Three families, each instance named by its parameters and seeded by the numbers of its name, in order:
  grid   n = N, d in DS, t in TS, RUNS instances per cell, out = OUT. d6_t70_r15 has the seed [6, 70, 15].
  sweep  n = N, d = SWEEP_D, t = SWEEP_T, out = k EPS for k in SWEEP_KS, RUNS instances per k. k enters the seed in
         tenths, so o1.5_d8_t40_r03 has the seed [15, 8, 40, 3], and each k must be a multiple of 0.1.
  scale  n = SCALE_N, d = SCALE_D, t = SCALE_T, out = OUT, RUNS instances. n1000_d8_t200_r03 has the seed
         [1000, 8, 200, 3].
The two-structure cell has a module of its own, two_structure.py.
Every other use of randomness on an instance seeds its own generator with the instance's seed plus STREAMS[use]
(common.py), and the repetition or start from 2 on (stream): A*'s start x0 (make), LO-RANSAC and the pruning
(synthetic/run.py), and WI and MBF-MaxCon, whose MATLAB code gets one 32-bit number derived from it (matlab_seed).

A file holds A, b, epsilon, outlier (the planted outliers, a MATLAB logical), theta (the true model) and x0. The MATLAB
side (A*, WI and MBF-MaxCon) reads the same files and never reads outlier or theta.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from ..common import STREAMS

ROOT = Path(__file__).resolve().parents[3]
INST = ROOT / "instances" / "synthetic"                   # one folder per instance set
N, EPS, OUT = 200, 0.1, 5.0
DS, TS, RUNS = (4, 6, 8), range(10, 81, 10), range(20)
SWEEP_D, SWEEP_T, SWEEP_KS = 8, 40, (1.5, 2.0, 5.0, 10.0, 50.0)
SCALE_N, SCALE_D, SCALE_T = 1000, 8, 200


def name(d, t, run):
    """The name of a grid instance: name(6, 70, 15) = 'd6_t70_r15'."""
    return f"d{d}_t{t:02d}_r{run:02d}"


def sweep_name(k, run):
    """The name of a sweep instance: sweep_name(1.5, 3) = 'o1.5_d8_t40_r03', and k = 2.0 prints as 'o2'."""
    return f"o{k:g}_d{SWEEP_D}_t{SWEEP_T}_r{run:02d}"


def scale_name(run):
    """The name of a scale instance: scale_name(3) = 'n1000_d8_t200_r03'."""
    return f"n{SCALE_N}_d{SCALE_D}_t{SCALE_T}_r{run:02d}"


GRID = [name(d, t, run) for d in DS for t in TS for run in RUNS]          # 480 names, in run order: d, t, run
SWEEP_NAMES = [sweep_name(k, run) for k in SWEEP_KS for run in RUNS]      # 100 names, in run order: k, run
SCALE_NAMES = [scale_name(run) for run in RUNS]                            # 20 names, in run order


def parse(key):
    """The numbers of an instance name, (d, t, run, k) with k None on the grid: 'd8_t50_r03' -> (8, 50, 3, None),
    'o1.5_d8_t40_r03' -> (8, 40, 3, 1.5), 'n1000_d8_t200_r03' -> (8, 200, 3, None). Anything after the third number,
    such as an A* arm ('_surv'), is ignored."""
    parts = key.split("_")
    k = None
    if parts[0].startswith("o"):
        k, parts = float(parts[0][1:]), parts[1:]
    elif parts[0].startswith("n"):
        parts = parts[1:]
    d, t, run = parts[:3]
    return int(d[1:]), int(t[1:]), int(run[1:]), k


def size(key):
    """The number of points of the named instance: SCALE_N for a scale name, which starts with it, else N."""
    return int(key.split("_")[0][1:]) if key.startswith("n") else N


def spec(key):
    """The seed of the named instance and its largest outlier offset out: [d, t, run] and OUT on the grid,
    [10 k, d, t, run] and k EPS on the sweep, [n, d, t, run] and OUT on the scale cell."""
    d, t, run, k = parse(key)
    if key.startswith("n"):
        return [size(key), d, t, run], OUT
    if k is None:
        return [d, t, run], OUT
    return [int(round(10 * k)), d, t, run], k * EPS


def stream(key, use, rep=1):
    """The seed of one use of randomness on the named instance: its seed with STREAMS[use] appended, and from
    repetition (or start) 2 on the repetition too. Since numpy pads seeds with 0, no later repetition can share the
    seed of repetition 1."""
    return spec(key)[0] + [STREAMS[use]] + ([rep] if rep > 1 else [])


def matlab_seed(key, use, rep=1):
    """A seed for MATLAB's rng: one 32-bit number drawn by numpy's SeedSequence from stream(key, use, rep)."""
    return int(np.random.SeedSequence(stream(key, use, rep)).generate_state(1)[0])


def path(key):
    """The file of the named instance, instances/synthetic/<grid, sweep or scale>/<name>.mat."""
    family = "scale" if key.startswith("n") else "grid" if parse(key)[3] is None else "sweep"
    return INST / family / f"{key}.mat"


def load(key):
    """The stored instance as a dict: name, d, t, run, out and seed from the name (parse, spec), and A, b, epsilon,
    outlier (bool), theta and x0 from its file."""
    d, t, run, k = parse(key)
    seed, out = spec(key)
    S = loadmat(path(key))
    return dict(name=key, d=d, t=t, run=run, out=out, seed=seed, A=S["A"].astype(float),
                b=S["b"].ravel().astype(float), epsilon=float(S["epsilon"].item()),
                outlier=S["outlier"].ravel().astype(bool), theta=S["theta"].ravel(), x0=S["x0"].ravel())


def make(key):
    """The named instance generated from its seed, as a dict with the fields of load(). Every draw is made for all n
    points, used or not, in this order, so reordering or skipping a draw changes every instance."""
    d, t, run, k = parse(key)
    seed, out = spec(key)
    n = size(key)
    rng = np.random.default_rng(seed)

    theta = rng.random(d) * 2 - 1
    A = np.column_stack([-1 + 2 * rng.random((n, d - 1)), np.ones(n)])    # coordinates in [-1, 1], a column of ones

    outlier = np.zeros(n, bool)
    outlier[rng.permutation(n)[:t]] = True                         # t outliers picked randomly

    noise = -EPS + 2 * EPS * rng.random(n)                         # inliers: uniform in [-EPS, EPS)
    offset = out - (out - EPS) * rng.random(n)                     # outliers: uniform in (EPS, out] ...
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)                # ... on either side
    noise = np.where(outlier, sign * offset, noise)

    x0 = np.random.default_rng(seed + [STREAMS["x0"]]).random(d)   # A*'s random start, a stream of its own

    return dict(A=A, b=A @ theta + noise, epsilon=EPS, outlier=outlier, theta=theta, x0=x0,
                name=key, d=d, t=t, run=run, out=out, seed=seed)


def generate(names):
    """Write the file of every named instance that has none yet; existing files are left alone."""
    made = 0
    for key in names:
        if not path(key).exists():
            path(key).parent.mkdir(parents=True, exist_ok=True)
            I = make(key)
            savemat(path(key), {f: I[f] for f in ("A", "b", "epsilon", "outlier", "theta", "x0")}, oned_as="column")
            made += 1
    print(f"generate: {made} written, {len(names) - made} already there ({INST})", flush=True)
