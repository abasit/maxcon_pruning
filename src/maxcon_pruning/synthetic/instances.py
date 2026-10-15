"""The synthetic regression instances, one .mat file each in instances/.

The generator is the one Cai, Chin and Koltun describe (ICCV 2019, Section 6.1): a random model theta, N points that
fit it exactly, and t of them picked at random as outliers. Inliers get noise uniform in [-EPS, EPS), outliers an
offset uniform in (EPS, out] with a random sign, so an outlier can sit just outside the inlier band. Their paper does
not say how the points are drawn, and we draw them as Tennakoon et al. do (CVPR 2021, genRandomLinearData.m): d - 1
coordinates uniform in [-1, 1] and an intercept column.

Two families, each instance named by its parameters and seeded by the numbers of its name, in order:
  grid   d in DS, t in TS, RUNS instances per cell, out = OUT. d6_t70_r15 has the seed [6, 70, 15].
  sweep  d = SWEEP_D, t = SWEEP_T, out = k EPS for k in SWEEP_KS, RUNS instances per k. k enters the seed in tenths,
         so o1.5_d8_t40_r03 has the seed [15, 8, 40, 3], and each k must be a multiple of 0.1.
Every other use of randomness on an instance seeds its own generator with the instance's seed plus STREAMS[use]: A*'s
start x0 (make), LO-RANSAC and the pruning (synthetic/run.py), and WI and MBF-MaxCon, whose MATLAB code gets one
32-bit number derived from that seed (matlab_seed).

A file holds A, b, epsilon, outlier (the planted outliers, a MATLAB logical), theta (the true model) and x0. The MATLAB
side (A*, WI and MBF-MaxCon) reads the same files and never reads outlier or theta.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[3]
INST = ROOT / "instances"
N, EPS, OUT = 200, 0.1, 5.0
DS, TS, RUNS = (4, 6, 8), range(10, 81, 10), range(20)
SWEEP_D, SWEEP_T, SWEEP_KS = 8, 40, (1.5, 2.0, 5.0, 10.0, 50.0)
STREAMS = dict(x0=1, lo_ransac=2, prune=3, wi=4, mbf=5)    # appended to a seed; not 0, numpy pads short seeds with 0


def name(d, t, run):
    """The name of a grid instance: name(6, 70, 15) = 'd6_t70_r15'."""
    return f"d{d}_t{t:02d}_r{run:02d}"


def sweep_name(k, run):
    """The name of a sweep instance: sweep_name(1.5, 3) = 'o1.5_d8_t40_r03', and k = 2.0 prints as 'o2'."""
    return f"o{k:g}_d{SWEEP_D}_t{SWEEP_T}_r{run:02d}"


GRID = [name(d, t, run) for d in DS for t in TS for run in RUNS]          # 480 names, in run order: d, t, run
SWEEP_NAMES = [sweep_name(k, run) for k in SWEEP_KS for run in RUNS]      # 100 names, in run order: k, run


def parse(key):
    """The numbers of an instance name, (d, t, run, k) with k None on the grid: 'd8_t50_r03' -> (8, 50, 3, None),
    'o1.5_d8_t40_r03' -> (8, 40, 3, 1.5). Anything after the third number, such as an A* arm ('_surv'), is ignored."""
    parts = key.split("_")
    k = None
    if parts[0].startswith("o"):
        k, parts = float(parts[0][1:]), parts[1:]
    d, t, run = parts[:3]
    return int(d[1:]), int(t[1:]), int(run[1:]), k


def spec(key):
    """The seed of the named instance and its largest outlier offset out: [d, t, run] and OUT on the grid,
    [10 k, d, t, run] and k EPS on the sweep."""
    d, t, run, k = parse(key)
    if k is None:
        return [d, t, run], OUT
    return [int(round(10 * k)), d, t, run], k * EPS


def matlab_seed(key, use):
    """A seed for MATLAB's rng from the named instance's seed and the stream of a use: one 32-bit number, drawn by
    numpy's SeedSequence from the seed with STREAMS[use] appended."""
    return int(np.random.SeedSequence(spec(key)[0] + [STREAMS[use]]).generate_state(1)[0])


def path(key):
    """The file of the named instance, instances/<name>.mat."""
    return INST / f"{key}.mat"


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
    """The named instance generated from its seed, as a dict with the fields of load(). Every draw is made for all N
    points, used or not, in this order, so reordering or skipping a draw changes every instance."""
    d, t, run, k = parse(key)
    seed, out = spec(key)
    rng = np.random.default_rng(seed)

    theta = rng.random(d) * 2 - 1
    A = np.column_stack([-1 + 2 * rng.random((N, d - 1)), np.ones(N)])    # coordinates in [-1, 1], a column of ones

    outlier = np.zeros(N, bool)
    outlier[rng.permutation(N)[:t]] = True                         # t outliers picked randomly

    noise = -EPS + 2 * EPS * rng.random(N)                         # inliers: uniform in [-EPS, EPS)
    size = out - (out - EPS) * rng.random(N)                       # outliers: uniform in (EPS, out] ...
    sign = np.where(rng.random(N) < 0.5, -1.0, 1.0)                # ... on either side
    noise = np.where(outlier, sign * size, noise)

    x0 = np.random.default_rng(seed + [STREAMS["x0"]]).random(d)   # A*'s random start, a stream of its own

    return dict(A=A, b=A @ theta + noise, epsilon=EPS, outlier=outlier, theta=theta, x0=x0,
                name=key, d=d, t=t, run=run, out=out, seed=seed)


def generate(names):
    """Write the file of every named instance that has none yet; existing files are left alone."""
    INST.mkdir(exist_ok=True)
    made = 0
    for key in names:
        if not path(key).exists():
            I = make(key)
            savemat(path(key), {f: I[f] for f in ("A", "b", "epsilon", "outlier", "theta", "x0")}, oned_as="column")
            made += 1
    print(f"generate: {made} written, {len(names) - made} already there ({INST})", flush=True)
