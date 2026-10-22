"""The two-structure instances, at d = 4: a main structure of m points within EPS of a model theta1, a second structure
of m points within EPS of a second model theta2, and g gross outliers, n = 2m + g points in all.

The draws are the grid's (instances.py) wherever the grid has them: theta1 uniform in [-1, 1]^d, d - 1 coordinates
uniform in [-1, 1] and an intercept column, noise uniform in [-EPS, EPS) for both structures, and the gross outliers
offset from theta1 by an amount uniform in (EPS, OUT] with a random sign. theta2 is one of two kinds:
  a = 0   drawn independently, uniform in [-1, 1]^d like theta1
  a > 0   theta1 + Delta, Delta = a EPS u / |u|_1 for u uniform in [-1, 1]^d. Over the data (x in [-1, 1]^(d-1) and
          the intercept) |a_i . Delta| reaches |Delta|_1, so the two planes are at most a EPS apart
Nothing keeps a point of one structure from falling within EPS of the other model, or a gross outlier within EPS of
theta2; those are the chance points that decide which structure is the larger.

The structures are labelled, not the optimum: the planted main structure need not be a maximum consensus set, so the
files hold `structure` (1 main, 2 second, 0 gross) and no `outlier`, and nothing here defines inliers removed.

The cell is CELL, RUNS instances in instances/synthetic/two/. Each instance is named by its parameters and seeded by
TAG and the numbers of its name: two_n120_m50_g20_a10_r03 has the seed [2, 120, 50, 20, 10, 3]. A*'s start x0
(uniform in [0, 1)^d) has a stream of its own, as on the grid.
"""
import re

import numpy as np
from scipy.io import loadmat, savemat

from ..common import STREAMS
from .instances import EPS, INST, OUT, RUNS

D = 4
CELL = (120, 50, 20, 10)                          # (n, m, g, a) of the cell
TAG = 2                                           # the first number of every seed
NAME = re.compile(r"two_n(\d+)_m(\d+)_g(\d+)(?:_a(\d+))?_r(\d+)")


def name(n, m, g, a, run):
    """The name of an instance: name(120, 50, 20, 10, 3) = 'two_n120_m50_g20_a10_r03'; a = 0 leaves out the a part."""
    return f"two_n{n}_m{m}_g{g}{f'_a{a}' if a else ''}_r{run:02d}"


CELL_NAMES = [name(*CELL, run) for run in RUNS]


def parse(key):
    """The parameters of an instance name as a dict: n, m, g, a and run."""
    x = NAME.fullmatch(key)
    if x is None:
        raise ValueError(f"not a two-structure name: {key}")
    n, m, g, a, run = (int(v or 0) for v in x.groups())
    if n != 2 * m + g:
        raise ValueError(f"{key}: n must be 2m + g")
    return dict(n=n, m=m, g=g, a=a, run=run)


def spec(key):
    """The seed of the named instance: TAG, then n, m, g, a and run."""
    s = parse(key)
    return [TAG, s["n"], s["m"], s["g"], s["a"], s["run"]]


def stream(key, use, rep=1):
    """The seed of one use of randomness on the named instance: its seed with STREAMS[use] appended, and from
    repetition 2 on the repetition too, as instances.stream does."""
    return spec(key) + [STREAMS[use]] + ([rep] if rep > 1 else [])


def matlab_seed(key, use, rep=1):
    """A seed for MATLAB's rng (WI, MBF-MaxCon): one 32-bit number drawn by numpy's SeedSequence from
    stream(key, use, rep)."""
    return int(np.random.SeedSequence(stream(key, use, rep)).generate_state(1)[0])


def path(key):
    """The file of the named instance, instances/synthetic/two/<name>.mat."""
    return INST / "two" / f"{key}.mat"


def make(key):
    """The named instance generated from its seed, as a dict with the fields of load(). Every draw is made for all n
    points, used or not, in this order, so reordering or skipping a draw changes every instance."""
    s = parse(key)
    n, m, a = s["n"], s["m"], s["a"]
    seed = spec(key)
    rng = np.random.default_rng(seed)

    theta = rng.random(D) * 2 - 1                                          # theta1, the main structure
    A = np.column_stack([-1 + 2 * rng.random((n, D - 1)), np.ones(n)])    # coordinates in [-1, 1], a column of ones

    perm = rng.permutation(n)                                              # m main, m second, the rest gross
    structure = np.zeros(n, int)
    structure[perm[:m]], structure[perm[m:2 * m]] = 1, 2

    noise = -EPS + 2 * EPS * rng.random(n)                                 # both structures: uniform in [-EPS, EPS)
    size = OUT - (OUT - EPS) * rng.random(n)                               # gross: uniform in (EPS, OUT] ...
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)                        # ... on either side of theta1
    u = rng.random(D) * 2 - 1                                              # theta2, or the direction of Delta
    theta2 = u if a == 0 else theta + a * EPS * u / np.abs(u).sum()

    b = A @ theta + np.where(structure == 0, sign * size, noise)
    second = structure == 2
    b[second] = A[second] @ theta2 + noise[second]

    x0 = np.random.default_rng(seed + [STREAMS["x0"]]).random(D)          # A*'s random start, a stream of its own

    return dict(A=A, b=b, epsilon=EPS, structure=structure, theta=theta, theta2=theta2, x0=x0,
                name=key, d=D, seed=seed, **s)


def generate(names):
    """Write the file of every named instance that has none yet; existing files are left alone."""
    made = 0
    for key in names:
        if not path(key).exists():
            path(key).parent.mkdir(parents=True, exist_ok=True)
            I = make(key)
            savemat(path(key), {f: I[f] for f in ("A", "b", "epsilon", "structure", "theta", "theta2", "x0")},
                    oned_as="column")
            made += 1
    print(f"generate: {made} written, {len(names) - made} already there", flush=True)


def load(key):
    """The stored instance as a dict: name, d, seed and the parameters from the name (parse, spec), and A, b, epsilon,
    structure, theta, theta2 and x0 from its file."""
    S = loadmat(path(key))
    return dict(name=key, d=D, seed=spec(key), **parse(key), A=S["A"].astype(float), b=S["b"].ravel().astype(float),
                epsilon=float(S["epsilon"].item()), structure=S["structure"].ravel().astype(int),
                theta=S["theta"].ravel(), theta2=S["theta2"].ravel(), x0=S["x0"].ravel())
