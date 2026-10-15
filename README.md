# Degree pruning for maximum consensus

The paper's Algorithm 1 (degree pruning) and its experiments: linear regression under the largest-residual
(Chebyshev) criterion on synthetic data, compared with A*, GORE, a trimming baseline, WI and MBF-MaxCon, and
homography estimation on real image pairs. The script that makes the paper's tables and figures is still to come.

## Layout

```
src/maxcon_pruning/
  prune.py                 Algorithm 1
  lo_ransac.py             LO-RANSAC, which gives the starting consensus set M0
  problems/regression.py   the regression problem: feasibility test, growth, residuals, GORE's rows
  problems/homography.py   the homography problem: the same, for point matches between two images
  trimming.py              the trimming baseline: keep the points within k eps of LO-RANSAC's model
  comparators/gore.py      GORE (Chin et al., CVPR 2016), with Gurobi
  astar.py                 A* (Cai et al., ICCV 2019), run in MATLAB
  common.py                result files, MATLAB sessions, seeds
  synthetic/instances.py   the synthetic instances and their seeds
  synthetic/run.py         the synthetic experiment, one method per command
  real/instances.py        the real datasets and the problem of each instance
  real/run.py              the real-data experiment, one method per command
instances/                 the instance files: the synthetic ones once generated, the real ones in real/<dataset>/
matlab/                    the MATLAB runners for A* and for WI/MBF-MaxCon
external/                  other authors' MATLAB code (see Setup)
```

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This installs the package in editable mode, which it needs: it finds `matlab/` and `external/` from its source folder.

- **Gurobi** (for GORE) comes with `uv sync` as `gurobipy`. Its pip licence is size-limited but enough for the
  synthetic grid and for the real pairs except Boston, whose GORE tests need a full licence (free for academics).
- **MATLAB** (for A*, WI and MBF-MaxCon), tested with R2026a, with the Optimization, Communications and Statistics
  toolboxes. If `matlab` is not on your PATH, point `MATLAB` at it:

  ```bash
  export MATLAB=/Applications/MATLAB_R2026a.app/bin/matlab
  ```

- **Other authors' code** is in `external/`, only the parts the MATLAB runners use: A* (MaxConTreeSearch,
  "free for non-commercial academic use"), MBF-MaxCon (MIT) with SeDuMi 1.3.4 (GPL v2), which A* calls, and WI's
  code, which came from its authors. It is here for our academic use only, not for redistribution;
  `external/README.md` gives each part's source, commit and licence.

## Running the synthetic experiment

```bash
uv run python -m maxcon_pruning.synthetic.run METHOD
```

| method | instances | what it does | writes |
|---|---|---|---|
| `generate` | grid, sweep | the instance files | `instances/*.mat` |
| `prune` | grid, sweep | LO-RANSAC (10,000 hypotheses), then Algorithm 1 | `lo.jsonl`, `prune.jsonl` |
| `trim` | grid, sweep | the trimming at 2, 3 and 5 eps | `trimming.jsonl` |
| `gore` | grid | GORE at its published settings | `gore.jsonl` |
| `wi-mbf` | grid, sweep | WI and MBF-MaxCon, the authors' code | `influence.jsonl` |
| `astar-survivors` | grid, sweep | A* on the survivors of the pruning | `astar.jsonl` |
| `astar-trimmed` | grid | A* on each trimmed set | `astar.jsonl` |
| `astar-gore` | grid | A* on what GORE leaves | `astar.jsonl` |
| `astar-alone` | grid | A* on the full instance | `astar.jsonl` |
| `all` | | all of the above, in this order | |

The grid has n = 200 points, d in {4, 6, 8}, t in {10, 20, ..., 80} outliers and 20 instances per (d, t), 480 in
all. The sweep has d = 8, t = 40 and outlier offsets up to k eps for k in {1.5, 2, 5, 10, 50}, 20 instances per k.
Results go to `results/` (grid) and `results/sweep/`, one JSON line per instance. A method skips the instances that
already have a line, so a stopped method restarts with the same command. Each method needs the ones above it in the
table that it reads from.

Rough times on an Apple M4: `prune` 2 to 2.5 h, `wi-mbf` about 15 min, `gore` probably 5 to 15 h, the A* runs on
survivors, trimmed sets and GORE's output seconds each, and `astar-alone` days (a run stops at 3,500 s, and three runs
in a row at that limit end a cell). Gurobi and MATLAB run on one thread; run one method at a time if the times matter.

## Running the real-data experiment

```bash
uv run python -m maxcon_pruning.real.run METHOD [DATASET] [--repeats N]
```

| method | what it does | writes |
|---|---|---|
| `prune` | N starts (default 20), each LO-RANSAC (10,000 hypotheses) and then Algorithm 1 | `lo.jsonl`, `prune.jsonl` |
| `trim` | the trimming at 2, 3 and 5 eps of every start | `trimming.jsonl` |
| `gore` | GORE at its published settings for real data (10 tests of at most 15 s) | `gore.jsonl` |
| `gore-tests` | GORE given the pruning's own time t as more tests: max(10, t / 15 s) tests of at most 15 s | `gore_tests.jsonl` |
| `gore-cap` | GORE given t as longer tests: 10 tests of at most max(15 s, t / 10) | `gore_cap.jsonl` |
| `wi-mbf` | WI and MBF-MaxCon, the authors' code, N repetitions (default 20) | `influence.jsonl` |
| `astar-survivors` | A* on the survivors of the pruning | `astar.jsonl` |
| `astar-trimmed` | A* on each trimmed set | `astar.jsonl` |
| `astar-gore` | A* on what GORE leaves | `astar.jsonl` |
| `astar-alone` | A* on the full instance | `astar.jsonl` |
| `all` | all of the above, in this order | |

The one dataset so far is `homogr`: five image pairs (city, BruggeTower, Brussels, adam and Boston, 87 to 678 matches)
and the homography problem at eps = 4 pixels. The matches are those Cai, Chin and Koltun ship with MaxConTreeSearch
(ICCV 2019), on pairs of the homogr dataset (Lebeda, Matas and Chum, BMVC 2012). Without a dataset a method runs on
every dataset. Results go to `results/real/<dataset>/`, and as above a stopped method restarts with the same command;
a larger N later only runs the missing starts or repetitions. GORE and A* use start 1. A GORE run with a Gurobi error,
such as a model too large for the pip licence, is printed but not recorded, so it runs again next time. A* starts from
LO-RANSAC's homography and runs each job in a MATLAB session of its own; `astar-alone` takes the pairs by their number
of LO-RANSAC outliers, fewest first, and stops a dataset after three runs in a row at the 3,500 s limit. On a
homography, WI and MBF-MaxCon test feasibility with our L-infinity program, so `wi-mbf` scores their returned sets in
the L2 transfer error: a set's size when our test finds it feasible, else the largest consensus set grown from the
models fitted to it.

Rough times on an Apple M4 for city and adam together: 38 s per start of `prune`, `gore` 96 s, and A* on the
survivors seconds; `gore-tests` and `gore-cap` take about as long as the pruning, or as `gore` when that is longer.

## Pruning your own data

```python
import numpy as np
from maxcon_pruning import lo_ransac, prune
from maxcon_pruning.problems.regression import Problem

P, rng = Problem(A, b, eps), np.random.default_rng(0)        # A: n x d, b: n, eps: the inlier threshold
model, M0 = lo_ransac.run(P, rng)
r = prune.run(P, P.p + 2, M0, model, rng)
r["keep"], r["cert_lower"], r["cert_upper"]                    # the survivors and the bounds on the optimum
```

With probability at least 1 - 1e-6 the survivors contain every maximum consensus set, so an exact method run on them
returns an optimum of the full problem. For homographies, use `Problem(x1, x2, eps)` from
`maxcon_pruning.problems.homography`, with the matches x1 and x2 as 3 x n homogeneous pixel coordinates.

## Third-party code

`lo_ransac.py` follows the local optimisation of MBF-MaxCon's LO-RANSAC; see `THIRD_PARTY.md`.
