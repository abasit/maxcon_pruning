# Degree pruning for maximum consensus

The paper's Algorithm 1 (degree pruning) and its synthetic experiments: linear regression under the largest-residual
(Chebyshev) criterion, compared with A*, GORE, a trimming baseline, WI and MBF-MaxCon. The real-data (homography)
experiments and the script that makes the paper's tables and figures are still to come.

## Layout

```
src/maxcon_pruning/
  prune.py                 Algorithm 1
  lo_ransac.py             LO-RANSAC, which gives the starting consensus set M0
  problems/regression.py   the regression problem: feasibility test, growth, residuals, GORE's rows
  trimming.py              the trimming baseline: keep the points within k eps of LO-RANSAC's model
  comparators/gore.py      GORE (Chin et al., CVPR 2016), with Gurobi
  astar.py                 A* (Cai et al., ICCV 2019), run in MATLAB
  common.py                result files, MATLAB sessions, seeds
  synthetic/instances.py   the synthetic instances and their seeds
  synthetic/run.py         the synthetic experiment, one method per command
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
  synthetic grid.
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
| `prune` | grid, sweep | LO-RANSAC for 2 s, then Algorithm 1 | `lo.jsonl`, `prune.jsonl` |
| `trim` | grid, sweep | the trimming at 2, 3 and 5 eps | `trimming.jsonl` |
| `astar-survivors` | grid, sweep | A* on the survivors of the pruning | `astar.jsonl` |
| `astar-trimmed` | grid | A* on each trimmed set | `astar.jsonl` |
| `gore` | grid | GORE at its published settings | `gore.jsonl` |
| `astar-gore` | grid | A* on what GORE leaves | `astar.jsonl` |
| `wi-mbf` | grid, sweep | WI and MBF-MaxCon, the authors' code | `influence.jsonl` |
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

## Pruning your own data

```python
import numpy as np
from maxcon_pruning import lo_ransac, prune
from maxcon_pruning.problems.regression import Problem

rng = np.random.default_rng(0)
model, M0, _ = lo_ransac.regression(A, b, eps, rng)          # A: n x d, b: n, eps: the inlier threshold
r = prune.run(Problem(A, b, eps), A.shape[1] + 2, M0, model, rng)
r["keep"], r["cert_lower"], r["cert_upper"]                    # the survivors and the bounds on the optimum
```

With probability at least 1 - 1e-6 the survivors contain every maximum consensus set, so an exact method run on them
returns an optimum of the full problem.

## Third-party code

`lo_ransac.py` ports MBF-MaxCon's LO-RANSAC and the Chebyshev fit of Chin et al. (CVPR 2015); see `THIRD_PARTY.md`.
