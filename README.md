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
  synthetic/two_structure.py  the two-structure cell's instances and their seeds
  synthetic/run.py         the synthetic experiment, one method per command
  synthetic/tables.py      the synthetic tables, from the results
  real/instances.py        the real datasets and the problem of each instance
  real/build.py            the real instance files, from the datasets' own files in data/
  real/fundamental.py      the fundamental-matrix instance files, their ground truth and labels
  real/eth3d_jld2.py       the ETH3D matches, from their Julia files into the numpy files fundamental.py reads
  real/run.py              the real-data experiment, one method per command
instances/                 the instance files: real/<dataset>/, and synthetic/ (grid/, sweep/, two/, scale/), which
                           `generate` writes and the repository does not hold
matlab/                    the MATLAB runners for A* and for WI/MBF-MaxCon
external/                  other authors' MATLAB code (see Setup)
```

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). The locked `gurobipy` 13.0.3 has wheels for Python 3.12 to 3.14
only, so name the version:

```bash
uv sync --python 3.12
```

This installs the package in editable mode, which it needs: it finds `matlab/` and `external/` from its source folder.

- **Gurobi** (for GORE) comes with `uv sync` as `gurobipy`. Its pip licence is size-limited (2,000 constraints) but
  enough for the synthetic grid. A GORE test on a homography has 4n + 3 constraints, so GORE on a real pair of 500
  matches or more, 80 of the 166 pairs `run_real.sh` runs, needs a full licence (free for academics).
- **MATLAB** (for A*, WI and MBF-MaxCon), tested with R2026a, with the Optimization, Communications and Statistics
  toolboxes. A method run by hand calls `matlab` on your PATH, and the two run scripts call
  `/Applications/MATLAB_R2026a.app/bin/matlab`; both use `MATLAB` instead when it is set:

  ```bash
  export MATLAB=/path/to/matlab
  ```

- **Other authors' code** is in `external/`, only the parts the MATLAB runners use: A* (MaxConTreeSearch,
  "free for non-commercial academic use"), MBF-MaxCon (MIT) with SeDuMi 1.3.4 (GPL v2), which A* calls, and WI's
  code, which came from its authors. It is here for our academic use only, not for redistribution;
  `external/README.md` gives each part's source, commit and licence.

## Running the synthetic experiment

The whole experiment, every method below in order, one process at a time with the machine kept awake (by
`caffeinate`, on macOS):

```bash
nohup bash run_synthetic.sh > /dev/null 2>&1 &
```

It logs each method to `results/logs/<method>.log`, and to `results/logs/protocol.log` the commit it runs from
(`git describe`, `-dirty` for uncommitted changes to tracked files; untracked files do not count) and each method's
start and end with its exit code. Started again it
resumes where it stopped. It runs Python's linear algebra on one thread, as MATLAB and Gurobi run. One method at a
time:

```bash
uv run python -m maxcon_pruning.synthetic.run METHOD [--repeats N]
```

| method | instances | what it does | writes |
|---|---|---|---|
| `generate` | grid, sweep, two, scale | the instance files | `instances/synthetic/` |
| `prune` | grid, sweep, two, scale, delta | LO-RANSAC (10,000 hypotheses), then Algorithm 1; the delta cell starts from the grid's LO-RANSAC | `lo.jsonl`, `prune.jsonl` |
| `trim` | grid, sweep, two, scale | the trimming at 2, 3 and 5 eps of LO-RANSAC's model | `trimming.jsonl` |
| `gore` | grid, two | GORE at its published settings | `gore.jsonl` |
| `wi-mbf` | grid, sweep, two, scale | WI and MBF-MaxCon, the authors' code, N repetitions per instance (default 10) | `influence.jsonl` |
| `astar-survivors` | grid, sweep, two, scale, delta | A* on the survivors of the pruning | `astar.jsonl` |
| `astar-trimmed` | grid, sweep, two, scale | A* on each trimmed set | `astar.jsonl` |
| `astar-gore` | grid, two | A* on what GORE leaves, where it removed something | `astar.jsonl` |
| `astar-alone` | grid, sweep, two, scale | A* on the full instance, on the sweep at k = 5, 10 and 50 only | `astar.jsonl` |
| `all` | | all of the above, in this order | |

The grid has n = 200 points, d in {4, 6, 8}, t in {10, 20, ..., 80} outliers and 20 instances per (d, t), 480 in
all. The sweep has d = 8, t = 40 and outlier offsets up to k eps for k in {1.5, 2, 5, 10, 50}, 20 instances per k.
The two-structure cell (`synthetic/two_structure.py`) has 20 instances at d = 4 and n = 120: a main structure and a
second structure of 50 points each, on planes at most 10 eps apart over the data, and 20 gross outliers. The scale
cell has 20 instances at n = 1,000, d = 8, t = 200. The delta cell prunes the grid's 20 instances at d = 8, t = 40 again at
delta = 0.01 and 0.1, with the grid's seeds and LO-RANSAC lines. Results go to `results/` (grid), `results/sweep/`,
`results/two/`, `results/scale/`, `results/delta_0.01/` and `results/delta_0.1/`, one JSON line per instance in each
method's file (per k for the trimming, per method and repetition for WI and MBF-MaxCon, per arm for A*). A method
skips the instances that already have a line, so a stopped method restarts with the same command. Each method needs
the ones above it in the table that it reads from.

A* runs one MATLAB session per cell and arm, and three runs in a row at its 3,500 s limit end it. A job on the same
points as another job of its instance shares that job's line instead of running again. When MATLAB crashes, the job
it was running gets a line in error and a new session runs the rest; two crashes in a row before any line end the
group. A line in error, of A* or of WI and MBF-MaxCon, runs again when the method is next started. GORE ends a cell
after three runs in a row in which every test timed out.

Times on an Apple M4 in the recorded run (started 20 October): `prune` 3.9 h, `gore` 9.8 h, `wi-mbf` about 7 min
per repetition, `astar-survivors` 8.6 h, and `astar-alone` days. Gurobi and MATLAB run on one thread, and Python too
under `run_synthetic.sh`; run one method at a time if the times matter.

The tables come from the results at any point of a run, with a dash for what is not there yet. They read the instance
files too, so `generate` comes first:

```bash
uv run python -m maxcon_pruning.synthetic.tables
```

writes to `results/tables/` a CSV with one row per instance and every number, the grid's tables per cell (the
pruning with A*, the trimming, WI and MBF-MaxCon, GORE) and the sweep's per instance, in LaTeX and in markdown. The
two-structure, scale and delta cells have no tables yet.

## Running the real-data experiment

The recorded run is one command, in three stages: every method on `homogr`, `evd`, `vgg` and `adelaide` with one
start and 10 repetitions of WI and MBF-MaxCon, then the `hpatches` summary, then the pruning's starts 2 to 10 on the
first four, start by start, so that the last stage can be stopped at any time:

```bash
nohup bash run_real.sh > /dev/null 2>&1 &
```

It logs each method to `results/logs/real_<method>.log`, and to `results/logs/real_protocol.log` the commit it runs
from and each method's start and end with its exit code. Like `run_synthetic.sh`, it runs Python's linear algebra on
one thread; a method run by hand uses every core unless you set the thread variables `run_real.sh` sets. One method
by hand:

```bash
uv run python -m maxcon_pruning.real.run METHOD [DATASET] [--repeats N]
```

| method | what it does | writes |
|---|---|---|
| `prune` | N starts (default 20), each LO-RANSAC (10,000 hypotheses) and then Algorithm 1 | `lo.jsonl`, `prune.jsonl` |
| `trim` | the trimming at 2, 3 and 5 eps of every start | `trimming.jsonl` |
| `gore` | GORE at its published settings for real data (10 tests of at most 15 s); a dataset ends after three runs in a row in which every test timed out and nothing was removed | `gore.jsonl` |
| `gore-tests` | GORE given the pruning's own time t as more tests: max(10, t / 15 s) tests of at most 15 s | `gore_tests.jsonl` |
| `gore-cap` | GORE given t as longer tests: 10 tests of at most max(15 s, t / 10) | `gore_cap.jsonl` |
| `wi-mbf` | WI and MBF-MaxCon, the authors' code, N repetitions (default 20) | `influence.jsonl` |
| `astar-survivors` | A* on the survivors of the pruning | `astar.jsonl` |
| `astar-trimmed` | A* on each trimmed set | `astar.jsonl` |
| `astar-gore` | A* on what GORE leaves, where it removed something | `astar.jsonl` |
| `astar-alone` | A* on the full instance | `astar.jsonl` |
| `all` | all of the above, in this order | |

The datasets are image pairs and the homography problem at eps = 4 pixels:

| dataset | pairs | matches | labels |
|---|---|---|---|
| `homogr` | 13 of the 16 of homogr (Lebeda, Matas and Chum, BMVC 2012); not LePoint1-3, where SIFT fails | ours | from the ground truth |
| `homogr_cai` | city, BruggeTower, Brussels, adam and Boston of homogr, not in the recorded run | Cai, Chin and Koltun's (ICCV 2019) | none |
| `vgg` | the rule's 28 of the 8 VGG affine sequences (Mikolajczyk et al., IJCV 2005) | ours | from the ground truth |
| `adelaide` | 8 homography sequences of AdelaideRMF (Wong et al., ICCV 2011), repeated matches dropped | the dataset's | hand labels |
| `evd` | the 15 of EVD (Mishkin et al., CVIU 2015) | the dataset's | from the ground truth |
| `hpatches` | 102, up to two of each of the 59 viewpoint sequences of HPatches (Balntas et al., CVPR 2017) | ours | from the ground truth |

`homogr_cai` has the matches Cai et al. ship with MaxConTreeSearch (`Homo/data/<pair>.mat`, copied as they are, not
built here), so their published A* results apply to it. The
other files are built from the datasets' own files, which are not in the repository; the builder's docstring gives
each source and the folder in `data/` it reads:

```bash
uv sync --extra build
uv run python -m maxcon_pruning.real.build DATASET
```

Our matches are OpenCV SIFT matches with the ratio test at 0.8 and one match per keypoint, labelled correct within
4 pixels of the ground-truth homography. `vgg` matches all 15 pairs of a sequence and keeps those with at least 100
matches and 25 to 85 percent correct, at most four per sequence, the lowest shares first. `hpatches` keeps per
sequence the pair 1-2, when it has at least 100 matches, and the hardest of 1-3 to 1-6 under that rule.

Without a dataset a method runs on every dataset. Results go to `results/real/<dataset>/`, and as above a stopped
method restarts with the same command; a larger N later only runs the missing starts or repetitions. GORE and A* use
start 1. A GORE run with a Gurobi error, such as a model too large for the pip licence, is printed but not recorded,
so it runs again next time. So does a line in error of A*, WI or MBF-MaxCon. A MATLAB crash gives the A* job it was
running such a line, and so does a finished A* run whose model is not finite; a WI and MBF-MaxCon session that
crashes, or that its watchdog stops, writes nothing for the repetitions it had left, which run when the method is next
started. A* starts from LO-RANSAC's homography and runs each job in a MATLAB session of its own.
`astar-survivors` runs on every pair. The other arms take a dataset's jobs (per k for the trimming) by their number of
points beyond LO-RANSAC's set, fewest first, and stop after three runs in a row at the 3,500 s limit; `astar-alone`
then also runs, with no stop, on the teaser pairs, where the pruning removed at least 90 matches and A* on the
survivors took at most 10 s. A match repeated exactly is dropped by the builders (AdelaideRMF has 69). On a
homography, WI and MBF-MaxCon run unmodified with our L-infinity program as their feasibility oracle, so a returned
set need not be L2-feasible; `wi-mbf` reports each returned set against the optimal set A* found: its size, the points
of the optimal set it lacks, the points outside that set it keeps, and whether our check confirms it L2-feasible. In
`run_real.sh` it runs before A*, so the two counts print as dashes there; `wi-mbf` run again after A* runs nothing new
and prints them.

`hpatches` is a summary experiment rather than a table (stage 2 of `run_real.sh`): one start, GORE at its published
setting only, five repetitions of WI and MBF-MaxCon, and A* alone on the fixed subset of 20 pairs in
`instances.ALONE`, every pair of it without the stopping rule (the subset is spread evenly over the pairs' labelled
outlier shares). `all`, or a method without a dataset, would give it the table datasets' settings (the default N,
both GORE variants, A* on GORE's output), though `astar-alone` keeps to the subset.

```bash
uv run python -m maxcon_pruning.real.run prune hpatches --repeats 1
uv run python -m maxcon_pruning.real.run trim hpatches
uv run python -m maxcon_pruning.real.run gore hpatches
uv run python -m maxcon_pruning.real.run wi-mbf hpatches --repeats 5
uv run python -m maxcon_pruning.real.run astar-survivors hpatches
uv run python -m maxcon_pruning.real.run astar-trimmed hpatches
uv run python -m maxcon_pruning.real.run astar-alone hpatches
```

Rough times on an Apple M4 for city and adam together: 38 s per start of `prune`, `gore` 96 s, and A* on the
survivors seconds; `gore-tests` and `gore-cap` take about as long as the pruning, or as `gore` when that is longer.

## Fundamental-matrix instances

`kitti`, `adelaide_f` and `eth3d` are the fundamental matrix in the linearised form of Cai, Chin and Koltun (ICCV 2019),
without the rank-2 constraint. Each image's matches are Hartley-normalised, and a match (x, y) <-> (x', y') gives the
nine columns m = [x'x, x'y, x', y'x, y'y, y', x, y, 1] of its epipolar value under the normalised F, one per entry of
F. One entry, `fixed`, is fixed to 1: the row a is the other eight columns and b = -m_fixed, so theta holds the other
eight entries, and a match is an inlier at |a theta - b| <= eps. In Cai et al.'s form, the one in `instances/real/`,
the fixed entry is F32 and eps is 0.03: a = [x'x, x'y, x', y'x, y'y, y', x, 1] and b = -y, the rows of
MaxConTreeSearch's `genMatrixLinearizeFundamental.m`. It is the regression problem in 8 unknowns (problem `linear`)
whatever the fixed entry: A*, WI and MBF-MaxCon run the authors' linear code on the file's A and b, and LO-RANSAC, the
pruning, the trimming and GORE `problems/regression.py` on the same rows. The pruning starts at level 10 (p + 2), and
A* starts from the file's random x0, not from LO-RANSAC's model.

The fixed entry makes eps a threshold of eps f on the epipolar value of the unit-norm normalised F, with
f = |F_fixed| / ||F|| the share of the fixed entry. Under forward motion with the matches centred on the epipole in x,
the share of F32 is near 0, eps is tight, many inliers of a ground-truth F fall outside eps, and the maximum
consensus of Cai's form can be drawn to models far from the epipolar geometry. `label_check` reports the share in the
ground truth: on kitti it is 0.08 to 0.39 for F32, and on kitti_104_108 and kitti_738_742, at 0.08, only 23 and 32
percent of the labelled inliers are within eps of the pose F.

A builder's gauge sets the fixed entry and eps. `cai` is Cai et al.'s form. `max` fixes on each pair the largest entry
of a reference F made without ground truth (the normalised 8-point of rank 2 over 500,000 hypotheses, scored by the
matches within 1 px of Sampson distance, and refitted on those matches only when the refit holds more of them; the
largest of nine entries has a share of 1/3 or more), and one eps per dataset by a rule fixed in advance: the 0.99
quantile of the residuals under the reference F of the matches within 3 px of it, pooled over the dataset's pairs.
The refit holds more on only 5 of the 24 pairs, so on the others the reference F is the best hypothesis itself. On
`kitti` the gauge fixes F12 on all five pairs at eps 0.0283, and on `adelaide_f` one of F13, F23, F31, F32 and F33 per
scene at eps 0.0338 without the repeated matches (0.0329 with them). The instances in `instances/real/` are in the
gauge `cai` without repeats; `--gauge max` and `--keep-repeats` write to `--out DIR`, as `DIR/<dataset>/`, and
`check --inst DIR` reads them.

| dataset | pairs | matches | labels |
|---|---|---|---|
| `kitti` | the five KITTI odometry pairs of MaxConTreeSearch (Cai, Chin and Koltun, ICCV 2019); 302 to 545 matches | Cai, Chin and Koltun's | the FM-Bench rule under F from the odometry poses, camera P2 |
| `adelaide_f` | the 19 fundamental-matrix scenes of AdelaideRMF (Wong et al., ICCV 2011); 164 to 341 matches | the dataset's | hand labels (0 a gross outlier, k the k-th motion) |
| `eth3d` | 12 pairs of the ETH3D test scenes (Schoeps et al., CVPR 2017), chosen by a rule over the labelled shares; 159 to 637 matches | SuperPoint+LightGlue, shipped by Pritts and Koeser (arXiv 2610.05743) | the FM-Bench rule under F from the shipped poses |

The builder reads `data/kitti/KITTI_<i>_<j>.mat`, MaxConTreeSearch's `Fund/data` at commit `1e198c7`, and reproduces
Cai et al.'s normalised points to rounding; and `data/adelaide_f/<scene>.mat`, RCMSA's `data/AdelaideRMF/F` at commit
`0d5f3e3`, whose matches and labels it keeps, without the exact repeats (the first of each kept with its label), as
`build.py` does for the homography scenes. 17 of the 19 scenes repeat 1 to 21 matches (122 in all), and on such rows
A* (`linearFitV2`) can return a model of NaN and a set our test finds infeasible, often every match: on book (2
repeats) and carchipscube (1) it does so within 3 s, and without the repeats it runs to its limit. `astar_jobs.m`
gives such a run a line in error, and `run.optimal_set` takes no line whose model is not finite. `--keep-repeats`
keeps them. `adelaide_f` has no ground-truth F.
`check` prints each instance's size, labels, normalisation and gauge, and where there is a ground truth how its labels
agree with eps:

```bash
uv run python -m maxcon_pruning.real.fundamental build kitti
uv run python -m maxcon_pruning.real.fundamental build adelaide_f
uv run python -m maxcon_pruning.real.fundamental check kitti
uv run python -m maxcon_pruning.real.fundamental build kitti --gauge max --out DIR
uv run python -m maxcon_pruning.real.fundamental build adelaide_f --keep-repeats --out DIR
uv run python -m maxcon_pruning.real.fundamental check kitti --inst DIR
```

`build kitti` writes the kitti files unlabelled, over the labelled ones (in `instances/real/kitti/`, or in `DIR`),
and `label` below labels them again from the KITTI poses; `check` before that finds them unlabelled.

A ground-truth F comes from the relative pose of the two cameras, and a match is labelled under it by the rule
`LABEL_RULE`. The default is FM-Bench's (Bian et al., BMVC 2019, `ComputeInlierRate.m`): a match is correct when both
one-sided epipolar distances are below 0.003 of its image's diagonal, which is 3.9 px on KITTI, 2.4 px on AdelaideRMF
and 22 px on ETH3D. The rule `sampson`, at most `LABEL_PX` = 1 px of Sampson distance, stays available. A build writes
the width and height of both images (`wh`) into every file, and a file built before that has them NaN when loaded.
`check` prints, for a labelled instance with a ground truth, the
labelled share, the shares within 1, 2 and 4 px of Sampson distance, the near outliers (labelled 0 within 5 px) and the
gross outliers (beyond 20 px).

The kitti pairs are frames of sequence 00 (Cai et al., Sec. 6.2 and Table 1), 0-based, and

```bash
uv run python -m maxcon_pruning.real.fundamental label kitti POSES CALIB [SEQ [CAMERA]] [--inst DIR] [--rule RULE]
```

labels the five pairs from `POSES/<SEQ>.txt` and `CALIB/<SEQ>/calib.txt` (KITTI's `data_odometry_poses.zip` and
`data_odometry_calib.zip`, in `data/kitti_odometry/dataset/poses` and `.../sequences`), sequence `00` and camera `P2`
by default, each file in its own gauge. The paper names no camera and the files' images are in colour; P2 and P3 fit
the matches about equally. Under P2 the FM-Bench rule labels 87 to 93 percent of the matches correct.

`eth3d` reads `data/eth3d_splg/npz/<scene>.npz`, which `eth3d_jld2.py` writes from the Julia files of Zenodo record
23240469. It needs only numpy and h5py, and the project's environment has no h5py:

```bash
python -I src/maxcon_pruning/real/eth3d_jld2.py data/eth3d_splg/datasets/eth3d/derived/splg/csponds data/eth3d_splg/npz
uv run python -m maxcon_pruning.real.fundamental select eth3d
uv run python -m maxcon_pruning.real.fundamental build eth3d
uv run python -m maxcon_pruning.real.fundamental build eth3d --gauge max --out DIR
```

The ground truth is the stored relative pose, with the principal points moved by 0.5 px into the frame of the matches
(`fundamental.py` gives the readings that were tested). The pairs come from the seven test scenes of the authors' split,
with 100 to 1,000 matches. Six are nearest to shares spread evenly over [min, 0.6), and six over [0.6, max], with at
most three per scene, the low shares first. `select` prints the rule's outcome. The test scenes hold only six such pairs
below 0.6, three each in playground and terrace, so the six low pairs are all of them, and the six high pairs come from
relief_2 and terrains. The label rule's 22 px is wide against eps here: under F_gt, 58 to 88 percent of the labelled
inliers are within eps in the gauge `max` (eps 0.0116 by the rule's 3 px window) and 13 to 100 percent in Cai's form, so
a method can drop labelled inliers that F_gt itself does not hold at eps.

The methods and their commands are those above, one dataset at a time (`run_real.sh` runs the homography sets only),
and a method without a dataset runs on `kitti`, `adelaide_f` and `eth3d` too. `adelaide_f` is built without its 122
repeated matches, on which A* returned no answer. On an Apple M4, kitti_104_108 takes 6 s per start of `prune`,
GORE under a second and A* alone 4 s.

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

With probability at least 1 - delta, where `prune.run` takes `delta=1e-6` by default, the survivors contain every
maximum consensus set, so an exact method run on them returns an optimum of the full problem. For homographies, use
`Problem(x1, x2, eps)` from `maxcon_pruning.problems.homography`, with the matches x1 and x2 as 3 x n homogeneous
pixel coordinates.

## Third-party code

`lo_ransac.py` follows the local optimisation of MBF-MaxCon's LO-RANSAC; see `THIRD_PARTY.md`.
