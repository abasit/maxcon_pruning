# External code

The parts of other authors' code that `matlab/astar_jobs.m` and `matlab/influence_jobs.m` put on MATLAB's path, copied
here for now from the sources below. Before release they are replaced by clone instructions in the README, and
WI-MaxCon by whatever its authors allow.

| folder | from | used by | licence |
|---|---|---|---|
| `MaxConTreeSearch/solvers` | https://github.com/ZhipengCai/MaxConTreeSearch, commit `1e198c7` | A*-NAPA-DIBP (`linearFitV2` on a linear problem, `genMatrixHomography` and `pseudoConvFit` on a homography) | "free for non-commercial academic use" (its README) |
| `MBF-MaxCon/linearASTAR/sedumi-master` | https://github.com/RuwanT/MBF-MaxCon, commit `1661bf8` | SeDuMi 1.3.4, which A* calls (MaxConTreeSearch's own SeDuMi 1.21 is not built and is taken off the path) | GNU GPL version 2 (its `COPYING`) |
| `MBF-MaxCon/maxcon_BMF`, `MBF-MaxCon/utils` | the same | MBF-MaxCon, and the Chebyshev fit `myFitTchebycheff.m` (Chin et al., CVPR 2015) | MIT |
| `WI-MaxCon/functions` | sent by the authors of WI (Zhang et al., TPAMI 2022) | WI | not stated; ask before release |

MATLAB toolboxes the code calls: Optimization (`linprog`, `optimoptions`, `fmincon`), Communications (`randsrc`, in
WI and MBF-MaxCon) and Statistics (`randsample`).
