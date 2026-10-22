"""The fundamental-matrix instances, into instances/real/<dataset>/: the fundamental matrix in the linearised form of
Cai, Chin and Koltun (ICCV 2019) and Tennakoon et al. (CVPR 2021), without the rank-2 constraint, with one entry of the
normalised F fixed to 1.

Each image's matches are Hartley-normalised on their own (hartley: centroid at the origin, mean distance sqrt 2), and
a match (x, y) <-> (x', y') of the two normalised frames gives the nine columns of the epipolar value x2n' Fn x1n,
m = [x'x, x'y, x', y'x, y'y, y', x, y, 1], one per entry of Fn row-major (design). The entry `fixed` (a flat index 0
to 8, row-major) is fixed to 1: a is the other eight columns and b = -m_fixed (linearise), so a theta - b is the
epipolar value of the normalised F with F_fixed = 1 and theta its other eight entries, and a match is an inlier when
|a theta - b| <= eps. fixed = 7, F32, at eps = EPS = 0.03 is Cai et al.'s form, the rows of
genMatrixLinearizeFundamental.m (external/MaxConTreeSearch/solvers), a = [x'x, x'y, x', y'x, y'y, y', x, 1] and
b = -y. This is a linear problem in 8 unknowns (problem 'linear'): A*, WI and MBF-MaxCon run the authors' linear code on
the rows A and b of the file (matlab/astar_jobs.m, influence_jobs.m), and LO-RANSAC, the pruning, the trimming and
GORE the regression problem (problems/regression.py) on the same rows, whatever the fixed entry. A* starts from x0,
uniform in [0, 1)^8 from the seed common.seed(name, 'x0'), as Cai et al.'s demo starts from rand(8, 1). theta_to_F
takes theta to F in pixels, F = T2' Fn T1, and F_to_theta takes F back, Fn = T2^-T F T1^-1 scaled to F_fixed = 1.

The fixed entry sets what eps means for an F. For any multiple Fn of the normalised F, |a theta - b| is
|x2n' Fn x1n| / |Fn_fixed|, so eps bounds the epipolar value of the unit-norm Fn by eps f, with f = |Fn_fixed| / ||Fn||
(Frobenius norm) the share of the fixed entry (share): a small f tightens eps, a model with a larger f counts more
matches at the same epipolar values, and f = 0 leaves F without a theta (F_to_theta raises at f <= SHARE_ZERO). Under a
pure translation, where F is a multiple of [e2]x in pixels with e2 the epipole of image 2, Fn32 is a multiple of e2's
x-coordinate in image 2's normalised frame, its offset in x from the centroid of image 2's matches; forward motion with
that centroid near the epipole gives F32 a share near 0, and the maximum consensus of Cai's form can then be drawn to
models far from the epipolar geometry. The largest entry of a 9-vector is at least 1/3 of its norm, so fixing the
largest entry of a good estimate keeps its share at 1/3 or more.

A builder writes a dataset in one of two gauges (GAUGES), the fixed entry and eps of every instance:
  cai  fixed 7 (F32) and eps EPS = 0.03, Cai et al.'s form
  max  per instance, the largest-magnitude entry of its reference F (reference_F) in the instance's normalised frames
       (largest), and one eps for the dataset by a rule fixed in advance (eps_rule): the EPS_Q = 0.99 quantile of the
       residuals |a theta - b|, theta being the reference F in each instance's gauge, of the matches within
       EPS_PX = 3 px of Sampson distance under it, pooled over the dataset's instances
The reference F uses no ground truth: REF_TRIALS hypotheses, each the normalised 8-point F of 8 matches drawn uniformly
(Hartley, TPAMI 1997: the singular vector of the smallest singular value of their design rows, in the normalised
frames, then the nearest F of rank 2 by SVD), scored by the number of matches within REF_PX = 1 px of Sampson
distance; the best is refitted by the same 8-point on all its matches within REF_PX while that adds matches, at most
REF_REFITS times, so it is the best hypothesis itself where the refit holds no more (on 19 of the 24 instances of
kitti and adelaide_f with every match). 500,000 hypotheses draw an all-inlier sample with probability 0.99 at an
inlier share of 0.24. Its seed is [crc32(name), REF_TAG] (reference_seed), REF_TAG being crc32 of 'reference_F':
common.STREAMS has the uses 1 to 5 and no other, so this stream is none of real.run's. With unique a builder drops a
source's exact repeated matches (equal x1 and x2), the first of each kept with its label (distinct), and its source
says how many; it does so by default, as build.py does for the homography scenes, and --keep-repeats keeps them. The
instances in instances/real/ are in the gauge 'cai' without repeats; a build in the gauge 'max' or with the repeats
writes elsewhere (--out, which it requires).

A ground-truth F comes from the relative pose of the two cameras: X2 = R X1 + t in camera coordinates gives
F = K2^-T [t]x R K1^-1, with x2' F x1 = 0 (F_from_poses; Hartley and Zisserman, Multiple View Geometry, 2004). A match
is labelled by it (label_by) under a rule of LABEL_RULES, LABEL_RULE by default:
  fmbench  the rule of FM-Bench (Bian et al., BMVC 2019, Evaluation/ComputeInlierRate.m): an inlier (label 1) when
           both one-sided epipolar distances under F_gt (line_distances), of x1 to the line F_gt' x2 in image 1 and of
           x2 to the line F_gt x1 in image 2, are below FMB_ALPHA = 0.003 times that image's diagonal, an outlier
           (label 0) otherwise; 3.9 px on KITTI's 1241 x 376, 2.4 px on AdelaideRMF's 640 x 480 and 22 px on ETH3D's
           about 6,200 x 4,140. The default
  sampson  an inlier when its Sampson distance, the first-order geometric distance of the match to F in pixels
           (sampson), is at most LABEL_PX = 1 px
A file holds the width and height of both images (wh), which the rule fmbench needs. label_check measures how such
labels agree with eps on an instance and with the geometry: over the points labelled 1, the share whose algebraic
residual under F_gt in the instance's normalised frame and gauge is within eps, those residuals' quantiles, and
whether the points labelled 1 are feasible as one set; and over all the points, the labelled share, the share the
rule gives now, the shares within SAMPSON_PX = 1, 2 and 4 px of Sampson distance under F_gt, the near outliers
(labelled 0 within NEAR_PX = 5 px) and the gross outliers (beyond GROSS_PX = 20 px).

Datasets:
  kitti       the five pairs of KITTI odometry (Geiger, Lenz and Urtasun, CVPR 2012) that Cai, Chin and Koltun ship
              with MaxConTreeSearch, data/kitti/KITTI_<i>_<j>.mat from github.com/ZhipengCai/MaxConTreeSearch at
              1e198c7 (Fund/data): the frames i and j of sequence 00, its first five crossroads (Cai et al., Sec. 6.2
              and Table 1); their matches data.matches.X1 and X2 in pixels, normalised and linearised here, which
              reproduces their data.x1, data.x2, T1 and T2 to rounding; named kitti_<i>_<j>; wh from their images
              data.matches.im1 and im2 (1241 x 376). Built unlabelled (label all 1, F_gt NaN), and label_kitti
              labels them from the odometry poses (Geiger et al.'s data_odometry_poses.zip and
              data_odometry_calib.zip, data/kitti_odometry/). The paper does not name the camera, and the files'
              images are in colour, so the frames are those of camera 2 or 3 (image_2, image_3); P2 and P3 fit the
              matches about equally, and the frames are 0-based
  adelaide_f  the 19 fundamental-matrix scenes of AdelaideRMF (Wong, Chin, Yu and Suter, ICCV 2011),
              data/adelaide_f/<scene>.mat from RCMSA's data (Pham, Chin, Yu and Suter, CVPR 2012;
              github.com/trungtpham/RCMSA at 0d5f3e3, data/AdelaideRMF/F): the dataset's matches and hand labels (0 a
              gross outlier, k a match of the k-th motion), the exact repeated matches of 17 scenes dropped (122 in
              all), the first of each kept with its label, as build.py does for the homography scenes (on repeated
              rows A*'s linearFitV2 can return a model of NaN and a set the feasibility test finds infeasible, often
              every match); labelled, label > 0 the inliers of some motion; no ground-truth F (F_gt NaN); wh from the
              file's img1 and img2 (640 x 480); named <scene>. book, biscuit, cube and game have one motion
  eth3d       12 pairs of the 13 high-resolution multi-view training scenes of ETH3D (Schoeps et al., CVPR 2017)
              with the SuperPoint+LightGlue matches of Pritts and Koeser (arXiv 2610.05743), eth3d_splg_csponds.tar.gz
              of Zenodo record 23240469, converted by eth3d_jld2.py into
              data/eth3d_splg/npz/<scene>.npz (ETH3D). Per pair the file gives the matches in pixels of the full
              undistorted images, K1 and K2, R_gt and t_gt, the pose of image 2's camera from image 1's
              (X2 = R_gt X1 + t_gt), both images' sizes, LightGlue's confidences and a depth-warp mask. The matches
              are LightGlue's, made at a long side of 1,024 px and scaled by (x + 0.5) s - 0.5, so the centre of the
              top-left pixel is at 0, and K is COLMAP's, where it is at 0.5: F_gt is F_from_poses with each K's
              principal point moved by -PIXEL_OFFSET = -0.5 into the matches' frame (eth3d_F). Of the readings of
              the pose (either direction, R or R', the images in either order, the cameras' own poses composed
              world-to-camera or camera-to-world), this one alone fits: a median Sampson distance of 3.0 px over the
              pairs, against 85 px or more for the others; a common offset of -1 to 1 px moves it by under 0.01 px,
              and the quantisation of the stored x, steps of s at the phase 0.5 - 0.5 / s, gives the 0.5. The mask
              holds 1.4 percent of the matches (the depth-warp test of their consolidation validates few SP+LG
              matches) and is not used as labels: a pair is labelled by LABEL_RULE under F_gt. The scenes of the
              test split, ETH3D_TEST, are those of their configs/splits.toml (VisualGeometryToolkitEval); they hold
              835 pairs, 797 of them with 7 matches or more, the paper's count (2,242 of the 2,246 of the validation
              scenes, likewise). The pairs are chosen by a rule fixed before any pilot ran (eth3d_picks): the
              candidates are the pairs of the test scenes with ETH3D_N = 100 to 1,000 matches after exact repeats are
              removed (there are none), each with its labelled share s; then 12 targets, ETH3D_PICK = 6 evenly spaced
              over [s_min, ETH3D_HIGH) = [s_min, 0.6) without its right end and 6 over [0.6, s_max] with both ends,
              s_min and s_max the smallest and largest share of a candidate, the low targets first, as the low
              shares are the scarce ones and the scene cap would otherwise spend their scenes on high targets, and
              each group upwards; for each target the candidate of its group (s below 0.6, or at 0.6 or more)
              nearest to it that is not chosen yet and whose scene has fewer than ETH3D_PER_SCENE = 3 chosen, or the
              nearest such candidate of either group when its group has none, ties to the smaller n and then to the
              earlier pair (scenes alphabetical, pairs as stored). With fewer than 6 candidates at 0.6 or more, all
              of them are chosen first and the targets below are 12 less their number. 6 of the 489 candidates are
              below 0.6, three each in playground and terrace, and the low targets take all six; the cap then leaves
              the high targets to the other scenes, and they take three pairs each of relief_2 and terrains, none of
              the other group (select prints each target and its pair). Named <scene>_<image 1>_<image 2>, the image
              names without .JPG

A file holds what instances.py reads for a 'linear' instance: name, set, problem ('linear'), model ('fundamental'),
eps, n, x1 and x2 (the matches, 3 x n homogeneous pixels), T1 and T2 (each image's normalisation), A (n x 8) and b
(n), x0, label, labelled and source; F_gt, the ground-truth F in pixels (x2' F_gt x1 = 0), NaN-filled when there is
none; fixed, gauge, and F_ref, the reference F in pixels in the gauge 'max', NaN-filled in the gauge 'cai'; and wh
(2 x 2), the width and height of image 1 and of image 2, a row each. instances.load gives a file without wh a
NaN-filled one.

    python -m maxcon_pruning.real.fundamental build DATASET [--gauge cai|max] [--keep-repeats] [--out DIR]
    python -m maxcon_pruning.real.fundamental check DATASET [--inst DIR]
    python -m maxcon_pruning.real.fundamental label kitti POSES CALIB [SEQ [CAMERA]] [--inst DIR] [--rule RULE]
    python -m maxcon_pruning.real.fundamental select eth3d
"""
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from .. import common
from ..problems import regression
from ..problems.homography import hartley_T
from ..prune import sample
from .instances import DATASETS, ROOT, load, path

EPS, FIXED = 0.03, 7                                    # Cai et al.'s eps on |a theta - b| and fixed entry, F32
GAUGES = ("cai", "max")
LABEL_RULES = ("fmbench", "sampson")
LABEL_RULE = "fmbench"                                  # label_by's rule by default
FMB_ALPHA = 0.003                                       # the rule fmbench: the share of each image's diagonal
LABEL_PX = 1.0                                          # the rule sampson: the threshold in pixels
QUANTILES = (0.5, 0.9, 0.99, 1.0)                       # label_check's quantiles of the residuals
SAMPSON_PX, NEAR_PX, GROSS_PX = (1.0, 2.0, 4.0), 5.0, 20.0                # label_check's Sampson distances
SHARE_ZERO = 1e-12                                      # the share of the fixed entry at or below which F has no theta
REF_PX, REF_TRIALS, REF_BATCH, REF_REFITS = 1.0, 500_000, 5_000, 10    # reference_F
REF_TAG = zlib.crc32(b"reference_F")                    # reference_F's seed, beside crc32(name); no use of STREAMS
EPS_PX, EPS_Q = 3.0, 0.99                               # eps_rule
DATA = ROOT / "data"
KITTI = ["104_108", "198_201", "417_420", "579_582", "738_742"]
ADELAIDE_F = ["biscuit", "biscuitbook", "biscuitbookbox", "boardgame", "book", "breadcartoychips", "breadcube",
              "breadcubechips", "breadtoy", "breadtoycar", "carchipscube", "cube", "cubebreadtoychips", "cubechips",
              "cubetoy", "dinobooks", "game", "gamebiscuit", "toycubecar"]
ETH3D = DATA / "eth3d_splg" / "npz"
ETH3D_TEST = ("office", "pipes", "playground", "relief", "relief_2", "terrace", "terrains")
ETH3D_MIN = 7                                           # their split's pairs: 7 matches or more
ETH3D_N, ETH3D_HIGH, ETH3D_PICK, ETH3D_PER_SCENE = (100, 1000), 0.6, 6, 3                  # eth3d_selection
PIXEL_OFFSET = 0.5                                      # ETH3D: K's frame less the matches' frame, in pixels


def hartley(x):
    """Hartley normalisation of 3 x n homogeneous points with last row 1, centroid at the origin and mean distance
    sqrt 2 (problems/homography.py's hartley_T): T and the normalised points T x."""
    T = hartley_T(x)
    return T, T @ x


def entry(fixed):
    """The name of a flat index into F, row-major: 'F32' for 7."""
    return f"F{fixed // 3 + 1}{fixed % 3 + 1}"


def design(x1n, x2n):
    """The nine columns of x2n' Fn x1n for normalised matches (3 x n, last row 1), one per entry of Fn row-major:
    [x'x, x'y, x', y'x, y'y, y', x, y, 1] (n x 9)."""
    (x, y), (xp, yp) = x1n[:2], x2n[:2]
    return np.column_stack([xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, np.ones_like(x)])


def linearise(x1n, x2n, fixed=FIXED):
    """The rows A (n x 8) and b (n) with F_fixed = 1: the other columns of the design and minus its column fixed; at
    fixed = 7 the rows of genMatrixLinearizeFundamental.m."""
    M = design(x1n, x2n)
    return np.delete(M, fixed, axis=1), -M[:, fixed]


def theta_to_F(theta, T1, T2, fixed=FIXED):
    """F in pixels of theta, the normalised F with F_fixed = 1 without that entry: T2' Fn T1."""
    return T2.T @ np.insert(np.asarray(theta, float).ravel(), fixed, 1.0).reshape(3, 3) @ T1


def normalised(F, T1, T2):
    """F in pixels in the normalised frames of T1 and T2: T2^-T F T1^-1."""
    return np.linalg.solve(T2.T, F) @ np.linalg.inv(T1)


def share(F, T1, T2, fixed=FIXED):
    """|Fn_fixed| / ||Fn|| of F's normalised Fn (Frobenius norm): eps on |a theta - b| is eps times this on the
    epipolar value of the unit-norm Fn."""
    Fn = normalised(F, T1, T2)
    return float(abs(Fn.flat[fixed]) / np.linalg.norm(Fn))


def largest(F, T1, T2):
    """The flat index of the largest-magnitude entry of F's normalised Fn (the first of equals)."""
    return int(np.argmax(np.abs(normalised(F, T1, T2))))


def F_to_theta(F, T1, T2, fixed=FIXED):
    """theta of F in pixels: Fn = T2^-T F T1^-1 scaled to F_fixed = 1, without that entry. A ValueError when the
    share of the fixed entry is at most SHARE_ZERO, as no theta holds such an F."""
    Fn = normalised(F, T1, T2).ravel()
    if not abs(Fn[fixed]) > SHARE_ZERO * np.linalg.norm(Fn):
        raise ValueError(f"{entry(fixed)} of the normalised F is 0 to rounding (share "
                         f"{share(F, T1, T2, fixed):.1e}): no theta holds it")
    return np.delete(Fn / Fn[fixed], fixed)


def skew(v):
    """The matrix [v]x with [v]x w = v x w."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def F_from_poses(K1, K2, R, t):
    """F in pixels for x2' F x1 = 0 of two cameras with the intrinsics K1 and K2, the second at X2 = R X1 + t in camera
    coordinates: K2^-T [t]x R K1^-1."""
    return np.linalg.solve(K2.T, skew(np.ravel(t)) @ R) @ np.linalg.inv(K1)


def kitti_poses(f):
    """The poses of a KITTI odometry sequence (poses/<seq>.txt): one 3 x 4 matrix [R | t] per frame, row-major, from
    camera 0's coordinates at that frame to its coordinates at frame 0, X0 = R X + t."""
    return np.loadtxt(f).reshape(-1, 3, 4)


def kitti_camera(f, cam):
    """The 3 x 4 projection matrix of one camera ('P0' to 'P3') from a KITTI odometry calib.txt."""
    for line in Path(f).read_text().splitlines():
        key, _, vals = line.partition(":")
        if key.strip() == cam:
            return np.array(vals.split(), float).reshape(3, 4)
    raise KeyError(f"{f}: no line {cam}")


def kitti_F(poses, P, i, j):
    """F in pixels from frame i to frame j of the camera with the projection P = K [I | c] of calib.txt, so that
    X = X0 + c in the camera's coordinates (its centre at -c in camera 0's; c = 0 for P0), and camera 0's poses:
    X_j = R X_i + t with R = R_j' R_i and t = R_j' (t_i - t_j) for camera 0, and t + c - R c for the camera of P."""
    K, c = P[:, :3], np.linalg.solve(P[:, :3], P[:, 3])
    (Ri, ti), (Rj, tj) = ((poses[k][:, :3], poses[k][:, 3]) for k in (i, j))
    R, t = Rj.T @ Ri, Rj.T @ (ti - tj)
    return F_from_poses(K, K, R, t + c - R @ c)


def eth3d_F(K1, K2, R, t):
    """F in the pixels of ETH3D's matches from the stored K1, K2 and pose (X2 = R X1 + t): F_from_poses with each
    K's principal point moved by -PIXEL_OFFSET, from COLMAP's frame to LightGlue's."""
    shift = np.array([[1.0, 0, -PIXEL_OFFSET], [0, 1, -PIXEL_OFFSET], [0, 0, 1]])
    return F_from_poses(shift @ K1, shift @ K2, R, t)


def epipolar(F, x1, x2):
    """x2' F x1 of every match (3 x n homogeneous pixels, scaled to last coordinate 1), and the lines F x1 and F' x2."""
    x1, x2 = x1 / x1[2], x2 / x2[2]
    l2, l1 = F @ x1, F.T @ x2
    return np.einsum("in,in->n", x2, l2), l2, l1


def sampson(F, x1, x2):
    """The Sampson distance of every match under F in pixels: |x2' F x1| / |((F x1)_1, (F x1)_2, (F' x2)_1, (F' x2)_2)|,
    the first-order distance of the match (x1, x2) in R^4 to the matches F holds exactly."""
    r, l2, l1 = epipolar(F, x1, x2)
    return np.abs(r) / np.sqrt(l2[0] ** 2 + l2[1] ** 2 + l1[0] ** 2 + l1[1] ** 2)


def sampsons(Fs, x1, x2):
    """The Sampson distances (m x n) of the matches x1 and x2 (3 x n, last row 1) under each of m Fs (m x 3 x 3)."""
    l2, l1 = Fs @ x1, np.swapaxes(Fs, 1, 2) @ x2
    r = np.einsum("in,min->mn", x2, l2)
    return np.abs(r) / np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2 + l1[:, 0] ** 2 + l1[:, 1] ** 2)


def symmetric_epipolar(F, x1, x2):
    """The symmetric epipolar distance of every match under F in pixels: the root mean square of the distances of x2
    to the line F x1 and of x1 to the line F' x2."""
    r, l2, l1 = epipolar(F, x1, x2)
    return np.abs(r) * np.sqrt((1 / (l2[0] ** 2 + l2[1] ** 2) + 1 / (l1[0] ** 2 + l1[1] ** 2)) / 2)


def line_distances(F, x1, x2):
    """The one-sided epipolar distances of every match under F in pixels: of x1 to the line F' x2 in image 1 and of
    x2 to the line F x1 in image 2."""
    r, l2, l1 = epipolar(F, x1, x2)
    return np.abs(r) / np.hypot(l1[0], l1[1]), np.abs(r) / np.hypot(l2[0], l2[1])


def eight_point(M):
    """The 8-point F of rank 2 of each set of design rows (M: m x k x 9, k >= 8, or k x 9 for one set), in their
    frame: the right singular vector of the smallest singular value, then the nearest matrix of rank 2 (Frobenius
    norm), the smallest singular value set to 0."""
    F = np.linalg.svd(M, full_matrices=M.shape[-2] < 9)[2][..., -1, :].reshape(*M.shape[:-2], 3, 3)
    U, s, Vt = np.linalg.svd(F)
    s[..., 2] = 0
    return U @ (s[..., :, None] * Vt)


def reference_seed(name):
    """The seed of reference_F on an instance: [crc32(name), REF_TAG], a stream of none of common.STREAMS' uses."""
    return [zlib.crc32(name.encode()), REF_TAG]


def reference_F(x1, x2, seed):
    """The reference F of the matches x1 and x2 (3 x n homogeneous pixels, last row 1), from no ground truth: in pixels,
    unit Frobenius norm, and its matches within REF_PX of Sampson distance. REF_TRIALS hypotheses in batches of
    REF_BATCH, each the 8-point F of rank 2 of 8 matches drawn uniformly (prune.sample, by the generator of seed) in
    the Hartley frames of all the matches; the one with the most matches within REF_PX wins (the first of equals), and
    is refitted by the 8-point on all of them while that adds matches, at most REF_REFITS times."""
    rng = np.random.default_rng(seed)
    (T1, x1n), (T2, x2n) = hartley(x1), hartley(x2)
    M, n, most, F = design(x1n, x2n), x1.shape[1], -1, None
    for drawn in range(0, REF_TRIALS, REF_BATCH):
        Fs = T2.T @ eight_point(M[sample(rng, min(REF_BATCH, REF_TRIALS - drawn), n, 8)]) @ T1
        count = (sampsons(Fs, x1, x2) <= REF_PX).sum(axis=1)
        j = int(np.argmax(count))
        if count[j] > most:
            most, F = int(count[j]), Fs[j]
    inl = np.flatnonzero(sampson(F, x1, x2) <= REF_PX)
    for _ in range(REF_REFITS):
        G = T2.T @ eight_point(M[inl]) @ T1
        g = np.flatnonzero(sampson(G, x1, x2) <= REF_PX)
        if len(g) <= len(inl):
            break
        F, inl = G, g
    return F / np.linalg.norm(F), inl


def reference_residual(I):
    """The residual |a theta - b| of every point of an instance (built or loaded) under its F_ref, in its gauge."""
    return np.abs(I["A"] @ F_to_theta(I["F_ref"], I["T1"], I["T2"], I["fixed"]) - np.ravel(I["b"]))


def eps_rule(instances, px=EPS_PX, q=EPS_Q):
    """eps of the gauge 'max' for a dataset's instances (each with F_ref and fixed): the q-quantile (numpy's linear
    interpolation) of the residuals |a theta - b| under F_ref in each instance's gauge of its matches within px of
    Sampson distance under F_ref, pooled over the instances."""
    return float(np.quantile(np.concatenate([reference_residual(I)[sampson(I["F_ref"], I["x1"], I["x2"]) <= px]
                                             for I in instances]), q))


def label_by(F_gt, x1, x2, wh, rule=None):
    """The labels of the matches under F_gt by a rule of LABEL_RULES (LABEL_RULE by default), 1 an inlier and 0 an
    outlier: 'fmbench', both one-sided distances below FMB_ALPHA times the diagonal of the image each is in (wh, the
    width and height of image 1 and of image 2, a row each); 'sampson', within LABEL_PX of Sampson distance."""
    rule = rule or LABEL_RULE
    if rule == "fmbench":
        d1, d2 = line_distances(F_gt, x1, x2)
        t1, t2 = FMB_ALPHA * np.hypot(*np.asarray(wh, float).T)
        return ((d1 < t1) & (d2 < t2)).astype(np.uint8)
    if rule == "sampson":
        return (sampson(F_gt, x1, x2) <= LABEL_PX).astype(np.uint8)
    raise ValueError(f"no label rule {rule!r}: {', '.join(LABEL_RULES)}")


def rule_text(rule=None):
    """A label rule in words, for a source."""
    rule = rule or LABEL_RULE
    return (f"the FM-Bench rule (Bian et al., BMVC 2019): both one-sided epipolar distances below {FMB_ALPHA:g} of the "
            f"image's diagonal" if rule == "fmbench" else f"{LABEL_PX:g} px of Sampson distance")


def quantiles(v):
    """The QUANTILES of v by name (q0.5 and so on), None for no values."""
    return dict(zip([f"q{x:g}" for x in QUANTILES], np.quantile(v, QUANTILES).tolist())) if len(v) else None


def label_check(I):
    """How the labels of a loaded instance agree with its eps and its geometry under F_gt: the share of its fixed
    entry in F_gt (share); over the points labelled 1 (inliers), the share whose residual |a theta - b| is within eps
    (within), theta being F_gt in the instance's normalised frame and gauge (F_to_theta), the QUANTILES of those
    residuals and of their Sampson distances; the number of points of the whole instance within eps of theta
    (gt_consensus, a consensus set, so a lower bound on the optimum); and decide on the inliers as one set (feasible,
    'feasible' for none). Over all the points, for a labelled instance: labelled_share, the share labelled 1; rule
    and rule_share, LABEL_RULE and the share it labels 1 now (None without wh); within_px, the shares within each of
    SAMPSON_PX of Sampson distance under F_gt; near_outliers, the points labelled 0 within NEAR_PX; and
    gross_outliers, the points beyond GROSS_PX. within and the quantiles are None without inliers, and within, the
    residuals and gt_consensus None when the share is at most SHARE_ZERO, as no theta holds F_gt then. An instance
    without F_gt gives its name, n and has_gt False."""
    if not np.isfinite(I["F_gt"]).all():
        return dict(name=I["name"], n=I["n"], has_gt=False)
    inl = np.flatnonzero(I["label"] == 1)
    P = regression.Problem(I["A"], I["b"], I["eps"])
    d = sampson(I["F_gt"], I["x1"], I["x2"])
    c = dict(name=I["name"], n=I["n"], has_gt=True, fixed=I["fixed"], share=share(I["F_gt"], I["T1"], I["T2"],
             I["fixed"]), inliers=len(inl), within=None, residual=None, sampson=quantiles(d[inl]), gt_consensus=None,
             feasible=P.decide(inl) if len(inl) else "feasible")
    if I["labelled"]:
        wh = I.get("wh", np.full((2, 2), np.nan))
        out = I["label"] == 0
        c.update(labelled_share=float(np.mean(I["label"] > 0)), rule=LABEL_RULE,
                 rule_share=float(label_by(I["F_gt"], I["x1"], I["x2"], wh).mean()) if np.isfinite(wh).all()
                 else None, within_px={f"{t:g}": float(np.mean(d <= t)) for t in SAMPSON_PX},
                 near_outliers=int((out & (d <= NEAR_PX)).sum()), gross_outliers=int((d > GROSS_PX).sum()))
    if c["share"] > SHARE_ZERO:
        r = P.residual(F_to_theta(I["F_gt"], I["T1"], I["T2"], I["fixed"]))
        c.update(within=float(np.mean(r[inl] <= I["eps"])) if len(inl) else None, residual=quantiles(r[inl]),
                 gt_consensus=int((r <= I["eps"]).sum()))
    return c


def instance(dataset, name, x1, x2, source, label=None, F_gt=None, eps=EPS, fixed=FIXED, F_ref=None, gauge="cai",
             wh=None):
    """The fields of one file from the matches x1 and x2 (3 x n homogeneous pixels, last row 1): the normalisations,
    the rows A and b with F_fixed = 1, A*'s start x0, and the labels when given (else label all 1 and labelled 0);
    F_gt and F_ref NaN-filled when not given, and wh (the width and height of each image, a row each) too."""
    (T1, x1n), (T2, x2n) = hartley(x1), hartley(x2)
    A, b = linearise(x1n, x2n, fixed)
    x0 = np.random.default_rng(common.seed(name, "x0")).random(A.shape[1])
    labelled = label is not None
    label = np.asarray(label, np.uint8) if labelled else np.ones(x1.shape[1], np.uint8)
    nan = np.full((3, 3), np.nan)
    return dict(name=name, set=dataset, problem="linear", model="fundamental", eps=eps, n=x1.shape[1], x1=x1, x2=x2,
                T1=T1, T2=T2, A=A, b=b[:, None], x0=x0[:, None], label=label[:, None], labelled=int(labelled),
                F_gt=nan if F_gt is None else F_gt, source=source, fixed=fixed, gauge=gauge,
                F_ref=nan if F_ref is None else F_ref,
                wh=np.full((2, 2), np.nan) if wh is None else np.asarray(wh, float).reshape(2, 2))


def kitti_source(pair):
    return (f"Cai, Chin and Koltun (ICCV 2019), github.com/ZhipengCai/MaxConTreeSearch at 1e198c7, "
            f"Fund/data/KITTI_{pair}.mat: data.matches.X1 and X2, their matches on the KITTI odometry pair {pair} "
            f"(Geiger, Lenz and Urtasun, CVPR 2012)")


def kitti():
    """The kitti pairs from Cai et al.'s files, unlabelled, checked against their normalisations: the arguments of
    instance per pair."""
    for pair in KITTI:
        d = loadmat(DATA / "kitti" / f"KITTI_{pair}.mat", squeeze_me=True, struct_as_record=False)["data"]
        x1, x2 = d.matches.X1.astype(float), d.matches.X2.astype(float)
        (T1, x1n), (T2, x2n) = hartley(x1), hartley(x2)
        gap = max(np.abs(T1 - d.T1).max(), np.abs(T2 - d.T2).max(), np.abs(x1n - d.x1).max(),
                  np.abs(x2n - d.x2).max())
        if gap > 1e-9:
            raise ValueError(f"kitti_{pair}: the normalisation differs from Cai et al.'s by {gap:.1e}")
        yield dict(name=f"kitti_{pair}", x1=x1, x2=x2, source=f"{kitti_source(pair)}; unlabelled",
                   wh=[d.matches.im1.shape[1::-1], d.matches.im2.shape[1::-1]])


def adelaide_f():
    """The AdelaideRMF scenes, every match and label as the dataset gives them: the arguments of instance per
    scene."""
    for scene in ADELAIDE_F:
        S = loadmat(DATA / "adelaide_f" / f"{scene}.mat")
        yield dict(name=scene, x1=S["data"][:3].astype(float), x2=S["data"][3:6].astype(float),
                   label=S["label"].ravel().astype(np.uint8),
                   source=f"AdelaideRMF (Wong, Chin, Yu and Suter, ICCV 2011), F/{scene}.mat from RCMSA's data "
                          f"(github.com/trungtpham/RCMSA at 0d5f3e3, data/AdelaideRMF/F): the dataset's matches and "
                          f"hand labels (0 a gross outlier, k the k-th motion); no ground-truth F",
                   wh=[S["img1"].shape[1::-1], S["img2"].shape[1::-1]])


def eth3d_pairs(scenes=ETH3D_TEST):
    """Every pair of the scenes' .npz files (ETH3D), scenes alphabetical and pairs as stored, each with F_gt
    (eth3d_F), its labels under LABEL_RULE, its matches without exact repeats (n, the first of each kept), their
    labelled share, gross outliers (beyond GROSS_PX of Sampson distance under F_gt) and depth-warp inliers, and its
    arguments of instance with every match (args)."""
    for scene in sorted(scenes):
        z = np.load(ETH3D / f"{scene}.npz", allow_pickle=False)
        o = z["offsets"]
        for k in range(len(o) - 1):
            s = slice(o[k], o[k + 1])
            x1 = np.vstack([z["x1"][s].T.astype(float), np.ones(o[k + 1] - o[k])])
            x2 = np.vstack([z["x2"][s].T.astype(float), np.ones(o[k + 1] - o[k])])
            wh = np.vstack([z["size1"][k], z["size2"][k]])
            F = eth3d_F(z["K1"][k], z["K2"][k], z["R_gt"][k], z["t_gt"][k])
            label = label_by(F, x1, x2, wh)
            names = [Path(str(z[f][k])).stem for f in ("source_name", "target_name")]
            name = f"{scene}_{names[0]}_{names[1]}"
            p = distinct(dict(name=name, x1=x1, x2=x2, label=label, source=""))
            mask = z["inlier_mask"][s]
            yield dict(name=name, scene=scene, n=p["x1"].shape[1],
                       share=float(p["label"].mean()) if len(p["label"]) else float("nan"),
                       gross=int((sampson(F, p["x1"], p["x2"]) > GROSS_PX).sum()), depth=int(mask.sum()),
                       args=dict(name=name, x1=x1, x2=x2, label=label, F_gt=F, wh=wh,
                                 source=f"ETH3D (Schoeps et al., CVPR 2017) {scene} {names[0]} and {names[1]}, the "
                                        f"SuperPoint+LightGlue matches of Pritts and Koeser (arXiv 2610.05743, Zenodo "
                                        f"23240469, eth3d_splg_csponds.tar.gz, {scene}.jld2; "
                                        f"data/eth3d_splg/npz/{scene}.npz); F_gt from their K and pose, the principal "
                                        f"points moved by -{PIXEL_OFFSET:g} px; labelled by {rule_text()} under "
                                        f"F_gt; {int(mask.sum())} of the matches in their depth-warp mask"))


def eth3d_picks(pairs):
    """The rule of the module docstring on eth3d_pairs (or any records with name, scene, n and share, in the files'
    order): per pick, in their order, its target (None for a pair taken whole), the group it is drawn for ('low',
    share below ETH3D_HIGH, or 'high') and the pair."""
    cand = [(j, p) for j, p in enumerate(pairs) if ETH3D_N[0] <= p["n"] <= ETH3D_N[1]]
    if not cand:
        return []
    groups = dict(low=[c for c in cand if c[1]["share"] < ETH3D_HIGH],
                  high=[c for c in cand if c[1]["share"] >= ETH3D_HIGH])
    shares = [p["share"] for _, p in cand]
    picks, per = [], {}

    def take(t, group, p):
        picks.append((t, group, p))
        per[p["scene"]] = per.get(p["scene"], 0) + 1

    if len(groups["high"]) < ETH3D_PICK:
        for _, p in groups["high"]:
            take(None, "high", p)
        high = []
    else:
        high = [(t, "high") for t in np.linspace(ETH3D_HIGH, max(shares), ETH3D_PICK)]
    low = [(t, "low") for t in np.linspace(min(shares), ETH3D_HIGH, 2 * ETH3D_PICK - len(picks) - len(high),
                                           endpoint=False)]
    for t, group in low + high:
        chosen = {p["name"] for _, _, p in picks}
        free = [[(j, p) for j, p in g if p["name"] not in chosen and per.get(p["scene"], 0) < ETH3D_PER_SCENE]
                for g in (groups[group], cand)]
        free = free[0] or free[1]
        if free:
            take(float(t), group, min(free, key=lambda c: (abs(c[1]["share"] - t), c[1]["n"], c[0]))[1])
    return picks


def eth3d_selection(pairs):
    """The names of the pairs the rule of the module docstring keeps, in the order of their picks (eth3d_picks)."""
    return [p["name"] for _, _, p in eth3d_picks(pairs)]


def eth3d():
    """The eth3d pairs of the selection rule (eth3d_selection), labelled under F_gt: the arguments of instance per
    pair, in the files' order."""
    pairs = list(eth3d_pairs())
    keep = set(eth3d_selection(pairs))
    for p in pairs:
        if p["name"] in keep:
            yield p["args"]


SOURCES = dict(kitti=kitti, adelaide_f=adelaide_f, eth3d=eth3d)


def distinct(p):
    """A source's arguments of instance without its exact repeated matches (equal x1 and x2), the first of each kept,
    in their order, with its label; the source says how many were dropped."""
    keep = np.sort(np.unique(np.vstack([p["x1"], p["x2"]]).T, axis=0, return_index=True)[1])
    q = dict(p, x1=p["x1"][:, keep], x2=p["x2"][:, keep],
             source=f"{p['source']}; {p['x1'].shape[1] - len(keep)} exact repeated matches dropped, the first of each "
                    f"kept")
    if p.get("label") is not None:
        q["label"] = p["label"][keep]
    return q


def build(dataset, gauge="cai", unique=True):
    """The instances of a dataset in a gauge (GAUGES), in the order of its source, without its exact repeated matches
    unless unique is False (distinct): 'cai', fixed 7 and eps EPS; 'max', per instance the largest entry of its
    reference F in its normalised frames, and eps by eps_rule over them all."""
    sources = [distinct(p) if unique else p for p in SOURCES[dataset]()]
    if gauge == "cai":
        return [instance(dataset, **p) for p in sources]
    out = []
    for p in sources:
        F = reference_F(p["x1"], p["x2"], reference_seed(p["name"]))[0]
        out.append(instance(dataset, **p, fixed=largest(F, hartley_T(p["x1"]), hartley_T(p["x2"])), F_ref=F,
                            gauge="max"))
    eps = eps_rule(out)
    for I in out:
        I["eps"] = eps
        I["source"] += gauge_note(I, len(out))
    return out


def off(T, x):
    """How far the normalised points T x are from centroid 0 and mean distance sqrt 2, the larger of the two."""
    y = (T @ x)[:2]
    return max(np.abs(y.mean(axis=1)).max(), abs(np.linalg.norm(y, axis=0).mean() - np.sqrt(2)))


def inliers_text(I):
    """The labelled inliers (points labelled above 0) and their share, or 'unlabelled'."""
    inl = np.ravel(I["label"]) > 0
    return f"labelled inliers {int(inl.sum()):5d} ({inl.mean():.0%})" if I["labelled"] else "unlabelled"


def gauge_text(I):
    return f"gauge {I['gauge']}, {entry(I['fixed'])} = 1, eps {I['eps']:.4g}"


def gauge_note(I, count):
    """What a source adds for an instance in the gauge 'max' of a dataset of count instances."""
    return (f"; the gauge 'max': {entry(I['fixed'])} = 1, the largest entry of the reference F (its share "
            f"{share(I['F_ref'], I['T1'], I['T2'], I['fixed']):.3f}; "
            f"{int((sampson(I['F_ref'], I['x1'], I['x2']) <= REF_PX).sum())} matches within {REF_PX:g} px), eps "
            f"{I['eps']:.6g} by the rule: the {EPS_Q:g} quantile of the residuals under the reference F of the matches "
            f"within {EPS_PX:g} px of it, over the {count} instances of {I['set']}")


def label_kitti(poses_dir, calib_dir, seq="00", cam="P2", inst=None, rule=None):
    """Label the kitti instances by F from the odometry poses under a label rule (label_by, LABEL_RULE by default)
    and rewrite their files (in inst, by default instances/real/), each in its own gauge and with the image sizes
    of Cai et al.'s files: camera 0's poses from poses_dir/<seq>.txt (kitti_poses; seq 00, Cai et al.'s, by
    default), the camera's projection from calib_dir/<seq>/calib.txt (cam: P2, the left colour camera, by default),
    and the frames i and j, 0-based, from the name kitti_<i>_<j>. A file whose matches are not those of Cai et al.'s
    file is refused."""
    poses = kitti_poses(Path(poses_dir) / f"{seq}.txt")
    P = kitti_camera(Path(calib_dir) / seq / "calib.txt", cam)
    src = {p["name"]: p for p in kitti()}
    for nm in DATASETS["kitti"]:
        I = load("kitti", nm, inst)
        if not (np.array_equal(I["x1"], src[nm]["x1"]) and np.array_equal(I["x2"], src[nm]["x2"])):
            raise ValueError(f"{path('kitti', nm, inst)}: its matches are not those of Cai et al.'s file")
        i, j = map(int, nm.split("_")[1:])
        F = kitti_F(poses, P, i, j)
        J = instance("kitti", nm, I["x1"], I["x2"],
                     f"{kitti_source(nm[len('kitti_'):])}; labelled by {rule_text(rule)} under F from the odometry "
                     f"poses of sequence {seq}, frames {i} and {j}, camera {cam}",
                     label=label_by(F, I["x1"], I["x2"], src[nm]["wh"], rule), F_gt=F, eps=I["eps"],
                     fixed=I["fixed"], F_ref=I["F_ref"], gauge=I["gauge"], wh=src[nm]["wh"])
        if J["gauge"] == "max":
            J["source"] += gauge_note(J, len(DATASETS["kitti"]))
        savemat(path("kitti", nm, inst), J)
        print(f"{nm:18s} n {J['n']:5d}, {inliers_text(load('kitti', nm, inst))}", flush=True)


def check(dataset, inst=None):
    """Per instance, without fitting anything: n, the labelled inliers, how far each image's normalised points are
    from centroid 0 and mean distance sqrt 2, the largest |b| and row norm of A, the gauge, and in the gauge 'max'
    the share of the fixed entry in F_ref and the matches within REF_PX and EPS_PX of it, those within eps; and
    label_check where there is a ground truth."""
    for nm in DATASETS[dataset]:
        I = load(dataset, nm, inst)
        o = max(off(I["T1"], I["x1"]), off(I["T2"], I["x2"]))
        print(f"{nm:18s} n {I['n']:5d}, {inliers_text(I)}, normalisation off by {o:.1e}, |b| <= "
              f"{np.abs(I['b']).max():.2f}, |a| <= {np.linalg.norm(I['A'], axis=1).max():.2f}, {gauge_text(I)}",
              flush=True)
        if np.isfinite(I["F_ref"]).all():
            d, r = sampson(I["F_ref"], I["x1"], I["x2"]), reference_residual(I) <= I["eps"]
            print(f"{'':18s} reference F: share {share(I['F_ref'], I['T1'], I['T2'], I['fixed']):.3f}, within "
                  f"{REF_PX:g} px {int((d <= REF_PX).sum())}, within {EPS_PX:g} px {int((d <= EPS_PX).sum())} "
                  f"({int(((d <= EPS_PX) & r).sum())} of them within eps), within eps {int(r.sum())}", flush=True)
        c = label_check(I)
        if c["has_gt"]:
            qs = {k: "-" if c[k] is None else ", ".join(f"{q} {v:.3g}" for q, v in c[k].items())
                  for k in ("residual", "sampson")}
            within = "-" if c["within"] is None else f"{c['within']:.1%}"
            gt = "-" if c["gt_consensus"] is None else c["gt_consensus"]
            print(f"{'':18s} share of {entry(c['fixed'])} {c['share']:.3f}, within eps {within} of {c['inliers']}, "
                  f"residual quantiles {qs['residual']}, Sampson quantiles {qs['sampson']}, gt consensus {gt}, "
                  f"inliers {c['feasible']}", flush=True)
            if "labelled_share" in c:
                now = "-" if c["rule_share"] is None else f"{c['rule_share']:.1%}"
                print(f"{'':18s} labelled {c['labelled_share']:.1%}, by {c['rule']} now {now}, within Sampson "
                      + ", ".join(f"{t} px {v:.1%}" for t, v in c["within_px"].items())
                      + f", near outliers {c['near_outliers']}, gross outliers {c['gross_outliers']}", flush=True)


def select():
    """The selection of eth3d (eth3d_picks) and the distribution of n, the labelled share and the gross outliers
    over the pairs of the test scenes, those of the split (ETH3D_MIN matches or more), the candidates and the pairs
    kept, then each pick with its target and group (and whether its pair is of the other group), and the pairs kept
    by decreasing share."""
    pairs = list(eth3d_pairs())
    picks = eth3d_picks(pairs)
    keep = [p["name"] for _, _, p in picks]
    cand = [p for p in pairs if ETH3D_N[0] <= p["n"] <= ETH3D_N[1]]
    groups = [(f"pairs of the test scenes ({', '.join(ETH3D_TEST)})", pairs),
              (f"test pairs, the {ETH3D_MIN} matches or more of the split", [p for p in pairs if p["n"] >= ETH3D_MIN]),
              (f"candidates, {ETH3D_N[0]} <= n <= {ETH3D_N[1]}", cand),
              ("kept", [p for p in pairs if p["name"] in keep])]
    for what, ps in groups:
        print(f"{what}: {len(ps)} pairs, {sum(p['share'] >= ETH3D_HIGH for p in ps)} with share >= {ETH3D_HIGH:g}")
        for k in ("n", "share", "gross"):
            v = np.array([p[k] for p in ps], float)
            v = v[np.isfinite(v)]
            q = np.quantile(v, (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)) if len(v) else []
            print(f"  {k:6s} min, q0.1, q0.25, median, q0.75, q0.9, max: " + ", ".join(f"{x:.3g}" for x in q)
                  + (f" (over the {len(v)} pairs with a match)" if len(v) < len(ps) else ""))
    scenes = sorted({p["scene"] for p in cand})
    print("candidates per scene, share >= / below " + f"{ETH3D_HIGH:g}: " + ", ".join(
        f"{s} {sum(p['scene'] == s and p['share'] >= ETH3D_HIGH for p in cand)} / "
        f"{sum(p['scene'] == s and p['share'] < ETH3D_HIGH for p in cand)}" for s in scenes))
    print(f"candidates below {ETH3D_HIGH:g}: " + ", ".join(f"{p['name']} {p['share']:.3f}" for p in cand
                                                         if p["share"] < ETH3D_HIGH))
    print("\npicks, in their order:")
    for t, group, p in picks:
        other = (p["share"] >= ETH3D_HIGH) != (group == "high")
        print(f"  {group:4s} target {'whole' if t is None else format(t, '.3f'):5s} {p['name']:34s} share "
              f"{p['share']:.3f}{', of the other group' if other else ''}")
    by = {p["name"]: p for p in pairs}
    print(f"\nkept, in the order of their picks: {keep}\nby decreasing share:")
    for nm in sorted(keep, key=lambda m: (-by[m]["share"], by[m]["n"])):
        p = by[nm]
        print(f"  {nm:34s} {p['scene']:10s} n {p['n']:5d}, share {p['share']:.3f}, gross {p['gross']:4d}, "
              f"depth-warp {p['depth']}")


def main(args):
    ap = argparse.ArgumentParser(prog="python -m maxcon_pruning.real.fundamental", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("dataset", metavar="DATASET", choices=list(SOURCES))
    b.add_argument("--gauge", choices=GAUGES, default="cai")
    b.add_argument("--keep-repeats", action="store_true", help="keep exact repeated matches (dropped by default)")
    b.add_argument("--out", metavar="DIR", default=None, help="the folder of the dataset folders (instances/real/)")
    c = sub.add_parser("check")
    c.add_argument("dataset", metavar="DATASET", choices=list(SOURCES))
    c.add_argument("--inst", metavar="DIR", default=None)
    lab = sub.add_parser("label")
    lab.add_argument("dataset", metavar="DATASET", choices=["kitti"])
    lab.add_argument("poses", metavar="POSES")
    lab.add_argument("calib", metavar="CALIB")
    lab.add_argument("seq", metavar="SEQ", nargs="?", default="00")
    lab.add_argument("cam", metavar="CAMERA", nargs="?", default="P2")
    lab.add_argument("--inst", metavar="DIR", default=None)
    lab.add_argument("--rule", choices=LABEL_RULES, default=LABEL_RULE)
    sel = sub.add_parser("select")
    sel.add_argument("dataset", metavar="DATASET", choices=["eth3d"])
    a = ap.parse_args(args)
    if a.command == "build":
        if a.out is None and (a.gauge != "cai" or a.keep_repeats):
            ap.error("instances/real/ holds Cai's form without repeats: give the gauge 'max' or --keep-repeats an "
                     "--out")
        for I in build(a.dataset, a.gauge, not a.keep_repeats):
            f = path(a.dataset, I["name"], a.out)
            f.parent.mkdir(parents=True, exist_ok=True)
            savemat(f, I)
            J = load(a.dataset, I["name"], a.out)
            print(f"{J['name']:18s} n {J['n']:5d}, {inliers_text(J)}, {gauge_text(J)}", flush=True)
    elif a.command == "check":
        check(a.dataset, a.inst)
    elif a.command == "select":
        select()
    else:
        label_kitti(a.poses, a.calib, a.seq, a.cam, a.inst, a.rule)


if __name__ == "__main__":
    main(sys.argv[1:])
