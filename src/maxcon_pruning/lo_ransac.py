"""LO-RANSAC, the source of M0, the consensus set that Algorithm 1 starts from. One procedure for every problem.

The loop is RANSAC: draw a minimal sample, fit its model, count the inliers, and stop after TRIALS hypotheses, the
budget GORE's paper gives RANSAC (Chin et al., CVPR 2016). RANSAC's adaptive criterion would stop after a handful of
hypotheses at high inlier ratios, and M0 is the input of the pruning, the trimming and GORE, so a fixed budget keeps
the comparison even and the result reproducible from the seed. Every hypothesis that beats the best raw inlier count
so far is locally optimised, the "inner RANSAC with iteration" of Chum, Matas and Kittler (DAGM 2003, their method 5),
written out by Lebeda, Matas and Chum (BMVC 2012, Algorithms 2 and 3), in the form of runLoRANSAC.m of MBF-MaxCon
(Tennakoon et al., CVPR 2021, MIT licence, see THIRD_PARTY.md):
  1. refit on the inliers of the hypothesis;
  2. REPS times, fit a random half of those inliers, then take ITERS reweighted least-squares steps on the points
     within a threshold that shrinks from K eps to eps, with weights 1 / |residual|;
  3. keep the model with the most inliers among all of these.
REPS, ITERS, K, the half-size inner sample and the 1 / |residual| weights are MBF-MaxCon's. Its stop after 2 s of
wall time, which depends on the machine, is replaced by the fixed budget.

A problem P supplies n, the inlier threshold eps, the minimal sample size p, and three methods, points 0-based:
  minimal(S)         the model of a minimal sample S, or None when the sample is degenerate
  fit(S, w=None)     the model fitted to the points S, weighted by w when given, or None when the fit fails.
                     Regression: the Chebyshev fit unweighted, weighted least squares with w. Homography: the
                     normalised DLT, its rows weighted by w
  residual(model)    the residual of every point under the model, inf where the model does not apply
A point is an inlier of a model at residual <= eps, as in the paper's definition of feasibility.
"""
import numpy as np

TRIALS = 10_000                                   # hypotheses per run
REPS, ITERS, K = 10, 10, 3.0                      # inner samples, reweighted steps per sample, threshold multiplier


def inliers(P, model, th):
    """The points with residual <= th under the model, none for a model of None."""
    if model is None:
        return np.empty(0, np.int64)
    return np.flatnonzero(P.residual(model) <= th)


def local(P, model, I, rng):
    """The local optimisation of a hypothesis with inliers I, steps 1 to 3 of the module docstring. Returns the best
    model found and its inliers."""
    eps = P.eps
    refit = P.fit(I)
    best = (refit, inliers(P, refit, eps)) if refit is not None else (model, I)
    base = best[1]
    if len(base) // 2 < P.p:                                   # too few inliers for inner samples
        return best
    for _ in range(REPS):
        theta = P.fit(rng.choice(base, size=len(base) // 2, replace=False))
        if theta is None:
            continue
        J = inliers(P, theta, eps)
        if len(J) > len(best[1]):
            best = (theta, J)
        for i in range(1, ITERS + 1):
            th = K * eps - i * (K - 1.0) * eps / ITERS          # K eps down to eps
            L = inliers(P, theta, th)
            if len(L) < P.p:
                break
            r = P.residual(theta)[L]
            theta_next = P.fit(L, 1.0 / np.maximum(r, 1e-12 * eps))
            if theta_next is None:
                break
            theta = theta_next
        J = inliers(P, theta, eps)
        if len(J) > len(best[1]):
            best = (theta, J)
    return best


def run(P, rng, trials=TRIALS):
    """LO-RANSAC on the problem P with the generator rng, `trials` hypotheses. Returns the best model and its inliers
    (sorted indices)."""
    n, p, eps = P.n, P.p, P.eps
    raw, best, model_best = 0, np.empty(0, np.int64), None
    for _ in range(trials):
        model = P.minimal(rng.choice(n, size=p, replace=False))
        I = inliers(P, model, eps)
        if len(I) > raw:                                       # a new best hypothesis: optimise it locally
            raw = len(I)
            theta, J = local(P, model, I, rng)
            if len(J) > len(best):
                model_best, best = theta, J
    return model_best, np.sort(best)
