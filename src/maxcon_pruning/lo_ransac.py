"""LO-RANSAC, the source of M0, Algorithm 1's starting consensus set.

regression() is our Python port of runLoRANSAC.m from Tennakoon et al.'s MBF-MaxCon (CVPR 2021; MIT licence, see
THIRD_PARTY.md), with the exchange algorithm of myFitTchebycheff.m by Chin, Purkait, Eriksson and Suter (CVPR 2015) as
its Chebyshev fit. It runs for a fixed wall time, so its result depends on the machine, and the drivers store it once
per instance. It differs from the MATLAB in two places: a point is an inlier at residual <= eps (the MATLAB has < eps),
as in the paper's definition of feasibility, and a minimal sample's model is solved exactly, which for d points is the
same model as their Chebyshev fit.
"""
import time

import numpy as np

from .problems.regression import chebyshev, inliers

SECONDS = 2.0                                     # the wall time of the regression LO-RANSAC


def steps(gap, rate):
    """The step lengths gap / rate, with NaN set to 0 and every value <= 0 set to Inf, so that argmin picks the
    smallest positive step."""
    t = gap / rate
    t[np.isnan(t)] = 0
    t[t <= 0] = np.inf
    return t


def exchange(X, Y, xn, itrn):
    """The body of myFitTchebycheff.m on the doubled system [X; -X], from the start xn. Returns the model."""
    n0, n = len(Y), X.shape[1]
    X, Y = np.vstack([X, -X]), np.concatenate([Y, -Y])
    r = X @ xn - Y
    d = r.max()
    M = list(np.flatnonzero(np.abs(r - d) < 1e-9))
    r[M] = d
    while len(M) < n + 1:                        # lower the largest residuals together until n + 1 are equal
        Am = X[M]
        y = Am.T @ -np.linalg.solve(Am @ Am.T, np.ones(len(M)))
        lam = steps(d - r, X @ y + 1)
        j = int(np.argmin(lam))
        xn = xn + lam[j] * y
        M.append(j)
        r = X @ xn - Y
        d = r.max()
        r[M] = d
    if len(M) != n + 1:
        return chebyshev(X[:n0], Y[:n0])[0]
    C = np.linalg.inv(np.hstack([np.ones((n + 1, 1)), X[M]]))
    while (C[0] / np.abs(C[0])).sum() < n and itrn:     # swap out the point with the most negative dual weight
        p1 = int(np.argmin(C[0]))
        y = C[1:, p1] / C[0, p1]
        t = steps(d - r, X @ y + 1)
        j = int(np.argmin(t))
        xn = xn + t[j] * y
        lam = np.concatenate([[1.0], X[j]]) @ C
        Cb = C[:, p1] / lam[p1]
        C = C - np.outer(Cb, lam)
        C[:, p1] = Cb
        M[p1] = j
        r = X @ xn - Y
        d = r.max()
        r[M] = d
        itrn -= 1
    return xn


def cheb_fit(X, Y, rng, itrn=50):
    """The Chebyshev fit of myFitTchebycheff.m: the exchange algorithm from a random start in [0, 1]^d, falling back to
    the linear program when the exchange meets a singular matrix."""
    xn = rng.random(X.shape[1])
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            return exchange(X, Y, xn, itrn)
    except np.linalg.LinAlgError:
        return chebyshev(X, Y)[0]


def local_opt(A, b, eps, I, rng):
    """The local optimisation of runLoRANSAC.m around the inlier set I: a refit, then 10 rounds of a fit to a random
    half of the inliers and 10 IRLS steps with the threshold shrinking from 3 eps to eps. Returns the best model, its
    inliers and the number of fits."""
    theta = cheb_fit(A[I], b[I], rng)
    base = inliers(A, b, theta, eps)
    best, fits = (theta, base), 1
    for _ in range(10):
        if len(base) == 0:
            break
        half = rng.choice(base, size=len(base) // 2, replace=False)
        theta = cheb_fit(A[half], b[half], rng)
        fits += 1
        J = inliers(A, b, theta, eps)
        if len(J) > len(best[1]):
            best = (theta, J)
        for s in range(1, 11):
            L = inliers(A, b, theta, eps * 3 - s * ((3 * eps - eps) / 10))
            fits += 1
            with np.errstate(divide="ignore", invalid="ignore"):
                w = 1.0 / np.abs(A[L] @ theta - b[L])
                try:
                    theta = np.linalg.solve(A[L].T @ (w[:, None] * A[L]), A[L].T @ (w * b[L]))
                except np.linalg.LinAlgError:
                    theta = np.full(A.shape[1], np.nan)
        J = inliers(A, b, theta, eps)
        if len(J) > len(best[1]):
            best = (theta, J)
    return best[0], best[1], fits


def regression(A, b, eps, rng, seconds=SECONDS):
    """LO-RANSAC on a regression instance for the given seconds of wall time: minimal samples of d points, and the
    local optimisation at each new best. Returns the model, its inlier set and the number of trials."""
    n, d = A.shape
    t0 = time.perf_counter()
    trials, best, model, I_lo = 0, 0, None, np.array([], dtype=np.int64)
    while True:
        S = rng.permutation(n)[:d]
        try:
            theta = np.linalg.solve(A[S], b[S])
        except np.linalg.LinAlgError:
            theta = np.full(d, np.nan)
        I = inliers(A, b, theta, eps)
        trials += 1
        if len(I) > best:
            best = len(I)
            th, J, fits = local_opt(A, b, eps, I, rng)
            trials += fits
            if len(J) > len(I_lo):
                model, I_lo = th, J
        if seconds < time.perf_counter() - t0:
            return model, I_lo, trials
