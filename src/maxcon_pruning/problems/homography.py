"""Homography estimation from point matches, one point per match. A set of matches is feasible when some homography
with positive depths maps every first point to within eps pixels of its second point in the L2 transfer error.

The feasibility test of a sampled set is one L-infinity linear program (oracle_lp): in the Hartley-normalised frames of
its matches, on the depth-weighted linearised constraints with the depths summing to the set size, it minimises the
largest constraint value t <= 0. The L-infinity box holds the L2 disc, so every set feasible in L2 passes. Only a
program proven infeasible makes a set infeasible; a program that fails counts as feasible and gives no model.

Problem holds an instance and the methods the other modules call: test, cover, grow and decide for the pruning
(prune.py), minimal, fit and residual for LO-RANSAC (lo_ransac.py), residual for the trimming, and gore for GORE.
"""
import math

import numpy as np
from scipy.optimize import linprog

GROW_MAX, BLOCK = 10, 256                         # refits per growth at most; models per block in cover


def dlt(x1, x2, sw=None):
    """Homography from at least four matches (3 x k homogeneous), h33 free, each match's two rows scaled by sw when
    given; returns H scaled to h33 = 1."""
    x, y = x1[0], x1[1]
    xp, yp = x2[0], x2[1]
    o, z = np.ones_like(x), np.zeros_like(x)
    A = np.vstack([np.column_stack([x, y, o, z, z, z, -xp * x, -xp * y, -xp]),
                   np.column_stack([z, z, z, x, y, o, -yp * x, -yp * y, -yp])])
    if sw is not None:
        A = A * np.r_[sw, sw][:, None]
    H = np.linalg.svd(A)[2][-1].reshape(3, 3)
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def hartley_T(x):
    """Hartley normalisation of 3 x k homogeneous points (last row 1): centroid 0, mean distance sqrt 2."""
    c = x[:2].mean(axis=1, keepdims=True)
    d = np.linalg.norm(x[:2] - c, axis=0).mean()
    s = math.sqrt(2) / d if d > 0 else 1.0
    return np.array([[s, 0, -s * c[0, 0]], [0, s, -s * c[1, 0]], [0, 0, 1.0]])


def positive_depths(H, x1):
    """H rescaled so that the depths h3 . x of the points x1 are positive (sign by the median depth): h33 = 1 when
    that keeps them positive, else mean |depth| = 1."""
    w = (H @ x1)[2]
    sg = np.sign(np.median(w)) or 1.0
    H, w = H * sg, w * sg
    if H[2, 2] > 1e-12 and np.all(w / H[2, 2] > 0):
        return H / H[2, 2]
    return H / max(np.abs(w).mean(), 1e-300)


def dlt_normalised(x1, x2, sw=None):
    """Hartley-normalised DLT: the fit in the normalised frames of the given matches (rows scaled by sw when given),
    mapped back to pixels, with positive depths."""
    x1, x2 = x1 / x1[2], x2 / x2[2]
    T1, T2 = hartley_T(x1), hartley_T(x2)
    H = np.linalg.inv(T2) @ dlt(T1 @ x1, T2 @ x2, sw) @ T1
    return positive_depths(H, x1)


def transfer(H, x1, x2):
    """L2 transfer error of every match in pixels; inf for a non-positive depth."""
    p = H @ x1
    w = p[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        e = p[:2] / w - x2[:2]
        e = np.sqrt((e ** 2).sum(axis=0))
    e[~np.isfinite(e)] = np.inf
    e[w <= 0] = np.inf
    return e


def oracle_lp(x1, x2, S, eps):
    """The feasibility test's linear program on the matches S: (status, t*, H in pixels or None). Infeasible iff
    status 2."""
    S = np.asarray(S, int)
    k = len(S)
    X1, X2 = x1[:, S] / x1[2, S], x2[:, S] / x2[2, S]
    T1, T2 = hartley_T(X1), hartley_T(X2)
    X1, X2, e = T1 @ X1, T2 @ X2, eps * T2[0, 0]
    x, y, xp, yp, o, z = X1[0], X1[1], X2[0], X2[1], np.ones(k), np.zeros(k)
    rows = []
    for sgn in (1.0, -1.0):
        c = sgn * xp + e                                   # sgn (h1 . x) - c (h3 . x) - t <= 0
        rows.append(np.column_stack([sgn * x, sgn * y, sgn * o, z, z, z, -c * x, -c * y, -c, -o]))
        c = sgn * yp + e
        rows.append(np.column_stack([z, z, z, sgn * x, sgn * y, sgn * o, -c * x, -c * y, -c, -o]))
    A_ub = np.vstack(rows)
    A_eq = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x.sum(), y.sum(), float(k), 0.0]])   # sum of the depths = |S|
    r = linprog(np.r_[np.zeros(9), 1.0], A_ub=A_ub, b_ub=np.zeros(len(A_ub)), A_eq=A_eq, b_eq=[float(k)],
                bounds=[(None, None)] * 9 + [(None, 0.0)], method="highs")
    if r.status != 0 or r.x is None:
        return r.status, None, None
    s2 = T2[0, 0]
    T2inv = np.array([[1.0 / s2, 0.0, -T2[0, 2] / s2], [0.0, 1.0 / s2, -T2[1, 2] / s2], [0.0, 0.0, 1.0]])
    return 0, float(r.x[9]), T2inv @ r.x[:9].reshape(3, 3) @ T1


def grow_dlt(x1, x2, eps, S, H):
    """From the set S of the model H: the normalised DLT on the set and all its L2 inliers, while the set grows (at
    most GROW_MAX refits). Returns the set (its matches within eps under the final H), that H and the refits."""
    S, it = np.asarray(S, int), 0
    while it < GROW_MAX and len(S) >= 4:
        Hn = dlt_normalised(x1[:, S], x2[:, S])
        Sn = np.flatnonzero(transfer(Hn, x1, x2) <= eps)
        if len(Sn) <= len(S):
            break
        S, H, it = Sn, Hn, it + 1
    return S[transfer(H, x1[:, S], x2[:, S]) <= eps], H, it


class Problem:
    """A homography instance: x1 and x2 (3 x n, pixels, last row 1) and the inlier threshold eps in pixels."""

    def __init__(self, x1, x2, eps):
        self.x1, self.x2 = x1 / x1[2], x2 / x2[2]          # last row 1, as the transfer error and GORE's rows need
        self.eps, self.n = eps, x1.shape[1]
        self.p = 4                               # the minimal sample size; the pruning starts at level p + 2

    def test(self, S):
        """The sets S (m x s indices): (feasible mask, models of the sets whose program was solved)."""
        feasible, models = np.ones(len(S), bool), []
        for i, Si in enumerate(S):
            st, t, H = oracle_lp(self.x1, self.x2, Si, self.eps)
            feasible[i] = st != 2
            if st == 0:
                models.append(H)
        return feasible, models

    def minimal(self, S):
        """The DLT homography of the four matches S, with positive depths on them. The DLT's own scaling, h33 = 1,
        gives every match a negative depth, and so no inliers, when pixel (0, 0) of the first image is beyond the
        plane's horizon."""
        return positive_depths(dlt(self.x1[:, S], self.x2[:, S]), self.x1[:, S])

    def fit(self, S, w=None):
        """The normalised DLT of the matches S, each match's rows scaled by sqrt(w) when weights are given, with
        positive depths."""
        return dlt_normalised(self.x1[:, S], self.x2[:, S], None if w is None else np.sqrt(w))

    def inliers(self, H):
        """The L2 inlier set of one model."""
        return np.flatnonzero(transfer(H, self.x1, self.x2) <= self.eps)

    def residual(self, H):
        """The L2 transfer error of every match under one model, in pixels (inf for a non-positive depth)."""
        return transfer(np.asarray(H, float).reshape(3, 3), self.x1, self.x2)

    def cover(self, Hs):
        """The L2 inlier matrix (n x models) of several models, BLOCK models at a time."""
        Hs, out = np.stack(Hs), np.empty((self.n, len(Hs)), bool)
        for i in range(0, len(Hs), BLOCK):
            P = np.matmul(Hs[i:i + BLOCK], self.x1)
            w = P[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                e = np.sqrt((P[:, 0] / w - self.x2[0]) ** 2 + (P[:, 1] / w - self.x2[1]) ** 2)
            out[:, i:i + BLOCK] = ((w > 0) & np.isfinite(e) & (e <= self.eps)).T
        return out

    def grow(self, cover, H, alive):
        """Two grown candidates, each restricted to the survivors: the inliers of the normalised-DLT refit of the set,
        and the set itself under its model; the larger wins, a tie goes to the refit. Returns the set, its model, the
        refits taken and the grown points outside `alive`."""
        cands = []
        if len(cover) >= 4:
            Hr = dlt_normalised(self.x1[:, cover], self.x2[:, cover])
            cands.append(grow_dlt(self.x1, self.x2, self.eps, self.inliers(Hr), Hr))
        cands.append(grow_dlt(self.x1, self.x2, self.eps, cover, H))
        G, gm, it = max(cands, key=lambda c: len(np.intersect1d(c[0], alive)))
        Ga = np.intersect1d(G, alive)
        return Ga, gm, it, len(G) - len(Ga)

    def decide(self, X):
        """One oracle program on X: 'infeasible' when proven infeasible, 'feasible' when its model keeps every match of
        X within eps in L2, else 'undecided'."""
        st, t, H = oracle_lp(self.x1, self.x2, X, self.eps)
        if st == 2:
            return "infeasible"
        if st != 0:
            return "undecided"
        return "feasible" if np.all(transfer(H, self.x1[:, X], self.x2[:, X]) <= self.eps) else "undecided"

    def transfer_linf(self, H):
        """The L-infinity transfer error of every match in pixels (GORE's family); inf for a non-positive depth."""
        p = np.asarray(H, float) @ self.x1
        w = p[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.abs(p[:2] / w - self.x2[:2]).max(axis=0)
        e[~np.isfinite(e)] = np.inf
        e[w <= 0] = np.inf
        return e

    def gore(self, H):
        """The instance as GORE takes it (comparators/gore.py), in the gauge-free form: rows G theta <= h in the
        Hartley frame of all n matches, L-infinity per coordinate and depth-weighted, four per match, their owner, the
        gauge w_k = 1 for the test of match k, the residual that orders the candidates (the L-infinity transfer error
        in pixels under H), and verify, the L2 inlier mask of a theta of that frame."""
        T1, T2 = hartley_T(self.x1), hartley_T(self.x2)
        x1n, x2n, e = T1 @ self.x1, T2 @ self.x2, self.eps * T2[0, 0]
        x, y, xp, yp = x1n[0], x1n[1], x2n[0], x2n[1]
        o, z = np.ones(self.n), np.zeros(self.n)
        G = np.empty((4 * self.n, 9))
        for j, (sgn, tgt, first) in enumerate([(1, xp, True), (-1, xp, True), (1, yp, False), (-1, yp, False)]):
            c = sgn * tgt + e
            blk = [sgn * x, sgn * y, sgn * o, z, z, z] if first else [z, z, z, sgn * x, sgn * y, sgn * o]
            G[j::4] = np.column_stack(blk + [-c * x, -c * y, -c])
        T2inv = np.linalg.inv(T2)

        def verify(theta):
            return transfer(T2inv @ np.asarray(theta, float).reshape(3, 3) @ T1, self.x1, self.x2) <= self.eps

        gauge = lambda k: (np.array([[0, 0, 0, 0, 0, 0, x[k], y[k], 1.0]]), np.array([1.0]))   # noqa: E731
        return G, np.zeros(4 * self.n), np.repeat(np.arange(self.n), 4), gauge, self.transfer_linf(H), verify
