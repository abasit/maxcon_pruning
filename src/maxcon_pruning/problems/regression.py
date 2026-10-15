"""Linear regression under the largest absolute residual, the Chebyshev criterion. A set of points is feasible when
some model theta keeps every residual |a_i . theta - b_i| within eps. Its Chebyshev value is the smallest largest
residual any model achieves, so the set is feasible when its value is at most eps.

The feasibility test is closed form for sets of d + 1 to d + 3 points. The value of d + 1 points is |b . v| / |v|_1,
for v spanning the vectors orthogonal to the columns of A on those points, and the value of a larger set is the largest
value of its (d + 1)-point subsets. Larger sets get one linear program each (HiGHS). A degenerate set, or a program
that fails, counts as feasible and gives no model, so a test that cannot decide never removes a point.

Problem holds an instance and the methods the other modules call: test, cover, grow and decide for the pruning
(prune.py), residual for the trimming, and gore for GORE.
"""
from itertools import combinations

import numpy as np
from scipy.optimize import linprog

GROW_MAX = 10                                     # line 6 of Algorithm 1: refits per growth at most


def chebyshev(A, b):
    """The Chebyshev fit of the rows (A, b) by linear program (HiGHS): the model and its value."""
    k, d = A.shape
    one = np.ones((k, 1))
    r = linprog(np.r_[np.zeros(d), 1.0], A_ub=np.block([[A, -one], [-A, -one]]), b_ub=np.r_[b, -b],
                bounds=[(None, None)] * d + [(0, None)], method="highs")
    if r.status != 0:
        raise RuntimeError(f"Chebyshev linear program failed: {r.message}")
    return r.x[:d], r.x[d]


def inliers(A, b, theta, th):
    """The points with |a_i . theta - b_i| <= th."""
    with np.errstate(invalid="ignore"):
        return np.flatnonzero(np.abs(A @ theta - b) <= th)


def closed_form(A, b, sets):
    """Chebyshev values of row sets of d+1 to d+3 points, batched, with a degenerate set given the value 0. Returns the
    values, the degenerate mask, and per set the positions left out by the subset attaining the value and its dual
    vector."""
    m, s = sets.shape
    d = A.shape[1]
    N = np.linalg.qr(A[sets], mode="complete")[0][:, :, d:]            # spans {v : A_S^T v = 0}
    if s == d + 1:
        V, out = N, np.empty((1, 0), int)
    elif s == d + 2:
        V, out = N @ np.stack([N[:, :, 1], -N[:, :, 0]], axis=1), np.arange(s)[:, None]
    else:
        J, L = np.array(list(combinations(range(s), 2))).T
        V, out = N @ np.transpose(np.cross(N[:, J], N[:, L]), (0, 2, 1)), np.column_stack([J, L])
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.abs(np.einsum("ms,msj->mj", b[sets], V)) / np.abs(V).sum(axis=1)
    bad = ~np.all(np.isfinite(vals), axis=1)
    vals[bad] = 0.0
    j = vals.argmax(axis=1)
    return vals[np.arange(m), j], bad, out[j], V[np.arange(m), :, j]


def cf_models(A, b, sets, out, v):
    """The Chebyshev fit of each set from its critical (d+1)-subset, where the residuals equioscillate."""
    m, s = sets.shape
    d = A.shape[1]
    keep = np.ones((m, s), bool)
    keep[np.arange(m)[:, None], out] = False
    rows = np.take_along_axis(sets, np.nonzero(keep)[1].reshape(m, d + 1), 1)
    vJ = np.take_along_axis(v, np.nonzero(keep)[1].reshape(m, d + 1), 1)
    sig = np.sign(vJ) * np.sign(np.einsum("mj,mj->m", vJ, b[rows]))[:, None]
    M, rhs = np.concatenate([A[rows], sig[:, :, None]], axis=2), b[rows][:, :, None]
    try:
        return np.linalg.solve(M, rhs)[:, :d, 0]
    except np.linalg.LinAlgError:
        return np.stack([np.linalg.lstsq(Mi, ri, rcond=None)[0][:d, 0] for Mi, ri in zip(M, rhs)])


def grow(A, b, eps, cover, model, alive):
    """Refit by the Chebyshev LP and take the inliers while the set grows (at most GROW_MAX refits), then restrict to
    the points in `alive`. Returns the set, its model, the refits taken and the grown points outside `alive`."""
    G, gm, it = cover, model, 0
    while it < GROW_MAX:
        try:
            gn = chebyshev(A[G], b[G])[0]
        except RuntimeError:
            break
        Gn = inliers(A, b, gn, eps)
        if len(Gn) <= len(G):
            break
        G, gm, it = Gn, gn, it + 1
    Ga = np.intersect1d(G, alive)
    return Ga, gm, it, len(G) - len(Ga)


class Problem:
    """A regression instance: A (n x d, one row per point), b (n) and the inlier threshold eps."""

    def __init__(self, A, b, eps):
        self.A, self.b, self.eps, self.n = A, b, eps, len(b)
        self.p = A.shape[1]                      # minimal sample size: every set of at most p points is feasible

    def test(self, S):
        """The row sets S (m x s indices): (feasible mask, models of the feasible sets the test decided)."""
        A, b, eps, d = self.A, self.b, self.eps, self.p
        if S.shape[1] <= d + 3:
            vals, bad, out, v = closed_form(A, b, S)
            feasible = vals <= eps                                      # a degenerate set has the value 0
            f = feasible & ~bad
            return feasible, (cf_models(A, b, S[f], out[f], v[f]) if f.any() else np.empty((0, d)))
        feasible, models = np.ones(len(S), bool), []
        for i, Si in enumerate(S):
            try:
                theta, value = chebyshev(A[Si], b[Si])
            except RuntimeError:
                continue                                                # a failed program counts as feasible
            if value <= eps:
                models.append(theta)
            else:
                feasible[i] = False
        return feasible, np.array(models).reshape(-1, d)

    def residual(self, theta):
        """The residual |a_i . theta - b_i| of every point under one model."""
        with np.errstate(invalid="ignore"):
            return np.abs(self.A @ np.asarray(theta, float) - self.b)

    def cover(self, Th):
        """The inlier matrix (n x models) of several models."""
        with np.errstate(invalid="ignore"):
            return np.abs(self.A @ np.asarray(Th).T - self.b[:, None]) <= self.eps

    def grow(self, cover, model, alive):
        """The Chebyshev-refit growth of an inlier set (see grow)."""
        return grow(self.A, self.b, self.eps, cover, model, alive)

    def decide(self, X):
        """One linear program on the points X: 'feasible', 'infeasible', or 'undecided' when the program fails."""
        try:
            return "feasible" if chebyshev(self.A[X], self.b[X])[1] <= self.eps else "infeasible"
        except RuntimeError:
            return "undecided"

    def gore(self, theta):
        """The instance as GORE takes it (comparators/gore.py): point i as the rows a_i theta <= b_i + eps and
        -a_i theta <= eps - b_i, the owner of each row, no gauge rows, the residuals under theta, and verify, a
        model's inlier mask."""
        n, d = self.A.shape
        G, h = np.empty((2 * n, d)), np.empty(2 * n)
        G[0::2], h[0::2] = self.A, self.b + self.eps
        G[1::2], h[1::2] = -self.A, self.eps - self.b
        return G, h, np.repeat(np.arange(n), 2), None, self.residual(theta), lambda th: self.cover([th])[:, 0]
