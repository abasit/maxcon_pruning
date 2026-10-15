"""GORE, guaranteed outlier removal (Chin, Kee, Eriksson and Neumann, CVPR 2016): Algorithm 1 of the paper, with
Gurobi, the solver it was published with.

A problem gives GORE its points as linear rows in the parameters. P.gore(model) returns the rows G theta <= h, the
point owning each row, optional gauge rows added to the test of a point, each point's residual under the model in
GORE's family, and verify(theta), a model's inliers in the exact residual. The test of a candidate k is problem (7)
of the paper: keep k an inlier and minimise the number of other remaining points relaxed by the big-M constant, with
the cutoff that at most u_hat may be relaxed, u_hat being the number of remaining points minus m0. Then
  - an infeasible program is Condition 1: k is a true outlier and is removed, and u_hat drops by one;
  - a solution is Condition 2: its model's inliers among the remaining points, in the exact residual, raise m0 when
    there are more of them, which lowers u_hat. The solver stops at its first solution, since every solution meets
    the cutoff;
  - running out of time leaves k.
The candidates are the points outside the model's inlier set, the largest residual first, at most T of them; by the
paper's Lemma 2 a point of the reference set can never be removed. The per-cell stopping rule of the experiments is
ours and lives in the driver.
"""
import time

import gurobipy as gp
import numpy as np
import scipy.sparse as sp


def program(G, h, owner, alive, k, ubar, M, gauge=None):
    """Problem (7) on the points alive (sorted; their rows in order), point k hard, the cutoff row sum z <= ubar, then
    the gauge rows of point k (if any); columns theta (free), then z_i (binary) for the other alive points in index
    order. Returns (A sparse csc, row lower, row upper, p, nz)."""
    o = alive[alive != k]
    p, nz = G.shape[1], len(o)
    rows = np.flatnonzero(np.isin(owner, alive))
    soft = np.flatnonzero(owner[rows] != k)
    Z = sp.csr_matrix((np.full(len(soft), -M), (soft, np.searchsorted(o, owner[rows][soft]))), shape=(len(rows), nz))
    blocks, lo, up = [[sp.csr_matrix(G[rows]), Z], [None, sp.csr_matrix(np.ones((1, nz)))]], [], []
    if gauge is not None:
        E, f = gauge(k)
        E = np.atleast_2d(E)
        blocks.append([sp.csr_matrix(E), sp.csr_matrix((E.shape[0], nz))])
        lo, up = list(f), list(f)
    A = sp.bmat(blocks, format="csc")
    return A, np.r_[np.full(len(rows) + 1, -np.inf), lo], np.r_[h[rows], ubar, up], p, nz


def test(G, h, owner, alive, k, ubar, M, cap, gauge=None):
    """One test of point k: Gurobi on one thread, seed 0, at most cap seconds, stopping at the first solution. Returns
    the status ('removed', 'incumbent', 'timeout' or 'error'), the incumbent's theta or None, the seconds (building
    the model included) and the branch-and-bound nodes. DualReductions 0 makes Gurobi report an infeasible program as
    infeasible rather than as infeasible or unbounded. A Gurobi error, such as a model larger than a size-limited
    licence allows, is printed and counts as 'error'."""
    t0 = time.perf_counter()
    A, lo, up, p, nz = program(G, h, owner, alive, k, ubar, M, gauge)
    m, theta, nodes = gp.Model(), None, 0
    try:
        for opt, val in (("OutputFlag", 0), ("Threads", 1), ("Seed", 0), ("TimeLimit", cap), ("SolutionLimit", 1),
                         ("DualReductions", 0)):
            m.setParam(opt, val)
        x = m.addMVar(p, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        z = m.addMVar(nz, vtype=gp.GRB.BINARY, obj=1.0)
        v = gp.MVar.fromlist(list(x.tolist()) + list(z.tolist()))
        fin_up, fin_lo = np.isfinite(up), np.isfinite(lo)
        if fin_up.any():
            m.addMConstr(A[fin_up], v, "<", up[fin_up])
        if fin_lo.any():
            m.addMConstr(A[fin_lo], v, ">", lo[fin_lo])
        m.optimize()
        nodes = int(m.NodeCount)
        if m.Status == gp.GRB.INFEASIBLE:
            status = "removed"
        elif m.SolCount > 0:
            status, theta = "incumbent", np.asarray(x.X)
        elif m.Status == gp.GRB.TIME_LIMIT:
            status = "timeout"
        else:
            status = "error"
    except gp.GurobiError as e:
        print(f"GORE: Gurobi error on point {k}: {e}", flush=True)
        status = "error"
    m.dispose()
    return status, theta, time.perf_counter() - t0, nodes


def run(P, model, m0, T, M, cap):
    """Algorithm 1 on the problem P from a model and the size m0 of its consensus set: at most T tests, big-M constant
    M, cap seconds per test. Returns the result line from 'n' on; seconds counts everything from building the rows."""
    t0 = time.perf_counter()
    G, h, owner, gauge, r, verify = P.gore(model)
    n = len(r)
    cand = [int(k) for k in np.argsort(-r, kind="stable") if r[k] > P.eps][:T]
    alive, m, tests = np.ones(n, bool), m0, []
    for k in cand:
        idx = np.flatnonzero(alive)
        status, theta, secs, nodes = test(G, h, owner, idx, k, len(idx) - m, M, cap, gauge)
        if status == "removed":
            alive[k] = False
        elif status == "incumbent":                    # Condition 2, verified in the exact residual
            m = max(m, int((verify(theta) & alive).sum()))
        tests.append([k, status, round(secs, 3), nodes])
    removed = sorted(x[0] for x in tests if x[1] == "removed")
    return dict(n=n, T=T, M=M, cap=cap, m0=m0, m0_final=m, candidates=len(cand), tests=tests,
                removed=removed, n_removed=len(removed), timeouts=sum(x[1] == "timeout" for x in tests),
                incumbents=sum(x[1] == "incumbent" for x in tests), errors=sum(x[1] == "error" for x in tests),
                at_limit=bool(tests) and not removed and all(x[1] == "timeout" for x in tests),
                seconds=time.perf_counter() - t0)
