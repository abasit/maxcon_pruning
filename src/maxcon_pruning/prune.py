"""Degree pruning, Algorithm 1 of the paper.

run() works in rounds on the remaining points. A round at level s draws uniform s-subsets in batches of BATCH, tests
each for feasibility, and counts for every point the samples through it and the feasible ones. After every batch it
replaces M0 by any larger consensus set that a feasible sample's model gives (line 6), and marks for removal every
point whose degree Robbins' mixture bound puts below the threshold tau_s(m0) (line 7). The level rule then says whether
the next round stays at s, rises to s + 1 or stops. Each remaining point outside M0 is tested on its own (line 14), and
one feasibility test on the survivors bounds the optimum. With probability at least 1 - DELTA no point of a maximum
consensus set is removed, whatever the batch size and the stopping rules (the paper's pruning theorem).

The problem P supplies P.n and four methods, with point indices 0-based into the full instance:
  test(S)                    the sets S (m x s indices): (feasible mask, models of the feasible sets); a set the test
                             cannot decide counts as feasible and gives no model
  cover(models)              the n x len(models) inlier matrix of several models
  grow(cover, model, alive)  the refit and growth of an inlier set: (the set restricted to alive, its model, ...)
  decide(X)                  one feasibility test on X: 'feasible', 'infeasible' or 'undecided'
"""
import math
import time

import numpy as np
from scipy.special import betaln, xlog1py, xlogy

DELTA, C = 1e-6, 8.0                              # delta, the failure probability; c, the constant of (samples)
BATCH = 20_000                                    # b, the samples drawn between two checks
T_MIN, T_MAX = 2_000, 2_000_000                   # a round's number of samples is clipped to [T_MIN, T_MAX]
L_MAX, EMPTY = 30, 3                              # ell_max rounds at most; EMPTY rounds without a removal stop the run
YIELD = 1 / 20                                    # line 11: a round that removes fewer than n_ell / 20 points rises
T_PT, POINT_CHUNK = 10_000, 200                   # line 14: samples per point at most, and per check


def threshold(m0, n, s):
    """tau_s(m0, n) = C(m0-1, s-1) / C(n-1, s-1), and 0 when m0 < s."""
    return math.comb(m0 - 1, s - 1) / math.comb(n - 1, s - 1) if m0 >= s else 0.0


def sample_size(m0, n, s, delta):
    """C ln(n/delta) n / (s tau), the number of samples of a round (samples); inf without a threshold."""
    tau = threshold(m0, n, s)
    return C * math.log(n / delta) * n / (s * tau) if tau > 0 else math.inf


def round_size(m0, n, s, delta):
    """The number of samples of a round, clipped to [T_MIN, T_MAX]."""
    return int(min(max(sample_size(m0, n, s, delta), T_MIN), T_MAX))


def log_mixture(Y, T, p, b):
    """log M_T(p) of Robbins' mixture with the prior 1/2 Beta(1/2, 1/2) + 1/2 Beta(1/2, b), elementwise in (Y, T).
    Under independent Bernoulli(p) indicators it is a nonnegative supermartingale in T with M_0 = 1."""
    comp = lambda bb: betaln(0.5 + Y, bb + T - Y) - betaln(0.5, bb)  # noqa: E731
    return np.logaddexp(comp(0.5), comp(b)) - math.log(2.0) - xlogy(Y, p) - xlog1py(T - Y, -p)


def sample(rng, m, k, s):
    """m uniform s-subsets of range(k), one per row: the positions of the s smallest of k uniform numbers."""
    return np.argpartition(rng.random((m, k)), s - 1, axis=1)[:, :s]


def level_rule(n, removed, end, capped, T_next):
    """Line 11 and the stops of line 13: 'stay', 'rise' or 'stop: <reason>'. A round of T_MAX samples that removed
    nothing stops the run; a round cut short for a restart repeats its level; a round that removed fewer than YIELD of
    its n points rises, unless the next level needs more than T_MAX samples, which stops the run."""
    if removed == 0 and end == "budget" and capped:
        return "stop: a round of T_MAX samples removed nothing"
    if end == "restart":
        return "stay"
    if removed < YIELD * n:
        return "rise" if T_next <= T_MAX else "stop: the next level needs more than T_MAX samples"
    return "stay"


def largest_consensus(P, models):
    """Line 6: the largest consensus set over all n points among the models, and its model (the first with that
    size); None without models."""
    if not len(models):
        return None
    R = P.cover(models)
    j = int(np.argmax(R.sum(axis=0)))
    return np.flatnonzero(R[:, j]), models[j]


def one_round(P, alive, M0, model, s, l, rng):
    """Round l at level s on the points alive (lines 3 to 9), with the share delta 2^-l of DELTA. Returns the points
    marked for removal, M0 and its model as the round leaves them, and the round's record."""
    t0, k, dl = time.perf_counter(), len(alive), DELTA * 2.0 ** -l
    T = T_start = round_size(len(M0), k, s, dl)
    m0_start, tau_start = len(M0), threshold(len(M0), k, s)
    b = 1.0 / tau_start if tau_start > 0 else 0.5               # the second shape of the mixture's prior
    thr = math.log(k / dl)                                      # log(n_l / delta_l): a bound at this level decides
    Ti, Yi = np.zeros(k, np.int64), np.zeros(k, np.int64)       # samples through each point, and the feasible ones
    state, fell = np.zeros(k, np.int8), np.zeros(k, np.int64)   # -1 marked (after `fell` samples), 1 cleared
    drawn, batches, adopted, end = 0, 0, False, "budget"
    while drawn < T and (state == 0).any():
        sets = sample(rng, min(BATCH, T - drawn), k, s)
        drawn, batches = drawn + len(sets), batches + 1
        feasible, models = P.test(alive[sets])
        Ti += np.bincount(sets.ravel(), minlength=k)
        Yi += np.bincount(sets[feasible].ravel(), minlength=k)
        best = largest_consensus(P, models)
        if best is not None and len(best[0]) > len(M0):
            G, gm = P.grow(best[0], best[1], alive)[:2]
            if len(G) > len(M0):
                M0, model, adopted = G, gm, True
        T, tau = round_size(len(M0), k, s, dl), threshold(len(M0), k, s)
        if tau > 0:                                                          # lines 7 and 8
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = Yi / Ti
                below = (Ti > 0) & (rate < tau) & (log_mixture(Yi, Ti, tau, b) >= thr)
                above = (Ti > 0) & (rate > tau / 2) & (log_mixture(Yi, Ti, tau / 2, b) >= thr)
            new = below & (state != -1)
            state[new], fell[new] = -1, drawn
            still = state != -1
            state[still] = np.where(above[still], 1, 0)
        marked = int((state == -1).sum())
        if marked and sample_size(len(M0), k - marked, s, dl) < T - drawn:   # a fresh round on fewer points is cheaper
            end = "restart"
            break
    if end != "restart" and not (state == 0).any():
        end = "decided"
    removed = alive[state == -1]
    T_next = sample_size(len(M0), k - len(removed), s + 1, DELTA * 2.0 ** -(l + 1))
    capped = sample_size(len(M0), k, s, dl) > T_MAX
    rec = dict(level=s, n=k, points=alive.tolist(), Ti=Ti.tolist(), Yi=Yi.tolist(), T_start=T_start, T_end=T,
               capped=capped, drawn=drawn, batches=batches, tau_start=tau_start, tau_end=threshold(len(M0), k, s),
               m0_start=m0_start, m0_end=len(M0), adopted=adopted, removed=removed.tolist(),
               fell=fell[state == -1].tolist(), cleared=int((state == 1).sum()), undecided=int((state == 0).sum()),
               end=end, T_next=None if math.isinf(T_next) else int(T_next),
               move=level_rule(k, len(removed), end, capped, T_next), seconds=time.perf_counter() - t0)
    return removed, M0, model, rec


def per_point(P, alive, M0, s0, rng, rounds_done):
    """The per-point tests on the survivors alive (line 14, Algorithm S1). Each survivor outside M0 (a suspect) is
    tested on its own, with sets made of it and s - 1 other survivors drawn uniformly, POINT_CHUNK at a time, at the
    largest level s from s0 at which the per-point number of samples C ln(1/alpha) / tau fits in T_PT. A suspect is
    removed when the mixture bound puts its degree below tau, no longer tested when it puts it above tau / 2, and left
    undecided after T_PT samples. alpha is the stage's share of DELTA over the number of suspects.
    Returns the removed points and the stage's record; `skipped` says why it tested nothing."""
    t0, k, m0 = time.perf_counter(), len(alive), len(M0)
    suspects = alive[~np.isin(alive, M0)]
    rec = dict(level=None, tau=None, suspects=len(suspects), samples=0, removed=[], cleared=0,
               undecided=len(suspects), seconds=0.0, skipped=None)
    if len(suspects) == 0:
        return alive[:0], dict(rec, skipped="no suspects")
    thr = math.log(len(suspects) / (DELTA * 2.0 ** -(rounds_done + 1)))       # log(1/alpha)
    affordable = lambda s: s <= min(m0, k) and C * thr / threshold(m0, k, s) <= T_PT  # noqa: E731
    s = s0
    if not affordable(s):
        return alive[:0], dict(rec, skipped="no affordable level")
    while affordable(s + 1):
        s += 1
    tau = threshold(m0, k, s)
    b, outcomes, samples = 1.0 / tau, {}, 0
    for i in suspects:
        others, Ti, Yi, outcome = alive[alive != i], 0, 0, "undecided"
        while outcome == "undecided" and Ti < T_PT:
            m = min(POINT_CHUNK, T_PT - Ti)
            S = others[sample(rng, m, k - 1, s - 1)]
            feasible = P.test(np.column_stack([np.full(m, i), S]))[0]
            Ti, Yi = Ti + m, Yi + int(feasible.sum())
            if Yi / Ti < tau and log_mixture(Yi, Ti, tau, b) >= thr:
                outcome = "removed"
            elif Yi / Ti > tau / 2 and log_mixture(Yi, Ti, tau / 2, b) >= thr:
                outcome = "cleared"
        outcomes[int(i)], samples = outcome, samples + Ti
    removed = np.array([i for i, o in outcomes.items() if o == "removed"], np.int64)
    out = list(outcomes.values())
    return removed, dict(rec, level=s, tau=tau, samples=samples, removed=removed.tolist(), cleared=out.count("cleared"),
                         undecided=out.count("undecided"), seconds=time.perf_counter() - t0)


def certificate(P, X, m0):
    """One feasibility test on the survivors X: (lower, upper, route). Feasible survivors are a maximum consensus set;
    infeasible ones bound the optimum by |X| - 1; undecided ones by |X|. M0 gives the lower bound."""
    verdict = P.decide(X)
    if verdict == "feasible":
        return len(X), len(X), "feasible"
    if verdict == "infeasible":
        return m0, len(X) - 1, "interval"
    return m0, len(X), "undecided"


def run(P, s0, M0, model, rng):
    """Algorithm 1 on the problem P from the level s0 and a consensus set M0 (indices) with its model; rng is the
    sampling generator. Returns the result line: the survivors (keep), M0 as it ends, the bounds on the optimum, the
    samples drawn, and the records of the rounds and of the per-point tests."""
    t0 = time.perf_counter()
    M0, model = np.asarray(M0), np.asarray(model)
    m0_initial, alive, s, rounds, empty, stop = len(M0), np.arange(P.n), s0, [], 0, "L_MAX rounds"
    for l in range(1, L_MAX + 1):
        if threshold(len(M0), len(alive), s) == 0:                          # no threshold: no round can remove a point
            stop = "M0 has fewer points than the subset size"
            break
        removed, M0, model, rec = one_round(P, alive, M0, model, s, l, rng)
        rounds.append(rec)
        alive = np.setdiff1d(alive, removed)                                  # line 10
        if rec["move"].startswith("stop"):
            stop = rec["move"][len("stop: "):]
            break
        empty = empty + 1 if len(removed) == 0 else 0
        if empty == EMPTY:
            stop = f"{EMPTY} rounds in a row removed nothing"
            break
        if rec["move"] == "rise":
            s += 1
    removed, stage = per_point(P, alive, M0, s0, rng, len(rounds))
    alive = np.setdiff1d(alive, removed)
    lower, upper, route = certificate(P, alive, len(M0))
    return dict(n=P.n, m0_initial=m0_initial, m0=len(M0), m0set=M0.tolist(), m0_model=model.tolist(),
                survivors=len(alive), keep=alive.tolist(), cert_lower=lower, cert_upper=upper, cert_route=route,
                exact=lower == upper, samples=sum(r["drawn"] for r in rounds) + stage["samples"],
                n_rounds=len(rounds), stop=stop, rounds=rounds, per_point=stage, seconds=time.perf_counter() - t0)
