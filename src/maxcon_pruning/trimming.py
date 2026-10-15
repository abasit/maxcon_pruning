"""The trimming, the uncertified baseline: keep the points whose residual under LO-RANSAC's model is at most k eps,
for k in KS. It certifies nothing; A* then runs on each trimmed set.
"""
import numpy as np

KS = (2, 3, 5)


def trim(P, model):
    """The trimmed sets of the problem P under a model, one per k in KS: {k: the points with residual <= k eps}."""
    r = P.residual(model)
    return {k: np.flatnonzero(r <= k * P.eps) for k in KS}
