"""The real instances, one .mat file each in instances/real/<dataset>/, and the problem of each.

Datasets:
  homogr   <pair>_e4 for city, BruggeTower, Brussels, adam and Boston: the matches Cai, Chin and Koltun ship with
           MaxConTreeSearch (ICCV 2019, their Table 2) on five pairs of homogr (Lebeda, Matas and Chum, BMVC 2012),
           the homography problem at eps = 4 pixels; unlabelled. Listed by size, 87 to 678 matches.

A file holds name, set (its dataset), problem ('homography'), eps, n, the matches x1 and x2 (3 x n, homogeneous
pixels), label, labelled (whether label carries labels) and source. The outliers of a labelled instance are its
points labelled 0. The randomness on an instance is seeded by its name (common.seed).
"""
from pathlib import Path

from scipy.io import loadmat

from ..problems import homography

ROOT = Path(__file__).resolve().parents[3]
INST = ROOT / "instances" / "real"
DATASETS = {
    "homogr": ["city_e4", "BruggeTower_e4", "Brussels_e4", "adam_e4", "Boston_e4"],
}


def path(dataset, name):
    return INST / dataset / f"{name}.mat"


def load(dataset, name):
    """An instance as a dict: the scalars as Python values, label as an int vector, the matches as float arrays."""
    I = {k: v for k, v in loadmat(path(dataset, name)).items() if not k.startswith("__")}
    for k in ("name", "set", "problem", "source", "eps", "n"):
        I[k] = I[k].item()
    I["labelled"], I["label"] = bool(I["labelled"].item()), I["label"].ravel().astype(int)
    I["x1"], I["x2"] = I["x1"].astype(float), I["x2"].astype(float)
    return I


def problem(I):
    """The problem object of an instance."""
    if I["problem"] == "homography":
        return homography.Problem(I["x1"], I["x2"], I["eps"])
    raise ValueError(f"{I['name']}: no problem class for {I['problem']!r}")


def outlier(I):
    """The points labelled 0 as a bool mask, or None for an unlabelled instance."""
    return I["label"] == 0 if I["labelled"] else None
