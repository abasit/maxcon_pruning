"""The real instances, one .mat file each in instances/real/<dataset>/, and the problem of each. All but kitti,
adelaide_f and eth3d are the homography problem at eps = 4 pixels, and kitti, adelaide_f and eth3d are the linearised
fundamental matrix at eps = 0.03 (problem 'linear'). build.py makes every homography dataset but homogr_cai, and
fundamental.py makes kitti, adelaide_f and eth3d; their docstrings give the sources and the rules. Each dataset is
listed by size (n, the number of matches).

Datasets:
  homogr      13 of the 16 pairs of homogr (Lebeda, Matas and Chum, BMVC 2012), our SIFT matches labelled by the
              ground truth; 84 to 2,746 matches. LePoint1, 2 and 3 are left out: SIFT finds 5, 1 and 0 correct
              matches there (of 40, 24 and 16)
  homogr_cai  city, BruggeTower, Brussels, adam and Boston of homogr with the matches Cai, Chin and Koltun ship with
              MaxConTreeSearch (ICCV 2019, their Table 2); unlabelled; 87 to 678 matches
  vgg         the 28 pairs the rule keeps from the 8 VGG affine sequences, our SIFT matches; 182 to 3,657
  adelaide    8 homography sequences of AdelaideRMF, the dataset's matches and labels; 103 to 363
  evd         the 15 pairs of EVD, the dataset's matches labelled by its ground truth; 83 to 1,164
  hpatches    102 pairs of the 59 viewpoint sequences of HPatches, our SIFT matches; 103 to 8,368. A summary
              experiment (README). ALONE["hpatches"] is the subset astar-alone runs on, fixed from the labels before
              any run: for each of 20 outlier shares evenly spaced from the lowest to the highest, the pair nearest
              to it not yet chosen (ties to the smaller n); 0.2 to 75 percent outliers, 117 to 4,875 matches.
  kitti       the five KITTI odometry pairs of sequence 00 Cai, Chin and Koltun ship with MaxConTreeSearch (ICCV
              2019), their matches, labelled by the FM-Bench rule under F from the odometry poses (camera P2); 302
              to 545 matches
  adelaide_f  the 19 fundamental-matrix scenes of AdelaideRMF, the dataset's matches and labels (0 a gross outlier,
              k the k-th motion); 164 to 341 matches
  eth3d       12 pairs of the test scenes of ETH3D with the SuperPoint+LightGlue matches of Pritts and Koeser (arXiv
              2610.05743), labelled by the FM-Bench rule under F from the shipped poses, chosen by fundamental.py's
              rule over the labelled shares; 159 to 637 matches, 50 to 100 percent labelled inliers

A file holds name, set (its dataset), problem ('homography'), eps, n, the matches x1 and x2 (3 x n, homogeneous
pixels), label, labelled (whether label carries labels) and source. The outliers of a labelled instance are its
points labelled 0. The randomness on an instance is seeded by its name (common.seed).

A 'linear' file (kitti, adelaide_f, eth3d) holds the same with problem 'linear', and model ('fundamental'), T1 and T2
(the Hartley normalisation of each image), A (n x 8) and b (n), the rows of the problem, x0 (A*'s start) and F_gt (the
ground-truth F in pixels, NaN-filled when there is none); and fixed, the entry of the normalised F fixed to 1 (a flat
index, row-major), gauge, F_ref, fundamental.py's reference F in pixels, and wh, the width and height of each image (a
row each). load gives a file without them fixed 7 (F32), gauge 'cai', F_ref and wh NaN-filled, Cai et al.'s form.
path and load read instances/real/ or, given inst, the folder inst of the same layout.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ..problems import homography, regression

ROOT = Path(__file__).resolve().parents[3]
INST = ROOT / "instances" / "real"
DATASETS = {
    "homogr": [
        "city_e4", "adam_e4", "WhiteBoard_e4", "BruggeSquare_e4", "ExtremeZoom_e4", "BruggeTower_e4",
        "CapitalRegion_e4", "boat_e4", "graf_e4", "BostonLib_e4", "Eiffel_e4", "Boston_e4", "Brussels_e4",
    ],
    "homogr_cai": ["city_e4", "BruggeTower_e4", "Brussels_e4", "adam_e4", "Boston_e4"],
    "vgg": [
        "bikes_1-6", "wall_2-6", "bikes_2-6", "graf_1-4", "trees_1-6", "trees_2-6", "bikes_1-5", "boat_1-6",
        "boat_2-6", "trees_3-6", "boat_3-6", "ubc_3-6", "ubc_5-6", "ubc_2-6", "graf_4-6", "ubc_4-6", "graf_3-5",
        "trees_4-6", "leuven_1-6", "wall_3-6", "graf_1-3", "boat_5-6", "bark_3-5", "wall_4-6", "bark_3-4",
        "bark_4-6", "bark_5-6", "wall_5-6",
    ],
    "adelaide": [
        "physics_e4", "bonython_e4", "ladysymon_e4", "barrsmith_e4", "sene_e4", "napiera_e4", "hartley_e4",
        "oldclassicswing_e4",
    ],
    "evd": [
        "shop_e4", "graf_e4", "mag_e4", "fox_e4", "adam_e4", "cafe_e4", "there_e4", "cat_e4", "vin_e4",
        "dum_e4", "face_e4", "index_e4", "pkk_e4", "girl_e4", "grand_e4",
    ],
    "hpatches": [
        "v_dogman_1-4", "v_maskedman_1-6", "v_beyus_1-5", "v_charing_1-3", "v_yuri_1-4", "v_wapping_1-6",
        "v_woman_1-6", "v_man_1-4", "v_soldiers_1-2", "v_churchill_1-6", "v_graffiti_1-4", "v_there_1-2",
        "v_wounded_1-5", "v_adam_1-2", "v_astronautis_1-5", "v_underground_1-5", "v_yuri_1-2", "v_beyus_1-2",
        "v_war_1-6", "v_gardens_1-3", "v_wormhole_1-6", "v_tabletop_1-5", "v_yard_1-6", "v_calder_1-4",
        "v_home_1-5", "v_talent_1-5", "v_circus_1-4", "v_wall_1-6", "v_birdwoman_1-3", "v_machines_1-5",
        "v_birdwoman_1-2", "v_blueprint_1-6", "v_bird_1-5", "v_grace_1-5", "v_dirtywall_1-6", "v_azzola_1-6",
        "v_maskedman_1-2", "v_courses_1-6", "v_charing_1-2", "v_woman_1-2", "v_abstract_1-6", "v_dogman_1-2",
        "v_boat_1-6", "v_wall_1-2", "v_eastsouth_1-6", "v_bark_1-2", "v_artisans_1-5", "v_coffeehouse_1-6",
        "v_fest_1-6", "v_cartooncity_1-5", "v_tempera_1-2", "v_war_1-2", "v_man_1-2", "v_strand_1-6",
        "v_apprentices_1-5", "v_sunseason_1-6", "v_bird_1-2", "v_vitro_1-4", "v_laptop_1-2", "v_graffiti_1-2",
        "v_wormhole_1-2", "v_bip_1-2", "v_wapping_1-2", "v_coffeehouse_1-2", "v_churchill_1-2", "v_calder_1-2",
        "v_home_1-2", "v_talent_1-2", "v_posters_1-6", "v_underground_1-2", "v_gardens_1-2", "v_abstract_1-2",
        "v_feast_1-6", "v_wounded_1-2", "v_circus_1-2", "v_grace_1-2", "v_machines_1-2", "v_astronautis_1-2",
        "v_courses_1-2", "v_pomegranate_1-2", "v_weapons_1-2", "v_artisans_1-2", "v_samples_1-2", "v_bees_1-2",
        "v_boat_1-2", "v_tabletop_1-2", "v_posters_1-2", "v_strand_1-2", "v_eastsouth_1-2", "v_yard_1-2",
        "v_vitro_1-2", "v_fest_1-2", "v_azzola_1-2", "v_cartooncity_1-2", "v_apprentices_1-2",
        "v_blueprint_1-2", "v_feast_1-2", "v_bricks_1-2", "v_dirtywall_1-2", "v_busstop_1-2", "v_sunseason_1-2",
        "v_london_1-2",
    ],
    "kitti": ["kitti_104_108", "kitti_198_201", "kitti_417_420", "kitti_738_742", "kitti_579_582"],
    "adelaide_f": [
        "carchipscube", "breadtoycar", "book", "toycubecar", "breadcubechips", "game", "breadcartoychips",
        "breadcube", "cubetoy", "biscuitbookbox", "boardgame", "cubechips", "breadtoy", "cube", "cubebreadtoychips",
        "biscuit", "gamebiscuit", "dinobooks", "biscuitbook",
    ],
    "eth3d": [
        "terrains_DSC_0618_DSC_0667", "relief_2_DSC_0460_DSC_0468", "playground_DSC_0590_DSC_0591",
        "relief_2_DSC_0458_DSC_0468", "playground_DSC_0587_DSC_0592", "terrains_DSC_0625_DSC_0629",
        "playground_DSC_0587_DSC_0590", "terrace_DSC_0277_DSC_0285", "terrains_DSC_0649_DSC_0668",
        "terrace_DSC_0276_DSC_0285", "relief_2_DSC_0482_DSC_0488", "terrace_DSC_0276_DSC_0284",
    ],
}
ALONE = {                                                 # a dataset's fixed subset for astar-alone, where it has one
    "hpatches": [
        "v_beyus_1-5", "v_woman_1-6", "v_astronautis_1-5", "v_yuri_1-2", "v_war_1-6", "v_gardens_1-3",
        "v_tabletop_1-5", "v_yard_1-6", "v_talent_1-5", "v_circus_1-4", "v_wall_1-6", "v_azzola_1-6",
        "v_abstract_1-6", "v_boat_1-6", "v_artisans_1-5", "v_fest_1-6", "v_man_1-2", "v_grace_1-2",
        "v_yard_1-2", "v_bricks_1-2",
    ],
}


def path(dataset, name, inst=None):
    return (INST if inst is None else Path(inst)) / dataset / f"{name}.mat"


def load(dataset, name, inst=None):
    """An instance as a dict: the scalars as Python values, label as an int vector, the matches as float arrays, and
    for a 'linear' instance its model, A, T1, T2, F_gt and F_ref as float arrays, b and x0 as float vectors, fixed and
    gauge."""
    I = {k: v for k, v in loadmat(path(dataset, name, inst)).items() if not k.startswith("__")}
    for k in ("name", "set", "problem", "source", "eps", "n"):
        I[k] = I[k].item()
    I["labelled"], I["label"] = bool(I["labelled"].item()), I["label"].ravel().astype(int)
    I["x1"], I["x2"] = I["x1"].astype(float), I["x2"].astype(float)
    if I["problem"] == "linear":
        I["model"] = I["model"].item()
        for k in ("A", "T1", "T2", "F_gt"):
            I[k] = I[k].astype(float)
        I["b"], I["x0"] = I["b"].ravel().astype(float), I["x0"].ravel().astype(float)
        I["fixed"] = int(I["fixed"].item()) if "fixed" in I else 7
        I["gauge"] = I["gauge"].item() if "gauge" in I else "cai"
        I["F_ref"] = I["F_ref"].astype(float) if "F_ref" in I else np.full((3, 3), np.nan)
        I["wh"] = I["wh"].astype(float) if "wh" in I else np.full((2, 2), np.nan)
    return I


def problem(I):
    """The problem object of an instance."""
    if I["problem"] == "homography":
        return homography.Problem(I["x1"], I["x2"], I["eps"])
    if I["problem"] == "linear":
        return regression.Problem(I["A"], I["b"], I["eps"])
    raise ValueError(f"{I['name']}: no problem class for {I['problem']!r}")


def outlier(I):
    """The points labelled 0 as a bool mask, or None for an unlabelled instance."""
    return I["label"] == 0 if I["labelled"] else None
