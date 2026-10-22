"""The instance files of the real datasets, from their images or from their own matches, into
instances/real/<dataset>/. All are homography instances at eps = 4 px.

Where a dataset gives images and a ground-truth homography (homogr, vgg, hpatches), the matches are made by one rule:
OpenCV 5.0.0 SIFT_create() with its defaults on the grey images, the nearest and second nearest descriptors of each
first-image keypoint (L2), the ratio test at 0.8, then one match per keypoint: by increasing descriptor distance, a
match is kept unless its first-image or its second-image position is taken; the matches are stored in the order of the
first image's keypoints. A match is labelled by the ground-truth homography H from the first image to the second: an
inlier (label 1) when its transfer error |H x1 - x2| is at most 4 px, an outlier (label 0) otherwise. OpenCV's pixel
coordinates are used as they are. Where a dataset gives its own matches (adelaide, evd), those are kept, since plain
SIFT finds almost no correct match on EVD's extreme views. adelaide keeps its labels. evd is labelled by its
ground-truth homographies at 4 px as above, since its own labels mark only some of the matches within 4 px. A match
that repeats an earlier one exactly (the same two points) is dropped, the first kept with its label; only adelaide has
any (69 in all).

  homogr    13 of the 16 pairs of homogr (Lebeda, Matas and Chum, BMVC 2012), not LePoint1, 2 and 3, where SIFT finds
            5, 1 and 0 correct matches; data/homogr/<pair>A and <pair>B, and <pair>_model.txt, the homography from B
            to A, inverted; from github.com/danini/magsac (data/homography); named <pair>_e4
  vgg       the 8 sequences of the VGG affine dataset (Mikolajczyk et al., IJCV 2005), data/vgg/<seq>/img<i>.ppm (boat
            .pgm) and H1to<j>p: all 15 pairs i-j of a sequence are matched, with the homography H1to<j>p H1to<i>p^-1,
            and a pair is kept when it has at least 100 matches and between 25 and 85 percent of them correct, at most
            four per sequence, the lowest shares first; named <seq>_<i>-<j>
  hpatches  the 59 viewpoint sequences of HPatches (Balntas et al., CVPR 2017),
            data/hpatches/hpatches-sequences-release/v_*/<i>.ppm and H_1_<k>: the pairs 1-2 to 1-6 of a sequence are
            matched, and two are kept, 1-2 when it has at least 100 matches and the one of 1-3 to 1-6 with the lowest
            correct share among those the vgg rule keeps; named <seq>_1-<k>
  adelaide  8 homography sequences of AdelaideRMF (Wong et al., ICCV 2011), data/adelaide/<seq>.mat from RCMSA's data:
            the dataset's matches and hand labels (0 a wrong match, k a point of the k-th plane), repeats dropped;
            named <seq>_e4
  evd       the 15 pairs of EVD (Mishkin et al., CVIU 2015), data/evd/<pair>_pts.txt from github.com/danini/magsac
            (data/homography/extremeview, the same as EVD_tentatives.zip): the dataset's tentative correspondences
            (Hessian-Affine and MSER, RootSIFT), labelled under data/evd/<pair>_H.txt, the homography from the first
            image to the second (EVD.zip, h/<pair>.txt, from cmp.felk.cvut.cz/wbs/datasets/); named <pair>_e4

A file holds what instances.py reads (name, set, problem, eps, n, x1, x2, label, labelled, source), H_gt with the
error of each match under it (gt_error), and the image sizes. adelaide has no ground-truth homography, so its H_gt is
the normalised DLT of the matches labelled 1, for reference only. Needs the build extra (uv sync --extra build).

    python -m maxcon_pruning.real.build DATASET
"""
import itertools
import sys

import cv2
import numpy as np
from scipy.io import loadmat, savemat

from ..problems.homography import dlt_normalised
from .instances import INST, ROOT

EPS, RATIO = 4.0, 0.8
MIN_N, SHARES, PER_SEQ = 100, (0.25, 0.85), 4             # the pair rule
DATA = ROOT / "data"
HOMOGR = ["adam", "boat", "Boston", "BostonLib", "BruggeSquare", "BruggeTower", "Brussels", "CapitalRegion", "city",
          "Eiffel", "ExtremeZoom", "graf", "WhiteBoard"]                    # not LePoint1-3 (the module docstring)
VGG = ["graf", "wall", "bark", "boat", "bikes", "trees", "leuven", "ubc"]
ADELAIDE = ["physics", "bonython", "hartley", "oldclassicswing", "barrsmith", "ladysymon", "napiera", "sene"]
HPATCHES = DATA / "hpatches" / "hpatches-sequences-release"


def match(im1, im2):
    """The matches of two grey images by the rule of the module docstring: (x1, x2) as 3 x n homogeneous pixels, and
    the counts (keypoints of each image, matches after the ratio test)."""
    sift = cv2.SIFT_create()
    (k1, d1), (k2, d2) = sift.detectAndCompute(im1, None), sift.detectAndCompute(im2, None)
    near = [m for m, n in cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2) if m.distance < RATIO * n.distance]
    taken1, taken2, kept = set(), set(), []
    for m in sorted(near, key=lambda m: m.distance):
        p1, p2 = k1[m.queryIdx].pt, k2[m.trainIdx].pt
        if p1 in taken1 or p2 in taken2:
            continue
        taken1.add(p1)
        taken2.add(p2)
        kept.append(m)
    kept.sort(key=lambda m: m.queryIdx)                    # in the order of the first image's keypoints
    x1 = np.array([[*k1[m.queryIdx].pt, 1.0] for m in kept]).T
    x2 = np.array([[*k2[m.trainIdx].pt, 1.0] for m in kept]).T
    return x1, x2, (len(k1), len(k2), len(near))


def label_by(x1, x2, H):
    """The transfer error of every match under H and the labels, 1 within EPS and 0 beyond."""
    p = H @ x1
    e = np.linalg.norm(p[:2] / p[2] - x2[:2], axis=0)
    return e, (e <= EPS).astype(np.uint8)


def distinct(x1, x2, label):
    """The matches without those that repeat an earlier one exactly (the same two points), the first kept with its
    label, and the number dropped."""
    keep = np.sort(np.unique(np.r_[x1[:2], x2[:2]].T, axis=0, return_index=True)[1])
    return x1[:, keep], x2[:, keep], label[keep], x1.shape[1] - len(keep)


def instance(dataset, name, x1, x2, label, H, source, sizes):
    """The fields of one file, its matches made distinct (the number dropped noted in source); gt_error is each
    match's transfer error under H."""
    x1, x2, label, dropped = distinct(x1, x2, label)
    if dropped:
        source = f"{source}; {dropped} repeated matches dropped"
    p = H @ x1
    e = np.linalg.norm(p[:2] / p[2] - x2[:2] / x2[2], axis=0)
    return dict(name=name, set=dataset, problem="homography", eps=EPS, n=x1.shape[1], x1=x1, x2=x2,
                label=label[:, None], labelled=1, H_gt=H, gt_error=e[:, None], sizes=np.array(sizes, float),
                source=source)


def sift_pair(dataset, name, fa, fb, H, where):
    """An instance made by the SIFT rule from the images fa and fb and the ground truth H (first image to second)."""
    ims = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in (fa, fb)]
    x1, x2, (n1, n2, near) = match(*ims)
    H = H / H[2, 2]
    source = (f"{where}; OpenCV {cv2.__version__} SIFT_create() defaults, ratio {RATIO}, keypoints [{n1}, {n2}], "
              f"after the ratio test {near}, one per keypoint {x1.shape[1]}")
    return instance(dataset, name, x1, x2, label_by(x1, x2, H)[1], H, source, [im.shape[:2] for im in ims])


def fitted(x1, x2, label):
    """The normalised DLT of the matches labelled 1, scaled to h33 = 1."""
    H = dlt_normalised(x1[:, label == 1], x2[:, label == 1])
    return H / H[2, 2]


def homogr():
    d = DATA / "homogr"
    for pair in HOMOGR:
        fa, fb = (next(d.glob(f"{pair}{s}.*")) for s in "AB")
        yield sift_pair("homogr", f"{pair}_e4", fa, fb, np.linalg.inv(np.loadtxt(d / f"{pair}_model.txt")),
                        f"homogr (Lebeda, Matas and Chum, BMVC 2012) via github.com/danini/magsac: {fa.name}, "
                        f"{fb.name}, {pair}_model.txt inverted")


def left_out(I, why):
    print(f"{I['name']:22s} n {I['n']:5d}, correct {I['label'].mean():4.0%}: left out, {why}", flush=True)


def selected(pairs, per=PER_SEQ):
    """The pairs the rule keeps, in their order: at least MIN_N matches and a correct share within SHARES, at most
    per of them, the lowest shares first. Prints each pair left out and why."""
    share = {I["name"]: I["label"].mean() for I in pairs}
    ok = [I["name"] for I in pairs if I["n"] >= MIN_N and SHARES[0] <= share[I["name"]] <= SHARES[1]]
    keep = sorted(ok, key=share.get)[:per]
    for I in pairs:
        if I["name"] not in keep:
            left_out(I, "lower shares kept" if I["name"] in ok else "outside the rule")
    return [I for I in pairs if I["name"] in keep]


def vgg():
    for seq in VGG:
        d = DATA / "vgg" / seq

        def H1(k):
            return np.eye(3) if k == 1 else np.loadtxt(d / f"H1to{k}p")

        def img(k):
            return next(d.glob(f"img{k}.p?m"))

        yield from selected([sift_pair("vgg", f"{seq}_{i}-{j}", img(i), img(j), H1(j) @ np.linalg.inv(H1(i)),
                                       f"VGG affine dataset (Mikolajczyk et al., IJCV 2005): data/vgg/{seq}/"
                                       f"{img(i).name}, {img(j).name}, H1to{j}p H1to{i}p^-1")
                             for i, j in itertools.combinations(range(1, 7), 2)])


def hpatches():
    for d in sorted(HPATCHES.glob("v_*")):
        easy, *hard = [sift_pair("hpatches", f"{d.name}_1-{k}", d / "1.ppm", d / f"{k}.ppm",
                                 np.loadtxt(d / f"H_1_{k}"),
                                 f"HPatches (Balntas et al., CVPR 2017): {d.name}/1.ppm, {k}.ppm, H_1_{k}")
                       for k in range(2, 7)]
        if easy["n"] >= MIN_N:
            yield easy
        else:
            left_out(easy, f"under {MIN_N} matches")
        yield from selected(hard, per=1)


def adelaide():
    for seq in ADELAIDE:
        S = loadmat(DATA / "adelaide" / f"{seq}.mat")
        x1, x2, label, dropped = distinct(S["data"][:3], S["data"][3:6], S["label"].ravel().astype(np.uint8))
        yield instance("adelaide", f"{seq}_e4", x1, x2, label, fitted(x1, x2, label),
                       f"AdelaideRMF (Wong et al., ICCV 2011), H/{seq}.mat from RCMSA's data: the dataset's matches "
                       f"and hand labels (0 a wrong match, k the k-th plane), {dropped} repeated matches dropped; "
                       f"H_gt the normalised DLT of plane 1", [S["img1"].shape[:2], S["img2"].shape[:2]])


def evd():
    d = DATA / "evd"
    for f in sorted(d.glob("*_pts.txt")):
        pair = f.name[:-len("_pts.txt")]
        P = np.loadtxt(f, usecols=(0, 1, 2, 3))
        x1, x2 = np.c_[P[:, :2], np.ones(len(P))].T, np.c_[P[:, 2:4], np.ones(len(P))].T
        H = np.loadtxt(d / f"{pair}_H.txt")
        H = H / H[2, 2]
        sizes = [cv2.imread(str(d / f"{pair}{s}.png"), cv2.IMREAD_UNCHANGED).shape[:2] for s in "AB"]
        yield instance("evd", f"{pair}_e4", x1, x2, label_by(x1, x2, H)[1], H,
                       f"EVD (Mishkin et al., CVIU 2015) via github.com/danini/magsac data/homography/extremeview: "
                       f"{f.name}, the dataset's tentative correspondences (Hessian-Affine and MSER, RootSIFT), "
                       f"labelled at 4 px under {pair}_H.txt, EVD's ground-truth homography (EVD.zip, h/{pair}.txt, "
                       f"cmp.felk.cvut.cz/wbs/datasets/)", sizes)


BUILDERS = dict(homogr=homogr, vgg=vgg, hpatches=hpatches, adelaide=adelaide, evd=evd)


def main(args):
    if len(args) != 1 or args[0] not in BUILDERS:
        sys.exit(__doc__)
    out = INST / args[0]
    out.mkdir(parents=True, exist_ok=True)
    for I in BUILDERS[args[0]]():
        savemat(out / f"{I['name']}.mat", I)
        inl = int((I["label"] > 0).sum())
        print(f"{I['name']:22s} n {I['n']:5d}, labelled inliers {inl:5d} ({inl / I['n']:.0%})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
