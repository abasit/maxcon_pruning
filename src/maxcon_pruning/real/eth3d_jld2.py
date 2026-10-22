"""The ETH3D SuperPoint+LightGlue correspondences of Pritts and Koeser (arXiv 2610.05743, ACCV 2026), from their Julia
files into one numpy .npz per scene, data/eth3d_splg/npz/<scene>.npz, which fundamental.py's builder eth3d reads.
This module needs numpy and h5py only and imports nothing of the package, so it runs with any Python that has those
two (the project's venv has no h5py); fundamental.py reads its output with numpy alone.

The source is eth3d_splg_csponds.tar.gz of Zenodo record 23240469, extracted to
data/eth3d_splg/datasets/eth3d/derived/splg/csponds/: one <scene>.jld2 per scene of the 13 high-resolution multi-view
training scenes of ETH3D (Schoeps et al., CVPR 2017), and manifest.tsv, the 3,081 pairs of the extraction (image 1
path, image 2 path, <scene>__<image 1>__<image 2>.h5, the names without .JPG). A .jld2 file is JLD2, which is HDF5:
its root holds 'pairs' and '_types' and no 'format', so the files are the first version of the consolidated record,
which serialises the pair's rig (VisualGeometryToolkitEval src/datasets/roma.jl, load_roma_pairs, the format-1
branch). '_types' holds one committed HDF5 compound type per Julia struct, with the attributes field_names and
field_types (object references to the field types) beside julia_type, which h5py cannot decode (a variable-length
list of references) and which nothing here needs: a struct is known by its field names. 'pairs' is a vector of object
references, one per pair, each to a scalar compound of the fields
  scene, source_name, target_name            strings; the names are the undistorted DSLR images, <name>.JPG
  source_img_path, target_img_path           the producer's paths, not kept
  csponds.components.first, .second          references to n x {data: (Float32, Float32)}, the matched pixels of
                                             image 1 (source) and image 2 (target)
  confidence                                 reference to n Float32, LightGlue's match confidence, higher better
  rig.source, rig.target                     each camera.model.intrinsics.{K, K_inv}.data (9 Float64, column-major,
                                             the pixel K and its inverse), camera.model.lens (empty: pinhole),
                                             camera.extrinsics.{R.mat.data (9, column-major), t.data (3)} and
                                             extent.{width, height}
  rig.extrinsics, rig.pose                   {R.mat.data, t.data} each; R_gt and t_gt are rig.extrinsics, the map
                                             from the source camera to the target camera (consolidate.jl,
                                             _consolidated_record), and rig.pose is its inverse
  matcher                                    'superpoint+lightglue'
  inlier_mask                                reference to a Julia BitVector {chunks, len, dims}: chunks a reference
                                             to UInt64 words, bit k - 1 of the vector at bit (k - 1) mod 64 of word
                                             (k - 1) div 64, least significant first; the depth-warp label of each
                                             match, from ETH3D's laser depth (validate_gt_inliers)
  gt_config                                  {gt_method, metric_tol (a reference to {val}, millimetres), rel_tol,
                                             bidirectional}
The matrices are Julia's column-major SMatrix data, so a 3 x 3 is reshape(3, 3, order='F'). The cameras'
extrinsics are world-to-camera (eth3d.jl, _eth3d_camera: COLMAP's qvec and tvec), so rig.extrinsics should be
R2 R1' and t2 - R2 R1' t1 of them; convert checks that to rounding, with K K_inv = I and every pair's match count,
and stops on any other layout. The val/test split and the conventions are not in the files: fundamental.py has them.

A scene's .npz holds, the pairs in the file's order and their matches concatenated in it:
  scene, matcher, gt_method        strings (one per scene; convert checks that gt_config is the same on every pair)
  metric_tol_mm, rel_tol, bidirectional    the depth-warp test's settings
  source_name, target_name         P strings, the image names with .JPG
  offsets                          P + 1 int64: pair k's matches are the rows offsets[k] to offsets[k + 1] - 1
  x1, x2                           N x 2 float32, the matched pixels in image 1 and image 2 as stored
  confidence                       N float32
  inlier_mask                      N bool
  K1, K2                           P x 3 x 3 float64
  size1, size2                     P x 2 int64, each image's (width, height)
  R_gt, t_gt                       P x 3 x 3 and P x 3, rig.extrinsics
  R_pose, t_pose                   rig.pose
  R1, t1, R2, t2                   the two cameras' world-to-camera extrinsics
No object array is written, so np.load reads the file with allow_pickle False.

    python -I eth3d_jld2.py CSPONDS_DIR OUT_DIR
"""
import sys
from pathlib import Path

import h5py
import numpy as np

FIELDS = ("scene", "source_name", "target_name", "source_img_path", "target_img_path", "csponds", "confidence",
          "rig", "matcher", "inlier_mask", "gt_config")
TOL = 1e-9                                              # convert's checks, on rotations, translations and K K_inv


def text(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def mat(data):
    """A 3 x 3 from Julia's column-major SMatrix data (9 numbers)."""
    return np.array(list(data), float).reshape(3, 3, order="F")


def vec(data):
    return np.array(list(data), float)


def bits(f, ref):
    """A Julia BitVector (its compound {chunks, len, dims}) as a bool vector."""
    b = f[ref][()]
    words = np.asarray(f[b["chunks"]][()], np.uint64)
    n = int(b["len"])
    k = np.arange(n)
    return ((words[k // 64] >> (k % 64).astype(np.uint64)) & np.uint64(1)).astype(bool)


def points(f, ref):
    """n x 2 float32 of a vector of {data: (Float32, Float32)}."""
    d = f[ref][()]["data"]
    return np.column_stack([d["1"], d["2"]]).astype(np.float32)


def camera(c):
    """K, K_inv, R and t (world to camera) and (width, height) of a rig's BoundedCamera."""
    m, e = c["camera"]["model"]["intrinsics"], c["camera"]["extrinsics"]
    return (mat(m["K"]["data"]), mat(m["K_inv"]["data"]), mat(e["R"]["mat"]["data"]), vec(e["t"]["data"]),
            np.array([int(c["extent"]["width"]), int(c["extent"]["height"])], np.int64))


def record(f, ref):
    """One pair of a scene file as a dict of numpy values, checked (module docstring)."""
    v = f[ref][()]
    if v.dtype.names != FIELDS:
        raise ValueError(f"{f.filename}: a pair with the fields {v.dtype.names}, not the first-version record")
    rig = v["rig"]
    K1, K1i, R1, t1, size1 = camera(rig["source"])
    K2, K2i, R2, t2, size2 = camera(rig["target"])
    R, t = mat(rig["extrinsics"]["R"]["mat"]["data"]), vec(rig["extrinsics"]["t"]["data"])
    Rp, tp = mat(rig["pose"]["R"]["mat"]["data"]), vec(rig["pose"]["t"]["data"])
    x1, x2 = points(f, v["csponds"]["components"]["first"]), points(f, v["csponds"]["components"]["second"])
    conf, mask = np.asarray(f[v["confidence"]][()], np.float32), bits(f, v["inlier_mask"])
    g = v["gt_config"]
    gaps = dict(K1=np.abs(K1 @ K1i - np.eye(3)).max(), K2=np.abs(K2 @ K2i - np.eye(3)).max(),
                R=np.abs(R - R2 @ R1.T).max(), t=np.abs(t - (t2 - R2 @ R1.T @ t1)).max(),
                pose_R=np.abs(Rp - R.T).max(), pose_t=np.abs(tp + R.T @ t).max())
    name = f"{text(v['scene'])} {text(v['source_name'])} {text(v['target_name'])}"
    bad = {k: x for k, x in gaps.items() if not x <= TOL}
    if bad:
        raise ValueError(f"{name}: the rig does not hold to rounding: {bad}")
    if not len(x1) == len(x2) == len(conf) == len(mask):
        raise ValueError(f"{name}: {len(x1)}, {len(x2)}, {len(conf)} and {len(mask)} matches, masks and scores")
    return dict(scene=text(v["scene"]), source_name=text(v["source_name"]), target_name=text(v["target_name"]),
                matcher=text(v["matcher"]), gt_method=text(g["gt_method"]),
                metric_tol_mm=float(f[g["metric_tol"]][()]["val"]), rel_tol=float(g["rel_tol"]),
                bidirectional=bool(g["bidirectional"]), x1=x1, x2=x2, confidence=conf, inlier_mask=mask, K1=K1,
                K2=K2, size1=size1, size2=size2, R_gt=R, t_gt=t, R_pose=Rp, t_pose=tp, R1=R1, t1=t1, R2=R2, t2=t2)


def scene_file(path):
    """The pairs of one <scene>.jld2, as records."""
    with h5py.File(path, "r") as f:
        if set(f.keys()) != {"pairs", "_types"}:
            raise ValueError(f"{path}: the root holds {sorted(f.keys())}, not the first version's pairs and _types")
        return [record(f, ref) for ref in f["pairs"][()]]


def pack(recs):
    """A scene's records as the arrays of its .npz."""
    one = {k: {r[k] for r in recs} for k in ("scene", "matcher", "gt_method", "metric_tol_mm", "rel_tol",
                                             "bidirectional")}
    if any(len(v) != 1 for v in one.values()):
        raise ValueError(f"settings that differ between the pairs of a scene: {one}")
    out = {k: np.array(v.pop()) for k, v in one.items()}
    out["offsets"] = np.concatenate([[0], np.cumsum([len(r["x1"]) for r in recs])]).astype(np.int64)
    for k in ("source_name", "target_name"):
        out[k] = np.array([r[k] for r in recs])
    for k in ("x1", "x2", "confidence", "inlier_mask"):
        out[k] = np.concatenate([r[k] for r in recs])
    for k in ("K1", "K2", "size1", "size2", "R_gt", "t_gt", "R_pose", "t_pose", "R1", "t1", "R2", "t2"):
        out[k] = np.stack([r[k] for r in recs])
    return out


def manifest_ids(path):
    """The pair ids <scene>__<image 1>__<image 2> of manifest.tsv (its third column without .h5)."""
    rows = [line.split("\t") for line in Path(path).read_text().splitlines() if line.strip()]
    return [Path(r[2]).name[:-len(".h5")] for r in rows]


def convert(src, out):
    """Every <scene>.jld2 of src into out/<scene>.npz, and the pairs against src/manifest.tsv."""
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ids = []
    for path in sorted(src.glob("*.jld2")):
        recs = scene_file(path)
        a = pack(recs)
        np.savez_compressed(out / f"{path.stem}.npz", **a)
        ids += [f"{r['scene']}__{Path(r['source_name']).stem}__{Path(r['target_name']).stem}" for r in recs]
        n = np.diff(a["offsets"])
        print(f"{path.stem:14s} pairs {len(recs):4d}, matches {n.min()} to {n.max()} (median {np.median(n):g}), "
              f"depth-warp inliers {a['inlier_mask'].mean():.1%}, sizes {sorted({tuple(s) for s in a['size1']})}",
              flush=True)
    listed = manifest_ids(src / "manifest.tsv")
    print(f"{len(ids)} pairs in the files, {len(listed)} in manifest.tsv, {len(set(ids) - set(listed))} of the files' "
          f"not in it, {len(set(listed) - set(ids))} of its not in the files, {len(ids) - len(set(ids))} repeated",
          flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    convert(*sys.argv[1:])
