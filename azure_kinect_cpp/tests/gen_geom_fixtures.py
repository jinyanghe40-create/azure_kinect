#!/usr/bin/env python3
"""Generate geometry-pipeline oracle fixtures from common/pc_pipeline.py.

Run from anywhere:
    python3 azure_kinect_cpp/tests/gen_geom_fixtures.py

Writes, under azure_kinect_cpp/tests/fixtures/geom/:
    manifest.json                 list of scenes (args + file names)
    <scene>.f32                   N x 3 little-endian float32 points
    <scene>.ply                   same points (binary PLY, for geom_replay)
    <scene>.expected.json         Python run_pipeline result (points stripped)
    <scene>.boxes.json            detection boxes, when the scene splits
"""
import json
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "fixtures", "geom")
sys.path.insert(0, os.path.join(ROOT, "common"))

import pc_pipeline as PC  # noqa: E402


def write_ply(path, pts):
    pts = np.ascontiguousarray(pts, dtype="<f4")
    n = len(pts)
    header = (f"ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              f"property float x\nproperty float y\nproperty float z\n"
              f"end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        f.write(rec.tobytes())


def write_f32(path, arr):
    np.ascontiguousarray(arr, dtype="<f4").tofile(path)


def clean(v):
    """JSON-ify numpy scalars/arrays and drop point arrays."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items() if k != "pts"}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return [clean(x) for x in v.tolist()]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def base_args(**kw):
    a = types.SimpleNamespace(
        edge=0.05,
        min_dist=0.3,
        max_dist=1.5,
        plane_thresh=0.003,
        table_thresh=0.010,
        voxel=0.02,
        min_cluster_points=0,
        max_faces=3,
        depth_noise_coef=0.008,
        face_select="most_facing",
        edge_auto=True,
        up_axis=None,
        cam_axis=None,
        cam_origin=None,
        gravity=False,
        split_uv_fn=None,
        split_boxes=None,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def args_json(a):
    return {
        "edge": a.edge,
        "edge_auto": a.edge_auto,
        "min_dist": a.min_dist,
        "max_dist": a.max_dist,
        "plane_thresh": a.plane_thresh,
        "table_thresh": a.table_thresh,
        "voxel": a.voxel,
        "min_cluster_points": a.min_cluster_points,
        "max_faces": a.max_faces,
        "depth_noise_coef": a.depth_noise_coef,
        "face_select": a.face_select,
        "up_axis": list(a.up_axis) if a.up_axis is not None else None,
        "cam_axis": list(a.cam_axis) if a.cam_axis is not None else None,
        "cam_origin": list(a.cam_origin) if a.cam_origin is not None else None,
    }


def synth_scene(seed=7, n_cubes=4, edge=0.05, noise=0.002):
    pts, _truth = PC.make_synthetic_scene(n_cubes=n_cubes, edge=edge, seed=seed,
                                          noise_sigma=noise)
    return pts


def touching_scene(edge=0.10, table_y=0.30, z0=0.75, seed=11, noise=0.0015,
                   per_face=700):
    """Two axis-aligned cubes touching along x, on a table, camera at origin.

    Only faces a depth camera could see are emitted: front (-z), top (-y) and
    the outer side (+x for the right cube, -x for the left cube).  Their front
    faces meet at x=0, so voxel clustering yields one connected component
    unless the 2D boxes split it.
    """
    rng = np.random.default_rng(seed)
    groups = []
    for sign in (-1.0, 1.0):
        cx = sign * edge / 2.0
        cy = table_y - edge / 2.0
        cz = z0
        c = np.array([cx, cy, cz])
        faces = [
            (np.array([0.0, 0.0, -1.0]), [0, 1]),   # front
            (np.array([0.0, -1.0, 0.0]), [0, 2]),   # top
            (np.array([sign, 0.0, 0.0]), [1, 2]),   # outer side
        ]
        pts = []
        for nrm, others in faces:
            local = np.zeros((per_face, 3))
            local[:, others[0]] = rng.uniform(-edge / 2, edge / 2, per_face)
            local[:, others[1]] = rng.uniform(-edge / 2, edge / 2, per_face)
            axis = [k for k in range(3) if k not in others][0]
            local[:, axis] = nrm[axis] * edge / 2.0
            pts.append(c + local + rng.normal(0, noise, local.shape))
        groups.append(np.vstack(pts))
    span = 0.30
    n_table = 20000
    xs = rng.uniform(-span, span, n_table)
    zs = rng.uniform(z0 - 0.20, z0 + 0.20, n_table)
    ys = np.full(n_table, table_y) + rng.normal(0, noise, n_table)
    table = np.stack([xs, ys, zs], axis=1)
    pts = np.vstack([table] + groups)
    labels = np.concatenate([np.full(len(table), -1)] + [
        np.full(len(g), i) for i, g in enumerate(groups)])
    return pts, labels


def grid_scene(edge=0.12, gap=0.16, table_y=0.45, z0=1.00, seed=23,
               noise=0.002, per_face=800):
    """Four yaw-rotated cubes on a table, well separated.

    Larger than the toy `make_synthetic_scene` cubes (12 cm at ~1 m) so the
    adaptive plane threshold stays small relative to a face.  Only the faces a
    camera at the origin could see are emitted.
    """
    rng = np.random.default_rng(seed)
    sp = edge + gap
    positions = [(-sp / 2, z0 - sp / 2), (sp / 2, z0 - sp / 2),
                 (-sp / 2, z0 + sp / 2), (sp / 2, z0 + sp / 2)]
    yaws = rng.uniform(-25.0, 25.0, len(positions))
    groups = []
    for (cx, cz), yaw_deg in zip(positions, yaws):
        c = np.array([cx, table_y - edge / 2.0, cz])
        yaw = math.radians(yaw_deg)
        Ry = np.array([[math.cos(yaw), 0.0, math.sin(yaw)],
                       [0.0, 1.0, 0.0],
                       [-math.sin(yaw), 0.0, math.cos(yaw)]])
        faces = [
            (np.array([0.0, 0.0, -1.0]), [0, 1]),   # front
            (np.array([0.0, -1.0, 0.0]), [0, 2]),   # top
            (np.array([1.0 if cx >= 0 else -1.0, 0.0, 0.0]), [1, 2]),  # outer
        ]
        pts = []
        for nrm, others in faces:
            local = np.zeros((per_face, 3))
            local[:, others[0]] = rng.uniform(-edge / 2, edge / 2, per_face)
            local[:, others[1]] = rng.uniform(-edge / 2, edge / 2, per_face)
            axis = [k for k in range(3) if k not in others][0]
            local[:, axis] = nrm[axis] * edge / 2.0
            pts.append(c + (Ry @ local.T).T + rng.normal(0, noise, local.shape))
        groups.append(np.vstack(pts))
    span_x, span_z = 0.55, 0.55
    n_table = 20000
    xs = rng.uniform(-span_x, span_x, n_table)
    zs = rng.uniform(z0 - span_z, z0 + span_z, n_table)
    ys = np.full(n_table, table_y) + rng.normal(0, noise, n_table)
    table = np.stack([xs, ys, zs], axis=1)
    pts = np.vstack([table] + groups)
    labels = np.concatenate([np.full(len(table), -1)] + [
        np.full(len(g), i) for i, g in enumerate(groups)])
    return pts, labels


def pinhole_uv_fn(fx, fy, cx, cy):
    def fn(P):
        P = np.asarray(P, np.float64)
        Z = P[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * P[:, 0] / Z + cx
            v = fy * P[:, 1] / Z + cy
        return np.stack([u, v], axis=1)
    return fn


def boxes_from_labels(pts, labels, n_cubes, fx, fy, cx, cy, pad=3.0):
    uv = pinhole_uv_fn(fx, fy, cx, cy)(pts)
    boxes = []
    for cid in range(n_cubes):
        m = labels == cid
        if not m.any():
            continue
        u, v = uv[m, 0], uv[m, 1]
        boxes.append((float(u.min() - pad), float(v.min() - pad),
                      float(u.max() + pad), float(v.max() + pad), cid, 0.9))
    return boxes


def emit_scene(manifest, name, pts, a, boxes=None, intr=None):
    pts = np.asarray(pts, np.float64)
    pts32 = np.ascontiguousarray(pts, dtype="<f4")
    pts = pts32.astype(np.float64)

    if boxes is not None:
        a.split_boxes = boxes
        a.split_uv_fn = pinhole_uv_fn(*intr[:4])

    res = PC.run_pipeline(pts, a, verbose=False)
    expected = clean(res["objects"])

    write_f32(os.path.join(OUT, name + ".f32"), pts32)
    write_ply(os.path.join(OUT, name + ".ply"), pts32)
    with open(os.path.join(OUT, name + ".expected.json"), "w") as f:
        json.dump(expected, f, indent=1, ensure_ascii=False)

    entry = {
        "name": name,
        "n": int(len(pts)),
        "pts": name + ".f32",
        "expected": name + ".expected.json",
        "args": args_json(a),
    }
    if boxes is not None:
        box_list = [{"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3],
                     "cls": int(b[4]), "score": b[5]} for b in boxes]
        with open(os.path.join(OUT, name + ".boxes.json"), "w") as f:
            json.dump(box_list, f, indent=1)
        entry["boxes"] = name + ".boxes.json"
        entry["intr"] = list(intr[:4])
        entry["size"] = [intr[4], intr[5]]
        entry["rotate"] = 0
        entry["px_fix"] = [0.0, 0.0]
    manifest.append(entry)
    print(f"{name}: {len(pts)} pts -> {len(expected)} objects")


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []

    fx, fy, cx, cy, W, H = 500.0, 500.0, 320.0, 240.0, 640, 480

    grid, glabels = grid_scene()
    gargs = base_args(edge=0.12, depth_noise_coef=0.002)
    emit_scene(manifest, "grid_default", grid, gargs)
    emit_scene(manifest, "grid_largest", grid,
               base_args(edge=0.12, depth_noise_coef=0.002, face_select="largest"))
    # Physical "up" in the camera frame is -Y (the table is at +Y).
    emit_scene(manifest, "grid_upaxis", grid,
               base_args(edge=0.12, depth_noise_coef=0.002, up_axis=[0.0, -1.0, 0.0],
                         cam_axis=[0.0, 0.0, 1.0]))
    gboxes = boxes_from_labels(grid, glabels, 4, fx, fy, cx, cy)
    emit_scene(manifest, "grid_split_on", grid, base_args(edge=0.12, depth_noise_coef=0.002),
               boxes=gboxes, intr=(fx, fy, cx, cy, W, H))
    # Same cloud expressed in a body frame (C0) with the optical centre at
    # (0.25, 0, 0.60).  Exercises cam_origin / cam_axis / up_axis together.
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    t = np.array([0.25, 0.0, 0.60])
    body = grid @ R.T + t
    emit_scene(manifest, "grid_bodyframe", body,
               base_args(edge=0.12, depth_noise_coef=0.002, up_axis=[0.0, 0.0, 1.0],
                         cam_axis=(R @ np.array([0.0, 0.0, 1.0])).tolist(),
                         cam_origin=t.tolist()))

    touch, labels = touching_scene()
    boxes = boxes_from_labels(touch, labels, 2, fx, fy, cx, cy)
    emit_scene(manifest, "touch_split_on", touch,
               base_args(edge=0.10, table_thresh=0.008, depth_noise_coef=0.002),
               boxes=boxes, intr=(fx, fy, cx, cy, W, H))
    emit_scene(manifest, "touch_split_off", touch,
               base_args(edge=0.10, table_thresh=0.008, depth_noise_coef=0.002))

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print("geometry fixtures ->", OUT)


if __name__ == "__main__":
    main()
