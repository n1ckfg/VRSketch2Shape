#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""shape-to-sketch.py -- the inverse direction of VRSketch2Shape.

VRSketch2Shape maps a *sequential VR sketch* to a 3D shape (see ARCHITECTURE.md).
This script runs the pipeline backwards: it takes a mesh, a point cloud, or one
of the repo's SDF grids and emits an ordered collection of brushstrokes, written
in the Latk format (https://lightningartist.org/spec/,
https://github.com/LightningArtist/latkpy).

It is a learning-free reimplementation of the synthetic sketch generation
pipeline of Sec. 3.2 / Fig. 2 of "Order Matters: 3D Shape Generation from
Sequential VR Sketches":

    surface sampling -> salient points -> spline strokes -> culling -> ordering

The paper's defaults are used throughout: 2048 uniform surface samples, a 15
degree curvature threshold for Sharp Edge Sampling, splines of maximum degree 2,
a minimum segment length of 12, a cosine-distance culling threshold of 0.04, an
endpoint merge radius of 2% of the normalized shape size, and a depth-first
stroke ordering that skips the nearest connection 10% of the time.

Two stages of the paper lean on external components -- Dora's Sharp Edge
Sampling and EMAP's neural edge reconstruction.  Both are replaced here with
geometric equivalents that need nothing beyond numpy/scipy; see the docstrings
of `extract_salient_points` and `recover_strokes` for what differs.

Usage
-----
    python shape-to-sketch.py -i chair.obj -o chair.latk
    python shape-to-sketch.py -i cloud.ply -o cloud.latk --json-out cloud.json
    python shape-to-sketch.py -i data/sdf/03001627/<id>/ori_sample_grid.h5 -o x.latk

`--json-out` writes the plain list-of-strokes JSON that
`dataloader.sketch_data.Sketch2ShapeDataset.read_syn_lines` consumes, so a
sketch produced here can be fed straight back into the sketch-to-shape model.

Output
------
The Latk document holds one layer with one frame -- a static drawing -- whose
strokes appear in the order the pipeline "drew" them, each stroke's points
ordered along the direction of travel.  By default stroke color ramps from blue
to red across that order, so the sequence is visible at a glance in Blender,
Unity, or any other Latk client.  Coordinates are the unit-sphere frame the
model expects, unless `--denormalize` restores the input's own placement.

Every distance threshold below is a fraction of the normalized shape size, so
the defaults transfer unchanged from one object to the next.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree

EPS = 1e-12

# Every threshold below is a fraction of the "normalized shape size", which is
# the diameter of the unit sphere the shape is normalized into (see
# `normalize_to_unit_sphere`), i.e. 2.0.
SHAPE_SIZE = 2.0

MESH_EXT = {".obj", ".ply", ".off", ".stl", ".glb", ".gltf", ".dae", ".fbx", ".3mf", ".x3d"}
CLOUD_EXT = {".xyz", ".pts", ".asc", ".txt", ".csv", ".npy", ".npz"}
SDF_EXT = {".h5", ".hdf5"}


@dataclass
class SketchParams:
    """Pipeline parameters.  Defaults follow Sec. 3.2 of the paper."""

    # Stage 1 -- surface sampling
    n_surface: int = 2048            # paper: 2048 uniform surface samples
    # Stage 2 -- salient points
    curvature_threshold: float = 15.0    # degrees, paper: 15
    salient_mode: str = "auto"           # auto | sharp | curvature | both
    n_salient: int = 6000                # budget of samples along sharp edges
    normal_knn: int = 12                 # neighbourhood for PCA normals / salience
    max_cloud: int = 120000              # cap on input cloud size for salience
    thin_knn: int = 16                   # neighbourhood for curve thinning
    thin_iterations: int = 3             # projections onto the local tangent line
    # Stage 3 -- stroke recovery
    knn: int = 8                         # k for the curve-network graph
    max_edge_factor: float = 3.0         # cap on graph edge length (x median NN dist)
    prune_branch: float = 0.03           # drop MST hairs shorter than this x shape size
    smooth_passes: int = 2               # Laplacian passes over a polyline before fitting
    bezier_error: float = 0.006          # fit tolerance, fraction of shape size
    resample_spacing: float = 0.01       # spline sampling step, fraction of shape size
    min_segment_points: int = 12         # paper: minimum segment length 12 (pre-fit)
    # Stage 4 -- culling
    cull_cosine: float = 0.04            # paper: cosine distance threshold 0.04
    merge_radius: float = 0.02           # paper: 2% of normalized shape size
    # Stage 5 -- ordering
    order_knn: int = 6                   # stroke-connectivity graph degree
    skip_prob: float = 0.1               # paper: skip nearest connection 10% of the time
    seed: int = 0


# --------------------------------------------------------------------------- #
# Geometry container and I/O
# --------------------------------------------------------------------------- #


class Shape:
    """A triangle mesh (points + faces) or a bare point cloud (points only)."""

    def __init__(self, points, faces=None, normals=None):
        # Contiguous throughout: a non-contiguous array makes numpy accumulate in a
        # different order, and the last-bit differences that follow are enough to
        # flip nearest-neighbour ties on a symmetric model.
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.faces = None if faces is None else np.ascontiguousarray(faces, dtype=np.int64)
        self.normals = None if normals is None else \
            np.ascontiguousarray(normals, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must be (N, 3), got {}".format(self.points.shape))

    @property
    def is_mesh(self):
        return self.faces is not None and len(self.faces) > 0

    def __repr__(self):
        kind = "mesh {} verts / {} faces".format(len(self.points), len(self.faces)) \
            if self.is_mesh else "point cloud {} points".format(len(self.points))
        return "<Shape {}>".format(kind)


def _load_obj(path):
    verts, faces = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):  # fan-triangulate n-gons
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return Shape(verts, faces if faces else None)


def _load_off(path):
    with open(path, "r", errors="ignore") as f:
        tokens = f.read().split()
    if tokens[0].upper().startswith("OFF"):
        tokens = tokens[1:] if tokens[0].upper() == "OFF" else [tokens[0][3:]] + tokens[1:]
    n_v, n_f = int(tokens[0]), int(tokens[1])
    pos = 3
    verts = np.array(tokens[pos:pos + 3 * n_v], dtype=np.float64).reshape(n_v, 3)
    pos += 3 * n_v
    faces = []
    for _ in range(n_f):
        k = int(tokens[pos])
        idx = [int(v) for v in tokens[pos + 1:pos + 1 + k]]
        pos += 1 + k
        for j in range(1, k - 1):
            faces.append([idx[0], idx[j], idx[j + 1]])
    return Shape(verts, faces if faces else None)


def _load_ply_ascii(path):
    with open(path, "rb") as f:
        raw = f.read()
    head_end = raw.find(b"end_header")
    if head_end < 0:
        raise ValueError("{}: not a PLY file".format(path))
    header = raw[:head_end].decode("ascii", errors="ignore").splitlines()
    if not any("ascii" in h for h in header):
        raise ValueError("{}: binary PLY needs trimesh installed".format(path))
    n_v = n_f = 0
    v_props = []
    element = None
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            element = parts[1]
            if element == "vertex":
                n_v = int(parts[2])
            elif element == "face":
                n_f = int(parts[2])
        elif parts[0] == "property" and element == "vertex" and parts[1] != "list":
            v_props.append(parts[-1])
    body = raw[raw.find(b"\n", head_end) + 1:].decode("ascii", errors="ignore").splitlines()
    verts = np.array([[float(x) for x in body[i].split()] for i in range(n_v)])
    cols = {name: k for k, name in enumerate(v_props)}
    points = verts[:, [cols.get("x", 0), cols.get("y", 1), cols.get("z", 2)]]
    normals = None
    if {"nx", "ny", "nz"} <= set(cols):
        normals = verts[:, [cols["nx"], cols["ny"], cols["nz"]]]
    faces = []
    for i in range(n_v, n_v + n_f):
        parts = [int(float(x)) for x in body[i].split()]
        idx = parts[1:1 + parts[0]]
        for j in range(1, len(idx) - 1):
            faces.append([idx[0], idx[j], idx[j + 1]])
    return Shape(points, faces if faces else None, normals)


def _load_text_cloud(path):
    delim = "," if path.lower().endswith(".csv") else None
    arr = np.loadtxt(path, delimiter=delim, ndmin=2)
    normals = arr[:, 3:6] if arr.shape[1] >= 6 else None
    return Shape(arr[:, :3], None, normals)


def _load_npy(path):
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        key = next((k for k in ("points", "pc", "xyz", "verts", "sdf") if k in arr), arr.files[0])
        arr = arr[key]
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == arr.shape[1] == arr.shape[2]:
        return _sdf_grid_to_shape(arr)          # a cubic array is an SDF volume
    arr = arr.reshape(-1, arr.shape[-1])
    normals = arr[:, 3:6] if arr.shape[1] >= 6 else None
    return Shape(arr[:, :3], None, normals)


def _sdf_grid_to_shape(sdf, level=0.02):
    """Marching-cubes an SDF volume, matching `utils.util_3d.sdf_to_mesh` scaling."""
    from skimage import measure

    sdf = np.asarray(sdf, dtype=np.float32)
    res = sdf.shape[-1]
    if sdf.min() > level or sdf.max() < level:
        level = float(np.median(sdf))
    verts, faces, _, _ = measure.marching_cubes(sdf, level=level)
    verts = verts / res - 0.5                    # same normalization as util_3d
    return Shape(verts, faces)


def _load_sdf_h5(path, key="pc_sdf_sample", level=0.02):
    import h5py

    with h5py.File(path, "r") as f:
        if key not in f:
            key = list(f.keys())[0]
        sdf = f[key][:].astype(np.float32)
    n = sdf.size
    res = int(round(n ** (1.0 / 3.0)))
    if res ** 3 != n:
        raise ValueError("{}: {} values is not a cubic grid".format(path, n))
    return _sdf_grid_to_shape(sdf.reshape(res, res, res), level=level)


def _load_with_trimesh(path):
    import trimesh

    obj = trimesh.load(path, process=False)
    if isinstance(obj, trimesh.Scene):
        geoms = list(obj.geometry.values())
        if not geoms:
            raise ValueError("{}: empty scene".format(path))
        obj = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    if isinstance(obj, trimesh.Trimesh):
        return Shape(obj.vertices, obj.faces)
    if isinstance(obj, trimesh.PointCloud):
        return Shape(obj.vertices)
    raise ValueError("{}: unsupported geometry {}".format(path, type(obj).__name__))


def load_shape(path, sdf_level=0.02, verbose=True):
    """Load a mesh, a point cloud, or an SDF grid as a `Shape`."""
    ext = os.path.splitext(path)[1].lower()
    if not os.path.exists(path):
        raise FileNotFoundError("{}: no such file".format(path))

    if ext in SDF_EXT:
        shape = _load_sdf_h5(path, level=sdf_level)
    elif ext in (".npy", ".npz"):
        shape = _load_npy(path)
    else:
        shape = None
        if ext in MESH_EXT or ext in CLOUD_EXT:
            try:
                shape = _load_with_trimesh(path)
            except ImportError:
                pass                              # fall through to the built-in readers
        if shape is None:
            if ext == ".obj":
                shape = _load_obj(path)
            elif ext == ".off":
                shape = _load_off(path)
            elif ext == ".ply":
                shape = _load_ply_ascii(path)
            elif ext in CLOUD_EXT:
                shape = _load_text_cloud(path)
            else:
                raise ValueError(
                    "{}: install trimesh to read {} files".format(path, ext))
    if verbose:
        print("[load] {} -> {}".format(path, shape))
    return shape


def normalize_to_unit_sphere(shape, reference=None):
    """Center on the mean and scale to unit radius.

    Mirrors `dataloader.sketch_data.normalize_lines_to_unit_sphere` and
    `utils.util_3d.get_normalize_mesh`, so the thresholds below (which are all
    expressed as fractions of the shape size) mean the same thing they do in
    the forward pipeline.  Returns (centroid, scale) so the caller can undo it.
    """
    ref = shape.points if reference is None else np.asarray(reference)
    centroid = ref.mean(axis=0)
    scale = float(np.max(np.linalg.norm(ref - centroid, axis=1)))
    if scale < EPS:
        scale = 1.0
    shape.points = np.ascontiguousarray((shape.points - centroid) / scale)
    return centroid, scale


# --------------------------------------------------------------------------- #
# Stage 1 -- uniform surface sampling
# --------------------------------------------------------------------------- #


def sample_surface(shape, n, rng):
    """Draw `n` area-weighted uniform samples from the surface (paper: 2048).

    A bare point cloud has no surface to sample, so its own points stand in for
    the sample; it is randomly thinned when it holds more than `n` points.
    """
    if not shape.is_mesh:
        pts = shape.points
        if len(pts) <= n:
            return pts.copy()
        return pts[rng.choice(len(pts), n, replace=False)]

    tri = shape.points[shape.faces]
    ab = tri[:, 1] - tri[:, 0]
    ac = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(ab, ac), axis=1)
    total = float(area.sum())
    if total < EPS:                                   # degenerate mesh
        return shape.points[rng.choice(len(shape.points), n)]

    f = rng.choice(len(area), size=n, p=area / total)
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    flip = (u + v > 1.0).ravel()                      # fold onto the triangle
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return tri[f, 0] + u * ab[f] + v * ac[f]


# --------------------------------------------------------------------------- #
# Stage 2 -- salient points (a stand-in for Dora's Sharp Edge Sampling)
# --------------------------------------------------------------------------- #


def estimate_normals(points, k):
    """Local-PCA normals plus the index of each point's k nearest neighbours.

    The normals are unoriented (sign is arbitrary), which is all the salience
    test below needs.
    """
    k = int(min(max(k, 3), len(points)))
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    idx = np.atleast_2d(idx)
    nb = points[idx]
    d = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", d, d) / max(k - 1, 1)
    _, evecs = np.linalg.eigh(cov)                    # ascending eigenvalues
    return evecs[:, :, 0], idx


def normal_variation(normals, idx):
    """Per-point salience in degrees: how much the normal turns locally.

    The 90th percentile of the neighbourhood angles is used rather than the
    maximum, which would be dominated by a single noisy neighbour.
    """
    dots = np.abs(np.einsum("nkj,nj->nk", normals[idx], normals)).clip(0.0, 1.0)
    return np.percentile(np.degrees(np.arccos(dots)), 90, axis=1)


def thin_to_curves(points, k, iterations):
    """Collapse a band of salient points onto the curve running through it.

    Thresholding curvature marks a *strip* either side of a crease, which the
    curve recovery below would zig-zag through.  Each point is repeatedly
    projected onto the local tangent line -- the first principal direction of
    its neighbourhood, through that neighbourhood's centroid -- which pulls the
    strip onto its centreline.  This stands in for the curve fitting that EMAP
    would otherwise do on the raw edge cloud.
    """
    P = np.array(points, dtype=np.float64, copy=True)
    k = int(min(max(k, 3), len(P)))
    for _ in range(max(iterations, 0)):
        _, idx = cKDTree(P).query(P, k=k)
        nb = P[np.atleast_2d(idx)]
        mean = nb.mean(axis=1)
        d = nb - mean[:, None, :]
        cov = np.einsum("nki,nkj->nij", d, d) / max(k - 1, 1)
        _, evecs = np.linalg.eigh(cov)
        tangent = evecs[:, :, 2]                      # largest eigenvalue last
        delta = P - mean
        P = mean + np.einsum("ni,ni->n", delta, tangent)[:, None] * tangent
    return P


def dedupe_on_grid(points, cell):
    """Keep one point per cell -- thinning piles many points onto one spot."""
    if len(points) == 0 or cell <= EPS:
        return points
    _, first = np.unique(np.round(points / cell).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(first)]


def sharp_edges(shape, angle_deg):
    """Mesh edges whose dihedral angle exceeds `angle_deg`, plus boundary edges.

    The two faces of an edge are compared after undoing any local winding
    inconsistency (detected from the direction each face traverses the shared
    edge), so the angle is a true crease angle even on the loosely authored
    meshes that ShapeNet is full of.  Edges used by a single face -- borders of
    open sheets and rims of holes -- always count as sharp.
    """
    f = shape.faces
    raw = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    forward = raw[:, 0] < raw[:, 1]                   # traversal direction
    key = np.sort(raw, axis=1)
    uniq, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = np.ravel(inv)                               # NumPy 2.0 returns a column

    tri = shape.points[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), EPS)
    face_of = np.tile(np.arange(len(f)), 3)

    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    faces_sorted = face_of[order]
    fwd_sorted = forward[order]

    boundary = uniq[counts == 1]

    two = np.flatnonzero(counts == 2)
    creased = np.empty((0, 2), dtype=np.int64)
    if len(two):
        s = starts[two]
        a, b = faces_sorted[s], faces_sorted[s + 1]
        n_a, n_b = fn[a], fn[b]
        same_winding = fwd_sorted[s] == fwd_sorted[s + 1]
        n_b = np.where(same_winding[:, None], -n_b, n_b)   # re-orient neighbour
        cos = np.clip(np.einsum("ij,ij->i", n_a, n_b), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos))
        creased = uniq[two[angle >= angle_deg]]

    return np.vstack([boundary, creased]) if len(creased) or len(boundary) else \
        np.empty((0, 2), dtype=np.int64)


def sample_along_edges(points, edges, n_target, min_spacing):
    """Resample a set of segments into a point cloud of roughly `n_target` points."""
    if len(edges) == 0:
        return np.empty((0, 3))
    p0, p1 = points[edges[:, 0]], points[edges[:, 1]]
    seg = p1 - p0
    lengths = np.linalg.norm(seg, axis=1)
    keep = lengths > EPS
    p0, seg, lengths = p0[keep], seg[keep], lengths[keep]
    if len(lengths) == 0:
        return np.empty((0, 3))

    spacing = max(float(lengths.sum()) / max(n_target, 1), min_spacing)
    steps = np.maximum(1, np.floor(lengths / spacing).astype(np.int64))
    per_edge = steps + 1                              # both endpoints included
    e = np.repeat(np.arange(len(lengths)), per_edge)
    offset = np.arange(len(e)) - np.repeat(np.concatenate([[0], np.cumsum(per_edge)[:-1]]),
                                           per_edge)
    t = (offset / steps[e])[:, None]
    pts = p0[e] + t * seg[e]

    # Vertices shared by several edges are sampled once per edge: snap to a
    # grid at a fraction of the spacing and drop the duplicates.
    cell = max(spacing * 0.35, EPS)
    _, first = np.unique(np.round(pts / cell).astype(np.int64), axis=0, return_index=True)
    return pts[np.sort(first)]


def extract_salient_points(shape, surface, params, rng, verbose=True):
    """Salient point cloud: sharp edges for meshes, high curvature for clouds.

    Paper stage: "extracting the salient point cloud using Sharp Edge Sampling
    (SES) and a curvature threshold of 15".  Dora's SES is a learning-free
    geometric operator that draws extra samples along sharp features; the mesh
    branch here reproduces that directly from the dihedral angles, while the
    point-cloud branch (no faces, no edges) thresholds local normal variation,
    which is the same 15 degree criterion applied to a PCA estimate.

    Curvature is measured on the densest cloud available -- the input points
    themselves for a point cloud, the uniform surface sample for a mesh --
    because PCA normals on a sparse sample are too noisy to separate a crease
    from a flat face.
    """
    mode = params.salient_mode
    if mode == "auto":
        mode = "sharp" if shape.is_mesh else "curvature"
    if mode in ("sharp", "both") and not shape.is_mesh:
        mode = "curvature"

    parts = []
    if mode in ("sharp", "both"):
        edges = sharp_edges(shape, params.curvature_threshold)
        pts = sample_along_edges(shape.points, edges, params.n_salient,
                                 params.resample_spacing * SHAPE_SIZE * 0.5)
        if verbose:
            print("[salient] {} sharp/boundary edges -> {} points".format(len(edges), len(pts)))
        parts.append(pts)

    thin = sum(len(p) for p in parts) < params.n_surface // 8
    if mode in ("curvature", "both") or thin:
        cloud, given = surface, None
        if not shape.is_mesh:
            cloud = shape.points
            given = shape.normals
            if len(cloud) > params.max_cloud:
                pick = rng.choice(len(cloud), params.max_cloud, replace=False)
                cloud = cloud[pick]
                given = None if given is None else given[pick]
        normals, idx = estimate_normals(cloud, params.normal_knn)
        if given is not None and len(given) == len(cloud):
            norm = np.linalg.norm(given, axis=1, keepdims=True)   # prefer file normals
            normals = given / np.maximum(norm, EPS)
        score = normal_variation(normals, idx)
        mask = score >= params.curvature_threshold
        if mask.sum() < max(64, len(cloud) // 50):    # smooth shape: keep the top 5%
            mask = score >= np.percentile(score, 95)
        pts = thin_to_curves(cloud[mask], params.thin_knn, params.thin_iterations)
        pts = dedupe_on_grid(pts, params.resample_spacing * SHAPE_SIZE * 0.5)
        if verbose:
            print("[salient] curvature >= {:g} deg keeps {}/{} points -> {} after thinning"
                  .format(params.curvature_threshold, int(mask.sum()), len(cloud), len(pts)))
        parts.append(pts)

    salient = np.vstack([p for p in parts if len(p)]) if any(len(p) for p in parts) else \
        np.empty((0, 3))
    if len(salient) == 0:
        raise RuntimeError("no salient points found; try --salient-mode curvature "
                           "or a lower --curvature-threshold")
    return salient


# --------------------------------------------------------------------------- #
# Stage 3 -- stroke recovery (a stand-in for EMAP)
# --------------------------------------------------------------------------- #


def build_curve_graph(points, knn, max_edge_factor):
    """Symmetric kNN proximity graph with long edges removed.

    The cap (a multiple of the median nearest-neighbour distance) keeps the
    spanning tree from bridging unrelated feature lines.
    """
    n = len(points)
    k = int(min(knn + 1, n))
    tree = cKDTree(points)
    dist, idx = tree.query(points, k=k)
    dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)

    nn = dist[:, 1] if k > 1 else np.array([1.0])
    median = float(np.median(nn[nn > EPS])) if np.any(nn > EPS) else 1.0
    max_len = max_edge_factor * median

    rows = np.repeat(np.arange(n), k - 1)
    cols = idx[:, 1:].ravel()
    vals = dist[:, 1:].ravel()
    keep = (vals > EPS) & (vals <= max_len)
    graph = coo_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n)).tocsr()
    return graph.maximum(graph.T), median            # union of both directions


def spanning_forest_adjacency(graph):
    """Adjacency lists of the minimum spanning forest of `graph`."""
    mst = minimum_spanning_tree(graph).tocoo()
    adj = [set() for _ in range(graph.shape[0])]
    for i, j in zip(mst.row, mst.col):
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _walk_branch(adj, start, first):
    """Follow degree-2 nodes from `start` through `first` to the next junction."""
    path = [start, first]
    prev, cur = start, first
    while len(adj[cur]) == 2:
        nxt = next(iter(adj[cur] - {prev}))
        if nxt == start:                              # closed loop
            path.append(nxt)
            break
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def prune_short_branches(points, adj, min_length, rounds=3):
    """Drop the hairs an MST grows on a noisy point cloud."""
    for _ in range(rounds):
        removed = 0
        for leaf in [i for i, a in enumerate(adj) if len(a) == 1]:
            if len(adj[leaf]) != 1:                   # already pruned this round
                continue
            path = _walk_branch(adj, leaf, next(iter(adj[leaf])))
            if len(adj[path[-1]]) < 3:                # a whole isolated component
                continue
            seg = points[path]
            if float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()) >= min_length:
                continue
            for a, b in zip(path[:-1], path[1:]):
                adj[a].discard(b)
                adj[b].discard(a)
            removed += 1
        if removed == 0:
            break
    return adj


def adjacency_to_polylines(points, adj):
    """Split a spanning forest into polylines running junction to junction."""
    seen = set()

    def mark(a, b):
        seen.add((a, b) if a < b else (b, a))

    def is_seen(a, b):
        return ((a, b) if a < b else (b, a)) in seen

    polylines = []
    for node in [i for i, a in enumerate(adj) if len(a) != 2 and len(a) > 0]:
        for nb in list(adj[node]):
            if is_seen(node, nb):
                continue
            path = _walk_branch(adj, node, nb)
            for a, b in zip(path[:-1], path[1:]):
                mark(a, b)
            polylines.append(points[path])

    for node in range(len(adj)):                      # leftover closed loops
        if len(adj[node]) != 2:
            continue
        nb = next((x for x in adj[node] if not is_seen(node, x)), None)
        if nb is None:
            continue
        path = _walk_branch(adj, node, nb)
        for a, b in zip(path[:-1], path[1:]):
            mark(a, b)
        polylines.append(points[path])

    return polylines


def smooth_polyline(P, passes, weight=0.5):
    """Laplacian smoothing with the endpoints pinned.

    A spanning tree walks its points in a slightly ragged order; a couple of
    passes take that staircase out before the spline fit, without moving a real
    corner by more than about one sample spacing.
    """
    if passes <= 0 or len(P) < 3:
        return P
    Q = np.array(P, dtype=np.float64, copy=True)
    for _ in range(passes):
        Q[1:-1] += weight * (0.5 * (Q[:-2] + Q[2:]) - Q[1:-1])
    return Q


def _chord_parameters(P):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    return u / u[-1] if u[-1] > EPS else np.linspace(0.0, 1.0, len(P))


def _fit_quadratic(P, u):
    """Least-squares quadratic Bezier through fixed endpoints (paper: degree 2)."""
    b0, b2 = P[0], P[-1]
    w = 2.0 * u * (1.0 - u)
    residual = P - ((1.0 - u) ** 2)[:, None] * b0 - (u ** 2)[:, None] * b2
    denom = float((w * w).sum())
    b1 = (w[:, None] * residual).sum(axis=0) / denom if denom > EPS else 0.5 * (b0 + b2)
    return np.stack([b0, b1, b2])


def _bezier_eval(ctrl, t):
    t = np.asarray(t, dtype=np.float64)[:, None]
    return ((1 - t) ** 2) * ctrl[0] + 2 * t * (1 - t) * ctrl[1] + (t ** 2) * ctrl[2]


def fit_bezier_spline(P, tol, depth=0, max_depth=12):
    """Piecewise quadratic Bezier fit, subdividing at the worst-fitting point."""
    if len(P) < 3:
        return [np.stack([P[0], 0.5 * (P[0] + P[-1]), P[-1]])]
    u = _chord_parameters(P)
    ctrl = _fit_quadratic(P, u)
    err = np.linalg.norm(P - _bezier_eval(ctrl, u), axis=1)
    if err.max() <= tol or depth >= max_depth or len(P) <= 4:
        return [ctrl]
    split = int(np.clip(int(err.argmax()), 1, len(P) - 2))
    return (fit_bezier_spline(P[:split + 1], tol, depth + 1, max_depth) +
            fit_bezier_spline(P[split:], tol, depth + 1, max_depth))


def sample_spline(segments, spacing):
    """Walk a spline at a roughly constant step, without repeating the joints."""
    out = []
    for k, ctrl in enumerate(segments):
        probe = _bezier_eval(ctrl, np.linspace(0.0, 1.0, 16))
        length = float(np.linalg.norm(np.diff(probe, axis=0), axis=1).sum())
        n = max(2, int(np.ceil(length / max(spacing, EPS))) + 1)
        pts = _bezier_eval(ctrl, np.linspace(0.0, 1.0, n))
        out.append(pts if k == 0 else pts[1:])
    return np.vstack(out)


def recover_strokes(points, params, verbose=True):
    """Salient points -> spline strokes.

    Paper stage: "fit Bezier splines to the salient point cloud using EMAP with
    a maximum degree of 2 and minimum segment length of 12".  EMAP is a learned
    edge reconstructor; the substitute here recovers the curve network
    combinatorially -- a capped kNN graph, its minimum spanning forest, and a
    junction-to-junction decomposition -- and then fits the same degree-2
    splines to each recovered polyline.
    """
    graph, median = build_curve_graph(points, params.knn, params.max_edge_factor)
    adj = spanning_forest_adjacency(graph)
    adj = prune_short_branches(points, adj, params.prune_branch * SHAPE_SIZE)
    polylines = adjacency_to_polylines(points, adj)
    if verbose:
        print("[strokes] curve graph: median spacing {:.4f}, {} raw polylines".format(
            median, len(polylines)))

    strokes = []
    for poly in polylines:
        if len(poly) < params.min_segment_points:     # EMAP's minimum segment length
            continue
        poly = smooth_polyline(poly, params.smooth_passes)
        segments = fit_bezier_spline(poly, params.bezier_error * SHAPE_SIZE)
        strokes.append(sample_spline(segments, params.resample_spacing * SHAPE_SIZE))
    if verbose:
        print("[strokes] {} splines of at least {} points".format(
            len(strokes), params.min_segment_points))
    return strokes


# --------------------------------------------------------------------------- #
# Stage 4 -- culling and merging (after Liu et al., StrokeAggregator)
# --------------------------------------------------------------------------- #


def cull_collinear(P, cos_threshold):
    """Drop points that add no turn, measured from the last point kept.

    Paper: "remove redundant points in near-linear segments with a cosine
    distance threshold of 0.04".  Measuring the incoming direction from the
    last *kept* point rather than the previous one stops a slow curve from
    being flattened away one negligible turn at a time.
    """
    if len(P) <= 2:
        return P
    keep = [0]
    for i in range(1, len(P) - 1):
        d1 = P[i] - P[keep[-1]]
        d2 = P[i + 1] - P[i]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < EPS or n2 < EPS:
            continue
        cosine_distance = 1.0 - float(np.dot(d1, d2) / (n1 * n2))
        if cosine_distance >= cos_threshold:
            keep.append(i)
    keep.append(len(P) - 1)
    return P[keep]


def merge_strokes(strokes, radius):
    """Concatenate strokes whose endpoints nearly touch (paper: within 2%)."""
    strokes = [s for s in strokes if len(s) >= 2]
    while True:
        if len(strokes) < 2:
            return strokes
        ends = np.stack([np.stack([s[0], s[-1]]) for s in strokes]).reshape(-1, 3)
        tree = cKDTree(ends)
        pairs = sorted(tree.query_pairs(radius),
                       key=lambda p: float(np.linalg.norm(ends[p[0]] - ends[p[1]])))
        used, merged = set(), []
        for a, b in pairs:
            sa, ea, sb, eb = a // 2, a % 2, b // 2, b % 2
            if sa == sb or sa in used or sb in used:
                continue
            A = strokes[sa][::-1] if ea == 0 else strokes[sa]    # tail at the joint
            B = strokes[sb][::-1] if eb == 1 else strokes[sb]    # head at the joint
            if np.linalg.norm(A[-1] - B[0]) < EPS:
                B = B[1:]
            merged.append(np.vstack([A, B]))
            used.update((sa, sb))
        if not merged:
            return strokes
        strokes = [s for i, s in enumerate(strokes) if i not in used] + merged


def _arc_length(P):
    return float(np.linalg.norm(np.diff(np.asarray(P), axis=0), axis=1).sum())


def cull_and_merge(strokes, params, verbose=True):
    """Simplify each stroke, join the ones that meet, drop what is left over.

    Culling comes first (as in the paper), so the point counts here are the
    ones a VR player would replay; the final filter is on stroke *length*,
    since a culled straight edge is a perfectly good two-point stroke.
    """
    culled = [cull_collinear(s, params.cull_cosine) for s in strokes]
    merged = merge_strokes(culled, params.merge_radius * SHAPE_SIZE)
    floor = params.prune_branch * SHAPE_SIZE
    kept = [s for s in merged if _arc_length(s) >= floor]
    if not kept:                                      # never return nothing
        kept = sorted(merged, key=_arc_length, reverse=True)[:max(1, len(merged) // 4)]
    if verbose:
        print("[cull] {} strokes -> {} merged -> {} longer than {:.3f}".format(
            len(strokes), len(merged), len(kept), floor))
    return kept


# --------------------------------------------------------------------------- #
# Stage 5 -- ordering
# --------------------------------------------------------------------------- #


def order_strokes(strokes, params, rng, verbose=True):
    """Sequence the strokes into a plausible drawing order.

    Paper: "connect stroke endpoints based on spatial proximity and perform a
    depth-first traversal of the resulting connectivity graph.  We introduce
    stochasticity by skipping nearest connections with a probability of 10%".
    The traversal starts from the longest stroke -- the closest thing to the
    big outline an artist lays down first -- and each stroke is flipped so it
    starts at the end nearest the previous stroke, so point order follows the
    hand as well.
    """
    n = len(strokes)
    if n <= 1:
        return list(strokes)

    ends = np.stack([np.stack([s[0], s[-1]]) for s in strokes])       # (n, 2, 3)
    flat = ends.reshape(-1, 3)
    tree = cKDTree(flat)
    k = int(min(2 * params.order_knn + 2, len(flat)))

    neighbours = [dict() for _ in range(n)]           # stroke -> {other: distance}
    dist, idx = tree.query(flat, k=k)
    for e in range(len(flat)):
        src = e // 2
        for d, j in zip(np.atleast_1d(dist[e]), np.atleast_1d(idx[e])):
            dst = int(j) // 2
            if dst == src:
                continue
            if d < neighbours[src].get(dst, np.inf):
                neighbours[src][dst] = float(d)
                neighbours[dst][src] = float(d)

    lengths = np.array([float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum())
                        for s in strokes])
    current = int(lengths.argmax())
    visited = {current}
    order = [current]
    stack = [current]
    skipped = 0

    def nearest_unvisited(from_stroke):
        free = [i for i in range(n) if i not in visited]
        if not free:
            return None
        d = [float(np.linalg.norm(ends[i][:, None] - ends[from_stroke][None], axis=-1).min())
             for i in free]
        return free[int(np.argmin(d))]

    while len(order) < n:
        candidates = sorted(((d, i) for i, d in neighbours[stack[-1]].items()
                             if i not in visited), key=lambda x: x[0])
        if candidates and len(candidates) > 1 and rng.random() < params.skip_prob:
            candidates = candidates[1:]               # skip the nearest connection
            skipped += 1
        if candidates:
            nxt = candidates[0][1]
        else:
            stack.pop()
            if stack:
                continue
            nxt = nearest_unvisited(order[-1])
            if nxt is None:
                break
        visited.add(nxt)
        order.append(nxt)
        stack.append(nxt)

    ordered = []
    cursor = None
    for i in order:
        s = strokes[i]
        if cursor is not None and np.linalg.norm(s[-1] - cursor) < np.linalg.norm(s[0] - cursor):
            s = s[::-1]
        ordered.append(s)
        cursor = s[-1]
    if verbose:
        print("[order] depth-first over {} strokes, {} nearest connections skipped".format(
            n, skipped))
    return ordered


# --------------------------------------------------------------------------- #
# Latk output
# --------------------------------------------------------------------------- #


def _hsv_to_rgb(h, s, v):
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def stroke_colors(n, mode, rng):
    """One RGBA per stroke.  `order` ramps blue -> red along the drawing order."""
    if mode == "white":
        return [(1.0, 1.0, 1.0, 1.0)] * n
    if mode == "random":
        return [tuple(_hsv_to_rgb(float(rng.random()), 0.85, 1.0)) + (1.0,) for _ in range(n)]
    out = []
    for i in range(n):
        t = i / max(n - 1, 1)
        r, g, b = _hsv_to_rgb(0.66 * (1.0 - t), 0.9, 1.0)
        out.append((r, g, b, 1.0))
    return out


def strokes_to_latk(strokes, colors, up="y", layer_name="sketch", precision=6):
    """Build the Latk document (see docs/LATK.md).

    All strokes go into layer 0, frame 0 -- a static drawing -- in traversal
    order, so a Latk player replays them in the order the pipeline "drew" them.
    Latk JSON is Y-up; pass `up="z"` for Z-up (Blender-style) input.
    """
    latk_strokes = []
    for stroke, color in zip(strokes, colors):
        pts = []
        for co in np.asarray(stroke, dtype=np.float64):
            x, y, z = (float(co[0]), float(co[2]), float(co[1])) if up == "z" else \
                      (float(co[0]), float(co[1]), float(co[2]))
            pts.append({
                "co": [round(x, precision), round(y, precision), round(z, precision)],
                "pressure": 1.0,
                "strength": 1.0,
                "vertex_color": [round(c, 6) for c in color],
            })
        latk_strokes.append({
            "color": [round(c, 6) for c in color],
            "fill_color": [0.0, 0.0, 0.0, 0.0],
            "points": pts,
        })
    return {
        "creator": "shape-to-sketch.py",
        "version": 2.9,
        "grease_pencil": [
            {"layers": [{"name": layer_name, "frames": [{"strokes": latk_strokes}]}]}
        ],
    }


def write_latk(path, document, zipped=None):
    """Write `.latk` (a zipped .json, the default) or a plain `.json`."""
    text = json.dumps(document, indent=1)
    if zipped is None:
        zipped = os.path.splitext(path)[1].lower() in (".latk", ".zip")
    if not zipped:
        with open(path, "w") as f:
            f.write(text)
        return
    inner = os.path.splitext(os.path.basename(path))[0] + ".json"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner, text)


def write_lines_json(path, strokes):
    """The flat list-of-strokes JSON read by `Sketch2ShapeDataset.read_syn_lines`."""
    with open(path, "w") as f:
        json.dump([np.asarray(s, dtype=np.float64).tolist() for s in strokes], f)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def shape_to_sketch(shape, params=None, verbose=True):
    """Run the five stages end to end.

    Returns `(strokes, (centroid, scale))`, where `strokes` is an ordered list
    of (n, 3) arrays in the unit-sphere frame, and the centroid/scale pair is
    what `normalize_to_unit_sphere` removed.
    """
    params = params or SketchParams()
    rng = np.random.default_rng(params.seed)

    centroid, scale = normalize_to_unit_sphere(shape)
    if verbose:
        print("[normalize] centroid {} scale {:.6f}".format(np.round(centroid, 4), scale))

    surface = sample_surface(shape, params.n_surface, rng)
    if verbose:
        print("[surface] {} uniform samples".format(len(surface)))

    salient = extract_salient_points(shape, surface, params, rng, verbose)
    if verbose:
        print("[salient] {} points total".format(len(salient)))

    strokes = recover_strokes(salient, params, verbose)
    strokes = cull_and_merge(strokes, params, verbose)
    if not strokes:
        raise RuntimeError(
            "no strokes recovered from {} salient points; try a lower "
            "--min-segment-points, a smaller --prune-branch, or a larger --n-salient"
            .format(len(salient)))
    strokes = order_strokes(strokes, params, rng, verbose)

    if verbose:
        total = sum(len(s) for s in strokes)
        print("[done] {} strokes, {} points, {:.1f} points/stroke".format(
            len(strokes), total, total / max(len(strokes), 1)))
    return strokes, (centroid, scale)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate a sequential VR sketch (Latk brushstrokes) from a "
                    "mesh, point cloud, or SDF grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d = SketchParams()

    p.add_argument("-i", "--input", required=True, help="mesh / point cloud / SDF file")
    p.add_argument("-o", "--output", default=None,
                   help="output .latk (zipped) or .json (plain); default: <input>.latk")
    p.add_argument("--json-out", default=None,
                   help="also write the flat stroke list read by read_syn_lines()")
    p.add_argument("--color", choices=("order", "random", "white"), default="order",
                   help="stroke color: 'order' ramps blue to red along drawing order")
    p.add_argument("--up", choices=("y", "z"), default="y",
                   help="up axis of the input; Latk JSON is written Y-up")
    p.add_argument("--denormalize", action="store_true",
                   help="write strokes in the input's original position and scale")
    p.add_argument("--sdf-level", type=float, default=0.02,
                   help="iso-level for marching cubes on SDF input")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")

    g = p.add_argument_group("pipeline (defaults follow Sec. 3.2 of the paper)")
    g.add_argument("--n-surface", type=int, default=d.n_surface)
    g.add_argument("--curvature-threshold", type=float, default=d.curvature_threshold,
                   help="degrees; sharp-edge / normal-variation cutoff")
    g.add_argument("--salient-mode", choices=("auto", "sharp", "curvature", "both"),
                   default=d.salient_mode)
    g.add_argument("--n-salient", type=int, default=d.n_salient)
    g.add_argument("--normal-knn", type=int, default=d.normal_knn)
    g.add_argument("--max-cloud", type=int, default=d.max_cloud)
    g.add_argument("--thin-knn", type=int, default=d.thin_knn)
    g.add_argument("--thin-iterations", type=int, default=d.thin_iterations)
    g.add_argument("--knn", type=int, default=d.knn)
    g.add_argument("--max-edge-factor", type=float, default=d.max_edge_factor)
    g.add_argument("--prune-branch", type=float, default=d.prune_branch)
    g.add_argument("--smooth-passes", type=int, default=d.smooth_passes)
    g.add_argument("--bezier-error", type=float, default=d.bezier_error)
    g.add_argument("--resample-spacing", type=float, default=d.resample_spacing)
    g.add_argument("--min-segment-points", type=int, default=d.min_segment_points,
                   help="shortest point run that is fit as a stroke")
    g.add_argument("--cull-cosine", type=float, default=d.cull_cosine)
    g.add_argument("--merge-radius", type=float, default=d.merge_radius)
    g.add_argument("--order-knn", type=int, default=d.order_knn)
    g.add_argument("--skip-prob", type=float, default=d.skip_prob)
    g.add_argument("--seed", type=int, default=d.seed)

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    verbose = not args.quiet

    params = SketchParams(**{f: getattr(args, f) for f in SketchParams.__dataclass_fields__})

    try:
        shape = load_shape(args.input, sdf_level=args.sdf_level, verbose=verbose)
        strokes, (centroid, scale) = shape_to_sketch(shape, params, verbose=verbose)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as err:
        print("shape-to-sketch: error: {}".format(err), file=sys.stderr)
        return 2

    if args.denormalize:
        strokes = [np.asarray(s) * scale + centroid for s in strokes]

    out = args.output or os.path.splitext(args.input)[0] + ".latk"
    rng = np.random.default_rng(params.seed + 1)
    document = strokes_to_latk(strokes, stroke_colors(len(strokes), args.color, rng),
                               up=args.up,
                               layer_name=os.path.splitext(os.path.basename(args.input))[0])
    write_latk(out, document)
    if verbose:
        print("[write] {}".format(out))

    if args.json_out:
        write_lines_json(args.json_out, strokes)
        if verbose:
            print("[write] {}".format(args.json_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
