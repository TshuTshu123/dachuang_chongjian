"""Bounded bilateral face-normal filtering and raw-observation support maps.

Traditional local filtering; not an exact reproduction of a specific paper.
Inputs/exports are millimeters. Algorithms run in meters. Face connectivity is
unchanged, boundary/nonmanifold vertices are fixed, and each vertex has a total
displacement cap. Support colors are diagnostics, never accuracy probabilities.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from reconstruct_human import geometry_report, transfer_colors


def bilateral_mesh_filter(mesh, normal_iterations=5, vertex_iterations=12,
                          max_move=.0015, sigma_normal=.45, workers=4, active_weights=None):
    if normal_iterations < 1 or vertex_iterations < 1 or max_move <= 0 or not np.isfinite(max_move):
        raise ValueError('Positive iterations and finite max_move > 0 required')
    if sigma_normal <= 0 or not np.isfinite(sigma_normal):
        raise ValueError('sigma_normal must be finite and positive')
    result = copy.deepcopy(mesh)
    original = np.asarray(mesh.vertices).copy()
    active = np.ones(len(original)) if active_weights is None else np.asarray(active_weights, dtype=np.float64)
    if active.shape != (len(original),) or not np.isfinite(active).all() or np.any((active < 0) | (active > 1)):
        raise ValueError('active_weights must have one finite weight in [0,1] per vertex')
    triangles = np.asarray(mesh.triangles)
    if not len(triangles) or not np.isfinite(original).all():
        raise ValueError('A nonempty finite triangle mesh is required')
    face_points = original[triangles]
    centers = face_points.mean(1)
    cross = np.cross(face_points[:, 1] - face_points[:, 0], face_points[:, 2] - face_points[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    if np.any(area2 < 1e-20):
        raise ValueError('Remove degenerate triangles before bilateral filtering')
    guidance = cross / area2[:, None]
    # Incidence restricts normal communication to the actual surface topology,
    # avoiding Euclidean shortcuts across lips, nearby hair, or fingers.
    incidence = csr_matrix((np.ones(triangles.size),
                            (triangles.ravel(), np.repeat(np.arange(len(triangles)), 3))),
                           shape=(len(original), len(triangles)))
    adjacency = (incidence.T @ incidence).tocoo()
    row, col = adjacency.row, adjacency.col
    distance2 = np.sum((centers[row] - centers[col]) ** 2, axis=1)
    positive = distance2[distance2 > 0]
    sigma_space2 = max(float(np.median(positive)) * 4 if len(positive) else 0, 1e-12)
    cosine = np.einsum('ij,ij->i', guidance[row], guidance[col])
    weights = (np.exp(-distance2 / (2 * sigma_space2))
               * np.exp(-(1 - np.clip(cosine, -1, 1)) / sigma_normal ** 2)
               * area2[col] * (cosine >= .5))
    # Frozen guidance retains sharp normal discontinuities throughout filtering.
    weight_sum = np.maximum(np.bincount(row, weights=weights, minlength=len(triangles)), 1e-30)
    normal = guidance.copy()
    for _ in range(normal_iterations):
        filtered = np.column_stack([np.bincount(row, weights=weights * normal[col, axis],
                                                minlength=len(triangles)) for axis in range(3)])
        filtered /= weight_sum[:, None]
        filtered /= np.maximum(np.linalg.norm(filtered, axis=1, keepdims=True), 1e-30)
        normal = filtered
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    fixed = np.zeros(len(original), dtype=bool)
    fixed[np.unique(unique_edges[counts != 2])] = True
    degree = np.asarray(incidence.sum(axis=1)).ravel()
    current = original.copy()
    prevented_flips = 0
    locally_blocked_vertices = np.zeros(len(original), dtype=bool)
    for _ in range(vertex_iterations):
        points = current[triangles]
        face_center = points.mean(1)
        height = np.einsum('fkj,fj->fk', face_center[:, None, :] - points, normal)
        updates = height[:, :, None] * normal[:, None, :]
        delta = np.column_stack([np.bincount(triangles.ravel(), weights=updates[:, :, axis].ravel(),
                                            minlength=len(original)) for axis in range(3)])
        delta /= np.maximum(degree[:, None], 1)
        displacement = current + .5 * delta * active[:, None] - original
        displacement *= np.minimum(1, max_move * active / np.maximum(np.linalg.norm(displacement, axis=1), 1e-30))[:, None]
        candidate = original + displacement
        candidate[fixed] = original[fixed]
        # Block vertices around threatened triangles locally. A tiny damaged
        # face must not reduce the step for an entire otherwise healthy head.
        # This does not test intersections between distant triangles.
        for attempt in range(12):
            p = candidate[triangles]
            new_cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            bad = np.einsum('ij,ij->i', new_cross, guidance) <= 1e-20
            if not bad.any():
                break
            prevented_flips += int(bad.sum())
            blocked = np.unique(triangles[bad])
            locally_blocked_vertices[blocked] = True
            candidate[blocked] = current[blocked]
        else:
            # If freezing propagated across a complex neighborhood, backtrack
            # the remaining step conservatively, reverting only as a last resort.
            for _ in range(12):
                candidate = (candidate + current) / 2
                p = candidate[triangles]
                cross_check = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
                if np.all(np.einsum('ij,ij->i', cross_check, guidance) > 1e-20):
                    break
            else:
                candidate = current.copy()
        current = candidate
    result.vertices = o3d.utility.Vector3dVector(current)
    result.compute_vertex_normals()
    movement = np.linalg.norm(current - original, axis=1) * 1000
    return result, {'method': 'topology-local bilateral normals + bounded plane projection',
                    'normal_iterations': normal_iterations, 'vertex_iterations': vertex_iterations,
                    'sigma_normal': sigma_normal, 'frozen_boundary_or_nonmanifold_vertices': int(fixed.sum()),
                    'active_vertices': int((active > 0).sum()),
                    'total_displacement_cap_mm': max_move * 1000,
                    'movement_median_mm': float(np.median(movement)),
                    'movement_p95_mm': float(np.quantile(movement, .95)),
                    'movement_max_mm': float(movement.max()), 'prevented_inverted_face_candidates': prevented_flips,
                    'locally_blocked_vertices': int(locally_blocked_vertices.sum())}


def observation_support(mesh, source_points, workers=4, batch_size=8192):
    """Red: insufficient nearby observations; amber: broad raw normal spread.

    Green means measured local support under these thresholds, not correct shape.
    Neighborhood spread includes real curvature as well as measurement noise.
    """
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    source_points = np.asarray(source_points)
    if len(source_points) < 6:
        raise ValueError('At least six source points are required')
    tree = cKDTree(source_points)
    nearest = np.empty(len(vertices))
    count = np.empty(len(vertices), dtype=np.int32)
    spread = np.empty(len(vertices))
    for start in range(0, len(vertices), batch_size):
        stop = min(start + batch_size, len(vertices))
        d, index = tree.query(vertices[start:stop], k=min(32, len(source_points)), workers=workers)
        nearest[start:stop] = d[:, 0]
        inside = d <= .008
        count[start:stop] = inside.sum(1)
        residual = np.einsum('nkj,nj->nk', source_points[index] - vertices[start:stop, None, :], normals[start:stop])
        # Include at least the closest point to avoid all-NaN unsupported rows.
        inside[:, 0] = True
        masked = np.where(inside, residual, np.nan)
        median = np.nanmedian(masked, axis=1)
        spread[start:stop] = 1.4826 * np.nanmedian(np.abs(masked - median[:, None]), axis=1)
    red = (nearest > .010) | (count < 6)
    amber = ~red & ((spread > .0015) | (count < 12))
    labels = np.where(red, 2, np.where(amber, 1, 0)).astype(np.uint8)
    colors = np.array([[.15, .65, .35], [1., .65, .12], [.85, .12, .12]])[labels]
    report = {'red_insufficient_support_vertices': int(red.sum()),
              'amber_broad_or_sparse_support_vertices': int(amber.sum()),
              'green_local_support_vertices': int((labels == 0).sum()),
              'nearest_raw_distance_p95_mm': float(np.quantile(nearest, .95) * 1000),
              'local_normal_spread_median_mm': float(np.median(spread) * 1000),
              'thresholds': {'neighborhood_radius_mm': 8, 'min_neighbors': 6,
                              'unsupported_nearest_distance_mm': 10, 'amber_spread_mm': 1.5},
              'note': 'Colors show observation support, not accuracy probabilities. Spread includes real surface curvature.'}
    return colors, labels, nearest, spread, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--normal-iterations', type=int, default=5)
    parser.add_argument('--vertex-iterations', type=int, default=12)
    parser.add_argument('--max-move-mm', type=float, default=1.5)
    parser.add_argument('--sigma-normal', type=float, default=.45)
    parser.add_argument('--active-min-x-mm', type=float,
                        help='Optional X-axis cutoff for these scans (body vertical is X); below cutoff unchanged.')
    parser.add_argument('--feather-mm', type=float, default=30,
                        help='Transition above the X cutoff; default 30 mm.')
    args = parser.parse_args()
    if args.mesh.resolve().parent == args.out.resolve():
        parser.error('Use a separate output folder to preserve input artifacts')
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    mesh.scale(.001, center=np.zeros(3))
    mesh.compute_vertex_normals()
    source = o3d.io.read_point_cloud(str(args.source))
    source.remove_non_finite_points()
    source.scale(.001, center=np.zeros(3))
    active = None
    if args.active_min_x_mm is not None:
        if not np.isfinite(args.active_min_x_mm) or not np.isfinite(args.feather_mm) or args.feather_mm <= 0:
            parser.error('Active X cutoff must be finite; feather-mm must be finite and positive')
        active = np.clip((np.asarray(mesh.vertices)[:, 0] * 1000 - args.active_min_x_mm) / args.feather_mm, 0, 1)
    output, report = bilateral_mesh_filter(mesh, args.normal_iterations, args.vertex_iterations,
                                            args.max_move_mm * .001, args.sigma_normal, active_weights=active)
    report['active_min_x_mm'] = args.active_min_x_mm
    report['feather_mm'] = args.feather_mm if active is not None else None
    if source.has_colors():
        transfer_colors(output, source, np.zeros(3), 1, 4)
        if active is not None and mesh.has_vertex_colors():
            colors = np.asarray(output.vertex_colors).copy()
            colors[active == 0] = np.asarray(mesh.vertex_colors)[active == 0]
            output.vertex_colors = o3d.utility.Vector3dVector(colors)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, geometry in [('before', mesh), ('after', output)]:
        colors, labels, distances, spread, support = observation_support(geometry, np.asarray(source.points))
        report[name + '_support'] = support
        report[name + '_topology'] = geometry_report(geometry, check_intersections=False)
        exported = copy.deepcopy(geometry).scale(1000, center=np.zeros(3))
        if name == 'after':
            o3d.io.write_triangle_mesh(str(args.out / 'refined_mesh.ply'), exported)
        exported.vertex_colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_triangle_mesh(str(args.out / f'{name}_support.ply'), exported)
        np.savez_compressed(args.out / f'{name}_support.npz', labels=labels,
                            nearest_distance_mm=distances * 1000, local_spread_mm=spread * 1000)
    report['input_mesh'] = str(args.mesh.resolve())
    report['source_observations'] = str(args.source.resolve())
    (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
