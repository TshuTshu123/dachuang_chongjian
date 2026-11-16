"""Experimental facial patch reconstruction using a fitted FLAME depth prior.

First run flame_face_trial.py. Requires triangle in the experiment's local
runtime directory; no changes to the conda environment or production pipeline.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'outputs/experiments/face_flame/runtime'))
import triangle
from tools.mano_join import boundary_loops
from tools.flame_face_trial import make_mesh, sha, comparison
from tools.export_presentation_images import mesh_scene


def interior(points, polygon):
    inside = np.zeros(len(points), bool)
    x, y = points.T
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if abs(b[1]-a[1]) < 1e-12:
            continue
        hit = ((a[1] > y) != (b[1] > y)) & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
        inside ^= hit
    return inside


def simple_cut(mesh, parts, nose):
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    d = v-nose
    inside = (((d[:, 0]-8)/80)**2+(d[:, 1]/68)**2 < 1) & (d[:, 2] < 85) & (d[:, 2] > -20)
    inside[parts['hair']['vertex_indices']] = False
    cut = np.all(inside[f], axis=1)
    for _ in range(30):
        # Omit isolated islands and retract point-contact triangles at the seam.
        sub = make_mesh(v, f[cut])
        groups, counts, _ = sub.cluster_connected_triangles()
        own = np.flatnonzero(cut)
        keep = np.asarray(groups) == int(np.argmax(counts))
        cut[own[~keep]] = False
        removed = f[cut]
        edges = np.sort(np.concatenate([removed[:, :2], removed[:, 1:], removed[:, [2, 0]]]), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        ids, degrees = np.unique(edges[counts == 1], return_counts=True)
        bad = ids[degrees != 2]
        if not len(bad):
            loops = boundary_loops(removed)
            if len(loops) != 1:
                raise ValueError(f'Expected one facial disk; got {len(loops)} boundaries')
            loop = loops[0]
            # A simple 3D seam can still overlap in the XY height-field plane.
            # Retract such folds so constrained triangulation preserves each edge.
            xy = v[loop, :2]
            a = xy[:, None, :]
            b = np.roll(xy, -1, axis=0)[:, None, :]
            c = xy[None, :, :]
            d = np.roll(xy, -1, axis=0)[None, :, :]
            def cross2(x, y):
                return x[..., 0]*y[..., 1]-x[..., 1]*y[..., 0]
            crossing = ((cross2(b-a, c-a)*cross2(b-a, d-a) < -1e-10)
                        & (cross2(d-c, a-c)*cross2(d-c, b-c) < -1e-10))
            affected = np.flatnonzero(crossing.any(1))
            if not len(affected):
                return cut, loop
            bad = np.unique(np.r_[loop[affected], loop[(affected+1) % len(loop)]])
        cut[np.any(np.isin(f, bad), axis=1)] = False
    raise ValueError('Face boundary could not be made simple')


def cast(mesh, xy, z0):
    scene = mesh_scene(mesh)
    rays = np.c_[xy, np.full(len(xy), z0), np.tile([0., 0., 1.], (len(xy), 1))].astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)
    t = hit['t_hit'].numpy()
    return z0+t, hit


def patch(mesh, fitted, parts, nose, strength=.9, spacing=2.):
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    cut, loop = simple_cut(mesh, parts, nose)
    polygon = v[loop, :2]-nose[:2]
    lo, hi = polygon.min(0), polygon.max(0)
    xx, yy = np.meshgrid(np.arange(lo[0]+spacing/2, hi[0], spacing),
                         np.arange(lo[1]+spacing/2, hi[1], spacing))
    points = np.c_[xx.ravel(), yy.ravel()]
    points = points[interior(points, polygon)]
    # Keep the first interior grid ring away from irregular source edges.
    samples = np.concatenate([a+(b-a)*np.linspace(0., 1., max(2, int(np.linalg.norm(b-a)/.5)+1))[:, None]
                              for a, b in zip(polygon, np.roll(polygon, -1, axis=0))])
    border = cKDTree(samples)
    points = points[border.query(points)[0] > .7]
    n = len(loop)
    tri = triangle.triangulate({'vertices': np.r_[polygon, points],
                               'segments': np.c_[np.arange(n), np.roll(np.arange(n), -1)]}, 'pQY')
    xy = np.asarray(tri['vertices'])+nose[:2]
    if not np.allclose(xy[:n], v[loop, :2], atol=1e-10, rtol=0):
        raise ValueError('Triangulator changed the source seam')
    face = np.asarray(tri['triangles'], int)
    # CCW in XY points toward +Z, but the front of this scan faces -Z.
    e1, e2 = xy[face[:, 1]]-xy[face[:, 0]], xy[face[:, 2]]-xy[face[:, 0]]
    if np.median(e1[:, 0]*e2[:, 1]-e1[:, 1]*e2[:, 0]) > 0:
        face = face[:, [0, 2, 1]]
    source_z, hit = cast(mesh, xy, nose[2]-250.)
    prior_z, _ = cast(fitted, xy, nose[2]-250.)
    if not np.isfinite(source_z).all():
        raise ValueError('Source face has unsupported interior rays')
    missing = ~np.isfinite(prior_z)
    if missing.any():
        if missing.mean() > .01:
            raise ValueError('FLAME does not cover the face patch')
        prior_z[missing] = prior_z[~missing][cKDTree(xy[~missing]).query(xy[missing])[1]]
    margin = border.query(xy-nose[:2])[0]
    weight = np.clip(margin/20., 0., 1.)
    weight = weight*weight*(3-2*weight)*strength
    z = source_z + weight*np.clip(prior_z-source_z, -18., 18.)
    xyz = np.c_[xy, z]
    xyz[:n] = v[loop]
    # Interpolate original RGB without treating texture as recovered geometry.
    ids = hit['primitive_ids'].numpy().astype(int)
    uv = hit['primitive_uvs'].numpy()
    colors = np.einsum('ij,ijk->ik', np.c_[1-uv.sum(1), uv], np.asarray(mesh.vertex_colors)[f[ids]])
    colors[:n] = np.asarray(mesh.vertex_colors)[loop]
    all_v = np.r_[v, xyz[n:]]
    mapping = np.r_[loop, np.arange(len(v), len(all_v))]
    retained = f[~cut]
    all_f = np.r_[retained, mapping[face]]
    all_c = np.r_[np.asarray(mesh.vertex_colors), colors[n:]]
    used, inverse = np.unique(all_f.ravel(), return_inverse=True)
    out_v, out_f, out_c = all_v[used], inverse.reshape(-1, 3), all_c[used]
    result = make_mesh(out_v, out_f, out_c)
    source_ids = np.where(used < len(v), used, -1)
    patch_start = len(retained)
    # Check patch intersections against every nearby retained triangle.
    lo3, hi3 = xyz.min(0)-3., xyz.max(0)+3.
    p = out_v[out_f]
    near = np.flatnonzero(np.all(p.max(1) >= lo3, axis=1) & np.all(p.min(1) <= hi3, axis=1))
    compact, inv = np.unique(out_f[near].ravel(), return_inverse=True)
    local = make_mesh(out_v[compact], inv.reshape(-1, 3))
    pairs = np.asarray(local.get_self_intersecting_triangles(), dtype=int).reshape(-1, 2)
    touching_patch = np.any(near[pairs] >= patch_start, axis=1)
    intersections = int(touching_patch.sum())
    # Every source vertex which remains is preserved bit for bit.
    existing = source_ids >= 0
    original_exact = bool(np.array_equal(out_v[existing], v[source_ids[existing]]))
    seam_exact = bool(np.array_equal(xyz[:n], v[loop]))
    closed_before = mesh.is_edge_manifold(allow_boundary_edges=False)
    closed_after = result.is_edge_manifold(allow_boundary_edges=False)
    vertex_before, vertex_after = mesh.is_vertex_manifold(), result.is_vertex_manifold()
    edge_ok = result.is_edge_manifold(allow_boundary_edges=True)
    valid = intersections == 0 and original_exact and seam_exact and edge_ok and (not closed_before or closed_after) and (not vertex_before or vertex_after)
    detail = {'method': 'face interior remesh; source boundary fixed; FLAME depth blended inside',
              'strength': strength, 'grid_spacing_mm': spacing, 'boundary_vertices': n,
              'removed_source_triangles': int(cut.sum()), 'new_patch_triangles': len(face),
              'new_patch_vertices': len(xyz)-n, 'template_missing_ray_fraction': float(missing.mean()),
              'intersections_touching_new_patch': intersections, 'source_vertices_retained_exact': original_exact,
              'seam_coordinates_exact': seam_exact, 'edge_manifold': edge_ok,
              'closed_before': closed_before, 'closed_after': closed_after,
              'vertex_manifold_before': vertex_before, 'vertex_manifold_after': vertex_after,
              'geometry_checks_passed': valid, 'is_measurement_accuracy_validation': False,
              'interior_depth_change_max_mm': float(np.max(abs(z[n:]-source_z[n:]))),
              'interior_depth_change_p95_mm': float(np.quantile(abs(z[n:]-source_z[n:]), .95)),
              'source_labels_reusable_without_remapping': False,
              'triangle_version': importlib.metadata.version('triangle')}
    return result, source_ids, detail


def trial(scan):
    start = time.perf_counter()
    folder = ROOT/'outputs/experiments/face_flame'/f'export_{scan:05d}'
    base = json.loads((folder/'report.json').read_text('utf-8'))
    source = Path(base['source'])
    if sha(source) != base['source_sha256']:
        raise ValueError('Source changed since fitting')
    mesh = o3d.io.read_triangle_mesh(str(source))
    mesh.compute_vertex_normals()
    fitted = o3d.io.read_triangle_mesh(str(folder/'fitted_flame.ply'))
    fitted.compute_vertex_normals()
    parts = json.loads((source.parent/'body_parts.json').read_text('utf-8'))['parts']
    nose = np.array(base['fit']['nose_initialization_mm'])
    candidate, source_ids, report = patch(mesh, fitted, parts, nose)
    report.update(status='experiment_only', source=str(source), source_sha256=base['source_sha256'],
                  source_unchanged=sha(source) == base['source_sha256'], seconds=time.perf_counter()-start)
    if report['geometry_checks_passed']:
        path = folder/'face_patch_90.ply'
        if not o3d.io.write_triangle_mesh(str(path), candidate):
            raise IOError('Export failed')
        loaded = o3d.io.read_triangle_mesh(str(path))
        report['export_exact'] = bool(np.array_equal(np.asarray(loaded.vertices), np.asarray(candidate.vertices))
                                    and np.array_equal(np.asarray(loaded.triangles), np.asarray(candidate.triangles)))
        report.update(file=path.name, sha256=sha(path))
        np.savez_compressed(folder/'face_patch_mapping.npz', source_vertex_index=source_ids)
    # Failed candidates can appear in diagnostics, but are never exported as meshes.
    comparison([mesh, candidate], ['原扫描模型', 'FLAME 脸部重建试验' + ('' if report['geometry_checks_passed'] else '（检查未通过）')],
               nose, folder/'patch_comparison.jpg')
    (folder/'patch_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), 'utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    for number in args.scan:
        trial(number)
