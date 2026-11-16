"""FLAME 面部拟合和局部重建；正式流程的纯几何实现。

输入毫米网格和同网格的语义标签。仅支持当前 X 向上、-Z 朝前的采集坐标。
固定脸部边界，保持其余原始顶点；所有候选都需经过相交和流形检查。
"""
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent


def mesh_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def load_model():
    folder = ROOT/'tools/flame'
    manifest = json.loads((folder/'download_manifest.json').read_text('utf-8'))
    path = folder/'flame2023_Open.pkl'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest['model']['sha256']:
        raise ValueError('FLAME model hash mismatch')
    with path.open('rb') as stream:
        data = pickle.load(stream, encoding='latin1')
    vertices = np.asarray(data['v_template'], float)
    unit = 1000. if np.ptp(vertices, axis=0).max() < 5 else 1.
    return vertices*unit, np.asarray(data['shapedirs'], float)*unit, np.asarray(data['f'], int), digest


def boundary_loops(faces):
    edges = np.sort(np.r_[faces[:, :2], faces[:, 1:], faces[:, [2, 0]]], axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    neighbors = {}
    for a, b in edges[counts == 1]:
        neighbors.setdefault(int(a), []).append(int(b))
        neighbors.setdefault(int(b), []).append(int(a))
    if any(len(v) != 2 for v in neighbors.values()):
        raise ValueError('Boundary is not a collection of simple loops')
    loops = []
    while neighbors:
        start = next(iter(neighbors))
        loop, previous, current = [start], None, start
        while True:
            nxt = next(v for v in neighbors[current] if v != previous)
            if nxt == start:
                break
            if nxt in loop:
                raise ValueError('Boundary loop repeats')
            loop.append(nxt)
            previous, current = current, nxt
        for v in loop:
            neighbors.pop(v)
        loops.append(np.array(loop, int))
    return loops


def make_mesh(vertices, faces, colors=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    return mesh


def detect_nose(mesh, parts):
    """Geometric initialization only; not an anatomical measurement landmark."""
    v = np.asarray(mesh.vertices)
    face = v[parts['face_neck']['vertex_indices']]
    hair = v[parts['hair']['vertex_indices']]
    top = np.quantile(hair[:, 0], .98)
    middle = np.median(face[:, 1])
    candidates = v[(v[:, 0] > top-200) & (v[:, 0] < top-100) & (abs(v[:, 1]-middle) < 25)]
    if len(candidates) < 20:
        raise ValueError('Insufficient central face surface for initialization')
    return candidates[candidates[:, 2] <= np.quantile(candidates[:, 2], .015)].mean(0)


def facial_regions(template, fitted):
    """Template region anchors, not detected anatomical measurement landmarks."""
    origin = template[int(np.argmax(template[:, 2]))]
    local = template-origin
    anchors = {}
    for name, x, y, radius_x, radius_y in [('left_eye', -30., 34., 16., 16.),
                                         ('right_eye', 30., 34., 16., 16.),
                                         ('mouth', 0., -40., 27., 13.),
                                         ('nose', 0., 0., 18., 20.)]:
        inside = (((local[:, 0]-x)/radius_x)**2+((local[:, 1]-y)/radius_y)**2 < 1) & (local[:, 2] > -55)
        if inside.sum() < 5:
            raise ValueError('Missing FLAME region support: ' + name)
        anchors[name] = fitted[inside].mean(0).tolist()
    return anchors


def fit_template(mesh, parts, template, basis, faces):
    v = np.asarray(mesh.vertices)
    nose = detect_nose(mesh, parts)
    tm = make_mesh(template, faces)
    tip_id = int(np.argmax(template[:, 2]))
    origin = template[tip_id]
    # FLAME: Y up, Z forward. Scan: X up, -Z forward (right-handed).
    frame = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    rel = template-origin
    visible = ((abs(rel[:, 0]) < 78) & (rel[:, 1] > -78) & (rel[:, 1] < 83)
               & (rel[:, 2] > -67) & (np.asarray(tm.vertex_normals)[:, 2] > .05))
    ids = np.flatnonzero(visible)
    ids = ids[np.linspace(0, len(ids)-1, min(1500, len(ids))).astype(int)]
    delta = v-nose
    region = ((delta[:, 0] > -92) & (delta[:, 0] < 100) & (abs(delta[:, 1]) < 100)
              & (delta[:, 2] > -15) & (delta[:, 2] < 110)
              & (np.asarray(mesh.vertex_normals)[:, 2] < -.1))
    # Hair is excluded even where it lies in front of the forehead.
    region[parts['hair']['vertex_indices']] = False
    target = v[region]
    if len(target) < 300 or len(ids) < 300:
        raise ValueError('Insufficient face fitting support')
    tree = cKDTree(target)
    count = min(20, basis.shape[2])
    p = np.zeros(7+count)
    lower = np.r_[[-.5, -.5, -.5], [-25., -25., -25.], np.log(.82), np.full(count, -2.5)]
    upper = np.r_[[.5, .5, .5], [25., 25., 25.], np.log(1.18), np.full(count, 2.5)]

    def evaluate(parameters, selected=None):
        x = template if selected is None else template[selected]
        b = basis[:, :, :count] if selected is None else basis[selected, :, :count]
        local = x-origin + np.einsum('vck,k->vc', b, parameters[7:])
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix() @ frame
        return np.exp(parameters[6])*(local @ rotation.T)+nose+parameters[3:6]

    history = []
    for step in range(18):
        current = evaluate(p, ids)
        distances, nearest = tree.query(current, workers=4)
        keep = distances < (30 if step < 5 else 18)
        if keep.sum() < 300:
            raise ValueError('Template initialization did not overlap the face')
        selected = ids[keep]
        matched = target[nearest[keep]]
        # Shape regularization keeps unsupported structures near the learned mean.
        def residual(parameters):
            return np.r_[(evaluate(parameters, selected)-matched).ravel(),
                         parameters[7:]*3., parameters[:3]*3., parameters[6]*25.,
                         (evaluate(parameters, [tip_id])[0]-nose)*1.5]
        if step < 5:
            def rigid(q):
                return residual(np.r_[q, np.zeros(count)])
            result = least_squares(rigid, p[:7], bounds=(lower[:7], upper[:7]),
                                   loss='soft_l1', f_scale=3., max_nfev=25)
            p[:7] = result.x
        else:
            result = least_squares(residual, p, bounds=(lower, upper),
                                   loss='soft_l1', f_scale=3., max_nfev=20)
            p = result.x
        history.append({'iteration': step, 'median_correspondence_mm': float(np.median(distances)),
                        'p90_correspondence_mm': float(np.quantile(distances, .9))})
    fitted = evaluate(p)
    detail = {'method': 'neutral FLAME shape fitted to 3D scan; no image network',
              'region_anchors': facial_regions(template, fitted),
              'shape_components': count, 'nose_initialization_mm': nose.tolist(),
              'geometric_initialization_is_measurement_landmark': False,
              'rotation_vector': p[:3].tolist(), 'translation_mm': p[3:6].tolist(),
              'scale': float(np.exp(p[6])), 'shape_coefficients': p[7:].tolist(), 'history': history}
    return fitted, nose, detail



def interior(points, polygon):
    inside = np.zeros(len(points), bool)
    x, y = points.T
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if abs(b[1]-a[1]) < 1e-12:
            continue
        hit = ((a[1] > y) != (b[1] > y)) & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
        inside ^= hit
    return inside


def simple_cut(mesh, parts, nose, radius_scale=1.):
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    d = v-nose
    inside = (((d[:, 0]-8)/(80*radius_scale))**2+(d[:, 1]/(68*radius_scale))**2 < 1) & (d[:, 2] < 85) & (d[:, 2] > -20)
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


def regional_target(xy, source_z, prior_z, anchors, strength=.9):
    """Keep broad scan-specific features without restoring high-frequency lumps."""
    residual = source_z-prior_z
    distance, neighbors = cKDTree(xy).query(xy, k=min(32, len(xy)), workers=4)
    values = residual[neighbors]
    median = np.median(values, axis=1)
    weights = np.exp(-.5*(distance/4.)**2)*(distance <= 9.)
    weights *= np.minimum(1., 3./np.maximum(abs(values-median[:, None]), 1e-9))
    low = np.sum(weights*values, axis=1)/np.maximum(weights.sum(1), 1e-12)
    # Coordinates are X-up, Y lateral. Smooth Gaussian masks avoid new seams.
    retention = np.full(len(xy), .25)
    for name, height, width, keep in [('left_eye', 18., 22., .50), ('right_eye', 18., 22., .50),
                                     ('mouth', 18., 32., .55), ('nose', 27., 21., .45)]:
        delta = xy-np.asarray(anchors[name])[:2]
        influence = np.exp(-.5*((delta[:, 0]/height)**2+(delta[:, 1]/width)**2))
        retention = np.maximum(retention, .25+(keep-.25)*influence)
    target = prior_z+retention*low+(1-strength)*(residual-low)
    return target, {'retention_low_frequency_min': float(retention.min()),
                    'retention_low_frequency_max': float(retention.max()),
                    'retention_high_frequency': 1-strength,
                    'residual_neighborhood_mm': 9.,
                    'template_region_anchors': anchors}


def patch(mesh, fitted, parts, nose, strength=.9, spacing=2., radius_scale=1., style='uniform', anchors=None):
    try:
        import triangle
    except ImportError:
        sys.path.insert(0, str(ROOT/'tools/flame/runtime'))
        import triangle
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    cut, loop = simple_cut(mesh, parts, nose, radius_scale)
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
    fade = weight*weight*(3-2*weight)
    regional = None
    if style == 'regional':
        if anchors is None:
            raise ValueError('Regional blending requires fitted template anchors')
        target, regional = regional_target(xy, source_z, prior_z, anchors, strength)
        z = source_z+fade*np.clip(target-source_z, -18.*strength, 18.*strength)
    elif style == 'uniform':
        z = source_z+fade*strength*np.clip(prior_z-source_z, -18., 18.)
    else:
        raise ValueError('Unknown face blend style: ' + style)
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
    # Verify actual winding at shared edges, not just whether a mesh could be oriented.
    directed = np.r_[out_f[:, :2], out_f[:, 1:], out_f[:, [2, 0]]]
    owners = np.tile(np.arange(len(out_f)), 3)
    unique, groups, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    orientation = np.bincount(groups, weights=np.where(directed[:, 0] < directed[:, 1], 1., -1.))
    patch_edge = np.bincount(groups, weights=(owners >= patch_start).astype(float)) > 0
    winding_conflicts = int(np.sum(patch_edge & (counts == 2) & (orientation != 0)))
    valid = (intersections == 0 and original_exact and seam_exact and edge_ok and winding_conflicts == 0
             and (not closed_before or closed_after) and (not vertex_before or vertex_after))
    detail = {'method': 'face interior remesh; source boundary fixed; FLAME depth blended inside',
              'blend_style': style, 'regional_detail': regional,
              'strength': strength, 'grid_spacing_mm': spacing, 'boundary_vertices': n, 'radius_scale': radius_scale,
              'removed_source_triangles': int(cut.sum()), 'new_patch_triangles': len(face),
              'new_patch_vertices': len(xyz)-n, 'template_missing_ray_fraction': float(missing.mean()),
              'intersections_touching_new_patch': intersections, 'source_vertices_retained_exact': original_exact,
              'seam_coordinates_exact': seam_exact, 'edge_manifold': edge_ok,
              'patch_winding_conflicts': winding_conflicts,
              'closed_before': closed_before, 'closed_after': closed_after,
              'vertex_manifold_before': vertex_before, 'vertex_manifold_after': vertex_after,
              'geometry_checks_passed': valid, 'is_measurement_accuracy_validation': False,
              'interior_depth_change_max_mm': float(np.max(abs(z[n:]-source_z[n:]))),
              'interior_depth_change_p95_mm': float(np.quantile(abs(z[n:]-source_z[n:]), .95)),
              'source_depth_gap_median_mm': float(np.median(abs(z[n:]-source_z[n:]))),
              'source_labels_reusable_without_remapping': False,
              'triangle_version': importlib.metadata.version('triangle')}
    return result, source_ids, detail

