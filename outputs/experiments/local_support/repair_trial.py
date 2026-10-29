"""0001 局部修复候选，仅用于当前姿态，不改默认流程和原模型。
从项目根目录执行：python outputs/experiments/local_support/repair_trial.py
鞋底在原观测最低点下 3 mm 处截断并对切口闭合；手部 ROI 是本批人工范围。
闭合底面只是缺失区域的平面近似，不是真实测量；手部最多移动 4 mm，不补造手指。
"""
from pathlib import Path
import copy
import json
import hashlib
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from reconstruct_human import geometry_report
from surface_detail import select_main_point_component

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'outputs/models/export_00001/20261029_214717_112915/export_00001_mesh.ply'


def signed2(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def triangulate(loop, vertices):
    """Ear clipping, retaining collinear boundary vertices to avoid T junctions."""
    poly = np.asarray(vertices)[loop, 1:3]
    area2 = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    direction = 1 if area2 > 0 else -1
    remaining = list(range(len(loop)))
    result = []
    while len(remaining) > 3:
        found = False
        for k, b in enumerate(remaining):
            a, c = remaining[k-1], remaining[(k+1) % len(remaining)]
            if direction * signed2(poly[a], poly[b], poly[c]) <= 1e-9:
                continue
            others = [i for i in remaining if i not in (a, b, c)]
            p = poly[others]
            ab = direction * signed2(poly[a], poly[b], p.T)
            bc = direction * signed2(poly[b], poly[c], p.T)
            ca = direction * signed2(poly[c], poly[a], p.T)
            if np.any((ab >= -1e-9) & (bc >= -1e-9) & (ca >= -1e-9)):
                continue
            result.append([loop[a], loop[b], loop[c]])
            remaining.pop(k)
            found = True
            break
        if not found:
            raise RuntimeError('Cut boundary cannot be triangulated safely; no final candidate written.')
    if abs(signed2(*poly[remaining])) <= 1e-9:
        raise RuntimeError('Degenerate final cap triangle')
    result.append([loop[i] for i in remaining])
    return result


def cut_and_cap(mesh, floor):
    v = np.asarray(mesh.vertices).copy()
    rgb = np.asarray(mesh.vertex_colors).copy()
    faces = np.asarray(mesh.triangles)
    vertices, colors = v.tolist(), rgb.tolist()
    inside = v[:, 0] >= floor
    all_in, all_out = inside[faces].all(1), (~inside[faces]).all(1)
    result = faces[all_in].tolist()
    cache = {}

    def intersection(a, b):
        edge = tuple(sorted((int(a), int(b))))
        if edge not in cache:
            ratio = (floor - v[a, 0]) / (v[b, 0] - v[a, 0])
            point = v[a] + ratio * (v[b] - v[a]); point[0] = floor
            cache[edge] = len(vertices)
            vertices.append(point.tolist())
            colors.append((rgb[a] + ratio * (rgb[b] - rgb[a])).tolist())
        return cache[edge]

    for tri in faces[~all_in & ~all_out]:
        polygon = []
        for a, b in zip(tri, np.roll(tri, -1)):
            if inside[a]:
                polygon.append(int(a))
            if inside[a] != inside[b]:
                polygon.append(intersection(a, b))
        for k in range(1, len(polygon) - 1):
            result.append([polygon[0], polygon[k], polygon[k+1]])
    f = np.asarray(result, dtype=np.int32)
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse] == 1]
    vertices_array = np.asarray(vertices)
    cut_edges = boundary[np.all(np.abs(vertices_array[boundary, 0] - floor) < 1e-7, axis=1)]
    next_vertex = {int(a): int(b) for a, b in cut_edges}
    if len(next_vertex) != len(cut_edges) or len(set(next_vertex.values())) != len(cut_edges):
        raise RuntimeError('Cut boundary is not a set of simple oriented loops')
    loops, caps = [], []
    while next_vertex:
        start = next(iter(next_vertex))
        loop = [start]
        current = next_vertex.pop(start)
        while current != start:
            loop.append(current)
            if current not in next_vertex:
                raise RuntimeError('Open cut contour')
            current = next_vertex.pop(current)
        if len(loop) < 3:
            raise RuntimeError('Invalid cut loop')
        loops.append(loop)
        caps.extend(triangulate(loop[::-1], vertices))
    if len(loops) not in (1, 2):
        raise RuntimeError(f'Unexpected number of cut contours: {len(loops)}')
    output = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices_array),
                                      o3d.utility.Vector3iVector(np.asarray(result + caps)))
    output.vertex_colors = o3d.utility.Vector3dVector(colors)
    output.remove_unreferenced_vertices()
    output.compute_vertex_normals()
    return output, {'floor_x_mm': floor, 'removed_original_vertices': int((~inside).sum()),
                     'new_cut_vertices': len(cache), 'closed_shoe_contours': len(loops), 'cap_triangles': len(caps),
                     'note': 'Planar inferred caps, not measured soles or print certification. A shared contour retains existing foot connectivity; feet are not separated.'}


def fit_hands(mesh, raw, max_move=4.):
    original = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.triangles)
    current = original.copy()
    normals = np.asarray(mesh.vertex_normals)
    roi = np.zeros(len(original))
    # Dataset-specific boxes, deliberately NOT applied to later scans.
    boxes = [([220, -250, 970], [380, -155, 1130]), ([240, 150, 960], [400, 250, 1110])]
    for low, high in boxes:
        distance = np.min(np.minimum(original - low, high - original), axis=1)
        roi = np.maximum(roi, np.clip(distance / 15, 0, 1))
    edges, counts = np.unique(np.sort(np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]]), axis=1), axis=0, return_counts=True)
    roi[np.unique(edges[counts != 2])] = 0
    indices = np.flatnonzero(roi > 0)
    tree = cKDTree(raw)
    before_d = tree.query(original[indices], workers=4)[0]
    tri = original[faces]
    guidance = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    affected_faces = np.any(roi[faces] > 0, axis=1)
    for _ in range(8):
        distance, nearest = tree.query(current[indices], k=24, workers=4)
        neighbors = raw[nearest]
        center = neighbors.mean(1)
        delta = neighbors - center[:, None]
        values, axes = np.linalg.eigh(np.einsum('nki,nkj->nij', delta, delta) / 24)
        normal = axes[:, :, 0]
        valid = ((distance[:, -1] < 14) & (values[:, 1] > 1e-8)
                 & (values[:, 0] / np.maximum(values[:, 1], 1e-20) < .12)
                 & (np.abs(np.einsum('ij,ij->i', normal, normals[indices])) > .6))
        offset = np.einsum('ij,ij->i', center - current[indices], normal)
        offset[~valid] = 0
        candidate = current.copy()
        candidate[indices] += .4 * offset[:, None] * normal * roi[indices, None]
        displacement = candidate[indices] - original[indices]
        displacement *= np.minimum(1, max_move * roi[indices] / np.maximum(np.linalg.norm(displacement, axis=1), 1e-20))[:, None]
        candidate[indices] = original[indices] + displacement
        for attempt in range(15):
            t = candidate[faces]
            cross = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            bad = (np.einsum('ij,ij->i', cross, guidance) <= 0) & affected_faces
            if not bad.any():
                break
            blocked = np.unique(faces[bad])
            candidate[blocked] = current[blocked]
        else:
            candidate = current.copy()
        current = candidate
    after_d = tree.query(current[indices], workers=4)[0]
    accepted = np.quantile(after_d, .95) <= np.quantile(before_d, .95) + .1
    output = copy.deepcopy(mesh)
    output.vertices = o3d.utility.Vector3dVector(current if accepted else original)
    output.compute_vertex_normals()
    move = np.linalg.norm((current if accepted else original) - original, axis=1)
    return output, {'accepted': bool(accepted), 'roi_vertices': len(indices), 'boxes_mm': boxes,
                     'hand_distance_p95_before_mm': float(np.quantile(before_d, .95)),
                     'hand_distance_p95_candidate_mm': float(np.quantile(after_d, .95)),
                     'movement_max_mm': float(move.max()), 'moved_vertices': int((move > 1e-9).sum()),
                     'non_hand_vertices_exactly_unchanged': bool(np.array_equal(current[roi == 0], original[roi == 0])),
                     'note': 'Conservative local projection; cannot recover unobserved fingers or remove all webbing.'}


if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh(str(SOURCE))
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    cloud = o3d.io.read_point_cloud(str(ROOT / 'data/export_00001.ply'))
    cloud, _, _ = select_main_point_component(cloud, 12, proxy_voxel=4)
    raw = np.asarray(cloud.points)
    floor = float(raw[:, 0].min() - 3)
    feet, foot_report = cut_and_cap(mesh, floor)
    topology = geometry_report(copy.deepcopy(feet).scale(.001, center=np.zeros(3)), False)
    if not topology['orientation_succeeded'] or topology['edges_with_more_than_two_faces']:
        raise RuntimeError('Unsafe foot repair topology')
    assert o3d.io.write_triangle_mesh(str(OUT / 'export_00001_feet_candidate.ply'), feet)
    combined, hand_report = fit_hands(feet, raw)
    assert o3d.io.write_triangle_mesh(str(OUT / 'export_00001_feet_hands_candidate.ply'), combined)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    report = {'source': str(SOURCE), 'source_unchanged': True, 'feet': foot_report,
              'foot_topology': topology, 'hands': hand_report}
    (OUT / 'repair_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
