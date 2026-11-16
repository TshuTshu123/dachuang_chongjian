"""躯干局部二次曲面去噪试验，不覆盖正式模型。

python tools/torso_surface_trial.py --scan 3 4
优先接续 wrist_observation 候选；没有时使用正式模型。语义标签须匹配正式模型。
仅上衣/躯干及沿同一网格表面补齐的未识别区域可动，保持面连接、颜色及其他部位。
半径、位移与截面均以毫米计；截面仅作形状变化检查，不是人工测量标记。
"""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from body_parts import file_hash
from hand_pipeline import smoothstep
from surface_evaluation import distances
from surface_quality import intersection_blocks, intersection_pairs
from tools.surface_trials import panel


def region(mesh, parts):
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    edges = np.unique(np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1), axis=0)
    lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    graph = coo_matrix((np.r_[lengths, lengths],
                       (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(len(v), len(v))).tocsr()
    known = np.zeros(len(v), bool)
    seed = known.copy()
    for name, part in parts.items():
        known[part['vertex_indices']] = True
        if name in ('torso', 'upper_clothing'):
            seed[part['vertex_indices']] = True
    distance, _, source = dijkstra(graph, indices=np.flatnonzero(known), min_only=True,
                                   return_predecessors=True, limit=60)
    allowed = seed | (~known & (source >= 0) & (distance < 60) & seed[np.maximum(source, 0)])
    ids = np.flatnonzero(allowed)
    _, components = connected_components(graph[ids][:, ids], directed=False)
    allowed[ids] = components == np.argmax(np.bincount(components))
    # Geodesic buffers hold collars, hems and other semantic boundaries fixed.
    border = np.zeros(len(v), bool)
    crossing = allowed[edges[:, 0]] != allowed[edges[:, 1]]
    border[np.unique(edges[crossing])] = True
    _, counts = np.unique(np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1),
                          axis=0, return_counts=True)
    border[np.unique(edges[counts != 2])] = True
    margin = dijkstra(graph, indices=np.flatnonzero(border), min_only=True, limit=12)
    weight = allowed * smoothstep(np.clip((margin - 2) / 10, 0, 1))
    adjacency = graph.copy()
    adjacency.data[:] = 1
    degree = np.maximum(np.asarray(adjacency.sum(1)).ravel(), 1)
    return weight, allowed, edges, adjacency, degree


def quadratic_step(vertices, normals, area, ids, radius):
    """Area-weighted robust quadratic MLS, in a PCA tangent frame.

    Quadratic terms retain broad curvature, unlike projecting onto a flat plane.
    All fitting uses the frozen mesh for this pass, not raw validation samples.
    """
    tree = cKDTree(vertices)
    displacement = np.zeros_like(vertices)
    supported = np.zeros(len(vertices), bool)
    for start in range(0, len(ids), 1000):
        take = ids[start:start + 1000]
        d, near = tree.query(vertices[take], k=192, workers=4)
        delta = vertices[near] - vertices[take, None]
        same_side = np.einsum('nkj,nj->nk', normals[near], normals[take]) > .25
        weights = np.exp(-.5 * (d / (radius * .5)) ** 2) * (d <= radius) * same_side * area[near]
        weights /= np.maximum(weights.sum(1, keepdims=True), 1e-20)
        center = np.einsum('nk,nkj->nj', weights, delta)
        centered = delta - center[:, None]
        cov = np.einsum('nk,nki,nkj->nij', weights, centered, centered)
        values, axes = np.linalg.eigh(cov)
        normal = axes[:, :, 0]
        normal *= np.where(np.einsum('nj,nj->n', normal, normals[take]) >= 0, 1., -1.)[:, None]
        u = axes[:, :, 1]
        w = np.cross(normal, u)
        x = np.einsum('nkj,nj->nk', delta, u) / radius
        y = np.einsum('nkj,nj->nk', delta, w) / radius
        h = np.einsum('nkj,nj->nk', delta, normal)
        design = np.stack((np.ones_like(x), x, y, x*x, x*y, y*y), axis=2)
        current = weights.copy()
        for _ in range(3):
            lhs = np.einsum('nk,nki,nkj->nij', current, design, design)
            lhs += np.eye(6)[None] * 1e-6
            rhs = np.einsum('nk,nki,nk->ni', current, design, h)
            coef = np.linalg.solve(lhs, rhs[..., None])[..., 0]
            residual = h - np.einsum('nki,ni->nk', design, coef)
            scale = np.maximum(.5, 1.4826 * np.median(np.abs(residual), axis=1))
            current = weights / (1 + (residual / (2.5 * scale[:, None])) ** 2)
        good = ((weights > 0).sum(1) >= 18) & (values[:, 1] > 1.)
        displacement[take] = normal * np.where(good, coef[:, 0], 0)[:, None]
        supported[take] = good
    return displacement, supported


def propose(mesh, weight, radius, cap):
    original = np.asarray(mesh.vertices).copy()
    f = np.asarray(mesh.triangles)
    cross = np.cross(original[f[:, 1]] - original[f[:, 0]], original[f[:, 2]] - original[f[:, 0]])
    area = np.bincount(f.ravel(), weights=np.repeat(np.linalg.norm(cross, axis=1) / 6, 3),
                       minlength=len(original))
    candidate = copy.deepcopy(mesh)
    ids = np.flatnonzero(weight > 0)
    for _ in range(2):
        candidate.compute_vertex_normals()
        xyz = np.asarray(candidate.vertices).copy()
        step, _ = quadratic_step(xyz, np.asarray(candidate.vertex_normals), area, ids, radius)
        delta = xyz + .8 * weight[:, None] * step - original
        delta *= np.minimum(1, cap * weight / np.maximum(np.linalg.norm(delta, axis=1), 1e-20))[:, None]
        candidate.vertices = o3d.utility.Vector3dVector(original + delta)
    candidate.compute_vertex_normals()
    return candidate


def face_quality(mesh, mask, edges):
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    p = v[f]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-20)
    all_edges = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)
    faces = np.tile(np.arange(len(f)), 3)
    order = np.lexsort(all_edges.T)
    all_edges, faces = all_edges[order], faces[order]
    paired = np.flatnonzero(np.all(all_edges[1:] == all_edges[:-1], axis=1))
    paired = paired[np.all(mask[all_edges[paired]], axis=1)]
    angle = np.degrees(np.arccos(np.clip(np.sum(normal[faces[paired]] * normal[faces[paired + 1]], axis=1), -1, 1)))
    return {'normal_angle_median_deg': float(np.median(angle)),
            'normal_angle_p90_deg': float(np.quantile(angle, .9))}


def sections(mesh, levels, center):
    """Intersect using mesh-edge identity, without rounding/welding nearby points.

    Very short segments are legitimate on dense meshes. Coordinate rounding can
    collapse one into a self-loop and reject an otherwise closed torso contour.
    Shared mesh edges (or an exact on-plane vertex) identify endpoints instead.
    Actual open/branched contours remain invalid; no gaps are filled here.
    These diagnostic levels are not anatomical measurement landmarks.
    """
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    height = v[:, 0]
    minimum = height[f].min(1)
    maximum = height[f].max(1)
    result = []
    for level in levels:
        triangles = f[(minimum < level) & (maximum >= level)]
        segments, keys = [], []
        for tri in triangles:
            points, endpoints = [], []
            for i, j in ((0, 1), (1, 2), (2, 0)):
                a, b = int(tri[i]), int(tri[j])
                if (height[a] < level) == (height[b] < level):
                    continue
                # Canonical edge order also makes interpolation identical across faces.
                a, b = min(a, b), max(a, b)
                t = (level - height[a]) / (height[b] - height[a])
                points.append((v[a] + t * (v[b] - v[a]))[1:])
                endpoints.append((0, a, a) if height[a] == level else
                                 (0, b, b) if height[b] == level else (1, a, b))
            if len(points) == 2 and endpoints[0] != endpoints[1]:
                segments.append(points)
                keys.extend(endpoints)
        if not segments:
            result.append(None)
            continue
        seg = np.asarray(segments)
        _, inverse = np.unique(np.asarray(keys), axis=0, return_inverse=True)
        ends = inverse.reshape(-1, 2)
        graph = coo_matrix((np.ones(len(ends)*2),
            (np.r_[ends[:, 0], ends[:, 1]], np.r_[ends[:, 1], ends[:, 0]]))).tocsr()
        _, group = connected_components(graph, directed=False)
        contours = []
        for label in np.unique(group):
            chosen = group[ends[:, 0]] == label
            points = seg[chosen].reshape(-1, 2)
            if len(points) < 20 or np.linalg.norm(np.mean(points, axis=0) - center[1:]) > 120:
                continue
            nodes = np.unique(ends[chosen])
            # Count incident segments, including duplicates, to reject branching.
            if np.any(np.asarray(graph[nodes][:, nodes].sum(1)).ravel() != 2):
                continue
            contours.append({'level_x_mm': float(level),
                             'width_mm': float(np.ptp(points[:, 0])), 'depth_mm': float(np.ptp(points[:, 1])),
                             'perimeter_mm': float(np.linalg.norm(seg[chosen, 1] - seg[chosen, 0], axis=1).sum())})
        result.append(max(contours, key=lambda x: x['perimeter_mm']) if contours else None)
    return result


def comparison(meshes, names, xyz, allowed, out, filename='comparison.jpg'):
    points = xyz[allowed]
    center = (points.min(0) + points.max(0)) / 2
    span = np.ptp(points, axis=0).max() * 1.05
    size = 520
    canvas = Image.new('RGB', (size * len(meshes), (size + 26) * 3), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (right, name) in enumerate(((np.array([0., 1., 0.]), 'Front'),
                                         (np.array([0., -1., 0.]), 'Back'),
                                         (np.array([0., 0., 1.]), 'Side'))):
        for col, (mesh, text) in enumerate(zip(meshes, names)):
            top = row * (size + 26)
            canvas.paste(panel(mesh, center, right, np.array([1., 0., 0.]), span, size), (col * size, top + 26))
            draw.text((col * size + 8, top + 7), f'{text} | {name}', fill='black')
    canvas.save(out / filename, quality=95)


def trial(scan):
    started = time.perf_counter()
    parent = ROOT / 'outputs/models' / f'export_{scan:05d}'
    rp = next(p for p in sorted(parent.glob('*/report.json'), reverse=True) if (p.parent/'body_parts.json').exists())
    report = json.loads(rp.read_text(encoding='utf-8'))
    formal = rp.parent / report['artifacts']['ply']['name']
    info = json.loads((rp.parent / 'body_parts.json').read_text(encoding='utf-8'))
    assert file_hash(formal) == info['mesh']['sha256']
    source = ROOT / 'outputs/experiments/wrist_observation' / parent.name / f'{parent.name}_wrist_observation.ply'
    if not source.exists():
        source = formal
    source_hash = file_hash(source)
    raw_path = Path(report['source'])
    raw_hash = file_hash(raw_path)
    mesh = o3d.io.read_triangle_mesh(str(source))
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices).copy()
    f = np.asarray(mesh.triangles)
    formal_mesh = o3d.io.read_triangle_mesh(str(formal))
    # Hand subdivision appends vertices; original torso indices must still match exactly.
    for name in ('upper_clothing', 'torso'):
        ids = info['parts'][name]['vertex_indices']
        assert np.array_equal(v[ids], np.asarray(formal_mesh.vertices)[ids])
    weight, allowed, edges, graph, degree = region(mesh, info['parts'])
    active = weight > 0
    mask = weight > .9
    raw = np.asarray(o3d.io.read_point_cloud(str(raw_path)).points).copy()
    raw *= 1000 if report['fingerprint']['unit'] == 'm' else 1
    raw = raw[np.isfinite(raw).all(1)]
    nearest = cKDTree(v).query(raw, workers=4)[1]
    selected = raw[mask[nearest]]
    samples = selected[np.random.default_rng(63).choice(len(selected), min(8000, len(selected)), replace=False)]
    before_distance = distances(mesh, samples)
    before = face_quality(mesh, mask, edges)
    points = v[allowed]
    center = (points.min(0) + points.max(0)) / 2
    levels = np.linspace(*np.quantile(points[:, 0], [.2, .7]), 7)
    base_sections = sections(mesh, levels, center)
    blocks = intersection_blocks(mesh, active, margin_mm=7.)
    base_pairs = intersection_pairs(mesh, blocks)
    cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    out = ROOT / 'outputs/experiments/torso_surface' / parent.name
    out.mkdir(parents=True, exist_ok=True)
    records, candidates, names = [], [mesh], ['Input']
    for radius, cap in ((12., 1.5), (20., 3.)):
        candidate = propose(mesh, weight, radius, cap)
        delta = np.asarray(candidate.vertices) - v
        reduced = np.zeros(len(v), bool)
        for _ in range(12):
            xyz = v + delta
            candidate.vertices = o3d.utility.Vector3dVector(xyz)
            candidate.compute_vertex_normals()
            new_cross = np.cross(xyz[f[:, 1]] - xyz[f[:, 0]], xyz[f[:, 2]] - xyz[f[:, 0]])
            bad = np.flatnonzero(np.einsum('ij,ij->i', cross, new_cross) <= 0)
            extra = intersection_pairs(candidate, blocks) - base_pairs
            if not len(bad) and not extra:
                break
            conflict = np.unique(np.r_[bad, np.asarray(list(extra), int).ravel()])
            halo = np.zeros(len(v), bool)
            halo[np.unique(f[conflict])] = True
            attenuation = np.ones(len(v))
            attenuation[halo] = 0
            for value in (.25, .5, .75):
                halo |= np.asarray(graph @ halo.astype(float)).ravel() > 0
                attenuation[halo] = np.minimum(attenuation[halo], value)
            reduced |= (attenuation < 1) & (np.linalg.norm(delta, axis=1) > 1e-9)
            delta *= attenuation[:, None]
        # Recompute after the final attenuation, including when the loop limit is reached.
        xyz = v + delta
        candidate.vertices = o3d.utility.Vector3dVector(xyz)
        candidate.compute_vertex_normals()
        new_cross = np.cross(xyz[f[:, 1]] - xyz[f[:, 0]], xyz[f[:, 2]] - xyz[f[:, 0]])
        flips = int((np.einsum('ij,ij->i', cross, new_cross) <= 0).sum())
        extra = intersection_pairs(candidate, blocks) - base_pairs
        after_distance = distances(candidate, samples)
        after = face_quality(candidate, mask, edges)
        candidate_sections = sections(candidate, levels, center)
        changes = [{key: b[key] - a[key] for key in ('width_mm', 'depth_mm', 'perimeter_mm')}
                   for a, b in zip(base_sections, candidate_sections) if a and b]
        max_width = max(abs(c['width_mm']) for c in changes)
        max_depth = max(abs(c['depth_mm']) for c in changes)
        perimeter_ratio = max(abs(b['perimeter_mm']/a['perimeter_mm'] - 1)
                              for a, b in zip(base_sections, candidate_sections) if a and b)
        gain = 1 - after['normal_angle_p90_deg'] / before['normal_angle_p90_deg']
        fit = {'before_median_mm': float(np.median(before_distance)), 'after_median_mm': float(np.median(after_distance)),
               'before_p95_mm': float(np.quantile(before_distance, .95)), 'after_p95_mm': float(np.quantile(after_distance, .95))}
        accepted = (not extra and not flips and gain > .05 and max_width <= 3 and max_depth <= 3
                    and perimeter_ratio <= .02 and fit['after_median_mm'] <= fit['before_median_mm'] + .4
                    and fit['after_p95_mm'] <= fit['before_p95_mm'] + .8 and len(changes) == len(levels))
        record = {'radius_mm': radius, 'cap_mm': cap, 'iterations': 2, 'accepted': bool(accepted),
                  'before': before, 'after': after, 'normal_angle_p90_decrease_percent': 100*gain,
                  'raw_consistency': fit, 'raw_consistency_samples': len(samples),
                  'validation_note': 'Raw samples are consistency checks, not independent ground truth or held-out data.',
                  'max_width_change_mm': max_width, 'max_depth_change_mm': max_depth,
                  'max_perimeter_change_percent': perimeter_ratio*100,
                  'sections_before': base_sections, 'sections_after': candidate_sections,
                  'movement_max_mm': float(np.linalg.norm(delta, axis=1).max()),
                  'locally_reduced_vertices': int(reduced.sum()), 'flipped_faces': flips,
                  'new_local_intersections': len(extra)}
        records.append(record)
        candidates.append(candidate)
        names.append(f'Quadratic {radius:g}mm / cap {cap:g}mm [{"PASS" if accepted else "REJECTED"}]')
        print(parent.name, json.dumps({k: record[k] for k in ('radius_mm', 'accepted', 'normal_angle_p90_decrease_percent',
              'movement_max_mm', 'max_width_change_mm', 'max_depth_change_mm', 'max_perimeter_change_percent',
              'flipped_faces', 'new_local_intersections')}, ensure_ascii=False), flush=True)
    comparison(candidates, names, v, allowed, out, 'candidate_comparison.jpg')
    passing = [i for i, r in enumerate(records) if r['accepted']]
    choice = max(passing, key=lambda i: records[i]['normal_angle_p90_decrease_percent']) if passing else None
    result = {'source': str(source), 'source_sha256': source_hash, 'formal_model': str(formal),
              'formal_sha256': info['mesh']['sha256'], 'raw_source': str(raw_path), 'raw_sha256': raw_hash,
              'method': 'area-weighted robust local quadratic surface fitting', 'length_unit': 'mm',
              'target_region': 'torso and upper clothing, including sleeves; other known body parts protected',
              'unknown_label_extension_geodesic_mm': 60, 'active_vertices': int(active.sum()),
              'boundary_buffer_mm': [2, 12], 'manual_measurement_landmarks': 'not_generated',
              'section_note': 'Seven fixed horizontal diagnostic contours, not anatomical measurement landmarks.',
              'candidates': records, 'selected_index': choice, 'elapsed_seconds': round(time.perf_counter() - started, 2)}
    if choice is not None:
        candidate = candidates[choice+1]
        comparison([mesh, candidate], ['Input', 'Selected torso correction'], v, allowed, out)
        path = out / f'{parent.name}_torso_surface.ply'
        assert o3d.io.write_triangle_mesh(str(path), candidate)
        exported = o3d.io.read_triangle_mesh(str(path))
        assert np.array_equal(np.asarray(exported.triangles), f)
        assert np.array_equal(np.asarray(exported.vertices), np.asarray(candidate.vertices))
        assert np.array_equal(np.asarray(exported.vertices)[~active], v[~active])
        assert np.array_equal(np.asarray(exported.vertex_colors), np.asarray(mesh.vertex_colors))
        for name in ('left_hand', 'right_hand', 'left_lower_arm', 'right_lower_arm', 'face_neck', 'lower_clothing'):
            ids = info['parts'][name]['vertex_indices']
            assert np.array_equal(np.asarray(exported.vertices)[ids], v[ids])
        assert exported.is_vertex_manifold() == mesh.is_vertex_manifold()
        assert len(exported.get_non_manifold_edges()) == len(mesh.get_non_manifold_edges())
        qa = {'candidate_sha256': file_hash(path), 'connectivity_unchanged': True,
              'outside_torso_region_exactly_unchanged': True, 'colors_unchanged': True,
              'finite_vertices': bool(np.isfinite(np.asarray(exported.vertices)).all()),
              'new_open_edges': 0, 'flipped_faces': records[choice]['flipped_faces'],
              'new_local_intersections': records[choice]['new_local_intersections'],
              'changed_vertices': int((np.linalg.norm(np.asarray(exported.vertices) - v, axis=1) > 1e-9).sum()),
              'protected_hands_arms_face_and_lower_clothing_exactly_unchanged': True,
              'manifold_checks_match_source': True,
              'input_experiment_unchanged': file_hash(source) == source_hash,
              'formal_model_unchanged': file_hash(formal) == info['mesh']['sha256'],
              'raw_data_unchanged': file_hash(raw_path) == raw_hash}
        (out / 'export_check.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
        result['selected_mesh'] = str(path)
    assert file_hash(source) == source_hash
    assert file_hash(raw_path) == raw_hash
    assert file_hash(formal) == info['mesh']['sha256']
    (out / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', parent.name, 'selected', choice, result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=int, nargs='+', default=[3, 4])
    args = parser.parse_args()
    for number in args.scan:
        trial(number)
