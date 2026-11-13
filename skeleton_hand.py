"""统一手部处理，由 run.py 默认自动调用。

--hand-skeleton 仅做原有局部贴合；--hand-density 额外过滤虚接、约束观测厚度。
默认自动模式按采样误差跳过正常手指，并拒绝会扩大开口的删面候选。
掌部默认采用椭圆截面参考的平滑体积限厚，取代逐顶点裁剪。
密度阈值和厚度参考随数据估计；缺少证据时跳过，不强制生成五根手指。
不改顶点编号和颜色；删面可能留下开口，不自动把缺失的指缝补回。
输入最终毫米网格、匹配的 body_parts.json、同坐标原始毫米点云。
"""

import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from body_parts import file_hash
from hand_centerlines import assess, hand_frame
from hand_pipeline import pairs, smoothstep
from surface_evaluation import distances

VERSION = 4


def distribution(values):
    values = np.asarray(values)
    return {'median_mm': float(np.median(values)), 'p95_mm': float(np.quantile(values, .95))}


def stabilize_offsets(vertices, faces, original_cross, displacement, graph):
    """Keep fragile triangles in place, with a small transition in displacement."""
    attenuation = np.ones(len(vertices))
    coords = vertices+displacement
    for _ in range(8):
        p = coords[faces]
        cross = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        old_size = np.linalg.norm(original_cross, axis=1)
        size = np.linalg.norm(cross, axis=1)
        bad = ((size < old_size*.05)
               | (np.einsum('ij,ij->i', cross, original_cross) < .1*size*old_size))
        if not bad.any():
            break
        ids = np.unique(faces[bad])
        attenuation[ids] = 0
        halo = np.zeros(len(vertices), bool); halo[ids] = True
        for value in (.4, .75):
            halo |= np.asarray(graph @ halo.astype(float)).ravel() > 0
            attenuation[halo] = np.minimum(attenuation[halo], value)
        coords = vertices+displacement*attenuation[:, None]
    return coords, attenuation


def propose_local(vertices, points, branches, palm):
    """Fit local observed planes only inside unambiguous finger mid-sections."""
    paths = [np.asarray(b['local_path_mm']) for b in branches]
    if not paths:
        return np.zeros_like(vertices), {'eligible_vertices': 0}
    projected = [cKDTree(p[:, :2]) for p in paths]
    queries = [tree.query(vertices[:, :2]) for tree in projected]
    distance = np.array([q[0] for q in queries])
    supported = np.array([b['status'] == 'supported_candidate' for b in branches])
    if not supported.any():
        return np.zeros_like(vertices), {'eligible_vertices': 0}
    owners = np.argmin(np.where(supported[:, None], distance, np.inf), axis=0)
    raw_distance = np.array([tree.query(points[:, :2])[0] for tree in projected])
    raw_owner = np.argmin(raw_distance, axis=0)
    delta = np.zeros_like(vertices)
    eligible = 0
    if palm is None:
        return delta, {'eligible_vertices': 0}
    pixel = np.rint((vertices[:, :2]-palm['origin'])/palm['voxel_mm']).astype(int)
    valid = np.all((pixel >= 0) & (pixel < palm['mask'].shape), axis=1)
    outside = np.zeros(len(vertices))
    field = ndi.distance_transform_edt(~palm['mask'])*palm['voxel_mm']
    outside[valid] = field[tuple(pixel[valid].T)]
    for branch_id in np.flatnonzero(supported):
        path = paths[branch_id]
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
        path_index = queries[branch_id][1]
        along = arc[path_index]
        weight = (smoothstep((along-8)/12)*smoothstep((arc[-1]-along-8)/10)
                  *smoothstep((outside-3)/6))
        inside = (owners == branch_id) & (distance[branch_id] <= 18) & (weight > 0)
        # Uncertain but recurrent neighboring branches are excluded, not absorbed.
        rivals = [i for i, b in enumerate(branches) if i != branch_id
                  and (supported[i] or b['stability'] >= .5)]
        if rivals:
            inside &= np.min(distance[rivals], axis=0) > distance[branch_id]+2
        ids = np.flatnonzero(inside)
        observed = points[(raw_owner == branch_id) & (raw_distance[branch_id] <= 22)]
        if len(ids) < 10 or len(observed) < 40:
            continue
        near, neighbor = cKDTree(observed).query(vertices[ids], k=20, workers=4)
        patch = observed[neighbor]
        weights = np.exp(-.5*(near/4.)**2)
        center = np.einsum('nk,nkj->nj', weights, patch)/weights.sum(1)[:, None]
        centered = patch-center[:, None, :]
        covariance = np.einsum('nk,nki,nkj->nij', weights, centered, centered)/weights.sum(1)[:, None, None]
        eigenvalue, axes = np.linalg.eigh(covariance)
        normals = axes[:, :, 0]
        offset = np.einsum('ij,ij->i', center-vertices[ids], normals)
        valid = ((near[:, -1] <= 10) & (near[:, 0] <= 8)
                 & (eigenvalue[:, 0]/np.maximum(eigenvalue[:, 1], 1.e-12) <= .25)
                 & (eigenvalue[:, 1] >= .3))
        ids, offset, normals = ids[valid], offset[valid], normals[valid]
        movement = np.clip(offset, -4., 4.)*weight[ids]
        delta[ids] = movement[:, None]*normals
        eligible += len(ids)
    return delta, {'eligible_vertices': eligible, 'maximum_normal_offset_mm': 4.,
                   'tip_protection_mm': 8., 'palm_fixed': True}


def refine_skeleton_hands(mesh_path, positions_path, raw_mm, *, density_constraints=False, conservative=False):
    mesh_path, positions_path = Path(mesh_path), Path(positions_path)
    info = json.loads(positions_path.read_text(encoding='utf-8'))
    if info.get('status') != 'complete' or info['mesh']['sha256'] != file_hash(mesh_path):
        raise ValueError('Hand centerlines require labels matching the exact input mesh.')
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.triangles)
    raw_mm = np.asarray(raw_mm)
    raw_mm = raw_mm[np.isfinite(raw_mm).all(1)]
    labels = np.full(len(vertices), 'unknown', dtype=object)
    for name, region in info['parts'].items():
        labels[region['vertex_indices']] = name
    known = np.flatnonzero(labels != 'unknown')
    closest_known = labels[known[cKDTree(vertices[known]).query(vertices, workers=4)[1]]]
    raw_labels = labels[cKDTree(vertices).query(raw_mm, workers=4)[1]]
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    graph = coo_matrix((np.ones(2*len(edges)), (np.r_[edges[:, 0], edges[:, 1]],
                       np.r_[edges[:, 1], edges[:, 0]])), shape=(len(vertices), len(vertices))).tocsr()
    degree = np.maximum(1., np.asarray(graph.sum(1)).ravel())
    selected = copy.deepcopy(mesh)
    records, positions = [], []
    changed = np.zeros(len(vertices), bool)
    for name in ('left_hand', 'right_hand'):
        record = {'part': name, 'applied': False}
        try:
            ids, arm_class, wrist, axes = hand_frame(vertices, info['parts'], name)
        except (KeyError, ValueError, np.linalg.LinAlgError) as error:
            records.append({**record, 'reason': str(error)}); continue
        local = (vertices-wrist) @ axes
        low = local[ids].min(0)-[12, 30, 15]; low[1] = -30
        high = local[ids].max(0)+[12, 12, 15]
        raw_local = (raw_mm-wrist) @ axes
        observed_mask = (np.all((raw_local >= low) & (raw_local <= high), axis=1)
            & (cKDTree(vertices[ids]).query(raw_mm, workers=4)[0] < 22)
            & np.isin(raw_labels, [name, arm_class, 'unknown']))
        points = raw_local[observed_mask]
        if len(points) < 100:
            records.append({**record, 'reason': 'Insufficient raw hand points'}); continue
        branches, palm, diagnostics = assess(points)
        for branch in branches:
            branch['world_path_mm'] = (np.asarray(branch['local_path_mm']) @ axes.T+wrist).tolist()
        positions.append({'part': name, 'wrist_mm': wrist.tolist(), 'frame_axes': axes.tolist(),
                          'branches': branches, 'diagnostics': diagnostics})
        record['supported_centerlines'] = sum(b['status']=='supported_candidate' for b in branches)
        if not record['supported_centerlines']:
            records.append({**record, 'reason': 'No sufficiently supported centerline'}); continue
        other = np.flatnonzero((labels != 'unknown') & ~np.isin(labels, [name, arm_class]))
        if len(other) and np.quantile(cKDTree(vertices[other]).query(vertices[ids])[0], .05) < 6:
            records.append({**record, 'reason': 'Hand touches another labeled part'}); continue
        allowed = (labels == name) | ((labels == 'unknown') & (closest_known == name))
        hand = allowed & np.all((local >= low) & (local <= high), axis=1)
        hand_ids = np.flatnonzero(hand)
        proposal, settings = propose_local(local[hand_ids], points, branches, palm)
        record.update(settings)
        displacement = np.zeros_like(vertices)
        displacement[hand_ids] = proposal @ axes.T
        active = np.linalg.norm(displacement, axis=1) > .03
        displacement[~active] = 0
        if active.sum() < 10:
            records.append({**record, 'reason': 'Too little unambiguous finger surface'}); continue
        # Smooth displacement, not the hand coordinates: avoid whole-finger shrinkage.
        initial = displacement.copy()
        for _ in range(3):
            displacement = .75*initial+.25*(graph @ displacement)/degree[:, None]
            displacement[~active] = 0
        record['proposed_vertices'] = int(active.sum())
        before_vertices = np.asarray(selected.vertices).copy()
        old_points = before_vertices[faces]
        old_cross = np.cross(old_points[:, 1]-old_points[:, 0], old_points[:, 2]-old_points[:, 0])
        touched = np.any(active[faces], axis=1)
        box = before_vertices[hand]
        face_ids = np.flatnonzero(np.all(old_points.max(1) >= box.min(0)-10, axis=1)
                                  & np.all(old_points.min(1) <= box.max(0)+10, axis=1))
        reference_pairs = pairs(selected, face_ids, active)
        evaluate = raw_mm[observed_mask]
        if len(evaluate) > 6000:
            evaluate = evaluate[np.random.default_rng(71).choice(len(evaluate), 6000, replace=False)]
        raw_tree = cKDTree(raw_mm[observed_mask])
        before_fit = distribution(distances(selected, evaluate))
        before_support = distribution(raw_tree.query(before_vertices[active])[0])
        if conservative:
            from hand_density_constraints import point_spacing
            tolerance = max(3., 2*point_spacing(points))
            if before_support['p95_mm'] <= tolerance:
                records.append({**record, 'reason': 'Finger fit already within local sampling tolerance',
                                'sampling_tolerance_mm': tolerance,
                                'before_moved_vertex_to_raw': before_support})
                continue
        attempts = []
        for scale in (1., .5, .25):
            candidate = copy.deepcopy(selected)
            coords, attenuation = stabilize_offsets(before_vertices, faces, old_cross, scale*displacement, graph)
            candidate.vertices = o3d.utility.Vector3dVector(coords)
            candidate.compute_vertex_normals()
            p = coords[faces[touched]]
            cross = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
            old = old_cross[touched]
            ratio = np.linalg.norm(cross, axis=1)/np.maximum(np.linalg.norm(old, axis=1), 1.e-15)
            cosine = np.einsum('ij,ij->i', cross, old)/np.maximum(np.linalg.norm(cross, axis=1)*np.linalg.norm(old, axis=1), 1.e-15)
            folded = int(np.sum((ratio < .05) | (cosine < .1)))
            extra = pairs(candidate, face_ids, active)-reference_pairs if not folded else None
            after_fit = distribution(distances(candidate, evaluate))
            after_support = distribution(raw_tree.query(coords[active])[0])
            fit_ok = (after_fit['median_mm'] <= before_fit['median_mm']+.04
                      and after_fit['p95_mm'] <= before_fit['p95_mm']+.10)
            support_ok = (after_support['median_mm'] < before_support['median_mm']-.03
                          and after_support['p95_mm'] <= before_support['p95_mm']+.15)
            check = {'scale': scale, 'folded_or_collapsed_faces': folded,
                     'new_local_intersections': len(extra) if extra is not None else None,
                     'locally_attenuated_vertices': int(np.sum(active & (attenuation < 1))),
                     'raw_to_mesh': after_fit,
                     'moved_vertex_to_raw': after_support,
                     'accepted': bool(not folded and not extra and fit_ok and support_ok)}
            attempts.append(check)
            if check['accepted']:
                assert np.array_equal(coords[~active], before_vertices[~active])
                selected = candidate
                moved = np.linalg.norm(coords-before_vertices, axis=1) > 1.e-9
                changed |= moved
                record.update(applied=True, accepted_scale=scale,
                              moved_vertices=int(moved.sum()),
                              movement_max_mm=float(np.linalg.norm(coords-before_vertices, axis=1).max()))
                break
        record.update(attempts=attempts, before_raw_to_mesh=before_fit,
                      before_moved_vertex_to_raw=before_support,
                      reason='Bounded finger surface fitting accepted' if record['applied'] else 'Quality checks rejected candidates; original retained')
        records.append(record)
    assert np.array_equal(np.asarray(selected.vertices)[~changed], vertices[~changed])
    assert np.array_equal(np.asarray(selected.triangles), faces)
    assert np.array_equal(np.asarray(selected.vertex_colors), np.asarray(mesh.vertex_colors))
    constraints = {'applied': False, 'parts': [], 'removed_faces': 0, 'enabled': density_constraints}
    palm_volume = {'applied': False, 'parts': [], 'enabled': False}
    if density_constraints:
        if conservative:
            from palm_volume import palm_candidate
            selected, palm_volume = palm_candidate(selected, raw_mm, info, positions)
            palm_volume['enabled'] = True
        from hand_density_constraints import constrain_density_and_thickness
        selected, constraints = constrain_density_and_thickness(selected, raw_mm, info, positions, stabilize_offsets,
                                                               allow_open_cuts=not conservative, enable_thickness=not conservative)
        constraints['enabled'] = True
    return selected, {'status': 'complete', 'applied': bool(changed.any() or constraints['applied'] or palm_volume['applied']), 'version': VERSION,
                      'method': 'centerline_density_and_observed_thickness' if density_constraints else 'stable_centerline_local_plane_fitting', 'parts': records,
                      'density_constraints': constraints,
                      'palm_volume': palm_volume,
                      'automatic_quality_policy': conservative,
                      'movement_max_mm': float(np.linalg.norm(np.asarray(selected.vertices)-vertices, axis=1).max()),
                      'new_local_intersections': 0, 'unchanged_vertex_indices_and_colors': True,
                      'topology_changed': bool(constraints['removed_faces']),
                      'notes': ['Unsupported gap bridges may be deleted; no missing-geometry synthesis.' if density_constraints else 'No face connections changed; no missing-geometry synthesis.',
                                'Thickness uses observed paired surfaces; confidence is not anatomical accuracy.' if density_constraints else 'Palm and tips excluded from local plane fitting.',
                                'Checks cover local intersections, not whole-body print certification.']}, {
                          'source_mesh_sha256': file_hash(mesh_path), 'unit': 'mm', 'parts': positions,
                          'scope': 'Centerlines computed before geometry processing; anatomical joints unknown.'}
