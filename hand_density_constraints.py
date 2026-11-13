"""骨架附近的自适应覆盖密度与手部厚度约束，供统一自动流程调用。

低密度用于拒绝跨过无观测指缝的三角面，不删除原始采集点。
密度阈值从留出点估计；厚度参考同一扫描的双侧观测和腕部，不固定套厘米数。
删面可能留下开口，不自动补回；保留顶点编号和颜色，供原部位文件继续使用。
自动模式拒绝新增开口；只有 --hand-density 实验允许导出此类候选。
缺少可靠双侧观测、存在接触或质量检查失败时保持原形。
"""

import copy

import numpy as np
import open3d as o3d
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from hand_centerlines import extract
from hand_pipeline import depths, pairs, scene_for, smoothstep
from surface_evaluation import distances


def point_spacing(points):
    unique = np.unique(np.round(points, 3), axis=0)
    sample = unique[np.linspace(0, len(unique)-1, min(3000, len(unique))).astype(int)]
    values = cKDTree(unique).query(sample, k=2, workers=4)[0][:, 1]
    return float(np.clip(np.median(values[values > 1.e-3]), .7, 4.))


class CoverageField:
    """Planar coverage normalized by nearby observed coverage, not raw point count."""
    def __init__(self, points, spacing):
        self.voxel = max(.6, spacing*.55)
        cells = np.floor(points[:, :2]/self.voxel).astype(int)
        _, ids = np.unique(cells, axis=0, return_index=True)
        xy = points[ids, :2]
        if len(xy) < 50:
            raise ValueError('Too few distinct projected hand samples.')
        self.origin = np.floor(xy.min(0)/self.voxel)*self.voxel-5*self.voxel
        pixel = np.rint((xy-self.origin)/self.voxel).astype(int)
        shape = pixel.max(0)+6
        if np.prod(shape) > 1_000_000:
            raise ValueError('Hand coverage grid exceeds memory bound.')
        sigma = max(.7, spacing*.75/self.voxel)
        def build(keep):
            grid = np.zeros(shape, float)
            np.add.at(grid, tuple(pixel[keep].T), 1)
            return ndi.gaussian_filter(grid, sigma)
        self.xy = xy
        self.tree = cKDTree(xy)
        self.grid = build(np.ones(len(xy), bool))
        self.reference = self.lookup(self.grid, xy)
        # Calibrate conservatively against an independent random holdout of cells.
        train = np.random.default_rng(193).random(len(xy)) >= .2
        test = ~train
        training = build(train)
        reference = self.lookup(training, xy[train])
        train_tree = cKDTree(xy[train])
        near = train_tree.query(xy[test], k=min(24, train.sum()))[1]
        local = np.median(reference[near], axis=1)
        held = self.lookup(training, xy[test])/np.maximum(local, 1.e-9)
        self.threshold = float(np.clip(np.quantile(held, .02)*.75, .04, .45))
        self.validation = {'unique_projected_cells': len(xy), 'held_out_cells': int(test.sum()),
                           'normalized_density_threshold': self.threshold,
                           'held_out_rejection_fraction': float((held < self.threshold).mean()),
                           'spacing_mm': spacing, 'kde_sigma_mm': sigma*self.voxel}

    def lookup(self, grid, xy):
        coordinates = ((np.asarray(xy)-self.origin)/self.voxel).T
        return ndi.map_coordinates(grid, coordinates, order=1, mode='constant', cval=0.)

    def score(self, xy):
        near = self.tree.query(xy, k=min(24, len(self.xy)))[1]
        reference = np.median(self.reference[near], axis=1)
        return self.lookup(self.grid, xy)/np.maximum(reference, 1.e-9)


def paired_bounds(points, xy, spacing):
    """Require two separated depth bands; a single visible surface is insufficient."""
    distance, ids = cKDTree(points[:, :2]).query(xy, k=min(40, len(points)), workers=4)
    lower = np.full(len(xy), np.nan)
    upper = lower.copy()
    radius = float(np.clip(spacing*3, 5, 10))
    for i, (ds, js) in enumerate(zip(distance, ids)):
        values = np.sort(points[js[ds <= radius], 2])
        if len(values) < 12:
            continue
        gaps = np.diff(values)
        eligible = np.arange(len(gaps))
        eligible = eligible[(eligible >= 3) & (eligible <= len(values)-5)]
        k = int(eligible[np.argmax(gaps[eligible])])
        if gaps[k] < max(1.5, spacing*.8):
            continue
        a, b = values[:k+1], values[k+1:]
        span = float(np.median(b)-np.median(a))
        if not 4 <= span <= 80:
            continue
        lower[i] = np.quantile(a, .15)
        upper[i] = np.quantile(b, .85)
    return lower, upper


def boundary_edges(mesh):
    faces = np.asarray(mesh.triangles)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def mesh_components(mesh):
    _, counts, _ = mesh.cluster_connected_triangles()
    return len(counts)


def prune_small_cut_patches(faces, mask, vertices, minimum_area):
    selected = np.flatnonzero(mask)
    if not len(selected):
        return mask
    ff = faces[selected]
    edges = np.sort(np.concatenate([ff[:, [0, 1]], ff[:, [1, 2]], ff[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(ff)), 3)
    order = np.lexsort(edges.T)
    edges, owner = edges[order], owner[order]
    shared = np.flatnonzero(np.all(edges[1:] == edges[:-1], axis=1))
    row, col = owner[shared], owner[shared+1]
    graph = coo_matrix((np.ones(len(row)*2), (np.r_[row, col], np.r_[col, row])), shape=(len(ff), len(ff))).tocsr()
    _, labels = connected_components(graph, directed=False)
    p = vertices[ff]
    area = .5*np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1)
    sizes = np.bincount(labels)
    areas = np.bincount(labels, weights=area)
    keep = (sizes >= 3) & (areas >= minimum_area)
    result = np.zeros_like(mask)
    result[selected[keep[labels]]] = True
    return result


def density_cut_mask(vertices, faces, hand, points, branches, spacing, field):
    supported = [b for b in branches if b['status'] == 'supported_candidate']
    if len(supported) < 2:
        return np.zeros(len(faces), bool), {'reason': 'Fewer than two supported branches'}
    paths = [np.asarray(b['local_path_mm']) for b in supported]
    candidate_ids = np.flatnonzero(np.all(hand[faces], axis=1))
    if not len(candidate_ids):
        return np.zeros(len(faces), bool), {'reason': 'No eligible hand faces'}
    p = vertices[faces[candidate_ids]]
    samples = np.stack([p.mean(1), (p[:, 0]+p[:, 1])/2,
                        (p[:, 1]+p[:, 2])/2, (p[:, 2]+p[:, 0])/2], axis=1)
    xy = samples[:, :, :2].reshape(-1, 2)
    distances_to_branch = np.array([cKDTree(path[:, :2]).query(xy)[0] for path in paths])
    order = np.argsort(distances_to_branch, axis=0)
    closest = np.take_along_axis(distances_to_branch, order[:2], axis=0)
    between = (closest[0] > max(3., spacing*1.5)) & (closest[0] < 28)
    between &= (closest[1]-closest[0]) < max(5., spacing*3)
    # Keep palm and distal fingertips outside the gap-opening operation.
    _, palm = extract(points, voxel=2., threshold_scale=1.)
    if palm is not None:
        pixels = np.rint((xy-palm['origin'])/palm['voxel_mm']).astype(int)
        valid = np.all((pixels >= 0) & (pixels < palm['mask'].shape), axis=1)
        inside = np.zeros(len(xy), bool)
        inside[valid] = palm['mask'][tuple(pixels[valid].T)]
        between &= ~inside
    for k, path in enumerate(paths):
        nearest = cKDTree(path[:, :2]).query(xy)[1]
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
        assigned = order[0] == k
        # A low-density gap may continue out between the fingertips. Protecting
        # an entire distal band would leave a false bridge over the gap's end.
        between[assigned & (arc[nearest] < 10)] = False
    scores = field.score(xy)
    low = (scores < field.threshold) & between
    low = low.reshape(-1, 4)
    # Two independent samples must indicate a gap, not a single edge fluctuation.
    cut = np.zeros(len(faces), bool)
    cut[candidate_ids[(low.sum(1) >= 2)]] = True
    # Preserve complete existing tip caps, not just their centerline samples.
    protected_tip = np.zeros(len(vertices), bool)
    for path in paths:
        near, nearest = cKDTree(path[:, :2]).query(vertices[:, :2])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
        protected_tip |= (near < 18) & ((arc[-1]-arc[nearest]) < 10)
    cut[np.any(protected_tip[faces], axis=1)] = False
    cut = prune_small_cut_patches(faces, cut, vertices, minimum_area=spacing**2*2)
    return cut, {'candidate_hand_faces': len(candidate_ids), 'low_density_faces': int(cut.sum()),
                 'threshold': field.threshold, 'large_gaps_only': True}


def constrain_density_and_thickness(mesh, raw_mm, info, frames, stabilize_offsets, *, allow_open_cuts=True, enable_thickness=True):
    original_vertices = np.asarray(mesh.vertices).copy()
    original_faces = np.asarray(mesh.triangles).copy()
    selected = copy.deepcopy(mesh)
    labels = np.full(len(original_vertices), 'unknown', dtype=object)
    for name, part in info['parts'].items(): labels[part['vertex_indices']] = name
    known = np.flatnonzero(labels != 'unknown')
    nearest_class = labels[known[cKDTree(original_vertices[known]).query(original_vertices)[1]]]
    raw_labels = labels[cKDTree(original_vertices).query(raw_mm, workers=4)[1]]
    records = []
    for frame in frames:
        name = frame['part']
        record = {'part': name, 'thickness_applied': False, 'density_cut_applied': False}
        ids = np.asarray(info['parts'][name]['vertex_indices'], int)
        wrist, axes = np.array(frame['wrist_mm']), np.array(frame['frame_axes'])
        local = (original_vertices-wrist) @ axes
        low = local[ids].min(0)-[12, 30, 15]; low[1] = -30
        high = local[ids].max(0)+[12, 12, 15]
        raw_local = (raw_mm-wrist) @ axes
        arm = name.replace('hand', 'lower_arm')
        if len(info['parts'].get(arm, {}).get('vertex_indices', [])) < 30:
            arm = 'upper_clothing'
        allowed = (labels == name) | ((labels == 'unknown') & (nearest_class == name))
        hand = allowed & np.all((local >= low) & (local <= high), axis=1) & (local[:, 1] > 0)
        other = np.flatnonzero((labels != 'unknown') & ~np.isin(labels, [name, arm]))
        if len(other) and np.quantile(cKDTree(original_vertices[other]).query(original_vertices[ids])[0], .05) < 6:
            records.append({**record, 'reason': 'Hand contacts another labeled part'}); continue
        take = (np.all((raw_local >= low) & (raw_local <= high), axis=1)
                & np.isin(raw_labels, [name, arm, 'unknown'])
                & (cKDTree(original_vertices[ids]).query(raw_mm, workers=4)[0] < 22))
        points = raw_local[take]
        if len(points) < 120 or hand.sum() < 30:
            records.append({**record, 'reason': 'Insufficient hand observations'}); continue
        spacing = point_spacing(points)
        field = CoverageField(points, spacing)
        record['coverage'] = field.validation
        xy_probes = points[points[:, 1] > 12, :2]
        xy_probes = xy_probes[np.linspace(0, len(xy_probes)-1, min(400, len(xy_probes))).astype(int)]
        world_probes = wrist+np.c_[xy_probes, np.zeros(len(xy_probes))] @ axes.T
        _, _, before_thickness, before_valid = depths(scene_for(selected, hand), world_probes, axes[:, 2])
        evaluate = raw_mm[take]
        if len(evaluate) > 6000:
            evaluate = evaluate[np.random.default_rng(37).choice(len(evaluate), 6000, replace=False)]
        # Estimate wrist reference only from paired depth observations.
        xx, yy = np.meshgrid(np.linspace(-12, 12, 5), np.linspace(-25, -5, 5))
        wrist_low, wrist_high = paired_bounds(points, np.c_[xx.ravel(), yy.ravel()], spacing)
        wrist_span = wrist_high-wrist_low
        good_wrist = np.isfinite(wrist_span) & (wrist_span >= 6) & (wrist_span <= 70)
        wrist_reference = float(np.median(wrist_span[good_wrist])) if good_wrist.sum() >= 10 else None
        if wrist_reference is not None and np.ptp(np.quantile(wrist_span[good_wrist], [.25, .75])) > wrist_reference*.3:
            wrist_reference = None
        hand_ids = np.flatnonzero(hand)
        lower, upper = paired_bounds(points, local[hand_ids, :2], spacing)
        reliable = np.isfinite(lower) & np.isfinite(upper)
        observed_pairs = reliable.copy()
        # Within covered palm regions only, interpolate a short missing section
        # from nearby paired observations. Never interpolate across an empty gap.
        supported_count = sum(b['status'] == 'supported_candidate' for b in frame['branches'])
        if reliable.sum() >= 100 and supported_count >= 3:
            missing = np.flatnonzero(~reliable)
            near, neighbors = cKDTree(local[hand_ids[reliable], :2]).query(local[hand_ids[missing], :2], k=4)
            donor_ids = np.flatnonzero(reliable)[neighbors]
            acceptable = ((near[:, -1] <= 10) & (field.score(local[hand_ids[missing], :2]) >= field.threshold*2))
            weights_ref = 1/np.maximum(near, spacing*.5)**2
            lo_interp = np.sum(weights_ref*lower[donor_ids], axis=1)/weights_ref.sum(1)
            hi_interp = np.sum(weights_ref*upper[donor_ids], axis=1)/weights_ref.sum(1)
            fill = missing[acceptable]
            lower[fill] = lo_interp[acceptable]; upper[fill] = hi_interp[acceptable]
            reliable[fill] = True
        padding = max(1., spacing*.8)
        lower -= padding; upper += padding
        if wrist_reference is not None:
            typical_observed = float(np.quantile((upper-lower)[reliable], .75)) if reliable.any() else 0.
            maximum = max(wrist_reference*1.5, wrist_reference+10, typical_observed)
            center = (lower+upper)/2
            half = np.minimum((upper-lower)/2, maximum/2)
            lower, upper = center-half, center+half
        record.update(wrist_reference_mm=wrist_reference, paired_hand_vertices=int(observed_pairs.sum()),
                      paired_hand_fraction=float(observed_pairs.mean()),
                      interpolated_hand_vertices=int((reliable & ~observed_pairs).sum()), envelope_padding_mm=padding)
        observed_limit = float(np.quantile((upper-lower)[reliable], .95)) if reliable.sum() >= 30 else None
        mesh_limit = float(np.quantile(before_thickness[before_valid], .95)) if before_valid.sum() >= 40 else None
        excessive = (observed_limit is not None and mesh_limit is not None
                     and mesh_limit > observed_limit*1.15
                     and mesh_limit-observed_limit > max(8., spacing*3))
        record.update(observed_thickness_limit_mm=observed_limit, mesh_thickness_p95_mm=mesh_limit,
                      excessive_thickness=bool(excessive))
        # Only cap depths outside an observed envelope; missing sides are not invented.
        delta_depth = np.zeros(len(original_vertices))
        current_local = (np.asarray(selected.vertices)-wrist) @ axes
        target = np.clip(current_local[hand_ids, 2], lower, upper)
        shifts = np.where(reliable, target-current_local[hand_ids, 2], 0.)
        weights = smoothstep(local[hand_ids, 1]/15)
        for branch in frame['branches']:
            path = np.asarray(branch['local_path_mm'])
            near, nn = cKDTree(path[:, :2]).query(local[hand_ids, :2])
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
            protect = (near < 16) & ((arc[-1]-arc[nn]) < 8)
            weights[protect] = 0
        if excessive and enable_thickness:
            delta_depth[hand_ids] = np.clip(shifts, -12, 12)*weights
        active = np.abs(delta_depth) > .1
        faces = np.asarray(selected.triangles)
        if active.sum() >= 12:
            edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
            graph = coo_matrix((np.ones(2*len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(len(local), len(local))).tocsr()
            degree = np.maximum(1., np.asarray(graph.sum(1)).ravel())
            base_depth = delta_depth.copy()
            eligible = np.zeros(len(local), bool)
            eligible[hand_ids] = reliable & (weights > 0)
            for _ in range(20):
                delta_depth = .12*base_depth+.88*(graph @ delta_depth)/degree
                delta_depth[~eligible] = 0
        active = np.abs(delta_depth) > .1
        thickness_attempts = []
        if active.sum() >= 12:
            before = np.asarray(selected.vertices).copy()
            p = before[faces]
            cross = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
            face_ids = np.flatnonzero(np.any(hand[faces], axis=1))
            old_pairs = pairs(selected, face_ids, active)
            old_fit = distances(selected, evaluate)
            old_support = cKDTree(raw_mm[take]).query(before[active])[0]
            for scale in (1., .5, .25):
                coords, attenuation = stabilize_offsets(before, faces, cross, scale*delta_depth[:, None]*axes[:, 2], graph)
                candidate = copy.deepcopy(selected); candidate.vertices = o3d.utility.Vector3dVector(coords)
                candidate.compute_vertex_normals()
                initial_coords = coords.copy()
                local_attenuation = np.ones(len(coords))
                for _ in range(4):
                    extra_pairs = pairs(candidate, face_ids, active)-old_pairs
                    if not extra_pairs:
                        break
                    affected = np.unique(faces[np.asarray(list(extra_pairs), int).ravel()])
                    affected = affected[active[affected]]
                    if not len(affected):
                        break
                    local_attenuation[affected] = 0
                    halo = np.zeros(len(coords), bool); halo[affected] = True
                    for value in (.4, .75):
                        halo |= np.asarray(graph @ halo.astype(float)).ravel() > 0
                        local_attenuation[halo] = np.minimum(local_attenuation[halo], value)
                    coords = before+(initial_coords-before)*local_attenuation[:, None]
                    candidate.vertices = o3d.utility.Vector3dVector(coords)
                    candidate.compute_vertex_normals()
                cp = coords[faces]
                cc = np.cross(cp[:, 1]-cp[:, 0], cp[:, 2]-cp[:, 0])
                touched = np.any(active[faces], axis=1)
                bad = int(np.sum(touched & ((np.linalg.norm(cc, axis=1) < .05*np.linalg.norm(cross, axis=1))
                    | (np.einsum('ij,ij->i', cc, cross) < .1*np.linalg.norm(cc, axis=1)*np.linalg.norm(cross, axis=1)))))
                extra = len(pairs(candidate, face_ids, active)-old_pairs) if not bad else None
                after_fit = distances(candidate, evaluate)
                support = cKDTree(raw_mm[take]).query(coords[active])[0]
                fit_ok = np.median(after_fit) <= np.median(old_fit)+.05 and np.quantile(after_fit, .95) <= np.quantile(old_fit, .95)+.2
                gain = np.quantile(old_support, .95)-np.quantile(support, .95)
                # Global thickness, rather than isolated dents, must decrease.
                _, _, trial_thickness, trial_valid = depths(scene_for(candidate, hand), world_probes, axes[:, 2])
                paired_rays = before_valid & trial_valid
                thickness_gain = (float(np.quantile(before_thickness[paired_rays], .95)-np.quantile(trial_thickness[paired_rays], .95))
                                  if paired_rays.sum() >= 40 else 0.)
                accepted = bool(not bad and extra == 0 and fit_ok and gain > .2 and thickness_gain >= 1.)
                thickness_attempts.append({'scale': scale, 'folded_faces': bad, 'new_local_intersections': extra,
                    'locally_retained_vertices': int(np.sum(active & (local_attenuation < 1))),
                    'moved_vertex_support_p95_gain_mm': float(gain), 'thickness_p95_gain_mm': thickness_gain, 'accepted': accepted})
                if accepted:
                    selected = candidate
                    record.update(thickness_applied=True, thickness_moved_vertices=int((np.linalg.norm(coords-before, axis=1) > .01).sum()),
                                  thickness_max_shift_mm=float(np.linalg.norm(coords-before, axis=1).max()))
                    break
        record['thickness_attempts'] = thickness_attempts
        if not enable_thickness:
            record['thickness_strategy'] = 'Handled by coherent palm volume stage; vertexwise cap disabled.'
        if not excessive:
            record['thickness_skip_reason'] = 'Observed thickness difference is not clearly excessive, or reference is unavailable.'
        # Delete only supported-branch gap bridges. No vertex compaction: labels stay valid.
        current = (np.asarray(selected.vertices)-wrist) @ axes
        faces = np.asarray(selected.triangles)
        cut, cut_info = density_cut_mask(current, faces, hand, points, frame['branches'], spacing, field)
        record['connection_filter'] = cut_info
        if cut.any():
            candidate = copy.deepcopy(selected)
            candidate.remove_triangles_by_mask(cut)
            candidate.compute_vertex_normals()
            # Remove only tiny, unsupported remnants of a cut bridge. A detached
            # finger with real observations must fail the component check below.
            baseline_components = mesh_components(selected)
            labels_cc, counts_cc, _ = candidate.cluster_connected_triangles()
            cleanup_faces = 0
            if len(counts_cc) > baseline_components:
                cf = np.asarray(candidate.triangles)
                labels_cc = np.asarray(labels_cc)
                extra_cut = np.zeros(len(cf), bool)
                for component, count in enumerate(counts_cc):
                    if count > 12:
                        continue
                    component_ids = np.flatnonzero(labels_cc == component)
                    if not np.all(hand[cf[component_ids]]):
                        continue
                    centroids = current[cf[component_ids]].mean(1)
                    if np.median(field.score(centroids[:, :2])) < field.threshold:
                        extra_cut[component_ids] = True
                cleanup_faces = int(extra_cut.sum())
                if cleanup_faces:
                    candidate.remove_triangles_by_mask(extra_cut)
                    candidate.compute_vertex_normals()
            before_fit = distances(selected, evaluate)
            after_fit = distances(candidate, evaluate)
            coverage_loss = float(((after_fit-before_fit) > 2*spacing).mean())
            faces_in_hand = max(1, int(np.all(hand[faces], axis=1).sum()))
            removed_count = int(cut.sum())+cleanup_faces
            before_boundary, after_boundary = boundary_edges(selected), boundary_edges(candidate)
            accepted = (removed_count <= faces_in_hand*.25 and coverage_loss <= .01
                and np.quantile(after_fit, .95) <= np.quantile(before_fit, .95)+.5
                and mesh_components(candidate) <= baseline_components
                and (allow_open_cuts or after_boundary <= before_boundary))
            record['connection_filter'].update(raw_coverage_loss_fraction=coverage_loss,
                raw_to_mesh_p95_before_mm=float(np.quantile(before_fit, .95)),
                raw_to_mesh_p95_after_mm=float(np.quantile(after_fit, .95)),
                boundary_edges_before=before_boundary, boundary_edges_after=after_boundary,
                unsupported_remnant_faces=cleanup_faces,
                open_boundary_rejected=bool(not allow_open_cuts and after_boundary > before_boundary),
                accepted=bool(accepted))
            if accepted:
                selected = candidate
                record.update(density_cut_applied=True, removed_faces=removed_count)
        _, _, after_thickness, after_valid = depths(scene_for(selected, hand), world_probes, axes[:, 2])
        common = before_valid & after_valid
        record['thickness_measurement'] = {'common_ray_samples': int(common.sum()),
            'lost_valid_ray_samples': int((before_valid & ~after_valid).sum()),
            'before_p95_mm': float(np.quantile(before_thickness[common], .95)) if common.any() else None,
            'after_p95_mm': float(np.quantile(after_thickness[common], .95)) if common.any() else None}
        records.append(record)
    assert len(selected.vertices) == len(original_vertices)
    assert np.array_equal(selected.vertex_colors, mesh.vertex_colors)
    protected = (labels != 'unknown') & ~np.isin(labels, ['left_hand', 'right_hand'])
    assert np.array_equal(np.asarray(selected.vertices)[protected], original_vertices[protected])
    applied = any(r['thickness_applied'] or r['density_cut_applied'] for r in records)
    return selected, {'applied': applied, 'parts': records,
        'removed_faces': len(original_faces)-len(selected.triangles), 'vertices_preserved': True,
        'source_points_removed': 0, 'gap_closure_after_filter': False,
        'allow_new_open_boundaries': allow_open_cuts}
