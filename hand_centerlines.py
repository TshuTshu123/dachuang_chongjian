"""手部中心线共享算法：掌部根区域、多尺度路径匹配、三维观测支撑。
由手部优化和 tools/hand_branch_trial.py 调用，不单独修改或输出人体网格。
输入毫米单位的手部局部点云；评分为启发式证据，不是关节识别概率。
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import linear_sum_assignment


def hand_frame(vertices, parts, name):
    ids = np.asarray(parts[name]['vertex_indices'], dtype=int)
    arm_class = name.replace('hand', 'lower_arm')
    arm = np.asarray(parts[arm_class]['vertex_indices'], dtype=int)
    if len(arm) < 30:
        arm_class = 'upper_clothing'
        arm = np.asarray(parts[arm_class]['vertex_indices'], dtype=int)
    if min(len(ids), len(arm)) < 30:
        raise ValueError('Insufficient hand or forearm labels.')
    nearest_arm = cKDTree(vertices[arm]).query(vertices[ids])[0]
    wrist = np.median(vertices[ids[nearest_arm <= np.quantile(nearest_arm, .08)+3]], axis=0)
    _, axes = np.linalg.eigh(np.cov(vertices[ids].T))
    normal = axes[:, 0]
    forward = vertices[ids].mean(0) - wrist
    forward -= (forward @ normal)*normal
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, normal)
    return ids, arm_class, wrist, np.column_stack((right, forward, normal))


def trace_mask(mask, origin, voxel):
    """Use a broad palm as a virtual root; follow interior-biased shortest paths."""
    xy = np.argwhere(mask)
    if len(xy) < 30:
        return [], None
    distance = ndi.distance_transform_edt(mask)*voxel
    yy = origin[1]+np.arange(mask.shape[1])*voxel
    # Limit palm search to the proximal half, excluding wrist and distal fingers.
    length = float(np.quantile(xy[:, 1]*voxel+origin[1], .98))
    proximal = mask & ((yy >= 8) & (yy <= max(25, length*.62)))[None, :]
    if not proximal.any():
        return [], None
    seed = np.unravel_index(np.argmax(np.where(proximal, distance, 0)), mask.shape)
    peak = distance[seed]
    if peak < 7:
        return [], None
    core = mask & (distance >= max(8, peak*.60))
    components, _ = ndi.label(core, np.ones((3, 3)))
    core = components == components[seed]
    palm = ndi.binary_dilation(core, iterations=max(1, round(5/voxel))) & mask
    # Wrist-facing area is never an endpoint candidate, including cropped forearm.
    wrist_limit = max(10., origin[1]+seed[1]*voxel-peak*.6)
    palm |= mask & (yy < wrist_limit)[None, :]
    lookup = np.full(mask.shape, -1, np.int32)
    lookup[tuple(xy.T)] = np.arange(len(xy))
    rows, cols, costs, lengths = [], [], [], []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        other = xy+np.array([dx, dy])
        valid = np.all((other >= 0) & (other < mask.shape), axis=1)
        ids = np.flatnonzero(valid)
        targets = lookup[tuple(other[valid].T)]
        ids, targets = ids[targets >= 0], targets[targets >= 0]
        # Do not connect diagonally across two empty corners.
        if dx and dy:
            keep = mask[xy[ids, 0]+dx, xy[ids, 1]] | mask[xy[ids, 0], xy[ids, 1]+dy]
            ids, targets = ids[keep], targets[keep]
        step = np.hypot(dx, dy)*voxel
        width = (distance[tuple(xy[ids].T)]+distance[tuple(xy[targets].T)])/2
        weight = step*(1+(4/np.maximum(width, voxel*.6))**2)
        rows.extend((ids, targets)); cols.extend((targets, ids))
        costs.extend((weight, weight)); lengths.extend((np.full(len(ids), step),)*2)
    roots = lookup[palm]
    virtual = len(xy)
    rows.extend((np.full(len(roots), virtual), roots))
    cols.extend((roots, np.full(len(roots), virtual)))
    costs.extend((np.full(len(roots), 1.e-6),)*2)
    lengths.extend((np.full(len(roots), 1.e-6),)*2)
    graph = coo_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(virtual+1, virtual+1)).tocsr()
    _, predecessor = dijkstra(graph, indices=virtual, return_predecessors=True)
    metric = coo_matrix((np.concatenate(lengths), (np.concatenate(rows), np.concatenate(cols))),
                        shape=graph.shape).tocsr()
    geodesic = dijkstra(metric, indices=virtual)[:-1]
    covered = palm.copy()
    branches = []
    rejected = []
    for _ in range(10):
        score = np.where(~covered[tuple(xy.T)] & np.isfinite(geodesic), geodesic, -1)
        tip = int(np.argmax(score))
        if score[tip] < 18:
            break
        chain, current = [], tip
        while current != virtual and current >= 0 and len(chain) <= len(xy):
            chain.append(current)
            if palm[tuple(xy[current])]:
                break
            current = int(predecessor[current])
        if not chain or current < 0:
            break
        pixels = xy[chain[::-1]]
        points = pixels*voxel+origin
        arc = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        width = distance[tuple(pixels.T)]
        distal_width = width[len(width)//2:]
        novel = float((~covered[tuple(pixels.T)]).sum()*voxel)
        # Invalidate an entire local tube rather than repeatedly selecting its boundary.
        for pixel, radius in zip(pixels, width):
            if palm[tuple(pixel)]:
                continue
            r = np.clip(radius*1.7, 5., 14.)/voxel
            lo = np.maximum(0, pixel-int(np.ceil(r)))
            hi = np.minimum(mask.shape, pixel+int(np.ceil(r))+1)
            grid = np.ogrid[lo[0]:hi[0], lo[1]:hi[1]]
            covered[lo[0]:hi[0], lo[1]:hi[1]] |= sum((a-b)**2 for a, b in zip(grid, pixel)) <= r*r
        if (arc < 18 or novel < 12 or np.quantile(distal_width, .65) > 18
                or arc < 3.2*np.median(width)):
            rejected.append({'tip_xy': points[-1].tolist(), 'arc_mm': arc,
                             'novel_mm': novel, 'median_radius_mm': float(np.median(width)),
                             'radius_q65_mm': float(np.quantile(width, .65))})
            continue
        branches.append({'xy': points, 'arc_mm': arc, 'radii_mm': width,
                         'tip_xy': points[-1], 'geodesic_mm': float(geodesic[tip])})
    return branches, {'mask': palm, 'origin': origin, 'voxel_mm': voxel,
                       'palm_center_xy': np.array(seed)*voxel+origin,
                       'palm_radius_mm': float(peak), 'rejected_paths': rejected}


def extract(points, voxel=2., threshold_scale=1.):
    # Equal-size 3D cells keep a dense camera overlap from dominating projected density.
    cells = np.floor(points/1.2).astype(np.int32)
    _, ids = np.unique(cells, axis=0, return_index=True)
    uniform = points[ids]
    origin = np.floor(uniform[:, :2].min(0)/voxel)*voxel-4*voxel
    indices = np.rint((uniform[:, :2]-origin)/voxel).astype(int)
    shape = indices.max(0)+5
    histogram = np.zeros(shape, float)
    np.add.at(histogram, tuple(indices.T), 1)
    density = ndi.gaussian_filter(histogram, sigma=max(.6, 1.2/voxel))
    threshold = np.median(density[tuple(indices.T)])*.40*threshold_scale
    mask = density >= threshold
    # Fill only small sampling pinholes; retain large enclosed gaps between fingers.
    holes = ndi.binary_fill_holes(mask) & ~mask
    cc, n = ndi.label(holes)
    sizes = np.bincount(cc.ravel())*voxel**2
    small = np.flatnonzero((sizes <= 20) & (np.arange(len(sizes)) > 0))
    mask |= np.isin(cc, small)
    yy = origin[1]+np.arange(mask.shape[1])*voxel
    mask[:, yy < -8] = False
    cc, n = ndi.label(mask, np.ones((3, 3)))
    if not n:
        return [], None
    sizes = np.bincount(cc.ravel()); sizes[0] = 0
    mask = cc == sizes.argmax()
    return trace_mask(mask, origin, voxel)


def resample(path, step=2.):
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if length[-1] < step:
        return path
    target = np.linspace(0, length[-1], max(2, int(length[-1]/step)+1))
    return np.column_stack([np.interp(target, length, path[:, i]) for i in range(path.shape[1])])


def match_cost(a, b):
    tip = np.linalg.norm(a['tip_xy']-b['tip_xy'])
    pa, pb = resample(a['xy']), resample(b['xy'])
    deviation = max(np.median(cKDTree(pa).query(pb)[0]), np.median(cKDTree(pb).query(pa)[0]))
    direction_a, direction_b = pa[-1]-pa[0], pb[-1]-pb[0]
    cosine = np.dot(direction_a, direction_b)/max(1.e-9, np.linalg.norm(direction_a)*np.linalg.norm(direction_b))
    return tip+deviation if tip <= 10 and deviation <= 4 and cosine >= .7 else np.inf


def match_branches(branches, other):
    if not branches or not other:
        return []
    costs = np.array([[match_cost(a, b) for b in other] for a in branches])
    row, col = linear_sum_assignment(np.where(np.isfinite(costs), costs, 1.e6))
    return [(int(i), int(j)) for i, j in zip(row, col) if np.isfinite(costs[i, j])]


def assess(points):
    configs = [(v, s) for v in (1.5, 2., 2.5) for s in (.8, 1., 1.2)]
    trials = [extract(points, *config) for config in configs]
    # Choose an actual cross-scale medoid, not the run with the most branches.
    similarities = np.zeros(len(trials))
    for i, (branches, _) in enumerate(trials):
        for j, (other, _) in enumerate(trials):
            if i == j:
                continue
            count = len(match_branches(branches, other))
            similarities[i] += 2*count/max(1, len(branches)+len(other))
    best = int(np.argmax(similarities))
    _, palm = trials[best]
    # A single representative run may miss a visible digit. Form branch consensus
    # across all runs, counting at most one vote per parameter configuration.
    pool = [(i, branch) for i, (branches, _) in enumerate(trials) for branch in branches]
    remaining = set(range(len(pool)))
    branches, consensus = [], []
    while remaining:
        groups = []
        for seed in sorted(remaining):
            members = {}
            for k in sorted(remaining):
                cost = match_cost(pool[seed][1], pool[k][1])
                config = pool[k][0]
                if np.isfinite(cost) and (config not in members or cost < members[config][0]):
                    members[config] = (cost, k)
            groups.append((len(members), -sum(v[0] for v in members.values()), -seed, members))
        _, _, negative_seed, members = max(groups, key=lambda item: item[:3])
        claimed = [item[1] for item in members.values()]
        remaining.difference_update(claimed)
        if len(members) < 2:
            continue
        branches.append(pool[-negative_seed][1])
        consensus.append([(config, pool[k][1]) for config, (_, k) in members.items()])
    raw_xy = cKDTree(points[:, :2])
    output = []
    for branch_id, branch in enumerate(branches):
        matched = consensus[branch_id]
        stability = len(matched)/len(configs)
        xy = resample(branch['xy'])
        # Retrieve nearby observed depth, never midpoint an arbitrary 40 mm slab.
        neighborhoods = raw_xy.query_ball_point(xy, 5.)
        depth = np.array([np.median(points[ids, 2]) if ids else np.nan for ids in neighborhoods])
        available = np.isfinite(depth)
        if available.any():
            depth = np.interp(np.arange(len(depth)), np.flatnonzero(available), depth[available])
        else:
            depth[:] = np.median(points[:, 2])
        tangent = np.gradient(xy, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.e-9)
        normal = np.c_[-tangent[:, 1], tangent[:, 0]]
        # Cross-sections away from the terminal 6 mm should contain both lateral sides.
        section_ok, depth_spans = [], []
        for k in range(1, max(1, len(xy)-3)):
            ids = raw_xy.query_ball_point(xy[k], 16.)
            delta = points[ids, :2]-xy[k]
            selected = np.abs(delta @ tangent[k]) <= 3
            across = (delta @ normal[k])[selected]
            z = points[ids, 2][selected]
            section_ok.append(len(across) >= 5 and np.quantile(across, .1) < -1.5
                              and np.quantile(across, .9) > 1.5)
            if len(z) >= 5:
                depth_spans.append(float(np.quantile(z, .9)-np.quantile(z, .1)))
        support = float(np.mean(section_ok)) if section_ok else 0.
        coverage = float(available.mean())
        jump = float(np.max(np.abs(np.diff(depth)))) if len(depth) > 1 else 0.
        span = float(np.median(depth_spans)) if depth_spans else 0.
        score = stability*.55+support*.3+coverage*.15
        reasons = []
        if stability < .75: reasons.append('unstable_across_scales')
        if support < .65: reasons.append('weak_cross_section_support')
        if coverage < .9: reasons.append('missing_observed_depth')
        if jump > 8: reasons.append('depth_discontinuity_or_overlap')
        if span > 35: reasons.append('thick_or_overlapping_surfaces')
        tip_spread = max((np.linalg.norm(b['tip_xy']-branch['tip_xy']) for _, b in matched), default=0.)
        output.append({'local_path_mm': np.c_[xy, depth].tolist(),
                       'length_mm': branch['arc_mm'], 'stability': stability,
                       'cross_section_support': support, 'observed_depth_coverage': coverage,
                       'max_depth_step_mm': jump, 'median_depth_span_mm': span,
                       'matched_tip_max_offset_mm': float(tip_spread), 'evidence_score': score,
                       'status': 'supported_candidate' if not reasons else 'uncertain',
                       'reasons': reasons})
    # Several stable tips can still be branches of the same unresolved thick tube.
    # Mark shared stems ambiguous instead of treating every stable tip as a digit.
    for i, first in enumerate(output):
        for second in output[i+1:]:
            if first['status'] != 'supported_candidate' or second['status'] != 'supported_candidate':
                continue
            a, b = np.array(first['local_path_mm'])[:, :2], np.array(second['local_path_mm'])[:, :2]
            shared = min((cKDTree(a).query(b)[0] < 6).mean(), (cKDTree(b).query(a)[0] < 6).mean())
            if shared > .55 and np.linalg.norm(a[-1]-b[-1]) < 30:
                for branch in (first, second):
                    if 'shared_stem_unresolved' not in branch['reasons']:
                        branch['reasons'].append('shared_stem_unresolved')
                    branch['status'] = 'uncertain'
    return output, palm, {'configurations': [{'voxel_mm': v, 'threshold_scale': s,
            'candidate_branches': len(result[0])} for (v, s), result in zip(configs, trials)],
            'representative_configuration': best,
            'density_threshold_median_factor': .40,
            'stability_minimum': .75, 'cross_section_support_minimum': .65,
            'minimum_depth_coverage': .9, 'maximum_depth_step_mm': 8,
            'maximum_depth_span_mm': 35,
            'score_is_probability': False, 'anatomical_joints_identified': False}


