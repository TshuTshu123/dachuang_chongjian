"""掌部整体限厚候选：用腕部参考、稳定指根和观测密度确定平滑体积压缩。

内部模块，统一流程验证通过后调用；不按扫描编号设置参数。
采用共享截面参考和平滑深度变换，避免逐顶点硬裁剪形成凹坑。
腕口和指根渐变，指尖固定；较有观测支持的一面作为锚面。
默认模式保留面连接；参考包络实验允许局部重新三角化。
厚度是外观约束，不是解剖测量；局部相交、翻折或观测退化时回退。
曲面锚定掌心实验：python tools/hand_skeleton_trial.py --mode focus --scan 3
该实验不是 run.py 的默认模式；手指 12 mm 实验与本模块分开。
参考面优先实验：python tools/hand_skeleton_trial.py --mode reference --scan 3
只用包络内观测作最多 1 mm 拟合；包络外点不作拟合或贴合检查依据。
整掌实验：python tools/hand_skeleton_trial.py --mode whole --scan 3 --keep-full 3
两侧都用 20 mm 简化参考厚度，掌根连续过渡；不是默认或解剖厚度标准。
"""

import copy

import numpy as np
import open3d as o3d
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.optimize import least_squares

from hand_centerlines import hand_frame
from hand_density_constraints import CoverageField, paired_bounds, point_spacing
from hand_pipeline import depths, pairs, scene_for, smoothstep
from surface_evaluation import distances


def ellipse_sections(points, palm_end, spacing, levels=None):
    """Fit observed cross-sectional rings, rejecting missing sectors and outliers."""
    sections = []
    for y in (np.r_[[-28., -18., -8.], np.linspace(10., palm_end, 7)] if levels is None else levels):
        cloud = points[np.abs(points[:, 1]-y) <= max(3., spacing*2), :][:, [0, 2]]
        if len(cloud) < 35:
            sections.append({'y_mm': float(y), 'valid': False, 'reason': 'Too few ring points'}); continue
        lo, hi = np.quantile(cloud, [.03, .97], axis=0)
        center = (lo+hi)/2
        radius = np.maximum((hi-lo)/2, 3.)
        seed = np.r_[center, np.log(radius), 0.]
        angles = np.arctan2((cloud[:, 1]-center[1])/radius[1], (cloud[:, 0]-center[0])/radius[0])
        bins = np.minimum(11, ((angles+np.pi)/(2*np.pi)*12).astype(int))
        balanced = np.concatenate([np.flatnonzero(bins == k)[::max(1, int((bins == k).sum()/20))] for k in range(12)])
        sample = cloud[balanced]
        def residual(params):
            cx, cz, la, lb, angle = params
            a, b = np.exp([la, lb]); c, s = np.cos(angle), np.sin(angle)
            x, z = (sample-[cx, cz]).T
            u, v = c*x+s*z, -s*x+c*z
            return (np.sqrt((u/a)**2+(v/b)**2)-1)*min(a, b)
        result = least_squares(residual, seed, loss='soft_l1', f_scale=max(1., spacing), max_nfev=120,
            bounds=(np.r_[center-radius*.6, np.log(radius*.6), -.35],
                    np.r_[center+radius*.6, np.log(radius*1.7), .35]))
        cx, cz, la, lb, angle = result.x
        a, b = np.exp([la, lb]); c, s = np.cos(angle), np.sin(angle)
        x, z = (cloud-[cx, cz]).T
        theta = np.arctan2((-s*x+c*z)/b, (c*x+s*z)/a)
        sector = np.minimum(11, ((theta+np.pi)/(2*np.pi)*12).astype(int))
        counts = np.bincount(sector, minlength=12)
        coverage = int((counts >= 3).sum())
        error = float(np.quantile(np.abs(residual(result.x)), .9))
        valid = bool(result.success and coverage >= 9 and error <= max(3., spacing*2)
                     and 6 <= b*2 <= 80 and 10 <= a*2 <= 160)
        sections.append({'y_mm': float(y), 'valid': valid, 'angular_sectors': coverage,
                         'width_mm': float(a*2), 'depth_mm': float(b*2),
                         'ring_fit_p90_mm': error, 'angle_degrees': float(np.degrees(angle)),
                         'center_xz_mm': [float(cx), float(cz)]})
    valid_ids = [i for i, p in enumerate(sections) if p['valid']]
    for k in range(1, len(valid_ids)-1):
        a, b, c = [sections[valid_ids[j]] for j in (k-1, k, k+1)]
        expected = np.median([a['depth_mm'], c['depth_mm']])
        if not .65*expected <= b['depth_mm'] <= expected*1.5:
            b['valid'] = False; b['reason'] = 'Isolated section thickness spike'
    return sections


def design(xy, scale):
    x, y = (xy/scale).T
    return np.c_[np.ones(len(x)), x, y, x*x, x*y, y*y]


def robust_anchor(xy, z, scale):
    matrix = design(xy, scale)
    weights = np.ones(len(z))
    for _ in range(6):
        coef = np.linalg.lstsq(matrix*weights[:, None]**.5, z*weights**.5, rcond=None)[0]
        residual = z-matrix @ coef
        sigma = max(.7, 1.4826*np.median(np.abs(residual-np.median(residual))))
        weights = np.minimum(1., 1.5*sigma/np.maximum(np.abs(residual), 1.e-9))
    return coef, float(np.quantile(np.abs(residual), .9))


def smooth_positive(values, width):
    """C1 positive part with derivative in [0,1]; the anchor shell stays fixed."""
    return np.where(values <= 0, 0., np.where(values < width, values*values/(2*width), values-width/2))


def region_fade(vertices, faces, region, width):
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    boundary = np.unique(edges[np.any(region[edges], axis=1) & ~np.all(region[edges], axis=1)])
    boundary = boundary[region[boundary]]
    if not len(boundary):
        return np.zeros(len(vertices))
    e = edges[np.all(region[edges], axis=1)]
    length = np.linalg.norm(vertices[e[:, 0]]-vertices[e[:, 1]], axis=1)
    n = len(vertices)
    graph = coo_matrix((np.r_[length, length, np.zeros(len(boundary))],
        (np.r_[e[:, 0], e[:, 1], np.full(len(boundary), n)], np.r_[e[:, 1], e[:, 0], boundary])), shape=(n+1, n+1)).tocsr()
    distance = dijkstra(graph, directed=True, indices=n)[:-1]
    fade = np.zeros(n)
    fade[region] = smoothstep(distance[region]/width)
    return fade


def palm_candidate(mesh, raw_mm, info, frames, palm_focus=False, reference_first=False, whole_palm=False):
    reference_first = reference_first or whole_palm
    palm_focus = palm_focus or reference_first
    original = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.triangles)
    labels = np.full(len(original), 'unknown', dtype=object)
    for name, part in info['parts'].items(): labels[part['vertex_indices']] = name
    known = np.flatnonzero(labels != 'unknown')
    nearest_class = labels[known[cKDTree(original[known]).query(original, workers=4)[1]]]
    raw_class = labels[cKDTree(original).query(raw_mm, workers=4)[1]]
    selected = copy.deepcopy(mesh)
    records = []
    for saved in frames:
        faces = np.asarray(selected.triangles)
        name = saved['part']
        record = {'part': name, 'applied': False}
        try:
            ids, arm, wrist, axes = hand_frame(original, info['parts'], name)
        except (KeyError, ValueError, np.linalg.LinAlgError) as error:
            records.append({**record, 'reason': str(error)}); continue
        local = (original-wrist) @ axes
        raw_local = (raw_mm-wrist) @ axes
        paths = [(np.asarray(b['world_path_mm'])-wrist) @ axes for b in saved['branches']
                 if b['status'] == 'supported_candidate']
        hand_length = float(np.quantile(local[ids, 1], .98))
        palm_end = float(np.median([p[0, 1] for p in paths])) if len(paths) >= 2 else hand_length*.55
        palm_end = float(np.clip(palm_end, hand_length*.3, hand_length*.7))
        low = local[ids].min(0)-[15, 45, 20]
        high = local[ids].max(0)+[15, 15, 20]
        allowed = np.isin(labels, [name, arm, 'unknown'])
        allowed &= (labels != 'unknown') | np.isin(nearest_class, [name, arm])
        region = allowed & np.all((local >= low) & (local <= high), axis=1)
        other = np.flatnonzero((labels != 'unknown') & ~np.isin(labels, [name, arm]))
        if len(other) and np.quantile(cKDTree(original[other]).query(original[ids])[0], .05) < 6:
            records.append({**record, 'reason': 'Contact with another labeled part'}); continue
        observed = (np.all((raw_local >= low) & (raw_local <= high), axis=1)
            & np.isin(raw_class, [name, arm, 'unknown'])
            & (cKDTree(original[ids]).query(raw_mm, workers=4)[0] < 25))
        points = raw_local[observed]
        if len(points) < 150 or palm_end < 30:
            records.append({**record, 'reason': 'Insufficient palm/wrist extent'}); continue
        spacing = point_spacing(points)
        cylinders = ellipse_sections(points, palm_end, spacing)
        record['elliptical_sections'] = cylinders
        coverage = CoverageField(points, spacing)
        surface_region = np.all((local >= low) & (local <= high), axis=1)
        scene = scene_for(selected, surface_region)
        # Follow the forearm direction at the wrist; hand/forearm axes can bend.
        arm_ids = np.asarray(info['parts'][arm]['vertex_indices'], int)
        arm_distance = np.linalg.norm(original[arm_ids]-wrist, axis=1)
        arm_near = original[arm_ids[(arm_distance > 10) & (arm_distance < 50)]]
        tangent = wrist-np.median(arm_near, axis=0) if len(arm_near) >= 20 else axes[:, 1].copy()
        tangent /= max(np.linalg.norm(tangent), 1.e-9)
        wrist_normal = axes[:, 2]-np.dot(axes[:, 2], tangent)*tangent
        if np.linalg.norm(wrist_normal) < .3:
            records.append({**record, 'reason': 'Wrist cross-section direction ambiguous'}); continue
        wrist_normal /= np.linalg.norm(wrist_normal)
        wrist_across = np.cross(tangent, wrist_normal)
        xx, yy = np.meshgrid(np.linspace(-10, 10, 7), np.linspace(-30, -5, 8))
        wrist_xy = np.c_[xx.ravel(), yy.ravel()]
        wrist_probes = wrist+xx.ravel()[:, None]*wrist_across+yy.ravel()[:, None]*tangent
        wrist_region = (np.linalg.norm(original-wrist, axis=1) < 60) & allowed
        _, _, wrist_gap, ok_wrist = depths(scene_for(selected, wrist_region), wrist_probes, wrist_normal)
        ok_wrist &= (wrist_gap >= 8) & (wrist_gap <= 80)
        if ok_wrist.sum() < 20:
            records.append({**record, 'reason': 'Insufficient complete wrist sections'}); continue
        wrist_ref = float(np.median(wrist_gap[ok_wrist]))
        # Raw paired wrist sections can tighten the reference only when stable.
        arm_axes = np.column_stack((wrist_across, tangent, wrist_normal))
        wrist_points = (raw_mm[observed]-wrist) @ arm_axes
        wrist_sections = ellipse_sections(wrist_points, palm_end, spacing, levels=[-28., -18., -8.])
        record['wrist_elliptical_sections'] = wrist_sections
        lo, hi = paired_bounds(wrist_points, wrist_xy, spacing)
        raw_gap = hi-lo
        good_raw = np.isfinite(raw_gap) & (raw_gap >= 8) & (raw_gap <= 80)
        raw_ref = None
        if good_raw.sum() >= 20:
            q = np.quantile(raw_gap[good_raw], [.25, .5, .75])
            if q[2]-q[0] <= q[1]*.3:
                raw_ref = float(q[1])
                wrist_ref = max(wrist_ref*.7, min(wrist_ref, raw_ref+spacing))
        cylinder_wrist = [p['depth_mm'] for p in wrist_sections if p['valid']]
        if len(cylinder_wrist) >= 2:
            ring_ref = float(np.median(cylinder_wrist))
            if .6*wrist_ref <= ring_ref <= wrist_ref*1.3:
                wrist_ref = max(wrist_ref*.8, min(wrist_ref, ring_ref+spacing))
                record['cylinder_wrist_reference_mm'] = ring_ref
        target = max(wrist_ref*1.1, wrist_ref+spacing)
        cylinder_palm = [p['depth_mm'] for p in cylinders if p['valid'] and 12 <= p['y_mm'] <= palm_end*.8]
        if len(cylinder_palm) >= 2:
            ring_target = float(np.quantile(cylinder_palm, .25)+spacing)
            if min(cylinder_palm) >= max(cylinder_palm)*.65:
                target = max(wrist_ref*.8, min(target, ring_target))
                record['cylinder_palm_reference_mm'] = ring_target
        if palm_focus:
            # Research mode: a wrist-sized appearance target for the PALM only.
            target = max(22., min(35., wrist_ref))
        if reference_first:
            target = max(20., min(30., wrist_ref*.9))
        if whole_palm:
            target = 20.
        palm_vertices = ids[(local[ids, 1] > 10) & (local[ids, 1] < palm_end+5)]
        if len(palm_vertices) < 50:
            records.append({**record, 'reason': 'Palm area too small'}); continue
        xlow, xhigh = np.quantile(local[palm_vertices, 0], [.12, .88])
        grid_x, grid_y = np.meshgrid(np.linspace(xlow, xhigh, 20), np.linspace(12, palm_end, 20))
        xy = np.c_[grid_x.ravel(), grid_y.ravel()]
        world_probes = wrist+np.c_[xy, np.zeros(len(xy))] @ axes.T
        lower, upper, gap, valid = depths(scene, world_probes, axes[:, 2])
        valid &= coverage.score(xy) >= coverage.threshold
        valid &= (gap >= 5) & (gap <= 120)
        if valid.sum() < 80:
            records.append({**record, 'reason': 'Palm sections incomplete'}); continue
        before_p85 = float(np.quantile(gap[valid], .85))
        before_p95 = float(np.quantile(gap[valid], .95))
        record.update(palm_end_mm=palm_end, wrist_reference_mm=wrist_ref,
                      raw_wrist_reference_mm=raw_ref, target_thickness_mm=target,
                      before_palm_p85_mm=before_p85, before_palm_p95_mm=before_p95,
                      valid_palm_rays=int(valid.sum()))
        if before_p85 <= target*1.1 or before_p85-target < max(4., spacing*2):
            records.append({**record, 'reason': 'Palm thickness is within the automatic reference range'}); continue
        # Choose the side with stronger original-point support as the anchor.
        raw_tree = cKDTree(raw_mm[observed])
        back_low = wrist+np.c_[xy[valid], lower[valid]] @ axes.T
        back_high = wrist+np.c_[xy[valid], upper[valid]] @ axes.T
        dlow = raw_tree.query(back_low)[0]; dhigh = raw_tree.query(back_high)[0]
        record['observed_side_support'] = {'lower_fraction': float((dlow <= spacing*2).mean()),
            'upper_fraction': float((dhigh <= spacing*2).mean()),
            'lower_median_mm': float(np.median(dlow)), 'upper_median_mm': float(np.median(dhigh))}
        if not reference_first and min((dlow <= spacing*2).mean(), (dhigh <= spacing*2).mean()) >= .85:
            records.append({**record, 'reason': 'Both palm surfaces have strong original observation support'}); continue
        sign = 1. if np.median(dlow) <= np.median(dhigh) else -1.
        z_anchor = lower[valid] if sign > 0 else upper[valid]
        anchor_support = dlow if sign > 0 else dhigh
        opposite_support = dhigh if sign > 0 else dlow
        strong = anchor_support <= max(4., spacing*2)
        if strong.sum() < 40:
            records.append({**record, 'reason': 'No sufficiently supported anchor side'}); continue
        scale_xy = max(40., palm_end)
        coef, anchor_error = robust_anchor(xy[valid][strong], z_anchor[strong], scale_xy)
        def anchor_values(query_xy):
            return design(query_xy, scale_xy) @ coef
        if palm_focus:
            # A curved supported back cannot always be represented by one quadratic.
            mask = np.zeros(len(xy)); values = np.zeros(len(xy))
            support_ids = np.flatnonzero(valid)[strong]
            mask[support_ids] = 1.; values[support_ids] = z_anchor[strong]
            denominator = ndi.gaussian_filter(mask.reshape(grid_x.shape), 1.3)
            numerator = ndi.gaussian_filter(values.reshape(grid_x.shape), 1.3)
            field = numerator/np.maximum(denominator, 1.e-9)
            missing = denominator < .01
            if missing.any() and (~missing).any():
                nearest = ndi.distance_transform_edt(missing, return_distances=False, return_indices=True)
                field[missing] = field[tuple(nearest[:, missing])]
            def anchor_values(query_xy):
                qx = (query_xy[:, 0]-xlow)/max(xhigh-xlow, 1.e-6)*19
                qy = (query_xy[:, 1]-12)/max(palm_end-12, 1.e-6)*19
                return ndi.map_coordinates(field, np.array([qy, qx]), order=1, mode='nearest')
            anchor_error = float(np.quantile(np.abs(anchor_values(xy[support_ids])-z_anchor[strong]), .9))
        if anchor_error > max(5., spacing*3):
            records.append({**record, 'reason': 'Supported side is too irregular for a smooth palm warp'}); continue
        # Density/support changes only a broad smooth field, never a vertex clamp.
        confidence = np.zeros(len(xy))
        confidence[valid] = np.exp(-.5*(opposite_support/max(2., spacing*2))**2)
        confidence_grid = ndi.gaussian_filter(confidence.reshape(grid_x.shape), 2.)
        vx = (local[:, 0]-xlow)/max(xhigh-xlow, 1.e-6)*19
        vy = (local[:, 1]-12)/max(palm_end-12, 1.e-6)*19
        conf = ndi.map_coordinates(confidence_grid, np.array([vy, vx]), order=1, mode='nearest')
        movable = region & ((labels == name) | ((labels == 'unknown') & (nearest_class == name)))
        if whole_palm:
            # Unlabeled generated palm/heel faces can be closer to forearm labels.
            # Include those inside the hand enclosure; the labeled arm stays fixed.
            movable |= region & (labels == 'unknown') & (nearest_class == arm) & (local[:, 1] > 0) & (local[:, 1] <= palm_end+10)
        fade = region_fade(original, faces, movable, 15.)
        fade *= smoothstep(local[:, 1]/25)
        fade *= smoothstep((palm_end+18-local[:, 1])/28)
        if palm_focus:
            # Same smooth XY field on both sides of the palm; no finger cap.
            fade = smoothstep(local[:, 1]/18)*smoothstep((palm_end+28-local[:, 1])/28)
        if whole_palm:
            # Entire palm shares one envelope; only the actual wrist mouth fades.
            fade = smoothstep(local[:, 1]/10)*smoothstep((palm_end+35-local[:, 1])/25)
        # Finger branches taper out before their distal part, including the thumb.
        for path in paths:
            near, nearest = cKDTree(path[:, :2]).query(local[:, :2])
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
            if palm_focus:
                tangent_xy = np.gradient(path[:, :2], axis=0)
                tangent_xy /= np.maximum(np.linalg.norm(tangent_xy, axis=1, keepdims=True), 1.e-9)
                along = np.clip(arc[nearest]+np.einsum('ij,ij->i', local[:, :2]-path[nearest, :2], tangent_xy[nearest]), 0, arc[-1])
                across = 1-smoothstep((near-10)/14)
                if whole_palm:
                    distal = smoothstep((along-25)/12)*smoothstep((local[:, 1]-palm_end*.75)/20)
                    fade *= 1-across*distal
                else:
                    fade *= 1-across*smoothstep((along-15)/15)
                tip = (near < 10) & ((arc[-1]-along) < max(8., spacing*2))
                fade[tip] = 0
                continue
            finger = (near < 16) & (arc[nearest] > 15)
            fade[finger] *= 1-smoothstep((arc[nearest][finger]-15)/15)
            tip = (near < 20) & ((arc[-1]-arc[nearest]) < max(8., spacing*2))
            fade[tip] = 0
        anchor = anchor_values(local[:, :2])
        shell = max(2., spacing)
        positive = smooth_positive(sign*(local[:, 2]-anchor)-shell, 3.)
        compression = float(np.clip(1-target/before_p85, 0., .7 if palm_focus else .4))
        strength = fade if palm_focus else fade*(1-.45*conf)
        delta = -sign*compression*strength*positive
        fit_observed = observed.copy()
        if reference_first:
            # Supported back + an offset reference surface define a new envelope.
            # Outside observations never enter fitting or the distance quality gate.
            signed_raw = sign*(points[:, 2]-anchor_values(points[:, :2]))
            retained = (signed_raw >= -3.) & (signed_raw <= target+2.)
            fit_observed[np.flatnonzero(observed)[~retained]] = False
            fit_points = points[retained]
            fit_depth = signed_raw[retained]
            near_reference = np.abs(fit_depth-target) <= 5.
            correction = np.zeros(len(local))
            if near_reference.sum() >= 15:
                source_xy = fit_points[near_reference, :2]
                source_residual = np.clip(fit_depth[near_reference]-target, -1., 1.)
                d, idx = cKDTree(source_xy).query(local[:, :2], k=min(12, len(source_xy)), workers=4)
                if d.ndim == 1:
                    d, idx = d[:, None], idx[:, None]
                weights = np.exp(-.5*(d/8.)**2)
                correction = (weights*source_residual[idx]).sum(1)/np.maximum(weights.sum(1), 1.e-12)
                correction *= np.exp(-.5*(d[:, 0]/12.)**2)
            depth_limit = np.maximum(12., target-shell+correction)
            # Smooth approach to the reference, with positive depth derivative.
            # Unlike a hard clamp this keeps side faces from all collapsing.
            capped = depth_limit*(-np.expm1(-positive/depth_limit))
            delta = sign*strength*(capped-positive)
            record.update(reference_first=True, reference_fit_max_mm=1.,
                          whole_palm=whole_palm,
                          retained_observations=int(retained.sum()),
                          ignored_observations=int((~retained).sum()),
                          outside_observations_used_for_fit=False,
                          fit_observation_scope='Inside the new reference envelope only')
        move_limit = 60. if reference_first else (30. if palm_focus else 16.)
        delta = np.clip(delta, -move_limit, move_limit)
        delta[~movable] = 0
        active = np.abs(delta) > .05
        delta[~active] = 0
        if active.sum() < 30:
            records.append({**record, 'reason': 'No coherent movable palm area'}); continue
        before = np.asarray(selected.vertices).copy()
        p = before[faces]
        old_cross = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        box = before[active]
        if reference_first:
            box = np.r_[box, before[active]+delta[active, None]*axes[:, 2]]
        face_ids = np.flatnonzero(np.all(p.max(1) >= box.min(0)-8, axis=1) & np.all(p.min(1) <= box.max(0)+8, axis=1))
        original_pairs = pairs(selected, face_ids, active)
        # Actual observed points near the well-supported anchor, not mesh samples.
        near_anchor = np.abs(points[:, 2]-anchor_values(points[:, :2])) <= max(4., spacing*2)
        near_anchor &= (points[:, 1] > 10) & (points[:, 1] < palm_end)
        if reference_first:
            near_anchor &= retained
        anchor_queries = raw_mm[observed][near_anchor]
        if len(anchor_queries) < 30:
            records.append({**record, 'reason': 'Insufficient observed anchor points'}); continue
        all_queries = raw_mm[fit_observed]
        before_anchor = distances(selected, anchor_queries)
        before_fit = distances(selected, all_queries)
        base_mesh = copy.deepcopy(selected)
        if reference_first:
            from finger_envelope import repair_local_folds, crossing_signatures
            original_signatures = crossing_signatures(base_mesh, face_ids, active)
        checks = []
        best_gain = 0.
        for amount in ((1., .9, .8, .75, .5) if palm_focus else (1., .75, .5)):
            column_fade = np.ones(len(before))
            coords = before+(amount*delta)[:, None]*axes[:, 2]
            candidate = copy.deepcopy(base_mesh); candidate.vertices = o3d.utility.Vector3dVector(coords)
            candidate.compute_vertex_normals()
            repair_iterations = 0
            retriangulated = 0
            for repair_iterations in range(10 if palm_focus else 4):
                p = coords[faces]
                cross = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
                touched = np.any(active[faces], axis=1)
                old_size = np.linalg.norm(old_cross, axis=1); size = np.linalg.norm(cross, axis=1)
                bad_ids = np.flatnonzero(touched & ((size < old_size*.05)
                    | (np.einsum('ij,ij->i', cross, old_cross) < .1*size*old_size)))
                if reference_first and 0 < len(bad_ids) <= (200 if whole_palm else 80):
                    bad_mask = np.zeros(len(faces), bool); bad_mask[bad_ids] = True
                    unpatched = copy.deepcopy(candidate)
                    repaired_ok = False
                    for _ in range(8 if whole_palm else 4):
                        repaired, ok, count = repair_local_folds(unpatched, bad_mask, movable, max_loop=512 if whole_palm else 128)
                        if not ok:
                            break
                        rf = np.asarray(repaired.triangles); rp = coords[rf]
                        ri = np.flatnonzero(np.all(rp.max(1) >= box.min(0)-8, axis=1) & np.all(rp.min(1) <= box.max(0)+8, axis=1))
                        conflicts = []
                        for a, b in pairs(repaired, ri, active):
                            signature = tuple(sorted((tuple(sorted(rf[a])), tuple(sorted(rf[b])))))
                            if signature not in original_signatures:
                                conflicts.extend(rf[[a, b]].ravel().tolist())
                        if not conflicts:
                            candidate = repaired; retriangulated = count; repaired_ok = True
                            break
                        expanded = np.any(np.isin(faces, np.unique(conflicts)), axis=1)
                        expanded &= np.all(movable[faces], axis=1)
                        extended_mask = bad_mask | expanded
                        if np.array_equal(extended_mask, bad_mask) or extended_mask.sum() > (1600 if whole_palm else 400):
                            break
                        bad_mask = extended_mask
                    if repaired_ok:
                        break
                added = pairs(candidate, face_ids, active)-original_pairs if not len(bad_ids) else set()
                if not len(bad_ids) and not added:
                    break
                affected = np.unique(np.r_[bad_ids, np.array(list(added), int).ravel()])
                centers = np.r_[local[faces[affected], :2].mean(1), local[np.unique(faces[affected]), :2]]
                # Whole XY columns fade together, avoiding vertexwise dents.
                nearest_column = cKDTree(centers).query(local[:, :2])[0]
                column_fade = np.minimum(column_fade, smoothstep(nearest_column/(12. if palm_focus else 8.)))
                coords = before+(amount*delta*column_fade)[:, None]*axes[:, 2]
                candidate.vertices = o3d.utility.Vector3dVector(coords)
                candidate.compute_vertex_normals()
            p = coords[faces]
            cross = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
            touched = np.any(active[faces], axis=1)
            old_size = np.linalg.norm(old_cross, axis=1); size = np.linalg.norm(cross, axis=1)
            bad = int(np.sum(touched & ((size < old_size*.05)
                | (np.einsum('ij,ij->i', cross, old_cross) < .1*size*old_size))))
            if retriangulated:
                triangles = np.asarray(candidate.triangles)
                xyz = coords[triangles]
                prior_faces = {tuple(sorted(t)) for t in faces.tolist()}
                added_faces = np.array([tuple(sorted(t)) not in prior_faces for t in triangles.tolist()])
                # Existing tiny faces outside the patch are not a repair regression.
                bad = int(np.sum(added_faces & (np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=1) < 1.e-8)))
            if reference_first and not bad:
                cp = coords[np.asarray(candidate.triangles)]
                ids_now = np.flatnonzero(np.all(cp.max(1) >= box.min(0)-8, axis=1) & np.all(cp.min(1) <= box.max(0)+8, axis=1))
                intersections = len(crossing_signatures(candidate, ids_now, active)-original_signatures)
            else:
                intersections = len(pairs(candidate, face_ids, active)-original_pairs) if not bad else None
            _, _, after_gap, after_valid = depths(scene_for(candidate, surface_region), world_probes, axes[:, 2])
            common = valid & after_valid
            after_p85 = float(np.quantile(after_gap[common], .85)) if common.any() else before_p85
            gain = float(np.quantile(gap[common], .85)-after_p85) if common.any() else 0.
            after_anchor = distances(candidate, anchor_queries)
            after_fit = distances(candidate, all_queries)
            anchor_loss = float(np.quantile(after_anchor, .95)-np.quantile(before_anchor, .95))
            full_loss = float(np.quantile(after_fit, .95)-np.quantile(before_fit, .95))
            accepted = bool(not bad and intersections == 0 and common.sum() >= valid.sum()*.98
                and gain >= max(2., before_p85*.04) and after_p85 >= wrist_ref*.65
                and anchor_loss <= max(.7, spacing*.5) and full_loss <= max(2., spacing*2))
            checks.append({'amount': amount, 'folded_faces': bad, 'new_local_intersections': intersections,
                           'column_repair_iterations': repair_iterations,
                           'palm_p85_gain_mm': gain, 'after_palm_p85_mm': after_p85,
                           'anchor_raw_distance_p95_increase_mm': anchor_loss,
                           'all_hand_raw_distance_p95_increase_mm': full_loss, 'accepted': accepted})
            if reference_first:
                checks[-1]['retriangulated_faces'] = retriangulated
            if palm_focus:
                print(f'Palm {"reference" if reference_first else "focus"} {name}: amount={amount}; folds={bad}; intersections={intersections}; P85={after_p85:.1f} mm; accepted={accepted}', flush=True)
            if accepted and gain > best_gain:
                selected = candidate
                best_gain = gain
                record.update(applied=True, movement_max_mm=float(np.abs(amount*delta*column_fade).max()),
                              moved_vertices=int((np.abs(amount*delta*column_fade) > .01).sum()), after_palm_p85_mm=after_p85,
                              after_palm_p95_mm=float(np.quantile(after_gap[common], .95)), selected_amount=amount)
                if not palm_focus:
                    break
        record.update(anchor_side='lower' if sign > 0 else 'upper', anchor_fit_p90_mm=anchor_error,
                      target_scale=1-compression, checks=checks,
                      reason='Coherent palm compression accepted' if record['applied'] else 'Coherent palm candidates rejected')
        records.append(record)
    protected = (labels != 'unknown') & ~np.isin(labels, ['left_hand', 'right_hand'])
    assert np.array_equal(np.asarray(selected.vertices)[protected], original[protected])
    if not reference_first:
        assert np.array_equal(selected.triangles, mesh.triangles)
    else:
        from hand_density_constraints import boundary_edges
        assert boundary_edges(selected) <= boundary_edges(mesh)
    assert np.array_equal(selected.vertex_colors, mesh.vertex_colors)
    method = 'whole_palm_reference_envelope' if whole_palm else ('reference_first_palm_envelope' if reference_first else ('curved_anchor_palm_focus_trial' if palm_focus else 'smooth_anchored_palm_volume_warp'))
    return selected, {'method': method, 'applied': any(p['applied'] for p in records),
                      'status': 'complete', 'new_local_intersections': 0,
                      'parts': records, 'topology_changed': not np.array_equal(selected.triangles, mesh.triangles), 'parameters_shared_across_scans': True,
                      'source_points_deleted': 0}
