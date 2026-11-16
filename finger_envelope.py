"""手指柱面厚度约束，默认外观目标 12 mm，不代表人体解剖标准。

沿可靠中心线和网格截面估计中心，按 12 mm 目标收缩厚度。
原始点云仅统计柱内外覆盖，不参与本实验的收缩拟合，也不阻止收紧。
保留投影宽度和指尖轮廓，指根渐变；只在可验证的骨架覆盖区域尝试。
几何检查失败或没有达到目标时回退，不把部分收紧报告成达到 12 mm。
实验入口：python tools/hand_skeleton_trial.py --mode tube --scan 0 1 2 3
尚未默认接入 run.py；不重建指缝连接，原始 PLY 不删除点。
"""
import copy
import numpy as np
import open3d as o3d
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from hand_pipeline import depths, scene_for, pairs, smoothstep

TARGET_MM = 12.


def crossing_signatures(mesh, face_ids, active):
    f=np.asarray(mesh.triangles)
    return {tuple(sorted((tuple(sorted(f[a])),tuple(sorted(f[b]))))) for a,b in pairs(mesh,face_ids,active)}


def repair_local_folds(candidate, bad_mask, hand, max_loop=128):
    """Ear-clip small boundary loops using existing vertices; no native hole filler."""
    f=np.asarray(candidate.triangles);remaining=f[~bad_mask];patch=f[bad_mask]
    edges={}
    for tid,tri in enumerate(patch):
        for u,v in zip(tri,np.roll(tri,-1)):
            key=tuple(sorted((int(u),int(v))))
            edges.setdefault(key,[]).append((int(u),int(v),tid))
    parent=list(range(len(patch)))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for items in edges.values():
        if len(items)==2:parent[find(items[0][2])]=find(items[1][2])
    boundary=[((find(items[0][2]),items[0][0]),(find(items[0][2]),items[0][1])) for items in edges.values() if len(items)==1]
    outgoing={}
    for u,v in boundary:
        if u in outgoing:return candidate,False,0
        outgoing[u]=v
    extra=[];vertices=np.asarray(candidate.vertices)
    while outgoing:
        start=next(iter(outgoing));loop=[];u=start
        while u in outgoing:
            loop.append(u[1]);u=outgoing.pop(u)
            if u==start:break
        if u!=start or len(loop)<3 or len(loop)>max_loop:return candidate,False,0
        xyz=vertices[loop];_,_,axes=np.linalg.svd(xyz-xyz.mean(0),full_matrices=False)
        xy=(xyz-xyz.mean(0))@axes[:2].T
        area=np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))
        if abs(area)<1.e-8:return candidate,False,0
        sign=1 if area>0 else -1;polygon=list(range(len(loop)))
        def cross(a,b):return a[0]*b[1]-a[1]*b[0]
        while len(polygon)>3:
            found=False
            for k,j in enumerate(polygon):
                i,h=polygon[k-1],polygon[(k+1)%len(polygon)]
                if sign*cross(xy[j]-xy[i],xy[h]-xy[j])<=1.e-9:continue
                others=[q for q in polygon if q not in (i,j,h)]
                blocked=False
                for q in others:
                    if min(sign*cross(xy[j]-xy[i],xy[q]-xy[i]),sign*cross(xy[h]-xy[j],xy[q]-xy[j]),sign*cross(xy[i]-xy[h],xy[q]-xy[h]))>=-1.e-9:
                        blocked=True;break
                if blocked:continue
                extra.append([loop[i],loop[j],loop[h]]);polygon.pop(k);found=True;break
            if not found:return candidate,False,0
        extra.append([loop[q] for q in polygon])
    if not extra:return candidate,False,0
    result=copy.deepcopy(candidate);result.triangles=o3d.utility.Vector3iVector(np.r_[remaining,np.array(extra,int)])
    from hand_density_constraints import boundary_edges
    if boundary_edges(result)>boundary_edges(candidate):return candidate,False,0
    result.compute_vertex_normals()
    return result,True,len(extra)


def constrain_fingers(mesh, raw_mm, info, frames, target_mm=TARGET_MM):
    original = np.asarray(mesh.vertices).copy()
    original_faces = np.asarray(mesh.triangles).copy()
    selected = copy.deepcopy(mesh)
    labels = np.full(len(original), 'unknown', dtype=object)
    for name, part in info['parts'].items(): labels[part['vertex_indices']] = name
    known = np.flatnonzero(labels != 'unknown')
    nearest_label = labels[known[cKDTree(original[known]).query(original, workers=4)[1]]]
    records = []
    for frame in frames:
        faces=np.asarray(selected.triangles)
        name = frame['part']; record = {'part': name, 'applied': False, 'target_mm': target_mm}
        branches = [b for b in frame['branches'] if b['status'] == 'supported_candidate']
        if not branches:
            records.append({**record, 'reason': 'No reliable digit centerline'}); continue
        wrist, axes = np.array(frame['wrist_mm']), np.array(frame['frame_axes'])
        local = (np.asarray(selected.vertices)-wrist) @ axes
        raw = (raw_mm-wrist) @ axes
        ids = np.asarray(info['parts'][name]['vertex_indices'], int)
        lo, hi = local[ids].min(0)-15, local[ids].max(0)+15
        hand = np.all((local >= lo) & (local <= hi), axis=1)
        hand &= (labels == name) | ((labels == 'unknown') & (nearest_label == name))
        arm = name.replace('hand', 'lower_arm')
        other = np.flatnonzero((labels != 'unknown') & ~np.isin(labels, [name, arm]))
        if len(other) and np.quantile(cKDTree(original[other]).query(original[ids])[0], .05) < 6:
            records.append({**record, 'reason': 'Hand contacts another body region'}); continue
        scene = scene_for(selected, hand)
        paths = [(np.asarray(b['world_path_mm'])-wrist) @ axes for b in branches]
        queries = [cKDTree(p[:, :2]).query(local[:, :2]) for p in paths]
        distance = np.array([q[0] for q in queries])
        raw_queries = [cKDTree(p[:, :2]).query(raw[:, :2]) for p in paths]
        raw_distance = np.array([q[0] for q in raw_queries]); raw_owner = raw_distance.argmin(0)
        coords = local.copy(); affected = np.zeros(len(local), bool)
        blend_a = np.zeros(len(local)); blend_b = np.zeros(len(local)); blend_sum = np.zeros(len(local)); blend_weight = np.zeros(len(local))
        profiles = []; raw_considered = 0; raw_kept = 0
        probes_all = []; before_all = []
        for k, path in enumerate(paths):
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
            probes = wrist+np.c_[path[:, :2], np.zeros(len(path))] @ axes.T
            lower, upper, gap, valid = depths(scene, probes, axes[:, 2])
            if valid.sum() < 8 or arc[-1] < 28:
                continue
            index = np.arange(len(path))
            center = np.interp(index, index[valid], (lower[valid]+upper[valid])/2)
            thickness = np.interp(index, index[valid], gap[valid])
            tangent = np.gradient(path[:, :2], axis=0)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.e-9)
            across = np.c_[-tangent[:, 1], tangent[:, 0]]
            for offset in (-10., -5., 5., 10.):
                side_xy = path[:, :2]+offset*across
                side_probes = wrist+np.c_[side_xy, np.zeros(len(path))] @ axes.T
                l, u, _, ok = depths(scene, side_probes, axes[:, 2])
                # Envelope the entire section instead of collapsing side vertices
                # against a plane inferred from a single central ray.
                lower[ok] = np.minimum(lower[ok], l[ok]); upper[ok] = np.maximum(upper[ok], u[ok])
            center = np.interp(index, index[valid], (lower[valid]+upper[valid])/2)
            thickness = np.interp(index, index[valid], upper[valid]-lower[valid])
            center = median_filter(center, size=5, mode='nearest')
            thickness = median_filter(thickness, size=5, mode='nearest')
            scale = np.minimum(1., target_mm/np.maximum(thickness, target_mm))
            at = queries[k][1]
            along = np.clip(arc[at]+np.einsum('ij,ij->i',local[:, :2]-path[at, :2],tangent[at]),0,arc[-1])
            center_v = np.interp(along,arc,center);scale_v=np.interp(along,arc,scale)
            # Wide axis is retained; the ellipse's thickness semi-axis is 6 mm.
            width = 12.
            raw_at = raw_queries[k][1]
            consider = (raw_owner == k) & (raw_distance[k] < width+4) & (arc[raw_at] >= 20)
            ellipse = (raw_distance[k]/width)**2+((raw[:, 2]-center[raw_at])/(target_mm/2))**2
            keep = consider & (ellipse <= 1)
            raw_considered += int(consider.sum()); raw_kept += int(keep.sum())
            root = smoothstep((along-10)/15)
            side = 1-smoothstep((distance[k]-width)/8)
            weight = root*side
            influence = np.exp(-.5*(distance[k]/6.)**2)
            blend_a += influence*scale_v
            blend_b += influence*center_v*(1-scale_v)
            blend_sum += influence
            blend_weight = np.maximum(blend_weight,weight)
            measure = valid & (arc >= 27)
            if measure.sum():
                probes_all.append(probes[measure]); before_all.append(gap[measure])
            profiles.append({'branch': k, 'length_mm': float(arc[-1]),
                             'considered_points': int(consider.sum()), 'retained_points': int(keep.sum())})
        proposal=(blend_a*local[:,2]+blend_b)/np.maximum(blend_sum,1.e-20)
        movable=hand & (blend_sum>1.e-8) & (blend_weight>0)
        coords[movable,2]=local[movable,2]+blend_weight[movable]*(proposal[movable]-local[movable,2])
        affected=movable & (np.abs(coords[:,2]-local[:,2])>.05)
        record.update(branches=profiles, considered_observations=raw_considered,
                      retained_observations=raw_kept, ignored_observations=raw_considered-raw_kept)
        if not probes_all or affected.sum() < 10:
            records.append({**record, 'reason': 'Insufficient covered finger core'}); continue
        candidate = copy.deepcopy(selected)
        before = np.asarray(selected.vertices).copy()
        world = coords @ axes.T+wrist
        world[~affected] = before[~affected]
        candidate.vertices = o3d.utility.Vector3dVector(world);candidate.compute_vertex_normals()
        p = before[faces]; old_cross = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        p = world[faces]; cross = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        touched = np.any(affected[faces], axis=1)
        bad_mask=touched & ((np.linalg.norm(cross, axis=1) < np.linalg.norm(old_cross, axis=1)*.02)
            | (np.einsum('ij,ij->i', cross, old_cross) <= 0))
        bad = int(bad_mask.sum());initial_bad=bad;repaired=0
        if bad:
            candidate,ok,repaired=repair_local_folds(candidate,bad_mask,hand)
            if ok:bad=0
        box = before[affected]
        p = before[faces]
        face_ids = np.flatnonzero(np.all(p.max(1) >= box.min(0)-8, axis=1) & np.all(p.min(1) <= box.max(0)+8, axis=1))
        cf=np.asarray(candidate.triangles);cp=world[cf]
        candidate_ids=np.flatnonzero(np.all(cp.max(1)>=box.min(0)-8,axis=1)&np.all(cp.min(1)<=box.max(0)+8,axis=1))
        added = len(crossing_signatures(candidate,candidate_ids,affected)-crossing_signatures(selected,face_ids,affected)) if not bad else None
        probes = np.concatenate(probes_all); before_depth = np.concatenate(before_all)
        _, _, after_depth, valid = depths(scene_for(candidate, hand), probes, axes[:, 2])
        reached = bool(valid.mean() >= .98 and np.quantile(after_depth[valid], .95) <= target_mm+.5
                       and np.max(after_depth[valid]) <= target_mm+1.) if valid.any() else False
        accepted = bool(not bad and added == 0 and reached)
        record.update(folded_faces=bad, new_local_intersections=added,
                      initial_folded_faces=initial_bad, retriangulated_faces=repaired,
                      before_core_p95_mm=float(np.quantile(before_depth, .95)),
                      after_core_p95_mm=float(np.quantile(after_depth[valid], .95)) if valid.any() else None,
                      after_core_max_mm=float(np.max(after_depth[valid])) if valid.any() else None,
                      target_reached=reached, applied=accepted,
                      reason='12 mm covered digit core accepted' if accepted else 'Strict tube candidate rejected; previous geometry retained')
        if accepted:
            selected = candidate
        records.append(record)
    assert len(selected.vertices)==len(original)
    assert np.array_equal(selected.vertex_colors, mesh.vertex_colors)
    protected = (labels != 'unknown') & ~np.isin(labels, ['left_hand', 'right_hand'])
    assert np.array_equal(np.asarray(selected.vertices)[protected], original[protected])
    return selected, {'applied': any(r['applied'] for r in records), 'target_mm': target_mm,
                      'method': 'strict_elliptical_digit_envelope', 'status': 'complete',
                      'topology_changed': not np.array_equal(selected.triangles, original_faces), 'new_local_intersections': 0,
                      'parts': records, 'source_points_deleted': 0,
                      'outside_observations_used_to_veto_target': False}
