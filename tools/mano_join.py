"""将 MANO 拟合实验的双手接到原模型腕口，保留独立试验输出。

项目根目录：python tools/mano_join.py --scan 3
先运行 tools/mano_fit.py；只写 experiments/mano_fit，不覆盖正式模型。
检查边界、非流形和接缝局部相交，失败不发布完整拼接网格。
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hand_centerlines import hand_frame
from hand_pipeline import pairs


def boundary_loops(faces):
    edges = np.sort(np.r_[faces[:, :2], faces[:, 1:], faces[:, [2, 0]]], axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    neighbors = {}
    for a, b in edges[counts == 1]:
        neighbors.setdefault(int(a), []).append(int(b)); neighbors.setdefault(int(b), []).append(int(a))
    if any(len(v) != 2 for v in neighbors.values()):
        raise ValueError('Boundary is not a collection of simple loops')
    loops = []
    while neighbors:
        start = next(iter(neighbors)); loop = [start]; previous = None; current = start
        while True:
            nxt = next(v for v in neighbors[current] if v != previous)
            if nxt == start: break
            if nxt in loop: raise ValueError('Boundary loop repeats')
            loop.append(nxt); previous, current = current, nxt
        for v in loop: neighbors.pop(v)
        loops.append(np.array(loop, int))
    return loops


def bridge_loops(vertices, body, hand):
    # Choose matching direction/start, then zip closed loops by normalized arc.
    best = None
    a = vertices[body]
    ta = np.r_[0., np.cumsum(np.linalg.norm(np.diff(np.r_[a, a[:1]], axis=0), axis=1))]; ta /= ta[-1]
    for direction in (hand, hand[::-1]):
        for shift in range(len(hand)):
            loop = np.roll(direction, shift); b = vertices[loop]
            tb = np.r_[0., np.cumsum(np.linalg.norm(np.diff(np.r_[b, b[:1]], axis=0), axis=1))]; tb /= tb[-1]
            probe = np.c_[*[np.interp(ta[:-1], tb, np.r_[b[:, k], b[0, k]]) for k in range(3)]]
            cost = np.mean(np.linalg.norm(a-probe, axis=1))
            if best is None or cost < best[0]: best = (cost, loop, tb)
    cost, hand, tb = best; result = []; i = j = 0
    while i < len(body) or j < len(hand):
        if j == len(hand) or (i < len(body) and ta[i+1] <= tb[j+1]):
            result.append([body[i % len(body)], body[(i+1) % len(body)], hand[j % len(hand)]]); i += 1
        else:
            result.append([body[i % len(body)], hand[(j+1) % len(hand)], hand[j % len(hand)]]); j += 1
    return np.array(result, int), float(cost)


def join(scan, output_folder=None, part_names=None, smooth_palm_wrist=False, wrist_cut_offset=0.):
    folder = ROOT/'outputs/experiments/mano_fit'/f'export_{scan:05d}' if output_folder is None else Path(output_folder)
    path = folder/'report.json'; report = json.loads(path.read_text(encoding='utf-8'))
    report.update(body_replaced=False, wrist_stitched=False)
    report.pop('full_preview', None)
    source = Path(report['source_mesh']); mesh = o3d.io.read_triangle_mesh(str(source))
    original = np.asarray(mesh.vertices).copy(); vertices = original.copy(); colors = np.asarray(mesh.vertex_colors).copy(); faces = np.asarray(mesh.triangles).copy()
    face_kind = np.full(len(faces), 'body', dtype=object)
    info = json.loads(Path(report.get('source_positions', str(source.parent/'body_parts.json'))).read_text(encoding='utf-8'))['parts']; records = []
    baseline_open_edges = sum(len(loop) for loop in boundary_loops(faces))
    for name in (['left_hand', 'right_hand'] if part_names is None else part_names):
        ids, arm, wrist, axes = hand_frame(original, info, name)
        local = (original-wrist) @ axes; centers = (vertices[faces].mean(1)-wrist) @ axes
        lo, hi = local[ids].min(0)-[40, 25, 60], local[ids].max(0)+[40, 60, 60]
        remove = (centers[:, 1] > wrist_cut_offset) & np.all((centers >= lo) & (centers <= hi), axis=1)
        # Avoid replacing a nearby torso or the other hand on contact poses.
        other = np.unique(np.concatenate([p['vertex_indices'] for key, p in info.items() if key not in (name, arm)]))
        if np.isin(faces[remove], other).any(): raise ValueError('Cut overlaps another labeled body region')
        # Clip intersected faces to the exact wrist plane instead of a jagged
        # centroid cut; the body stays on y<=0 and the bridge on y>=0.
        inside = np.all((centers >= lo) & (centers <= hi), axis=1)
        y = ((vertices-wrist) @ axes)[:, 1]-wrist_cut_offset
        y[np.abs(y)<1.e-7] = 0
        crossing = inside & (y[faces].min(1) < 0) & (y[faces].max(1) > 0)
        drop = inside & (y[faces].max(1) > 0)
        kept = faces[~drop]
        new_vertices = []; new_colors = []; edge_cache = {}; clipped = []
        for tri in faces[crossing]:
            polygon = []
            for u, v in zip(tri, np.roll(tri, -1)):
                if y[u] <= 0: polygon.append(int(u))
                if (y[u] < 0 < y[v]) or (y[v] < 0 < y[u]):
                    edge = tuple(sorted((int(u), int(v))))
                    if edge not in edge_cache:
                        t = y[u]/(y[u]-y[v]); edge_cache[edge] = len(vertices)+len(new_vertices)
                        new_vertices.append(vertices[u]+t*(vertices[v]-vertices[u]))
                        new_colors.append(colors[u]+t*(colors[v]-colors[u]))
                    polygon.append(edge_cache[edge])
            for j in range(1, len(polygon)-1):clipped.append([polygon[0], polygon[j], polygon[j+1]])
        if new_vertices:
            vertices = np.r_[vertices, np.array(new_vertices)]; colors = np.r_[colors, np.array(new_colors)]
        kept = np.r_[kept, np.asarray(clipped, int).reshape(-1, 3)]
        kept_kind = np.r_[face_kind[~drop], np.full(len(clipped), 'body')]
        loops = boundary_loops(kept)
        near = [loop for loop in loops if np.linalg.norm(vertices[loop].mean(0)-wrist) < 80 and np.max(np.abs(((vertices[loop]-wrist)@axes)[:,1]-wrist_cut_offset))<1.e-5]
        if not near: raise ValueError('No wrist opening on cut plane')
        def loop_area(loop):
            xy=((vertices[loop]-wrist)@axes)[:,[0,2]]
            return abs(np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1)))/2
        body_loop=max(near,key=loop_area)
        auxiliary=[loop for loop in near if loop is not body_loop]
        if len(auxiliary)>2 or sum(loop_area(loop) for loop in auxiliary)>.35*loop_area(body_loop):raise ValueError('Ambiguous multiple wrist openings')
        if auxiliary:
            # Small auxiliary contours come from scan cavities crossing the cut.
            # Close only these coplanar contours; do not move the forearm surface.
            from finger_envelope import repair_local_folds
            cap_seed=np.array([[loop[0],loop[j],loop[j+1]] for loop in auxiliary for j in range(1,len(loop)-1)],int)
            temporary=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(np.r_[kept,cap_seed]))
            mask=np.r_[np.zeros(len(kept),bool),np.ones(len(cap_seed),bool)]
            temporary,ok,_=repair_local_folds(temporary,mask,np.ones(len(vertices),bool),max_loop=512)
            if not ok:raise ValueError('Could not close auxiliary wrist contour')
            kept=np.asarray(temporary.triangles).copy();kept_kind=np.full(len(kept),'body',dtype=object)
        ring = vertices[body_loop].copy()
        for _ in range(8):ring = .65*ring+.175*(np.roll(ring,1,axis=0)+np.roll(ring,-1,axis=0))
        cut_nodes = body_loop >= len(original)
        vertices[body_loop[cut_nodes]] = ring[cut_nodes]
        hand = o3d.io.read_triangle_mesh(str(folder/(name+'_template.ply')))
        hv = np.asarray(hand.vertices).copy(); hf = np.asarray(hand.triangles).copy(); hand_loops = boundary_loops(hf)
        if len(hand_loops) != 1: raise ValueError('MANO wrist boundary is not unique')
        hand_loop = hand_loops[0]
        # Anchor the template's open wrist to the scan's actual cut, fading toward palm.
        body_center = vertices[body_loop].mean(0)
        shift = body_center-hv[hand_loop].mean(0)
        local_hand = (hv-hv[hand_loop].mean(0)) @ axes
        weight = np.clip(1-local_hand[:, 1]/35, 0, 1)
        weight = weight*weight*(3-2*weight)
        body_ring = (vertices[body_loop]-body_center) @ axes
        hand_ring = local_hand[hand_loop]
        radii = np.ptp(body_ring[:, [0, 2]], axis=0)/np.maximum(np.ptp(hand_ring[:, [0, 2]], axis=0), 1.)
        radii = np.clip(radii*.85, .6, 2.5)
        warped = local_hand.copy(); warped[:, [0, 2]] *= radii
        plane = np.linalg.lstsq(np.c_[np.ones(len(hand_ring)), hand_ring[:, [0, 2]]], hand_ring[:, 1], rcond=None)[0]
        warped[:, 1] += 6.-np.c_[np.ones(len(warped)), local_hand[:, [0, 2]]] @ plane
        warped = warped @ axes.T+body_center
        hv += weight[:, None]*(warped-hv)
        offset = len(vertices); vertices = np.r_[vertices, hv]; colors = np.r_[colors, np.asarray(hand.vertex_colors)]
        bridge, distance = bridge_loops(vertices, body_loop, hand_loop+offset)
        if distance > 25: raise ValueError('Wrist bridge is too long')
        faces = np.r_[kept, hf+offset, bridge]
        face_kind = np.r_[kept_kind, np.full(len(hf), name), np.full(len(bridge), name+'_bridge')]
        records.append({'part': name, 'removed_faces': int(remove.sum()), 'wrist_cut_offset_mm': wrist_cut_offset, 'auxiliary_wrist_contours_capped':len(auxiliary), 'wrist_body_vertices': len(body_loop), 'wrist_template_vertices': len(hand_loop), 'bridge_faces': len(bridge), 'mean_bridge_span_mm': distance, 'wrist_alignment_shift_mm': shift.tolist()})
    result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    result.vertex_colors = o3d.utility.Vector3dVector(colors)
    if smooth_palm_wrist:
        from tools.wrist_cut_trial import cut_wrist
        for record in records:
            frame=next(p for p in report['parts'] if p['part']==record['part'])
            failure=None
            for low in [0.,2.,4.,6.]:
                try:
                    smoothed,smoothing=cut_wrist(result,np.array(frame['wrist_mm']),np.array(frame['frame_axes']),low_level=low,high_level=35.,contour_match=True)
                    result=smoothed;failure=None;break
                except ValueError as error:failure=error
            if failure is not None:raise failure
            record['palm_side_wrist_smoothing']=smoothing
        vertices=np.asarray(result.vertices).copy();faces=np.asarray(result.triangles).copy();colors=np.asarray(result.vertex_colors).copy()
        face_kind=np.full(len(faces),'palm_side_wrist_checked',dtype=object)
    if len(result.get_non_manifold_edges()) or len(result.get_non_manifold_vertices()): raise ValueError('Nonmanifold join')
    if sum(len(loop) for loop in boundary_loops(faces)) > baseline_open_edges: raise ValueError(f'Unclosed seam: {baseline_open_edges} -> {sum(len(loop) for loop in boundary_loops(faces))} boundary edges')
    active = np.arange(len(vertices)) >= len(original)
    points = vertices[active]; xyz = vertices[faces]
    # Each wrist/hand checked separately, including the retained forearm context.
    crossings = set()
    for record in records:
        ids, arm, wrist, axes = hand_frame(original, info, record['part'])
        box = np.r_[original[ids], points[np.linalg.norm(points-wrist, axis=1) < 250]]
        selected = np.flatnonzero(np.all(xyz.max(1) >= box.min(0)-25, axis=1) & np.all(xyz.min(1) <= box.max(0)+25, axis=1))
        crossings |= pairs(result, selected, active)
    if crossings:
        from finger_envelope import repair_local_folds
        base = result
        bad = np.zeros(len(faces), bool)
        for _ in range(4):
            bad[np.array(list(crossings), int).ravel()] = True
            repaired, ok, count = repair_local_folds(base, bad, active, max_loop=512)
            if not ok: break
            rf = np.asarray(repaired.triangles); xyz = vertices[rf]
            all_crossings = set()
            for record in records:
                ids, arm, wrist, axes = hand_frame(original, info, record['part'])
                box = np.r_[original[ids], points[np.linalg.norm(points-wrist, axis=1) < 250]]
                selected = np.flatnonzero(np.all(xyz.max(1) >= box.min(0)-25, axis=1) & np.all(xyz.min(1) <= box.max(0)+25, axis=1))
                all_crossings |= pairs(repaired, selected, active)
            if not all_crossings:
                result = repaired; crossings = set(); break
            conflict_vertices = np.unique(rf[np.array(list(all_crossings), int).ravel()])
            expanded = np.any(np.isin(faces, conflict_vertices), axis=1)
            if (bad | expanded).sum() > 500: break
            bad |= expanded
        if len(result.get_non_manifold_vertices()) or len(result.get_non_manifold_edges()) or sum(len(loop) for loop in boundary_loops(np.asarray(result.triangles))) > baseline_open_edges:
            raise ValueError('Seam repair failed topology checks')
    report['join_attempt'] = {'parts': records, 'new_hand_local_intersections': len(crossings)}
    report['join_attempt']['intersection_types'] = [list((face_kind[a], face_kind[b])) for a, b in sorted(crossings)]
    if crossings:
        report['wrist_stitched'] = False
        report['join_attempt']['reason'] = 'Intersections found; complete replacement not published'
    else:
        assert np.array_equal(vertices[:len(original)], original)
        result.remove_unreferenced_vertices(); result.orient_triangles(); result.compute_vertex_normals()
        output = folder/f'export_{scan:05d}_mano_trial.ply'
        if not o3d.io.write_triangle_mesh(str(output), result): raise OSError('PLY export failed')
        report.update(body_replaced=True, wrist_stitched=True, full_preview=output.name)
        report['join_attempt'].update(unchanged_retained_body_coordinates=True, new_open_edges=0, new_nonmanifold_vertices=0)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(report['join_attempt'], flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--scan', type=int, default=3)
    join(parser.parse_args().scan)
