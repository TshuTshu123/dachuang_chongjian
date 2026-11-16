"""正式表面整理的纯几何步骤；输入毫米网格和点云，不读写实验目录。"""
import copy
import json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from hand_pipeline import smoothstep, scene_for
from surface_quality import intersection_blocks, intersection_pairs
from tools.mano_surface_trial import subdivide, thickness, distance
from tools.wrist_observation_trial import warp_wrist
from tools.torso_surface_trial import region, propose, face_quality, sections
from surface_evaluation import distances


class QualityRejection(ValueError):
    """Deterministic geometry rejection; unchanged inputs may reuse it."""


def refine_mano_surface(original, raw, parts, fit, palm_by):
    n_original = len(original.vertices)
    frame_by={p['part']:p for p in fit['parts']}
    labels=np.full(n_original,-1,int);frames=[]
    for i,name in enumerate(['left_hand','right_hand']):
        labels[parts[name]['vertex_indices']]=i
        f=frame_by[name];frames.append({'name':name,'wrist':np.array(f['wrist_mm']),
                                      'axes':np.array(f['frame_axes']),'end':float(palm_by[name]['palm_end_mm'])})
    reference,labels=subdivide(original,labels,frames)
    assert reference.is_vertex_manifold()==original.is_vertex_manifold()
    assert len(reference.get_non_manifold_edges())==len(original.get_non_manifold_edges())
    assert abs(reference.get_surface_area()/original.get_surface_area()-1)<1e-10
    v=np.asarray(reference.vertices).copy();faces=np.asarray(reference.triangles);normals=np.asarray(reference.vertex_normals)
    edges=np.unique(np.sort(np.r_[faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]],axis=1),axis=0)
    graph=coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(v),len(v))).tocsr()
    degree=np.maximum(np.asarray(graph.sum(1)).ravel(),1)
    offset=np.zeros(len(v));fade=np.zeros(len(v));records=[];heldout=[]
    for i,frame in enumerate(frames):
        ids=np.flatnonzero(labels==i);local=(v-frame['wrist'])@frame['axes']
        hand=v[ids];mask=np.all((raw>=hand.min(0)-8)&(raw<=hand.max(0)+8),axis=1)
        obs=raw[mask];scene=scene_for(reference,labels==i)
        hit=scene.compute_closest_points(o3d.core.Tensor(obs.astype(np.float32)),nthreads=4)
        close=hit['points'].numpy();gap=np.linalg.norm(close-obs,axis=1)
        normal=hit['primitive_normals'].numpy()
        y=((obs-frame['wrist'])@frame['axes'])[:,1]
        valid=(gap<=3.)&(y>35)&(y<frame['end']+8)
        trusted=obs[valid];trusted_n=normal[valid]
        if len(trusted)<50:
            records.append({'part':frame['name'],'status':'insufficient_support','trusted_points':len(trusted)});continue
        perm=np.random.default_rng(42+i).permutation(len(trusted));test=perm[:max(10,len(perm)//5)];train=perm[len(test):]
        heldout.append(trusted[test]);tree=cKDTree(trusted[train])
        d,nn=tree.query(v[ids],k=min(16,len(train)),workers=4)
        obs_nn=trusted[train][nn];nnorm=trusted_n[train][nn]
        delta=obs_nn-v[ids,None,:];signed=np.einsum('nkj,nj->nk',delta,normals[ids])
        tangential=np.sqrt(np.maximum(d*d-signed*signed,0))
        good=(tangential<=6)&(np.abs(signed)<=3)&(np.abs(np.einsum('nkj,nj->nk',nnorm,normals[ids]))>=.7)
        weights=np.exp(-.5*(tangential/3)**2)*good
        estimate=np.sum(weights*signed,axis=1)/np.maximum(weights.sum(1),1e-20)
        supported=(good.sum(1)>=3)&(weights.sum(1)>=1.)
        offset[ids]=np.where(supported,estimate,0)
        fade[ids]=supported*smoothstep((local[ids,1]-35)/15)*smoothstep((frame['end']+8-local[ids,1])/15)
        records.append({'part':frame['name'],'status':'supported','nearby_points':len(obs),'trusted_points':len(trusted),
                        'rejected_far_points':int((gap>3).sum()),'heldout_points':len(test),'supported_vertices':int(supported.sum())})
    # Remove broad offsets: add measured local variation, never the scan's thick envelope.
    low=offset.copy()
    for _ in range(16):low=.5*low+.5*np.asarray(graph@low)/degree;low[labels<0]=0
    residual=offset-low
    # Fade semantic boundary and protect the existing wrist connection and distal fingers.
    fade[np.asarray(graph@(labels<0).astype(float)).ravel()>0]=0
    for _ in range(2):fade=np.minimum(fade,np.asarray(graph@fade)/degree)
    active=fade>0
    if not active.any() or not heldout:raise ValueError('No reliable local observations')
    test=np.concatenate(heldout);before_d=distance(reference,test)
    blocks=intersection_blocks(reference,active);before_pairs=intersection_pairs(reference,blocks)
    before_thick=[thickness(reference,labels==i,frame) for i,frame in enumerate(frames)]
    base_cross=np.cross(v[faces[:,1]]-v[faces[:,0]],v[faces[:,2]]-v[faces[:,0]])
    choices=[];accepted=[]
    for cap in [.5,1.]:
        shift=np.clip(residual,-cap,cap)*fade
        candidate=copy.deepcopy(reference);xyz=v+normals*shift[:,None]
        candidate.vertices=o3d.utility.Vector3dVector(xyz);candidate.compute_vertex_normals()
        cross=np.cross(xyz[faces[:,1]]-xyz[faces[:,0]],xyz[faces[:,2]]-xyz[faces[:,0]])
        flipped=int((np.sum(base_cross*cross,axis=1)<=0).sum())
        extra=intersection_pairs(candidate,blocks)-before_pairs
        initial_conflicts={'new_intersections':len(extra),'flipped_faces':flipped}
        retracted=np.zeros(len(v),bool)
        for _ in range(8):
            bad=np.flatnonzero(np.sum(base_cross*cross,axis=1)<=0)
            if not len(bad) and not extra:break
            bad=np.unique(np.r_[bad,np.asarray(list(extra),int).ravel()])
            halo=np.zeros(len(v),bool);halo[np.unique(faces[bad])]=True
            attenuation=np.ones(len(v));attenuation[halo]=0
            for factor in [.25,.5,.75]:
                halo|=np.asarray(graph@halo.astype(float)).ravel()>0
                attenuation[halo]=np.minimum(attenuation[halo],factor)
            retracted|=(attenuation<1)&(np.abs(shift)>1e-5)
            shift*=attenuation;xyz=v+normals*shift[:,None]
            candidate.vertices=o3d.utility.Vector3dVector(xyz);candidate.compute_vertex_normals()
            cross=np.cross(xyz[faces[:,1]]-xyz[faces[:,0]],xyz[faces[:,2]]-xyz[faces[:,0]])
            flipped=int((np.sum(base_cross*cross,axis=1)<=0).sum())
            extra=intersection_pairs(candidate,blocks)-before_pairs
        after_d=distance(candidate,test);th=[];thickness_ok=True
        for i,frame in enumerate(frames):
            a,ok_a=before_thick[i];b,ok_b=thickness(candidate,labels==i,frame);ok=ok_a&ok_b
            if ok.sum()<30:thickness_ok=False;th.append({'part':frame['name'],'common_rays':int(ok.sum())});continue
            delta=float(np.quantile(b[ok],.85)-np.quantile(a[ok],.85))
            thickness_ok&=delta<=.6
            th.append({'part':frame['name'],'common_rays':int(ok.sum()),'before_p85_mm':float(np.quantile(a[ok],.85)),
                       'after_p85_mm':float(np.quantile(b[ok],.85)),'change_mm':delta})
        fit_ok=np.quantile(after_d,.95)<=np.quantile(before_d,.95)+.15 and np.median(after_d)<=np.median(before_d)+.05
        moved=np.abs(shift)>1e-5
        qa=not flipped and not extra and thickness_ok and fit_ok and moved.sum()>20
        record={'cap_mm':cap,'accepted':bool(qa),'changed_vertices':int(moved.sum()),'max_movement_mm':float(np.abs(shift).max()),
                'p95_movement_mm':float(np.quantile(np.abs(shift[moved]),.95)) if moved.any() else 0.,
                'new_intersections':len(extra),'flipped_faces':flipped,'thickness':th,
                'initial_conflicts':initial_conflicts,'locally_retracted_vertices':int(retracted.sum()),
                'heldout_before_median_mm':float(np.median(before_d)),'heldout_after_median_mm':float(np.median(after_d)),
                'heldout_before_p95_mm':float(np.quantile(before_d,.95)),'heldout_after_p95_mm':float(np.quantile(after_d,.95))}
        choices.append(record)
        print('MANO surface',json.dumps(record),flush=True)
        if qa:accepted.append((float(np.median(after_d)),candidate,record))
    selected=min(accepted,key=lambda x:x[0]) if accepted else None
    chosen=selected[1] if selected else reference
    assert np.array_equal(np.asarray(chosen.vertices)[:n_original][labels[:n_original]<0],np.asarray(original.vertices)[labels[:n_original]<0])
    assert np.array_equal(np.asarray(chosen.triangles),faces)
    print('MANO surface topology checked',flush=True)
    assert chosen.is_vertex_manifold()==reference.is_vertex_manifold()
    assert len(chosen.get_non_manifold_edges())==len(reference.get_non_manifold_edges())
    updated = copy.deepcopy(parts)
    if selected:
        for i, frame in enumerate(frames):
            part = updated[frame['name']]
            added = np.flatnonzero((labels == i) & (np.arange(len(labels)) >= n_original))
            part['vertex_indices'] = list(part['vertex_indices']) + added.tolist()
            part['vertex_count'] = len(part['vertex_indices'])
            part['label_origin'] = 'Transported labels; subdivided MANO palm vertices inherit hand assignment'
    return (chosen if selected else original), {
        'status': 'complete', 'applied': bool(selected),
        'method': 'Conforming subdivision and bounded observation-supported high-pass residuals',
        'topology_changed': bool(selected), 'input_vertices': n_original,
        'output_vertices': len(chosen.vertices) if selected else n_original,
        'parts': records, 'candidates': choices, 'selected': selected[2] if selected else None,
        'manual_measurement_landmarks': 'not_generated',
        'note': 'Common palm ray thickness and held-out nearby observations are not anatomical ground truth.'
    }, updated


def refine_wrists(mesh, raw, parts, fit, scanmesh):
    selected = mesh
    records = []
    xyz = np.asarray(mesh.vertices)
    names = list(parts)
    labels = np.full(len(xyz), -1, int)
    for index, name in enumerate(names):
        labels[parts[name]['vertex_indices']] = index
    known = np.flatnonzero(labels >= 0)
    tree = cKDTree(xyz[known])
    for frame in fit['parts']:
        name = frame['part']
        wrist = np.asarray(frame['wrist_mm'])
        axes = np.asarray(frame['frame_axes'])
        allowed = [names.index(n) for n in (name, name.replace('hand', 'lower_arm')) if n in names]
        def allowed_fn(points):
            return np.isin(labels[known[tree.query(points, workers=4)[1]]], allowed)
        choices, attempts = [], []
        for cap in (6., 10., 14.):
            try:
                candidate, detail = warp_wrist(selected, raw, wrist, axes, allowed_fn, cap, scanmesh)
                detail['displacement_cap_mm'] = cap
                choices.append((candidate, detail))
            except (ValueError, RuntimeError) as error:
                attempts.append({'cap_mm': cap, 'reason': str(error)})
        if choices:
            selected, detail = min(choices, key=lambda pair: pair[1]['scan_surface_after_median_mm'])
            detail.update(part=name, applied=True, attempts=attempts)
            detail['accepted_candidates'] = [{'cap_mm': d['displacement_cap_mm'],
                'scan_distance_median_mm': d['scan_surface_after_median_mm']} for _, d in choices]
        else:
            detail = {'part': name, 'applied': False, 'attempts': attempts}
        records.append(detail)
        print('Wrist correction', name, 'applied', detail['applied'], flush=True)
    return selected, {'status': 'complete', 'applied': any(p['applied'] for p in records),
        'method': 'original_scan_section_affine_warp', 'parts': records,
        'topology_changed': False, 'manual_measurement_landmarks': 'not_generated',
        'note': 'Only checked, connectivity-preserving warps; failed wrists retain their input geometry.'}, parts


def refine_torso(mesh, raw, parts, axis='auto', up_sign=1):
    original = mesh
    world_xyz = np.asarray(mesh.vertices).copy()
    extent = np.ptp(world_xyz, axis=0)
    order = np.argsort(extent)
    if axis == 'auto' and extent[order[-1]] < 1.3 * extent[order[-2]]:
        raise QualityRejection('Torso vertical axis ambiguous; specify --seg-axis.')
    up = int(order[-1]) if axis == 'auto' else 'xyz'.index(axis)
    rotation = np.eye(3)[:, [up, (up+1)%3, (up+2)%3]].copy()
    rotation[:, 0] *= up_sign
    rotation[:, 2] *= up_sign
    rotated = not np.array_equal(rotation, np.eye(3))
    if rotated:
        mesh = copy.deepcopy(mesh)
        mesh.vertices = o3d.utility.Vector3dVector(world_xyz @ rotation)
        raw = raw @ rotation
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices).copy()
    f = np.asarray(mesh.triangles)
    weight, allowed, edges, graph, degree = region(mesh, parts)
    active = weight > 0
    if active.sum() < 100 or (weight > .9).sum() < 100:
        raise QualityRejection('Insufficient torso region')
    mask = weight > .9
    nearest = cKDTree(v).query(raw, workers=4)[1]
    selected = raw[mask[nearest]]
    samples = selected[np.random.default_rng(63).choice(len(selected), min(8000, len(selected)), replace=False)]
    if len(samples) < 100:
        raise QualityRejection('Insufficient torso observations')
    before_distance = distances(mesh, samples)
    before = face_quality(mesh, mask, edges)
    points = v[allowed]
    center = (points.min(0) + points.max(0)) / 2
    levels = np.linspace(*np.quantile(points[:, 0], [.2, .7]), 7)
    base_sections = sections(mesh, levels, center)
    if any(section is None for section in base_sections):
        raise QualityRejection('Torso input diagnostic contours incomplete')
    blocks = intersection_blocks(mesh, active, margin_mm=7.)
    base_pairs = intersection_pairs(mesh, blocks)
    cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    records, candidates, names = [], [mesh], ['Input']
    def assess(radius, cap):
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
        if len(changes) != len(levels):
            raise QualityRejection('Torso candidate diagnostic contours incomplete')
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
        return candidate, record

    for radius, cap in ((12., 1.5), (20., 3.)):
        try:
            candidate, record = assess(radius, cap)
        except (ValueError, RuntimeError) as error:
            candidate = None
            record = {"radius_mm": radius, "cap_mm": cap, "accepted": False,
                      "reason": str(error), "exception_type": type(error).__name__,
                      "retryable": not isinstance(error, QualityRejection)}
        records.append(record)
        candidates.append(candidate)
        names.append(f"Quadratic {radius:g}mm [{record['accepted']}]")
        print("Torso", str({key: record[key] for key in ("radius_mm", "accepted", "reason",
              "normal_angle_p90_decrease_percent", "movement_max_mm") if key in record}), flush=True)
    passing = [i for i, r in enumerate(records) if r['accepted']]
    choice = max(passing, key=lambda i: records[i]['normal_angle_p90_decrease_percent']) if passing else None
    if choice is None and any(record.get('retryable') for record in records):
        raise RuntimeError('No torso candidate passed; unexpected candidate error: ' +
                           '; '.join(record.get('reason', '') for record in records if record.get('retryable')))
    candidate = candidates[choice+1] if choice is not None else original
    if rotated and choice is not None:
        positions = world_xyz.copy()
        positions[active] = np.asarray(candidate.vertices)[active] @ rotation.T
        candidate.vertices = o3d.utility.Vector3dVector(positions)
        candidate.compute_vertex_normals()
    return candidate, {'status': 'complete', 'applied': choice is not None,
        'method': 'area-weighted robust local quadratic surface fitting', 'topology_changed': False,
        'candidates': records, 'selected_index': choice, 'active_vertices': int(active.sum()),
        'diagnostic_section_method': 'mesh_edge_identity_without_coordinate_welding',
        'diagnostic_section_count': len(levels),
        'boundary_buffer_mm': [2, 12], 'unknown_label_extension_geodesic_mm': 60,
        'axis': 'xyz'[up], 'up_sign': up_sign, 'manual_measurement_landmarks': 'not_generated',
        'note': 'Diagnostic contours are not anatomical landmarks; raw consistency is not measurement accuracy.'}, parts
