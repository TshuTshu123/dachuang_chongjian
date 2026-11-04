"""当前手部策略：以整手优化前网格为准，只处理明显过厚的掌侧。
由 run.py 调用；--no-hand-refine 关闭。掌面 P95 同时超过腕厚 1.35 倍和 8mm 才处理。
背侧固定；限厚延伸到指尖，仅沿厚度方向压缩；100轮平滑只作用于掌面内部。
不使用今日整手平滑/指部回补结果，厚度正常、方向不明确或检查失败时保持基准。
"""
import copy
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from body_parts import load_part_indices
from finger_restore import restore_fingers
from hand_pipeline import smoothstep,scene_for,depths,stats,pairs,relax

RATIO=1.35
EXCESS_MM=8.


def refine_hands(mesh_path,position_file):
    mesh=o3d.io.read_triangle_mesh(str(mesh_path));mesh.compute_vertex_normals()
    v=np.asarray(mesh.vertices).copy();f=np.asarray(mesh.triangles);normals=np.asarray(mesh.vertex_normals);n=len(v)
    info=json.loads(Path(position_file).read_text(encoding='utf-8'))
    labels=np.full(n,'unknown',dtype=object)
    for name,part in info['parts'].items():labels[part['vertex_indices']]=name
    for name in ('left_hand','right_hand'):load_part_indices(position_file,name,mesh_path)
    known=np.flatnonzero(labels!='unknown');nearest=labels[known[cKDTree(v[known]).query(v,workers=4)[1]]]
    edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    graph=coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    selected=copy.deepcopy(mesh);records=[];all_changed=np.zeros(n,bool);all_back=np.zeros(n,bool)
    for name in ('left_hand','right_hand'):
        record={'part':name,'applied':False,'thickness_cap_applied':False}
        seed=np.asarray(info['parts'][name]['vertex_indices'],dtype=np.int64)
        arm_name=name.replace('hand','lower_arm');arm=np.asarray(info['parts'][arm_name]['vertex_indices'],dtype=np.int64)
        if min(len(seed),len(arm))<30:
            records.append({**record,'reason':'Insufficient hand/forearm labels; baseline retained'});continue
        _,axes=np.linalg.eigh(np.cov(v[seed].T));back=axes[:,0];mean=float(np.mean(normals[seed]@back))
        if mean<0:back=-back
        palm=-back
        distance=cKDTree(v[arm]).query(v[seed],workers=4)[0]
        wrist=np.median(v[seed[distance<=np.quantile(distance,.08)+3]],axis=0)
        forward=v[seed].mean(0)-wrist;forward-=np.dot(forward,palm)*palm
        if np.linalg.norm(forward)<10:
            records.append({**record,'reason':'Wrist direction ambiguous; baseline retained'});continue
        forward/=np.linalg.norm(forward);right=np.cross(forward,palm)
        local=v-wrist;t=local@forward;y=local@right;d=local@palm
        length=float(np.quantile(t[seed],.97));lo,hi=np.quantile(y[seed],[.01,.99]);half=(hi-lo)/2
        if length<45 or half<10:
            records.append({**record,'reason':'Unreliable hand frame; baseline retained'});continue
        other=np.flatnonzero((labels!='unknown')&~np.isin(labels,[name,arm_name]))
        if len(other):
            contact=float(np.quantile(cKDTree(v[other]).query(v[seed],workers=4)[0],.05))
            record['other_body_distance_q05_mm']=contact
            if contact<6:
                records.append({**record,'reason':'Hand contacts another body part; independent thickness unreliable, baseline retained'});continue
        enclosure=(t>-65)&(t<length*1.35)&(y>lo-35)&(y<hi+35)&(d>-100)&(d<140)
        allowed=np.isin(labels,['unknown',name,arm_name])&((labels!='unknown')|np.isin(nearest,[name,arm_name]))
        measured=enclosure&allowed
        scene=scene_for(mesh,measured);wrist_scene=scene_for(mesh,enclosure);sections=[]
        for t0 in (-40,-35,-30,-25,-20,-15):
            tt,yy=np.meshgrid(np.linspace(t0,t0+5,5),np.linspace(-8,8,20))
            points=wrist+tt.ravel()[:,None]*forward+yy.ravel()[:,None]*right
            _,_,gap,ok=depths(wrist_scene,points,palm);values=gap[ok&(gap<85)]
            if len(values)>=30:sections.append((float(np.median(values)),t0))
        if not sections:
            records.append({**record,'reason':'No complete wrist section; baseline retained'});continue
        target,section=min(sections)
        tt,yy=np.meshgrid(np.linspace(20,length*.70,30),np.linspace(lo+half*.2,hi-half*.2,35))
        probes=wrist+tt.ravel()[:,None]*forward+yy.ravel()[:,None]*right
        _,_,before,valid=depths(scene,probes,palm)
        record.update(wrist_center_mm=wrist.tolist(),target_thickness_mm=target,wrist_section_t_mm=section,
                      before_thickness=stats(before[valid]))
        if valid.sum()<60:
            records.append({**record,'reason':'Palm coverage insufficient; baseline retained'});continue
        p95=float(np.quantile(before[valid],.95));thick=p95>target*RATIO and p95-target>EXCESS_MM
        record.update(thickness_ratio=p95/target,excess_mm=p95-target,excessive=bool(thick))
        if not thick:
            records.append({**record,'reason':'Thickness difference within tolerance; entire hand unchanged'});continue
        if abs(mean)<.35:
            records.append({**record,'reason':'Palm/back direction ambiguous; do not move back surface'});continue
        hand=measured&(t>0)&np.isin(labels,['unknown',name]);ids=np.flatnonzero(hand)
        points=wrist+t[ids,None]*forward+y[ids,None]*right
        lower,upper,gap,ok=depths(scene,points,palm)
        protected=(normals[ids]@back>.25)|(~ok)
        protected|=ok&(np.abs(d[ids]-lower)<1.)
        scale=np.ones(len(ids));scale[ok]=np.minimum(1,target/gap[ok])
        new=d[ids].copy();new[ok]=lower[ok]+(d[ids][ok]-lower[ok])*scale[ok]
        shift=np.maximum(d[ids]-new,0)*smoothstep(t[ids]/10)
        shift[protected]=0
        cap=copy.deepcopy(selected);xyz=np.asarray(selected.vertices).copy();xyz[ids]-=shift[:,None]*palm
        cap.vertices=o3d.utility.Vector3dVector(xyz);cap.compute_vertex_normals()
        # Finger geometry is used only as an exclusion mask. No restoration or
        # smoothing is applied to fingers; authorized depth-only cap remains.
        _,_,finger=restore_fingers(mesh,mesh,info,[record])
        palm_mask=hand&(finger==0)&(normals@palm>.3)&(t<length*.68)
        full_back=np.zeros(n,bool);full_back[ids[protected]]=True
        palm_mask[full_back]=False
        boundary=np.unique(edges[np.any(palm_mask[edges],axis=1)&~np.all(palm_mask[edges],axis=1)])
        boundary=boundary[palm_mask[boundary]];e=edges[np.all(palm_mask[edges],axis=1)]
        lengths=np.linalg.norm(v[e[:,0]]-v[e[:,1]],axis=1)
        paths=coo_matrix((np.r_[lengths,lengths,np.zeros(len(boundary))],
                        (np.r_[e[:,0],e[:,1],np.full(len(boundary),n)],np.r_[e[:,1],e[:,0],boundary])),shape=(n+1,n+1)).tocsr()
        distance=dijkstra(paths,directed=True,indices=n)[:-1]
        weights=np.zeros(n);weights[palm_mask]=smoothstep(distance[palm_mask]/8)*smoothstep(t[palm_mask]/10);weights[boundary]=0
        box=np.r_[v[hand],xyz[hand]];points=v[f]
        face_ids=np.flatnonzero(np.all(points.max(1)>=box.min(0)-5,axis=1)&np.all(points.min(1)<=box.max(0)+5,axis=1))
        reference_pairs=pairs(selected,face_ids,hand);attempts=[];chosen=None
        raw_shift=np.zeros(n);raw_shift[ids]=shift;movable_depth=raw_shift>0
        degree=np.maximum(np.asarray(graph.sum(1)).ravel(),1)
        for depth_cycles in (0,10,35):
            field=raw_shift.copy()
            for _ in range(depth_cycles):
                field=.25*raw_shift+.75*(graph@field/degree)
                field[~movable_depth|full_back]=0
            cap=copy.deepcopy(selected);coords=np.asarray(selected.vertices).copy();coords[ids]-=field[ids,None]*palm
            cap.vertices=o3d.utility.Vector3dVector(coords);cap.compute_vertex_normals()
            for smoothing_cap in (8.,4.,0.):
                candidate=relax(cap,graph,weights,100,smoothing_cap,[(ids,palm)]) if smoothing_cap else copy.deepcopy(cap)
                cv=np.asarray(candidate.vertices).copy();cv[full_back]=v[full_back]
                candidate.vertices=o3d.utility.Vector3dVector(cv);candidate.compute_vertex_normals()
                # Retain baseline locally around residual folds. Attenuating the
                # already constrained displacement never smooths fingers/back.
                original_candidate=cv.copy();attenuation=np.ones(n)
                for repair_step in range(8):
                    extra=pairs(candidate,face_ids,hand)-reference_pairs
                    if not extra:break
                    affected=np.unique(f[np.asarray(list(extra),dtype=np.int64).ravel()])
                    affected=affected[hand[affected]]
                    if not len(affected):break
                    attenuation[affected]=0
                    halo=np.zeros(n,bool);halo[affected]=True
                    for level in (.25,.5,.75):
                        halo|=np.asarray(graph@halo.astype(float)).ravel()>0
                        attenuation[halo]=np.minimum(attenuation[halo],level)
                    cv=np.asarray(selected.vertices)+(original_candidate-np.asarray(selected.vertices))*attenuation[:,None]
                    candidate.vertices=o3d.utility.Vector3dVector(cv);candidate.compute_vertex_normals()
                old=v[f];old_area=np.linalg.norm(np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0]),axis=1)
                points=cv[f];area=np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]),axis=1)
                collapsed=int(np.sum(area<old_area*.001));extra=pairs(candidate,face_ids,hand)-reference_pairs
                attempts.append({'depth_field_cycles':depth_cycles,'palm_smoothing_cap_mm':smoothing_cap,'new_local_intersections':len(extra),'collapsed_faces':collapsed,
                                 'locally_retained_vertices':int(np.sum(hand&(attenuation<1)))})
                if not extra and not collapsed:chosen=candidate;break
            if chosen is not None:break
        record['attempts']=attempts
        if chosen is None:
            records.append({**record,'reason':'Palm-only candidates failed local checks; baseline retained'});continue
        cv=np.asarray(chosen.vertices)
        assert np.array_equal(cv[full_back],v[full_back])
        # Cap reaches the fingertips without shifting their projected outline.
        finger_ids=np.flatnonzero(hand&(finger>0))
        outline=float(np.max(np.abs((cv[finger_ids]-v[finger_ids])@np.c_[forward,right]))) if len(finger_ids) else 0.
        assert outline<1e-6
        _,_,after,ok_after=depths(scene_for(chosen,measured),probes,palm);common=valid&ok_after
        record.update(applied=True,thickness_cap_applied=True,reason='Excessive palm thickness: depth-only cap and palm-only smoothing',
                      after_thickness=stats(after[common]),palm_smoothed_vertices=int((weights>0).sum()),
                      smoothing_iterations=100 if smoothing_cap else 0,finger_outline_shift_mm=outline,
                      depth_field_cycles=depth_cycles,
                      locally_retained_vertices=int(np.sum(hand&(attenuation<1))),
                      back_vertices_fixed=int(full_back.sum()),cap_extends_to_fingertips=True)
        selected=chosen;all_changed|=hand;all_back|=full_back;records.append(record)
    result=np.asarray(selected.vertices)
    assert np.array_equal(result[~all_changed],v[~all_changed])
    assert np.array_equal(result[all_back],v[all_back])
    assert np.array_equal(np.asarray(selected.triangles),f)
    assert np.array_equal(np.asarray(selected.vertex_colors),np.asarray(mesh.vertex_colors))
    applied=any(part['applied'] for part in records)
    return selected,{'status':'complete','applied':applied,'method':'conditional_palm_only' if applied else 'pre_hand_baseline_retained',
                     'parts':records,'trigger_ratio':RATIO,'trigger_excess_mm':EXCESS_MM,
                     'new_local_intersections':0,'movement_max_mm':float(np.linalg.norm(result-v,axis=1).max()),
                     'reason':'Only excessive palms processed; back/finger smoothing disabled',
                     'notes':['Baseline is the mesh before all whole-hand optimization.',
                              'Finger outline/back preserved; excessive thickness can be reduced along the depth axis through fingertips.',
                              'Unreliable direction/coverage or failed checks retain baseline. Local checks are not full-body certification.']}
