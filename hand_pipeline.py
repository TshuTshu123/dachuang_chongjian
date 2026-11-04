"""历史整手后处理及共用几何工具；当前默认策略位于 palm_pipeline.py。
整手强平滑与指部回补已撤回默认，仅保留历史实现供对比。
输入最终毫米网格与匹配的 body_parts.json；不改顶点编号、面连接和 RGB。
缺少腕部参考时跳过限厚；新增相交时尝试平滑回退，仍失败则保留原模型。
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


def smoothstep(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


def scene_for(mesh,mask):
    part=o3d.geometry.TriangleMesh();part.vertices=mesh.vertices
    faces=np.asarray(mesh.triangles);part.triangles=o3d.utility.Vector3iVector(faces[np.any(mask[faces],axis=1)])
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(part));return scene


def depths(scene,points,normal):
    hits=[]
    for sign in (-1,1):
        rays=np.c_[points+sign*300*normal,np.tile(-sign*normal,(len(points),1))].astype(np.float32)
        hits.append(scene.cast_rays(o3d.core.Tensor(rays),nthreads=4)['t_hit'].numpy())
    back=-300+hits[0];front=300-hits[1];gap=front-back
    return back,front,gap,np.isfinite(gap)&(gap>1)&(gap<150)


def stats(values):
    return {'samples':len(values),'median_mm':float(np.median(values)),
            'p95_mm':float(np.quantile(values,.95)),'max_mm':float(values.max())} if len(values) else {'samples':0}


def pairs(mesh,face_ids,active):
    part=o3d.geometry.TriangleMesh();part.vertices=mesh.vertices
    f=np.asarray(mesh.triangles);part.triangles=o3d.utility.Vector3iVector(f[face_ids])
    hit=np.asarray(part.get_self_intersecting_triangles(),dtype=np.int64).reshape(-1,2)
    hit=face_ids[hit];affected=np.any(active[f],axis=1)
    hit=hit[np.any(affected[hit],axis=1)]
    return {tuple(sorted(p)) for p in hit.tolist()}


def relax(mesh,graph,weights,cycles,cap,palm_frames=()):
    original=np.asarray(mesh.vertices).copy();current=original.copy()
    degree=np.maximum(np.asarray(graph.sum(1)).ravel(),1)
    for _ in range(cycles):
        delta=current+.45*weights[:,None]*(graph@current/degree[:,None]-current)-original
        for ids,normal in palm_frames:
            delta[ids]-=np.maximum(delta[ids]@normal,0)[:,None]*normal
        delta*=np.minimum(1,cap*weights/np.maximum(np.linalg.norm(delta,axis=1),1e-20))[:,None]
        current=original+delta
    result=copy.deepcopy(mesh);result.vertices=o3d.utility.Vector3dVector(current);result.compute_vertex_normals()
    return result


def refine_hands(mesh_path,position_file):
    mesh=o3d.io.read_triangle_mesh(str(mesh_path));mesh.compute_vertex_normals()
    v=np.asarray(mesh.vertices).copy();f=np.asarray(mesh.triangles);normals=np.asarray(mesh.vertex_normals);n=len(v)
    info=json.loads(Path(position_file).read_text(encoding='utf-8'))
    labels=np.full(n,'unknown',dtype=object)
    for name,part in info['parts'].items():labels[part['vertex_indices']]=name
    for name in ('left_hand','right_hand'):load_part_indices(position_file,name,mesh_path)
    labeled=np.flatnonzero(labels!='unknown')
    _,nearest=cKDTree(v[labeled]).query(v,workers=4);nearest_class=labels[labeled[nearest]]
    edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    graph=coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    xyz=v.copy();region=np.zeros(n,bool);weights=np.zeros(n);frames=[];records=[];face_mask=np.zeros(len(f),bool);points=v[f]
    for name in ('left_hand','right_hand'):
        seed=np.asarray(info['parts'][name]['vertex_indices'],dtype=np.int64)
        arm_name=name.replace('hand','lower_arm');arm=np.asarray(info['parts'][arm_name]['vertex_indices'],dtype=np.int64)
        record={'part':name,'applied':False}
        if min(len(seed),len(arm))<30:
            records.append({**record,'reason':'Insufficient hand/forearm labels'});continue
        _,axes=np.linalg.eigh(np.cov(v[seed].T));back=axes[:,0]
        mean=float(np.mean(normals[seed]@back))
        # When both surfaces are labeled, sign is ambiguous. Symmetric thickness
        # compression then avoids guessing which surface is the palm.
        symmetric=abs(mean)<.35
        if mean<0:back=-back
        palm=-back
        distances=cKDTree(v[arm]).query(v[seed],workers=4)[0]
        wrist=np.median(v[seed[distances<=np.quantile(distances,.08)+3]],axis=0)
        forward=v[seed].mean(0)-wrist;forward-=np.dot(forward,palm)*palm
        if np.linalg.norm(forward)<10:
            records.append({**record,'reason':'Wrist direction ambiguous'});continue
        forward/=np.linalg.norm(forward);right=np.cross(forward,palm)
        local=v-wrist;t=local@forward;y=local@right;d=local@palm
        length=float(np.quantile(t[seed],.97));lo,hi=np.quantile(y[seed],[.01,.99]);half=(hi-lo)/2
        if length<45 or half<10:
            records.append({**record,'reason':'Hand frame implausible'});continue
        enclosure=(t>-65)&(t<length*1.12)&(y>lo-40)&(y<hi+40)&(d>-100)&(d<140)
        allowed=np.isin(labels,['unknown',name,arm_name])
        # Protect unclassified scalp/clothing near hands touching head/body.
        allowed&=(labels!='unknown')|np.isin(nearest_class,[name,arm_name])
        mask=enclosure&(t>-50)&allowed;ids=np.flatnonzero(mask)
        region|=mask;weights[mask]=np.maximum(weights[mask],smoothstep((t[mask]+50)/50))
        scene=scene_for(mesh,enclosure);sections=[]
        for t0 in (-40,-35,-30,-25,-20,-15):
            tt,yy=np.meshgrid(np.linspace(t0,t0+5,5),np.linspace(-8,8,20))
            probes=wrist+tt.ravel()[:,None]*forward+yy.ravel()[:,None]*right
            _,_,g,ok=depths(scene,probes,palm);good=g[ok&(g<85)]
            if len(good)>=30:sections.append((float(np.median(good)),t0))
        target,section=min(sections) if sections else (None,None)
        hand_ids=ids[t[ids]>-12]
        if target is not None:
            probes=wrist+t[hand_ids,None]*forward+y[hand_ids,None]*right
            lower,upper,gap,ok=depths(scene,probes,palm)
            scale=np.ones(len(hand_ids));scale[ok]=np.minimum(1,target/gap[ok])
            anchor=(lower+upper)/2 if symmetric else lower
            new=d[hand_ids].copy();new[ok]=anchor[ok]+(d[hand_ids][ok]-anchor[ok])*scale[ok]
            shift=d[hand_ids]-new
            shift*=smoothstep((t[hand_ids]+2)/10)*smoothstep((length*1.05-t[hand_ids])/(length*.12))
            xyz[hand_ids]-=shift[:,None]*palm
        tt,yy=np.meshgrid(np.linspace(20,length*.73,30),np.linspace(lo+half*.2,hi-half*.2,35))
        probes=wrist+tt.ravel()[:,None]*forward+yy.ravel()[:,None]*right
        _,_,before,valid=depths(scene,probes,palm)
        frames.append((name,ids,enclosure,probes,palm,before,valid,target))
        records.append({**record,'wrist_center_mm':wrist.tolist(),'target_thickness_mm':target,
                        'wrist_section_t_mm':section,'symmetric_cap':symmetric,'roi_vertices':len(ids),
                        'cap_skipped_reason':None if target is not None else 'Incomplete wrist cross-section'})
        box=v[enclosure];face_mask|=np.all(points.max(1)>=box.min(0)-5,axis=1)&np.all(points.min(1)<=box.max(0)+5,axis=1)
    if not frames:return mesh,{'status':'complete','applied':False,'method':'wrist cap + strong hand smoothing','parts':records,'reason':'No usable hand frame'}
    boundary=np.unique(edges[np.any(region[edges],axis=1)&~np.all(region[edges],axis=1)]);boundary=boundary[region[boundary]]
    e=edges[np.all(region[edges],axis=1)];lengths=np.linalg.norm(v[e[:,0]]-v[e[:,1]],axis=1)
    paths=coo_matrix((np.r_[lengths,lengths,np.zeros(len(boundary))],
                    (np.r_[e[:,0],e[:,1],np.full(len(boundary),n)],np.r_[e[:,1],e[:,0],boundary])),shape=(n+1,n+1)).tocsr()
    distance=dijkstra(paths,directed=True,indices=n)[:-1];weights*=np.minimum(distance/15,1);weights[boundary]=0
    active=weights>0;xyz[~active]=v[~active]
    capped=copy.deepcopy(mesh);capped.vertices=o3d.utility.Vector3dVector(xyz);capped.compute_vertex_normals()
    core=np.linalg.norm(xyz-v,axis=1)>.1;pre_weights=core.astype(float)
    first=np.asarray(graph@core.astype(float)).ravel()>0;pre_weights[first&~core]=.6
    second=np.asarray(graph@first.astype(float)).ravel()>0;pre_weights[second&(pre_weights==0)]=.25
    pre_weights*=weights
    # Mild palm pass before the approved whole-hand 100-cycle pass.
    prepared=relax(capped,graph,pre_weights,24,4)
    proposals=[('wrist_cap_and_strong_smooth',relax(prepared,graph,weights,100,10)),
               ('strong_smooth_only',None),('reduced_smooth_50',None),('reduced_smooth_25',None)]
    selected=mesh;method=None;checks=[];face_ids=np.flatnonzero(face_mask)
    reference=pairs(mesh,face_ids,region)
    smoothing_only=None
    for name,candidate in proposals:
        if candidate is None:
            if smoothing_only is None:smoothing_only=relax(mesh,graph,weights,100,10)
            candidate=copy.deepcopy(smoothing_only)
            if name.startswith('reduced'):
                scale=.5 if name.endswith('50') else .25
                candidate.vertices=o3d.utility.Vector3dVector(v+scale*(np.asarray(smoothing_only.vertices)-v))
        old=v[f];old_cross=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0])
        stabilized=set()
        for _ in range(8):
            cv=np.asarray(candidate.vertices).copy();p=cv[f];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
            bad=np.linalg.norm(cross,axis=1)<np.linalg.norm(old_cross,axis=1)*.001
            if not bad.any():break
            vertices=np.unique(f[bad]);stabilized.update(vertices.tolist())
            # Local 5% backtracking addresses isolated almost-zero triangles;
            # it must still pass the subsequent crossing/thickness checks.
            cv[vertices]=np.where(active[vertices,None],.95*cv[vertices]+.05*v[vertices],v[vertices])
            candidate.vertices=o3d.utility.Vector3dVector(cv)
        candidate.compute_vertex_normals()
        candidate,finger_report,_=restore_fingers(mesh,candidate,info,records)
        cv=np.asarray(candidate.vertices);p=cv[f];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        collapsed=int(np.sum(np.linalg.norm(cross,axis=1)<np.linalg.norm(old_cross,axis=1)*.001))
        new=pairs(candidate,face_ids,region)-reference;thickness=[];thickness_ok=True
        for part,ids,enclosure,probes,normal,before,valid,target in frames:
            _,_,after,ok=depths(scene_for(candidate,enclosure),probes,normal);common=valid&ok
            a=stats(before[common]);b=stats(after[common])
            if not b['samples']:thickness_ok=False
            elif b['p95_mm']>a['p95_mm']+1:thickness_ok=False
            thickness.append({'part':part,'before':a,'after':b,'target_mm':target})
        check={'attempt':name,'new_local_intersections':len(new),'collapsed_faces':collapsed,
               'stabilized_vertices':len(stabilized),'finger_protection':finger_report,
               'thickness_not_increased':thickness_ok,'thickness':thickness}
        checks.append(check)
        if not new and not collapsed and thickness_ok:
            selected=candidate;method=name;break
    movement=np.linalg.norm(np.asarray(selected.vertices)-v,axis=1)
    assert np.array_equal(np.asarray(selected.vertices)[~active],v[~active])
    assert np.array_equal(np.asarray(selected.triangles),f)
    assert np.array_equal(np.asarray(selected.vertex_colors),np.asarray(mesh.vertex_colors))
    for record in records:
        record['applied']=bool(method and record.get('roi_vertices',0))
        record['thickness_cap_applied']=bool(method=='wrist_cap_and_strong_smooth' and record.get('target_thickness_mm') is not None)
    return selected,{'status':'complete','applied':bool(method and movement.max()>.01),
                     'method':method or 'original retained','parts':records,'attempts':checks,
                     'smoothing_iterations':100,'smoothing_cap_mm':10,'wrist_transition_mm':50,
                     'movement_max_mm':float(movement.max()),'new_local_intersections':checks[-1]['new_local_intersections'] if method else 0,
                     'reason':'Local safety checks passed' if method else 'Candidates rejected; original retained',
                     'notes':['Geometric appearance approximation; palm/back details reduced, original finger geometry protected.',
                              'Known other regions protected; unknown contact areas use nearest semantic class.',
                              'Local intersection checks, not whole-body print certification.']}
