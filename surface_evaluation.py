"""共享表面候选评估，由 run.py 与 tools/surface_trials.py 使用。
固定样本检查贴合距离、面部起伏、翻面和局部新增相交；收益不足保留原形。
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from surface_quality import propose,regions,intersection_blocks,intersection_pairs,intersection_groups

def distances(mesh,points):
    scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(points.astype(np.float32)),nthreads=4).numpy()

def metrics(mesh,raw,samples,edges,mask,face):
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);p=v[f];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);normal=cross/np.maximum(np.linalg.norm(cross,axis=1)[:,None],1e-20)
    e=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);fi=np.tile(np.arange(len(f)),3);order=np.lexsort(e.T);e=e[order];fi=fi[order]
    paired=np.flatnonzero(np.all(e[1:]==e[:-1],axis=1));angle=np.degrees(np.arccos(np.clip(np.sum(normal[fi[paired]]*normal[fi[paired+1]],axis=1),-1,1)))
    def rough(region):
        ids=np.all(region[e[paired]],axis=1);return float(np.quantile(angle[ids],.9)) if ids.any() else None
    d=distances(mesh,samples);nearest=cKDTree(raw).query(v[mask],workers=4)[0]
    return {'raw_to_mesh_p95_mm':float(np.quantile(d,.95)),'raw_to_mesh_median_mm':float(np.median(d)),
            'mesh_to_raw_p95_mm':float(np.quantile(nearest,.95)),'normal_angle_p90_deg':rough(mask),'face_angle_p90_deg':rough(face)}


def prepare_quality(mesh,raw,parts,target='surface'):
    v=np.asarray(mesh.vertices);tree=cKDTree(v);rng=np.random.default_rng(7)
    weights,face,edges=regions(mesh,parts,target);mask=weights>.1
    supported=tree.query(raw,workers=4)[1];selected=raw[mask[supported]]
    if not len(selected):
        raise ValueError('No supported vertices in the selected region.')
    samples=selected[rng.choice(len(selected),min(5000,len(selected)),replace=False)]
    before=metrics(mesh,raw,samples,edges,mask,face)
    blocks=intersection_blocks(mesh,weights>0)
    base_pairs=intersection_pairs(mesh,blocks)
    return dict(weights=weights,face=face,edges=edges,mask=mask,samples=samples,before=before,blocks=blocks,base_pairs=base_pairs,target=target)


def assess_surface(mesh,raw,parts,method,context):
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles)
    weights,face,edges,mask,samples,before,blocks,base_pairs=(context[k] for k in ('weights','face','edges','mask','samples','before','blocks','base_pairs'))
    candidate,detail,_,_,_=propose(mesh,raw,parts,method,context.get('target','surface'))
    after=metrics(candidate,raw,samples,edges,mask,face);groups=intersection_groups(candidate,blocks);extra=set().union(*groups.values())-base_pairs
    graph=None;total_reduced=np.zeros(len(v),bool)
    for backoff in range(6):
        if not extra:break
        # Local backoff only; original hand/lip/edge protection still applies.
        if graph is None:graph=coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(v),len(v))).tocsr()
        attenuation=np.ones(len(v));halo=np.zeros(len(v),bool);halo[np.unique(f[np.asarray(list(extra)).ravel()])]=True;attenuation[halo]=0
        for scale in (0.,.25,.5,.75):
            halo|=np.asarray(graph@halo.astype(float)).ravel()>0;attenuation[halo]=np.minimum(attenuation[halo],scale)
        prior=np.asarray(candidate.vertices).copy();xyz=v+(prior-v)*attenuation[:,None]
        candidate.vertices=o3d.utility.Vector3dVector(xyz);candidate.compute_vertex_normals()
        changed=np.linalg.norm(xyz-prior,axis=1)>1e-12;total_reduced|=changed
        groups=intersection_groups(candidate,blocks,previous=groups,dirty_faces=np.any(changed[f],axis=1))
        extra=set().union(*groups.values())-base_pairs
    if total_reduced.any():
        after=metrics(candidate,raw,samples,edges,mask,face);detail['locally_reduced_vertices']=int(total_reduced.sum())
    baseline=before['normal_angle_p90_deg']
    improvement=1-after['normal_angle_p90_deg']/baseline if baseline and after['normal_angle_p90_deg'] is not None else 0.
    face_ok=before['face_angle_p90_deg'] is None or after['face_angle_p90_deg']<=before['face_angle_p90_deg']+1e-6
    accepted=(not extra and face_ok and improvement>.02 and after['raw_to_mesh_p95_mm']<=before['raw_to_mesh_p95_mm']+.15 and after['raw_to_mesh_median_mm']<=before['raw_to_mesh_median_mm']+.05)
    cv=np.asarray(candidate.vertices);p=cv[f];base=v[f];flipped=int(np.sum(np.sum(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])*np.cross(base[:,1]-base[:,0],base[:,2]-base[:,0]),axis=1)<=0))
    accepted=accepted and flipped==0
    detail.update(before=before,after=after,roughness_decrease_percent=improvement*100,new_intersections=len(extra),flipped_faces=flipped,accepted=accepted)
    assert np.array_equal(np.asarray(candidate.vertices)[weights==0],v[weights==0])
    detail['movement_max_mm']=float(np.linalg.norm(np.asarray(candidate.vertices)-v,axis=1).max())
    return candidate,detail


def refine_surface(mesh,raw,parts):
    context=prepare_quality(mesh,raw,parts)
    candidate,detail=assess_surface(mesh,raw,parts,'local_taubin',context)
    return (candidate if detail['accepted'] else mesh),{'status':'complete','applied':detail['accepted'],**detail}
