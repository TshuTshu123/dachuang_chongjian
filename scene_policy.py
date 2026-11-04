"""run.py 的人体/通用物体分流工具，不单独执行。
依据 Sapiens 部位组合做保守启发式判断；不是经过标定的物体分类器。
通用物体只按原始观测距离清理外扩表面，保留所有受支持的连通部件。
"""
import copy
from collections import deque
import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


def classify_parts(data):
    if data.get('status')!='complete':
        return {'kind':'unknown','reason':'Human recognition unavailable; use conservative generic processing.'}
    n=max(1,data['mesh']['vertex_count']);parts=data['parts']
    def count(names):return sum(parts.get(name,{}).get('vertex_count',0) for name in names)
    face=count(['face_neck'])/n;head=count(['face_neck','hair'])/n
    trunk=count(['torso','upper_clothing'])/n
    lower=count(['lower_clothing','left_upper_leg','right_upper_leg','left_lower_leg','right_lower_leg'])/n
    limbs=count(['left_hand','right_hand','left_lower_arm','right_lower_arm','left_foot','right_foot','left_shoe','right_shoe'])/n
    human=face>=.003 and head>=.012 and trunk>=.05 and lower>=.04 and limbs>=.003
    return {'kind':'human' if human else 'object','method':'Sapiens coherent part coverage heuristic',
            'reason':'Head, torso, lower body and limb evidence present.' if human else 'No reliable combination of human parts; generic object branch.',
            'evidence':{'face_fraction':face,'head_fraction':head,'trunk_fraction':trunk,'lower_body_fraction':lower,'limb_fraction':limbs},
            'note':'Not a calibrated person detector; --kind human/object overrides ambiguous scans.'}


def generic_observation_trim(mesh,raw_mm,voxel_mm):
    result=copy.deepcopy(mesh);limit=max(12.,voxel_mm*6)
    distance=cKDTree(raw_mm).query(np.asarray(mesh.vertices),workers=4)[0]
    remove=distance>limit
    if remove.mean()>.35:
        return result,{'applied':False,'reason':'More than 35% would be removed; baseline retained.','distance_limit_mm':limit}
    before=len(result.triangles);result.remove_vertices_by_mask(remove);result.compute_vertex_normals()
    return result,{'applied':bool(remove.any()),'method':'Remove unsupported extrapolated vertices, preserve supported components',
                   'distance_limit_mm':limit,'removed_vertices':int(remove.sum()),'removed_faces':before-len(result.triangles)}


def repair_generic_topology(mesh):
    """Bounded face-only cleanup for object meshes; never move vertices.
    Cut conflicting orientation cycles instead of inventing closed surfaces.
    Accept at most 1% face/area loss, otherwise return the original mesh.
    """
    candidate=copy.deepcopy(mesh);before=len(mesh.triangles);area=mesh.get_surface_area()
    candidate.remove_non_manifold_edges()
    for _ in range(4):
        f=np.asarray(candidate.triangles);v=np.asarray(candidate.vertices);p=v[f]
        face_area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
        edge=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);owner=np.tile(np.arange(len(f)),3)
        direction=np.where(edge[:,0]<edge[:,1],1,-1);edge=np.sort(edge,axis=1)
        order=np.lexsort(edge.T);edge=edge[order];owner=owner[order];direction=direction[order]
        k=np.flatnonzero(np.all(edge[1:]==edge[:-1],axis=1));a=owner[k];b=owner[k+1];relation=-direction[k]*direction[k+1]
        graph=coo_matrix((np.r_[relation,relation],(np.r_[a,b],np.r_[b,a])),shape=(len(f),len(f))).tocsr()
        signs=np.zeros(len(f),np.int8);bad=set()
        for root in range(len(f)):
            if signs[root]:continue
            signs[root]=1;queue=deque([root])
            while queue:
                current=queue.popleft()
                for pos in range(graph.indptr[current],graph.indptr[current+1]):
                    other=graph.indices[pos];expected=signs[current]*graph.data[pos]
                    if signs[other]==0:signs[other]=expected;queue.append(other)
                    elif signs[other]!=expected:bad.add(current if face_area[current]<face_area[other] else other)
        if bad:
            candidate.remove_triangles_by_index(sorted(bad))
            continue
        nonmanifold=np.asarray(candidate.get_non_manifold_vertices())
        if len(nonmanifold):
            candidate.remove_triangles_by_mask(np.any(np.isin(f,nonmanifold),axis=1));continue
        break
    removed=before-len(candidate.triangles);lost=max(0.,area-candidate.get_surface_area())
    if removed>.01*before or lost>.01*area or not candidate.is_orientable() or not candidate.is_vertex_manifold():
        return mesh,{'applied':False,'reason':'Object topology cleanup exceeded budget or remained non-orientable.','proposed_removed_faces':removed}
    candidate.remove_unreferenced_vertices();candidate.orient_triangles();candidate.compute_vertex_normals()
    return candidate,{'applied':bool(removed),'method':'Bounded non-manifold and orientation-conflict face cleanup',
                      'removed_faces':removed,'removed_face_fraction':removed/before,'removed_area_fraction':lost/area,
                      'positions_preserved':True,'orientable':True,'vertex_manifold':True,
                      'note':'May leave open seams; does not certify self-intersection freedom or watertightness.'}
