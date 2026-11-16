"""手部共用几何工具，由掌部处理、MANO 诊断和实验脚本复用。
提供厚度射线采样、局部相交检查和受限平滑；不包含独立整手后处理入口。
"""
import copy
import numpy as np
import open3d as o3d


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
