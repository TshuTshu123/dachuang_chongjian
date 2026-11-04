"""表面细节候选：语义保护、曲率保护及限幅去噪；不单独运行。
输入毫米网格、原始毫米点和匹配的部位数据。保留手、鞋、头发、唇部及面连接。
只做小尺度表面整理，不补造缺失五官；是否采用由独立质量检查决定。
"""
import copy
import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from mesh_refine import bilateral_mesh_filter
from surface_detail import fit_local_planes


def intersection_blocks(mesh,active,size_mm=45.,margin_mm=2.):
    """Spatially partition moving faces; include every overlapping fixed face.
    Bounds use full triangles, not centers alone. Margin exceeds twice the
    displacement cap, so both meshes use identical complete comparison sets.
    """
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);p=v[f]
    lower=p.min(1);upper=p.max(1);moving=np.flatnonzero(np.any(active[f],axis=1))
    keys=np.floor(p[moving].mean(1)/size_mm).astype(np.int64)
    _,group=np.unique(keys,axis=0,return_inverse=True);blocks=[]
    for i in range(int(group.max())+1):
        own=moving[group==i];lo=lower[own].min(0)-margin_mm;hi=upper[own].max(0)+margin_mm
        ids=np.flatnonzero(np.all(upper>=lo,axis=1)&np.all(lower<=hi,axis=1))
        blocks.append((own,ids))
    return blocks


def intersection_groups(mesh,blocks,previous=None,dirty_faces=None):
    groups={} if previous is None else dict(previous);f=np.asarray(mesh.triangles);v=np.asarray(mesh.vertices)
    for i,(own,ids) in enumerate(blocks):
        if dirty_faces is not None and not dirty_faces[ids].any():continue
        used,inverse=np.unique(f[ids].ravel(),return_inverse=True)
        part=o3d.geometry.TriangleMesh();part.vertices=o3d.utility.Vector3dVector(v[used]);part.triangles=o3d.utility.Vector3iVector(inverse.reshape(-1,3))
        hit=np.asarray(part.get_self_intersecting_triangles(),dtype=np.int64).reshape(-1,2)
        pairs=ids[hit];pairs=pairs[np.any(np.isin(pairs,own),axis=1)]
        groups[i]={tuple(sorted(pair)) for pair in pairs.tolist()}
    return groups


def intersection_pairs(mesh,blocks):
    return set().union(*intersection_groups(mesh,blocks).values())


def regions(mesh,parts):
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);n=len(v)
    edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    graph=coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    degree=np.maximum(np.asarray(graph.sum(1)).ravel(),1)
    weights=np.zeros(n);face=np.zeros(n,bool);protected=np.zeros(n,bool)
    if parts:
        for name,part in parts.items():
            ids=part['vertex_indices']
            if name in ('face_neck','torso','upper_clothing','lower_clothing','left_upper_arm','right_upper_arm','left_lower_arm','right_lower_arm','left_upper_leg','right_upper_leg','left_lower_leg','right_lower_leg'):
                weights[ids]=.65 if name=='face_neck' else 1.
            else:protected[ids]=True
        face[parts['face_neck']['vertex_indices']]=True
    else:weights[:]=1.
    mesh.compute_vertex_normals();normal=np.asarray(mesh.vertex_normals)
    spread=1-np.linalg.norm(graph@normal/degree[:,None],axis=1)
    weights*=np.exp(-np.maximum(spread,0)/.08);weights[spread>.18]=0
    for _ in range(3):weights=np.minimum(weights,graph@weights/degree)
    weights[protected]=0
    return weights,face,edges


def propose(mesh,raw,parts,method):
    original=np.asarray(mesh.vertices).copy();f=np.asarray(mesh.triangles)
    weights,face,edges=regions(mesh,parts)
    if method in ('bilateral_light','bilateral_medium'):
        result=copy.deepcopy(mesh);result.scale(.001,center=np.zeros(3))
        strength=method=='bilateral_medium'
        result,detail=bilateral_mesh_filter(result,normal_iterations=5 if strength else 3,vertex_iterations=10 if strength else 5,
                                           max_move=.0008 if strength else .00045,active_weights=weights)
        result.scale(1000,center=np.zeros(3))
        # Exact coordinates at every protected vertex, including round-trip units.
        xyz=np.asarray(result.vertices).copy();xyz[weights==0]=original[weights==0]
        result.vertices=o3d.utility.Vector3dVector(xyz)
    elif method=='local_taubin':
        result=mesh.filter_smooth_taubin(number_of_iterations=4)
        result.vertex_colors=mesh.vertex_colors
        delta=(np.asarray(result.vertices)-original)*weights[:,None]
        delta*=np.minimum(1,.5*weights/np.maximum(np.linalg.norm(delta,axis=1),1e-20))[:,None]
        result.vertices=o3d.utility.Vector3dVector(original+delta)
        detail={'iterations':4,'max_move_mm':.5,'semantic_protection':True}
    elif method=='normal_relax':
        graph=coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(original),len(original))).tocsr()
        degree=np.maximum(np.asarray(graph.sum(1)).ravel(),1);guide=np.asarray(mesh.vertex_normals).copy();xyz=original.copy()
        for _ in range(8):
            laplace=graph@xyz/degree[:,None]-xyz
            delta=xyz+.4*weights[:,None]*np.sum(laplace*guide,axis=1)[:,None]*guide-original
            delta*=np.minimum(1,.6*weights/np.maximum(np.linalg.norm(delta,axis=1),1e-20))[:,None]
            xyz=original+delta
        result=copy.deepcopy(mesh);result.vertices=o3d.utility.Vector3dVector(xyz)
        detail={'iterations':8,'max_move_mm':.6,'direction':'original surface normal only'}
    elif method=='local_plane_5mm':
        tree=cKDTree(raw);distance,nearest=tree.query(original,workers=4)
        normal,offset,valid,flat,_=fit_local_planes(raw*.001,24,.005,4)
        guide=normal[nearest];anchors=raw[nearest]*.001+offset[nearest,None]*guide
        height=np.einsum('ij,ij->i',anchors-original*.001,guide)*1000
        allowed=(distance<3)&valid[nearest]&flat[nearest]&(np.abs(np.sum(guide*np.asarray(mesh.vertex_normals),axis=1))>.8)
        shift=np.clip(height*.45,-.8,.8)*weights*allowed
        xyz=original+shift[:,None]*guide;result=copy.deepcopy(mesh);result.vertices=o3d.utility.Vector3dVector(xyz)
        detail={'radius_mm':5,'neighbors':24,'max_move_mm':.8}
    else:raise ValueError(method)
    xyz=np.asarray(result.vertices).copy()
    xyz[(weights==0)|~np.isfinite(xyz).all(1)]=original[(weights==0)|~np.isfinite(xyz).all(1)]
    old=original[f];normal=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0])
    for _ in range(8):
        p=xyz[f];bad=np.sum(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])*normal,axis=1)<=0
        if not bad.any():break
        blocked=np.unique(f[bad]);xyz[blocked]=original[blocked]
    else:xyz=original.copy()
    result.vertices=o3d.utility.Vector3dVector(xyz);result.compute_vertex_normals()
    assert np.array_equal(xyz[weights==0],original[weights==0])
    assert np.array_equal(np.asarray(result.triangles),f)
    assert np.array_equal(np.asarray(result.vertex_colors),np.asarray(mesh.vertex_colors))
    return result,{'method':method,'parameters':detail,'protected_vertices':int(np.sum(weights==0)),
                  'movement_max_mm':float(np.linalg.norm(xyz-original,axis=1).max())},weights,face,edges
