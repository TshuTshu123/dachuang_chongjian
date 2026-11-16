"""指部保护：恢复强平滑前的指尖/细长指部，掌面手背其余顶点完全不动。
由 palm_pipeline.py 借用分区掩码保护指部；不是手指识别模型。按原网格投影细长分支和远端定位，
在指根内侧渐变，完整恢复末端。网格拓扑、颜色和顶点编号保持不变。
"""
import copy
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt,binary_fill_holes,label
from scipy.spatial import cKDTree

VERSION=2

def smoothstep(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


def skeletonize(mask):
    """Zhang-Suen thinning using NumPy only; no new environment dependency."""
    image=np.pad(mask.astype(bool),1)
    for _ in range(200):
        changed=False
        for phase in (0,1):
            p=[image[:-2,1:-1],image[:-2,2:],image[1:-1,2:],image[2:,2:],
               image[2:,1:-1],image[2:,:-2],image[1:-1,:-2],image[:-2,:-2]]
            count=sum(x.astype(np.uint8) for x in p)
            transitions=sum((~p[i]&p[(i+1)%8]).astype(np.uint8) for i in range(8))
            if phase==0:corner=~(p[0]&p[2]&p[4])&~(p[2]&p[4]&p[6])
            else:corner=~(p[0]&p[2]&p[6])&~(p[0]&p[4]&p[6])
            remove=image[1:-1,1:-1]&(count>=2)&(count<=6)&(transitions==1)&corner
            if remove.any():image[1:-1,1:-1][remove]=False;changed=True
        if not changed:break
    return image[1:-1,1:-1]


def restore_fingers(reference,processed,info,parts):
    v=np.asarray(reference.vertices);current=np.asarray(processed.vertices).copy();f=np.asarray(reference.triangles)
    if v.shape!=current.shape or not np.array_equal(np.sort(f,axis=1),np.sort(np.asarray(processed.triangles),axis=1)):
        raise ValueError('Finger reference topology does not match processed mesh')
    names=np.full(len(v),'unknown',dtype=object)
    for name,part in info['parts'].items():names[part['vertex_indices']]=name
    known=np.flatnonzero(names!='unknown');nearest=names[known[cKDTree(v[known]).query(v,workers=4)[1]]]
    weights=np.zeros(len(v));records=[]
    for part in parts:
        name=part['part'];seed=np.asarray(info['parts'][name]['vertex_indices'],dtype=np.int64)
        if len(seed)<30 or not part.get('wrist_center_mm'):continue
        wrist=np.array(part['wrist_center_mm']);_,axes=np.linalg.eigh(np.cov(v[seed].T));normal=axes[:,0]
        forward=v[seed].mean(0)-wrist;forward-=np.dot(forward,normal)*normal
        if np.linalg.norm(forward)<10:continue
        forward/=np.linalg.norm(forward);right=np.cross(forward,normal)
        delta=v-wrist;t=delta@forward;y=delta@right;d=delta@normal
        length=float(np.quantile(t[seed],.97));lo,hi=np.quantile(y[seed],[.01,.99])
        region=(t>0)&(t<length*1.2)&(y>lo-25)&(y<hi+25)&(np.abs(d-np.median(d[seed]))<80)
        region&=np.isin(names,['unknown',name])&((names!='unknown')|np.isin(nearest,[name,name.replace('hand','lower_arm')]))
        ids=np.flatnonzero(region)
        if len(ids)<30:continue
        # One-millimeter projected silhouette, independent of scan orientation.
        lower=np.array([lo-28,-5]);upper=np.array([hi+28,length*1.2+5])
        size=np.ceil(upper-lower).astype(int)+1;pixels=np.c_[y,t]-lower
        canvas=Image.new('1',(int(size[0]),int(size[1])),0);draw=ImageDraw.Draw(canvas)
        for tri in f[np.all(region[f],axis=1)]:draw.polygon([tuple(pixels[i]) for i in tri],fill=1)
        silhouette=binary_fill_holes(np.asarray(canvas,dtype=bool))
        radius=float(np.clip((hi-lo)*.12,8,13));core=distance_transform_edt(silhouette)>=radius
        digit=np.zeros(len(ids))
        if core.any():
            distance=distance_transform_edt(~core);thin=silhouette&(distance>radius+2)
            components,count=label(thin);keep=np.zeros(count+1,bool)
            for component in range(1,count+1):
                points=np.argwhere(components==component)
                if len(points)>8 and distance[components==component].max()>radius+10 and np.ptp(points,axis=0).max()>15:keep[component]=True
            xy=np.rint(pixels[ids]).astype(int);xy[:,0]=np.clip(xy[:,0],0,size[0]-1);xy[:,1]=np.clip(xy[:,1],0,size[1]-1)
            digit=smoothstep((distance[xy[:,1],xy[:,0]]-radius-2)/8)*keep[components[xy[:,1],xy[:,0]]]
            # Recover the complete thin branch, not just a rounded tip on an
            # over-smoothed narrow stalk. Thick palm skeleton remains protected.
            medial=skeletonize(silhouette);inside=distance_transform_edt(silhouette)
            rows,cols=np.indices(silhouette.shape);thin_medial=medial&(inside<radius)&((rows+lower[1])>length*.25)
            branches,count=label(thin_medial,structure=np.ones((3,3),bool));kept=np.zeros_like(medial)
            palm_row,palm_col=np.unravel_index(np.argmax(inside),inside.shape)
            for component in range(1,count+1):
                points=np.argwhere(branches==component)
                if len(points)<10:continue
                distal_branch=(points[:,0]+lower[1]).max()>length*.70
                thumb_branch=np.abs(points[:,1]-palm_col).max()>inside[palm_row,palm_col]+8
                if distal_branch or thumb_branch:kept|=branches==component
            other=medial&~kept
            if kept.any() and other.any():
                finger_distance=distance_transform_edt(~kept);palm_distance=distance_transform_edt(~other)
                branch_weight=smoothstep((palm_distance-finger_distance)/5)
                branch_weight[inside>=radius+2]=0
                digit=np.maximum(digit,branch_weight[xy[:,1],xy[:,0]])
        # Broad fused fingers may not form narrow silhouette branches. Restore
        # their distal end too; the proximal palm remains outside this band.
        distal=smoothstep((t[ids]/length-.68)/.12)
        weight=np.maximum(digit,distal);weights[ids]=np.maximum(weights[ids],weight)
        records.append({'part':name,'finger_region_vertices':int((weight>0).sum()),
                        'fully_restored_vertices':int((weight>=1-1e-12).sum()),'silhouette_core_radius_mm':radius})
    before=current.copy();current+=(v-current)*weights[:,None]
    current[weights==1]=v[weights==1]
    assert np.array_equal(current[weights==0],before[weights==0])
    out=copy.deepcopy(processed);out.vertices=o3d.utility.Vector3dVector(current);out.compute_vertex_normals()
    return out,{'version':VERSION,'parts':records,'changed_vertices':int((np.linalg.norm(current-before,axis=1)>1e-8).sum()),
                'nonfinger_vertices_bitwise_preserved':True},weights
