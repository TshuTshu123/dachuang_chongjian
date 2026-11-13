"""原始观测约束的腕部连接试验，不改变正式模型。
运行：python tools/wrist_observation_trial.py --scan 3 4
以已有 mano_surface 候选为输入，保留远端手掌手指，以原扫描重建的腕部为参考。
优先保留面连接，按原扫描截面校正中心与宽厚；原始点云用于复核贴合。
仅在变形不通过时尝试短段重接，检查失败则保留输入。
输出 experiments/wrist_observation：单份候选、对照图、原始观测参考点及 QA。
人工测量点由用户指定，本脚本不生成测量标记点。
"""
from pathlib import Path
import argparse,copy,json,sys,time
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from body_parts import file_hash
from hand_pipeline import smoothstep,pairs
from tools.mano_join import boundary_loops,bridge_loops
from tools.wrist_cut_trial import clip_keep
from tools.surface_trials import panel
from sole_repair import triangulate
from surface_quality import intersection_blocks,intersection_pairs

THETA=np.linspace(-np.pi,np.pi,64,endpoint=False)
UNIT=np.c_[np.cos(THETA),np.sin(THETA)]


def ellipse(points):
    center=np.median(points,axis=0);delta=points-center
    cov=np.cov(delta.T);val,vec=np.linalg.eigh(cov)
    radius=np.sqrt(np.maximum(val,9)*2);angle=np.arctan2(vec[1,0],vec[0,0])
    x=np.r_[center,np.log(radius),angle]
    def evaluate(x):
        c,s=np.cos(x[4]),np.sin(x[4]);rot=np.array([[c,-s],[s,c]])
        d=(points-x[:2])@rot;r=np.exp(x[2:4])
        return (np.sqrt(np.sum((d/r)**2,axis=1))-1)*np.mean(r)
    fit=least_squares(evaluate,x,loss='soft_l1',f_scale=2.,max_nfev=70,
                      bounds=(np.r_[center-20,np.log([6,6]),-10],np.r_[center+20,np.log([65,65]),10]))
    x=fit.x;c,s=np.cos(x[4]),np.sin(x[4]);rot=np.array([[c,-s],[s,c]])
    shape=rot@np.diag(np.exp(2*x[2:4]))@rot.T
    r=1/np.sqrt(np.einsum('ni,ij,nj->n',UNIT,np.linalg.inv(shape),UNIT))
    return x[:2],r,float(np.median(np.abs(evaluate(x))))


def raw_profile(local,levels):
    profiles=[]
    for y in levels:
        pts=local[np.abs(local[:,1]-y)<5][:,[0,2]]
        if len(pts)<35:raise ValueError('Insufficient section observations')
        center,pred,error=ellipse(pts)
        d=pts-center;angles=np.arctan2(d[:,1],d[:,0]);r=np.linalg.norm(d,axis=1)
        values=[];coverage=0
        for j,theta in enumerate(THETA):
            delta=np.abs(np.angle(np.exp(1j*(angles-theta))))
            near=delta<np.deg2rad(15)
            if near.sum()>=3:
                coverage+=1;values.append(np.clip(np.median(r[near]),pred[j]-3,pred[j]+3))
            else:values.append(pred[j])
        if coverage<40 or error>5:raise ValueError(f'Section unreliable at {y:.1f}: coverage={coverage}, residual={error:.2f}')
        radii=gaussian_filter1d(values,1.2,mode='wrap')
        profiles.append({'y':float(y),'center':center,'radius':radii,'points':len(pts),'covered_angles':coverage,'residual_mm':error})
    centers=gaussian_filter1d(np.array([p['center'] for p in profiles]),1,axis=0,mode='nearest')
    radii=gaussian_filter1d(np.array([p['radius'] for p in profiles]),1,axis=0,mode='nearest')
    for i,p in enumerate(profiles):p['center']=centers[i];p['radius']=radii[i]
    return profiles


def section(mesh,wrist,axes,y,allowed=None):
    v=(np.asarray(mesh.vertices)-wrist)@axes;f=np.asarray(mesh.triangles)
    p=v[f];keep=(p[:,:,1].min(1)<y)&(p[:,:,1].max(1)>y)&np.all(np.abs(p[:,:,[0,2]]).mean(1)<90,axis=1)
    ff=f[keep]
    edge=np.sort(np.stack([ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]],axis=1),axis=2)
    crossing=(v[edge[:,:,0],1]-y)*(v[edge[:,:,1],1]-y)<0
    edge=edge[crossing]
    if len(edge)<10 or len(edge)%2:raise ValueError('Section unavailable')
    edges,inverse=np.unique(edge,axis=0,return_inverse=True);links=inverse.reshape(-1,2)
    a=v[edges[:,0]];b=v[edges[:,1]];t=(y-a[:,1])/(b[:,1]-a[:,1]);points=(a+t[:,None]*(b-a))[:,[0,2]]
    neighbors={}
    for a,b in links:
        neighbors.setdefault(int(a),[]).append(int(b));neighbors.setdefault(int(b),[]).append(int(a))
    loops=[];seen=set()
    for start in neighbors:
        if start in seen or len(neighbors[start])!=2:continue
        loop=[];previous=None;current=start
        while current not in loop:
            loop.append(current);seen.add(current)
            choices=[n for n in neighbors.get(current,[]) if n!=previous]
            if not choices:break
            previous,current=current,choices[0]
        if current!=start or len(loop)<8:continue
        xy=points[loop];area=abs(np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1)))/2
        if np.linalg.norm(xy.mean(0))<40:loops.append((area,xy))
    if not loops:raise ValueError('No isolated outer wrist contour')
    _,xy=max(loops,key=lambda x:x[0]);center,_,_=ellipse(xy)
    angle=np.arctan2(xy[:,1]-center[1],xy[:,0]-center[0]);order=np.argsort(angle)
    xy=xy[order];ang=angle[order]
    return np.c_[*[np.interp(THETA,np.r_[ang-2*np.pi,ang,ang+2*np.pi],np.tile(xy[:,k],3)) for k in range(2)]]


def ring_sample(v,loop,wrist,axes):
    xy=((v[loop]-wrist)@axes)[:,[0,2]];center=np.mean(xy,axis=0)
    angle=np.arctan2(xy[:,1]-center[1],xy[:,0]-center[0]);order=np.argsort(angle);angle=angle[order];xy=xy[order]
    return np.c_[*[np.interp(THETA,np.r_[angle-2*np.pi,angle,angle+2*np.pi],np.tile(xy[:,k],3)) for k in range(2)]]


def crop(mesh,wrist,axes,low=-55,high=65):
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);local=(v-wrist)@axes;p=local[f]
    mask=(p[:,:,1].max(1)>low)&(p[:,:,1].min(1)<high)&np.all(p[:,:,[0,2]].max(1)>-65,axis=1)&np.all(p[:,:,[0,2]].min(1)<65,axis=1)
    out=copy.deepcopy(mesh);out.remove_triangles_by_mask(~mask);out.remove_unreferenced_vertices();return out


def distances(mesh,points):
    scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(points.astype(np.float32)),nthreads=4).numpy()


def rebuild(mesh,raw,wrist,axes,low,high,strength,allowed_fn,scanmesh):
    original=np.asarray(mesh.vertices).copy();v=original.copy();f=np.asarray(mesh.triangles);colors=np.asarray(mesh.vertex_colors).copy()
    local=(v-wrist)@axes;fc=local[f].mean(1)
    region=(np.abs(fc[:,0])<55)&(np.abs(fc[:,2])<60)&(fc[:,1]>low-15)&(fc[:,1]<high+15)
    allowed=allowed_fn(v);region&=np.all(allowed[f],axis=1)
    selected=f[region];outside=f[~region]
    # Nearby torso triangles can share semantic labels by mistake; keep only
    # the connected wrist patch inside this local band, not every nearby face.
    edges=np.r_[selected[:,[0,1]],selected[:,[1,2]],selected[:,[2,0]]]
    graph=coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(v),len(v))).tocsr()
    _,components=connected_components(graph,directed=False)
    selected_ids=np.flatnonzero(region)
    nearest=selected_ids[np.argmin(np.linalg.norm(fc[selected_ids]-[0,0,0],axis=1))]
    owner=components[f[nearest,0]]
    region&=components[f[:,0]]==owner
    selected=f[region];outside=f[~region]
    v,c,lower=clip_keep(v,colors,selected,wrist,axes,low,True,np.ones(len(selected),bool))
    v,c,upper=clip_keep(v,c,selected,wrist,axes,high,False,np.ones(len(selected),bool))
    f=np.r_[outside,lower,upper];loops=boundary_loops(f);ends=[]
    for level in [low,high]:
        matches=[loop for loop in loops if np.max(np.abs(((v[loop]-wrist)@axes)[:,1]-level))<1e-5 and np.linalg.norm(((v[loop].mean(0)-wrist)@axes)[[0,2]])<60]
        if not matches:raise ValueError('Wrist boundary unavailable')
        def area(loop):
            xy=((v[loop]-wrist)@axes)[:,[0,2]]
            return abs(np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1)))/2
        matches.sort(key=area,reverse=True)
        if len(matches)>3 or sum(area(l) for l in matches[1:])>.2*area(matches[0]):raise ValueError('Ambiguous wrist boundary loops')
        for loop in matches[1:]:
            local_cap=(v-wrist)@axes
            cap_vertices=local_cap[:,[1,0,2]]
            f=np.r_[f,np.asarray(triangulate(loop.tolist(),cap_vertices.tolist()),int)]
        ends.append(matches[0])
    lo,hi=[ring_sample(v,e,wrist,axes) for e in ends]
    # Retained surface derivatives use nearby sections without endpoint phase search.
    d0=(lo-section(mesh,wrist,axes,low-5,allowed))/5
    d1=(section(mesh,wrist,axes,high+5,allowed)-hi)/5
    d0=np.clip(gaussian_filter1d(d0,1.5,axis=0,mode='wrap'),-.8,.8)
    d1=np.clip(gaussian_filter1d(d1,1.5,axis=0,mode='wrap'),-.8,.8)
    raw=raw[allowed_fn(raw)];rlocal=(raw-wrist)@axes
    rlocal=rlocal[(np.abs(rlocal[:,0])<55)&(np.abs(rlocal[:,2])<55)&(rlocal[:,1]>low-8)&(rlocal[:,1]<high+8)]
    perm=np.random.default_rng(23).permutation(len(rlocal));test_mask=np.zeros(len(rlocal),bool);test_mask[perm[::5]]=True
    train=rlocal[~test_mask];test=rlocal[test_mask&(rlocal[:,1]<=10)]
    if len(test)<60:raise ValueError('Insufficient wrist validation points')
    levels=np.linspace(low,high,int(np.ceil((high-low)/3))+1)
    prof=raw_profile(train,levels)
    scan_allowed=allowed_fn(np.asarray(scanmesh.vertices))
    previous=ends[0];newfaces=[];allids=[];actual=[]
    for j,y in enumerate(levels[1:-1],1):
        t=(y-low)/(high-low);L=high-low
        prior=(2*t**3-3*t*t+1)*lo+(t**3-2*t*t+t)*L*d0+(-2*t**3+3*t*t)*hi+(t**3-t*t)*L*d1
        p=prof[j]
        observed=section(scanmesh,wrist,axes,y,scan_allowed)
        observed=gaussian_filter1d(observed,.5,axis=0,mode='wrap')
        # Preserve the raw-supported section center and dimensions rather than
        # carrying over the old template's wrist offset or scan corrugations.
        observed+=p['center']-observed.mean(0)
        observed=.75*(p['center']+UNIT*p['radius'][:,None])+.25*observed
        # Raw observations dominate around the wrist, fading before the thick palm.
        weight=strength*smoothstep((y-low)/8)*(1-smoothstep((y-10)/(high-10)))
        xy=(1-weight)*prior+weight*observed
        xyz=np.c_[xy[:,0],np.full(64,y),xy[:,1]]@axes.T+wrist
        ids=np.arange(len(v),len(v)+64);allids.extend(ids);v=np.r_[v,xyz]
        c=np.r_[c,np.asarray(mesh.vertex_colors)[cKDTree(original).query(xyz)[1]]]
        if j==1:
            tri,_=bridge_loops(v,previous,ids);newfaces.extend(tri)
        else:
            nxt=np.roll(ids,-1);prev=np.roll(previous,-1)
            newfaces.extend(np.c_[previous,prev,ids]);newfaces.extend(np.c_[prev,nxt,ids])
        previous=ids;actual.append((y,xy))
    tri,_=bridge_loops(v,previous,ends[1]);newfaces.extend(tri);f=np.r_[f,np.asarray(newfaces,int)]
    candidate=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(f));candidate.vertex_colors=o3d.utility.Vector3dVector(c)
    if not candidate.orient_triangles():raise ValueError('Orientation conflict')
    candidate.compute_vertex_normals()
    if len(candidate.get_non_manifold_edges()) or len(candidate.get_non_manifold_vertices()):raise ValueError('Non-manifold wrist')
    if sum(map(len,boundary_loops(f)))>sum(map(len,boundary_loops(np.asarray(mesh.triangles)))):
        details=[{'n':len(loop),'center_local':((v[loop].mean(0)-wrist)@axes).round(2).tolist()} for loop in boundary_loops(f)]
        raise ValueError('New wrist opening '+str(details))
    active=np.arange(len(v))>=len(original);p=v[f];box=v[np.unique(np.r_[ends[0],ends[1],allids])]
    roi=np.flatnonzero(np.all(p.max(1)>=box.min(0)-4,axis=1)&np.all(p.min(1)<=box.max(0)+4,axis=1))
    intersections=pairs(candidate,roi,active)
    if intersections:raise ValueError(f'{len(intersections)} local intersections')
    # Only validate the reconstructed, observed wrist zone. Exclude remote palm
    # bulges and raw outliers using the training profile, never the candidate.
    valid_test=(test[:,1]>=low+8)&(test[:,1]<=5)
    for k,pt in enumerate(test):
        j=int(np.argmin(np.abs(levels-pt[1])));pr=prof[j];delta=pt[[0,2]]-pr['center']
        theta=np.arctan2(delta[1],delta[0]);rad=np.interp(theta,np.r_[THETA,THETA[0]+2*np.pi],np.r_[pr['radius'],pr['radius'][0]])
        valid_test[k]&=abs(np.linalg.norm(delta)-rad)<=4
    test=test[valid_test]
    if len(test)<40:raise ValueError('Too few supported held-out wrist observations')
    testworld=test@axes.T+wrist
    before=distances(crop(mesh,wrist,axes),testworld);after=distances(crop(candidate,wrist,axes),testworld)
    # Independent held-out observations must agree with the proposed wrist surface.
    if np.median(after)>np.median(before)+.25 or np.quantile(after,.95)>np.quantile(before,.95)+.3:
        raise ValueError(f'Held-out fit median {np.median(before):.3f}->{np.median(after):.3f}, P95 {np.quantile(before,.95):.3f}->{np.quantile(after,.95):.3f}')
    search=[(j,p) for j,p in enumerate(prof) if -30<=p['y']<=5]
    perimeter=[float(np.linalg.norm(np.roll(p['center']+UNIT*p['radius'][:,None],-1,axis=0)-(p['center']+UNIT*p['radius'][:,None]),axis=1).sum()) for j,p in search]
    index=int(np.argmin(perimeter));j,ref=search[index]
    center=np.r_[ref['center'][0],ref['y'],ref['center'][1]]@axes.T+wrist
    tangent=np.r_[prof[min(j+1,len(prof)-1)]['center'][0]-prof[max(j-1,0)]['center'][0],levels[min(j+1,len(prof)-1)]-levels[max(j-1,0)],prof[min(j+1,len(prof)-1)]['center'][1]-prof[max(j-1,0)]['center'][1]]@axes.T;tangent/=np.linalg.norm(tangent)
    comparisons=[]
    for y,xy in actual:
        if not -25<=y<=5:continue
        target=prof[np.argmin(np.abs(levels-y))];prior=section(mesh,wrist,axes,y,allowed)
        cen=xy.mean(0);old=prior.mean(0);rawcenter=target['center']
        per=lambda a:float(np.linalg.norm(np.roll(a,-1,axis=0)-a,axis=1).sum())
        comparisons.append({'level_mm':float(y),'center_before_to_raw_mm':float(np.linalg.norm(old-rawcenter)),'center_after_to_raw_mm':float(np.linalg.norm(cen-rawcenter)),
                            'perimeter_before_mm':per(prior),'perimeter_after_mm':per(xy),'raw_perimeter_mm':per(rawcenter+UNIT*target['radius'][:,None])})
    assert np.array_equal(v[:len(original)],original)
    if comparisons:
        before_center=np.median([p['center_before_to_raw_mm'] for p in comparisons]);after_center=np.median([p['center_after_to_raw_mm'] for p in comparisons])
        before_size=np.median([abs(p['perimeter_before_mm']-p['raw_perimeter_mm']) for p in comparisons]);after_size=np.median([abs(p['perimeter_after_mm']-p['raw_perimeter_mm']) for p in comparisons])
        if after_center>before_center+.25 or after_size>before_size+1:
            raise ValueError(f'Section center/perimeter regressed: {before_center:.2f}->{after_center:.2f} mm, {before_size:.2f}->{after_size:.2f} mm')
        if after_center>before_center-.3 and after_size>before_size-1:
            raise ValueError('No meaningful section-center or circumference gain')
    result={'low_mm':low,'high_mm':high,'raw_strength':strength,'heldout_points':len(test),'before_median_mm':float(np.median(before)),'after_median_mm':float(np.median(after)),
            'before_p95_mm':float(np.quantile(before,.95)),'after_p95_mm':float(np.quantile(after,.95)),'new_intersections':0,'new_open_edges':0,'retained_vertices_unchanged':True,
            'section_comparisons':comparisons,'reference_point_mm':center.tolist(),'reference_axis':tangent.tolist(),'reference_level_mm':ref['y'],
            'reference_search_boundary':index in [0,len(search)-1],'reference_ellipse_residual_mm':ref['residual_mm'],'raw_sections':[{'level_mm':p['y'],'points':p['points'],'covered_angles':p['covered_angles'],'ellipse_residual_mm':p['residual_mm']} for p in prof],
            'reference_note':'Raw-supported narrow-section geometric reference; not a validated anatomical landmark. Boundary minimum means axial wrist location remains ambiguous.'}
    return candidate,result


def warp_wrist(mesh,raw,wrist,axes,allowed_fn,cap,scanmesh):
    """Move only the observed wrist band; preserve all faces and distal hand coordinates."""
    v=np.asarray(mesh.vertices).copy();f=np.asarray(mesh.triangles);local=(v-wrist)@axes
    allowed=allowed_fn(v);radial=np.linalg.norm(local[:,[0,2]],axis=1)
    weight=smoothstep((local[:,1]+5)/10)*(1-smoothstep((local[:,1]-10)/30))
    weight*=allowed&(radial<50)
    active=weight>.001
    r=raw[allowed_fn(raw)];r=(r-wrist)@axes
    r=r[(np.abs(r[:,0])<50)&(np.abs(r[:,2])<50)&(r[:,1]>-32)&(r[:,1]<27)]
    perm=np.random.default_rng(23).permutation(len(r));test=r[perm[::5]];train=r[np.delete(perm,np.arange(0,len(perm),5))]
    levels=np.linspace(-25,22,17);prof=raw_profile(train,levels)
    old_centers=[];old_radii=[]
    for j,y in enumerate(levels):
        pts=section(mesh,wrist,axes,y,allowed)
        ec,er,_=ellipse(pts);old_centers.append(ec);old_radii.append(er)
        target=section(scanmesh,wrist,axes,y)
        center,radius,_=ellipse(target)
        prof[j]['center']=center;prof[j]['radius']=radius
    centers=gaussian_filter1d(np.array(old_centers),1,axis=0,mode='nearest')
    ids=np.flatnonzero(active);yy=local[ids,1]
    oldcenter=np.c_[np.interp(yy,levels,centers[:,0]),np.interp(yy,levels,centers[:,1])]
    rawcenters=np.array([p['center'] for p in prof]);newcenter=np.c_[np.interp(yy,levels,rawcenters[:,0]),np.interp(yy,levels,rawcenters[:,1])]
    def sqrt_shape(r):
        design=np.c_[UNIT[:,0]**2,2*UNIT[:,0]*UNIT[:,1],UNIT[:,1]**2]
        a,b,c=np.linalg.lstsq(design,1/np.maximum(r,6)**2,rcond=None)[0]
        val,vec=np.linalg.eigh([[a,b],[b,c]])
        return (vec/np.sqrt(np.maximum(val,1/65**2)))@vec.T
    transforms=np.array([sqrt_shape(p['radius'])@np.linalg.inv(sqrt_shape(er)) for p,er in zip(prof,old_radii)])
    transforms=gaussian_filter1d(transforms,1.5,axis=0,mode='nearest')
    mat=np.empty((len(ids),2,2))
    for a in range(2):
        for b in range(2):mat[:,a,b]=np.interp(yy,levels,transforms[:,a,b])
    xy=newcenter+np.einsum('nij,nj->ni',mat,local[ids][:,[0,2]]-oldcenter)
    offset=np.zeros_like(v);offset[ids]=np.c_[xy[:,0]-local[ids,0],np.zeros(len(ids)),xy[:,1]-local[ids,2]]@axes.T
    offset*=weight[:,None]
    offset*=min(1,cap/max(np.linalg.norm(offset,axis=1).max(),1e-10))
    # Smooth displacement over the existing mesh while holding the band boundary fixed.
    edges=np.unique(np.sort(np.r_[f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]],axis=1),axis=0)
    graph=coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(v),len(v))).tocsr()
    degree=np.maximum(np.asarray(graph.sum(1)).ravel(),1)
    anchor=offset.copy()
    for _ in range(4):offset=.7*anchor+.3*(graph@offset)/degree[:,None];offset[~active]=0
    blocks=intersection_blocks(mesh,active,margin_mm=cap*2+1);existing=intersection_pairs(mesh,blocks)
    base=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    candidate=copy.deepcopy(mesh);retracted=np.zeros(len(v),bool)
    for step in range(12):
        xyz=v+offset;candidate.vertices=o3d.utility.Vector3dVector(xyz);candidate.compute_vertex_normals()
        cross=np.cross(xyz[f[:,1]]-xyz[f[:,0]],xyz[f[:,2]]-xyz[f[:,0]])
        bad=np.flatnonzero(np.sum(cross*base,axis=1)<=0);extra=intersection_pairs(candidate,blocks)-existing
        if not len(bad) and not extra:break
        if step==11:raise ValueError('Could not safely deform wrist without crossings')
        conflict=np.unique(np.r_[bad,np.asarray(list(extra),int).ravel()]);halo=np.zeros(len(v),bool);halo[np.unique(f[conflict])]=True
        atten=np.ones(len(v));atten[halo]=0
        for factor in [.25,.5,.75]:halo|=np.asarray(graph@halo.astype(float)).ravel()>0;atten[halo]=np.minimum(atten[halo],factor)
        retracted|=(atten<1)&active;offset*=atten[:,None]
    valid=(test[:,1]>=-20)&(test[:,1]<=10)
    for k,pt in enumerate(test):
        j=int(np.argmin(abs(levels-pt[1])));p=prof[j];dd=pt[[0,2]]-p['center'];theta=np.arctan2(dd[1],dd[0])
        rr=np.interp(theta,np.r_[THETA,THETA[0]+2*np.pi],np.r_[p['radius'],p['radius'][0]])
        valid[k]&=abs(np.linalg.norm(dd)-rr)<4
    testworld=test[valid]@axes.T+wrist
    if len(testworld)<45:raise ValueError('Insufficient held-out wrist points')
    before=distances(crop(mesh,wrist,axes),testworld);after=distances(crop(candidate,wrist,axes),testworld)
    comparisons=[]
    for j,y in enumerate(levels):
        if not 0<=y<=16:continue
        p=prof[j];a=section(mesh,wrist,axes,y,allowed);b=section(candidate,wrist,axes,y,allowed)
        per=lambda a:float(np.linalg.norm(np.roll(a,-1,axis=0)-a,axis=1).sum())
        comparisons.append({'level_mm':float(y),'center_before_to_reference_mm':float(np.linalg.norm(a.mean(0)-p['center'])),'center_after_to_reference_mm':float(np.linalg.norm(b.mean(0)-p['center'])),
                            'perimeter_before_mm':per(a),'perimeter_after_mm':per(b),'reference_perimeter_mm':per(p['center']+UNIT*p['radius'][:,None])})
    bc=np.median([p['center_before_to_reference_mm'] for p in comparisons]);ac=np.median([p['center_after_to_reference_mm'] for p in comparisons])
    bs=np.median([abs(p['perimeter_before_mm']-p['reference_perimeter_mm']) for p in comparisons]);az=np.median([abs(p['perimeter_after_mm']-p['reference_perimeter_mm']) for p in comparisons])
    band=active&(local[:,1]>=0)&(local[:,1]<=20);query_ids=np.flatnonzero(band)
    d0=distances(crop(scanmesh,wrist,axes),v[query_ids]);d1=distances(crop(scanmesh,wrist,axes),np.asarray(candidate.vertices)[query_ids])
    if np.median(after)>np.median(before)+.5 or np.quantile(after,.95)>np.quantile(before,.95)+.5 or ac>bc+.25 or az>bs+1 or (ac>bc-.3 and az>bs-1) or np.median(d1)>np.median(d0)-.1:
        raise ValueError(f'Wrist warp QA: fit {np.median(before):.2f}->{np.median(after):.2f}, P95 {np.quantile(before,.95):.2f}->{np.quantile(after,.95):.2f}; center {bc:.2f}->{ac:.2f}, size {bs:.2f}->{az:.2f}')
    assert np.array_equal(np.asarray(candidate.vertices)[~active],v[~active])
    assert np.array_equal(np.asarray(candidate.triangles),f)
    return candidate,{'method':'original_scan_section_affine_warp','section_reference':'original_scan_reconstruction_not_anatomical_ground_truth','scan_surface_before_median_mm':float(np.median(d0)),'scan_surface_after_median_mm':float(np.median(d1)),'max_movement_mm':float(np.linalg.norm(offset,axis=1).max()),'changed_vertices':int((np.linalg.norm(offset,axis=1)>.001).sum()),'locally_retracted_vertices':int(retracted.sum()),
                      'heldout_points':len(testworld),'before_median_mm':float(np.median(before)),'after_median_mm':float(np.median(after)),'before_p95_mm':float(np.quantile(before,.95)),'after_p95_mm':float(np.quantile(after,.95)),
                      'center_error_before_mm':float(bc),'center_error_after_mm':float(ac),'perimeter_error_before_mm':float(bs),'perimeter_error_after_mm':float(az),'section_comparisons':comparisons,
                      'new_intersections':0,'flipped_faces':0,'new_open_edges':0,'connectivity_unchanged':True,'manual_measurement_landmarks':'not_generated',
                      'reference_note':'Original scan sections constrain the transition. Manual measurement landmarks are not inferred. Original proximal wrist and forearm below -5 mm retained.'}


def trial(scan):
    start=time.perf_counter();parent=ROOT/'outputs/experiments/mano_surface'/f'export_{scan:05d}'
    upstream=json.loads((parent/'report.json').read_text(encoding='utf-8'));source=Path(upstream['output']);digest=file_hash(source)
    rp=Path(upstream['source']).parent;rawpath=Path(upstream['raw_source']);rawhash=file_hash(rawpath)
    raw=np.asarray(o3d.io.read_point_cloud(str(rawpath)).points).copy()
    formal=json.loads((rp/'report.json').read_text(encoding='utf-8'))
    if formal['fingerprint']['unit']=='m':raw*=1000
    fit=json.loads((rp/'intermediate/mano/report.json').read_text(encoding='utf-8'))
    original=o3d.io.read_triangle_mesh(str(source));selected=copy.deepcopy(original);records=[]
    scanmesh=o3d.io.read_triangle_mesh(str(rp/formal['artifacts']['base']['name']))
    original_formal=o3d.io.read_triangle_mesh(str(rp/formal['artifacts']['ply']['name']))
    info=json.loads((rp/'body_parts.json').read_text(encoding='utf-8'))
    labels=np.full(len(original_formal.vertices),'unknown',dtype=object)
    for name,part in info['parts'].items():labels[part['vertex_indices']]=name
    known=np.flatnonzero(labels!='unknown');known_tree=cKDTree(np.asarray(original_formal.vertices)[known])
    for frame in fit['parts']:
        wrist=np.array(frame['wrist_mm']);axes=np.array(frame['frame_axes']);attempts=[];chosen=None
        allowed_names=[frame['part'],frame['part'].replace('hand','lower_arm')]
        def allowed_fn(xyz):return np.isin(labels[known[known_tree.query(xyz,workers=4)[1]]],allowed_names)
        warp_candidates=[]
        for cap in [6.,10.,14.]:
            try:
                candidate,record=warp_wrist(selected,raw,wrist,axes,allowed_fn,cap,scanmesh)
                record['displacement_cap_mm']=cap;warp_candidates.append((candidate,record))
            except ValueError as e:attempts.append({'method':'raw_section_warp','cap_mm':cap,'reason':str(e)})
        if warp_candidates:
            chosen=min(warp_candidates,key=lambda pair:pair[1]['scan_surface_after_median_mm'])
            chosen[1]['accepted_candidates']=[{'cap_mm':r['displacement_cap_mm'],'scan_distance_median_mm':r['scan_surface_after_median_mm'],'section_center_difference_mm':r['center_error_after_mm']} for _,r in warp_candidates]
        for low,high in [(-18.7,25.3),(-8.7,32.3),(-28.7,30.3)]:
            if chosen:break
            for strength in [1.,.85]:
                try:
                    candidate,record=rebuild(selected,raw,wrist,axes,low,high,strength,allowed_fn,scanmesh)
                    chosen=(candidate,record);break
                except ValueError as e:attempts.append({'low_mm':low,'high_mm':high,'strength':strength,'reason':str(e)})
            if chosen:break
        if chosen:selected,result=chosen;result.update(part=frame['part'],applied=True,attempts=attempts)
        else:result={'part':frame['part'],'applied':False,'attempts':attempts}
        records.append(result);print(scan,frame['part'],json.dumps(result),flush=True)
    out=ROOT/'outputs/experiments/wrist_observation'/f'export_{scan:05d}';out.mkdir(parents=True,exist_ok=True)
    canvas=Image.new('RGB',(1200,4*425),'white');draw=ImageDraw.Draw(canvas)
    for i,frame in enumerate(fit['parts']):
        wrist=np.array(frame['wrist_mm']);axes=np.array(frame['frame_axes']);center=wrist+axes[:,1]*5
        meshes=[crop(m,wrist,axes) for m in [scanmesh,original,selected]]
        for angle,right in enumerate([axes[:,0],axes[:,2]]):
            forward=np.cross(axes[:,1],right)
            if np.dot(forward,wrist-original.get_center())>0:right=-right
            row=2*i+angle
            for col,(mesh,name) in enumerate(zip(meshes,['Scan surface','Previous MANO wrist','Raw-constrained wrist'])):
                canvas.paste(panel(mesh,center,right,axes[:,1],130,400),(col*400,row*425+25));draw.text((col*400+4,row*425+4),frame['part']+' | '+name,fill='black')
    canvas.save(out/'comparison.jpg',quality=95)
    output=None
    if any(r['applied'] for r in records):
        selected.remove_unreferenced_vertices();selected.compute_vertex_normals();output=out/f'export_{scan:05d}_wrist_observation.ply'
        assert o3d.io.write_triangle_mesh(str(output),selected,write_vertex_normals=True)
        loaded=o3d.io.read_triangle_mesh(str(output));assert len(loaded.vertices)==len(selected.vertices) and np.isfinite(np.asarray(loaded.vertices)).all()
    else:
        stale=out/f'export_{scan:05d}_wrist_observation.ply'
        if stale.is_file():stale.unlink()
    result={'source':str(source),'source_sha256':digest,'raw_source':str(rawpath),'raw_sha256':rawhash,'formal_model_unchanged':True,'parts':records,'output':str(output) if output else None,'seconds':round(time.perf_counter()-start,2),
            'notes':['Rebuilt band includes a small proximal forearm section; palms beyond high cut and fingers retained.','Reference point is not validated against anatomical wrist landmarks.','Only local intersections checked.','Old body_parts.json indices must not be reused on candidate.']}
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'alignment_diagnostics.json').write_text(json.dumps({'coordinate_unit':'mm','not_measurement_landmarks':True,'source_raw_sha256':rawhash,'parts':[{k:r[k] for k in ['part','applied','method','reference_point_mm','reference_axis','reference_level_mm','reference_search_boundary','reference_note'] if k in r} for r in records]},ensure_ascii=False,indent=2),encoding='utf-8')
    if (out/'wrist_references.json').is_file():(out/'wrist_references.json').unlink()
    assert file_hash(source)==digest and file_hash(rawpath)==rawhash
    print('DONE',scan,result['seconds'],output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scan',type=int,nargs='+',default=[3,4]);a=p.parse_args()
    for s in a.scan:trial(s)
