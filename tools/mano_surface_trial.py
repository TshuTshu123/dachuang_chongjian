"""MANO 参考面融合原始点云的小尺度变化，独立试验，不修改正式模型。
根目录：python tools/mano_surface_trial.py --scan 3 4
输出 outputs/experiments/mano_surface/export_编号：比较图、报告和通过检查的一份 PLY。
先局部细分，保留参考面的厚度；仅采用距参考面 3 mm 内的观测。
腕口与指尖固定，沿法线限幅融合高频残差；留出点、厚度、翻面与相交检查失败则回退。
"""
from pathlib import Path
import argparse, copy, json, sys, time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from body_parts import file_hash
from tools.surface_trials import panel
from hand_pipeline import smoothstep, scene_for, depths
from surface_quality import intersection_blocks, intersection_pairs


def subdivide(mesh, labels, frames, passes=2):
    """Split selected edges conformingly; midpoints do not change the reference surface."""
    v=np.asarray(mesh.vertices).copy(); f=np.asarray(mesh.triangles).copy()
    colors=np.asarray(mesh.vertex_colors).copy()
    for _ in range(passes):
        selected=np.zeros(len(f),bool)
        for k,frame in enumerate(frames):
            y=((v-frame['wrist'])@frame['axes'])[:,1]
            inside=(labels==k)&(y>35)&(y<frame['end']+8)
            selected|=np.all(inside[f],axis=1)
        edges=np.unique(np.sort(np.concatenate([f[selected][:,[0,1]],f[selected][:,[1,2]],f[selected][:,[2,0]]]),axis=1),axis=0)
        lookup={tuple(e):len(v)+i for i,e in enumerate(edges)}
        new=[]
        for a,b,c in f:
            ab=lookup.get(tuple(sorted((a,b))));bc=lookup.get(tuple(sorted((b,c))));ca=lookup.get(tuple(sorted((c,a))))
            count=sum(x is not None for x in (ab,bc,ca))
            if count==0:new.append((a,b,c))
            elif count==3:new.extend([(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)])
            elif count==1:
                if ab is not None:new.extend([(a,ab,c),(ab,b,c)])
                elif bc is not None:new.extend([(b,bc,a),(bc,c,a)])
                else:new.extend([(c,ca,b),(ca,a,b)])
            else:
                if ab is None:new.extend([(c,ca,bc),(ca,a,b),(ca,b,bc)])
                elif bc is None:new.extend([(a,ab,ca),(ab,b,c),(ab,c,ca)])
                else:new.extend([(b,bc,ab),(bc,c,a),(bc,a,ab)])
        v=np.r_[v,v[edges].mean(1)];colors=np.r_[colors,colors[edges].mean(1)]
        labels=np.r_[labels,labels[edges[:,0]]]
        f=np.asarray(new,int)
    result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(f))
    result.vertex_colors=o3d.utility.Vector3dVector(colors);result.compute_vertex_normals()
    return result,labels


def distance(mesh,points):
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(points.astype(np.float32)),nthreads=4).numpy()


def thickness(mesh,mask,frame):
    v=np.asarray(mesh.vertices);local=(v-frame['wrist'])@frame['axes'];ids=np.flatnonzero(mask)
    palm=ids[(local[ids,1]>38)&(local[ids,1]<frame['end'])]
    xlo,xhi=np.quantile(local[palm,0],[.15,.85])
    xx,yy=np.meshgrid(np.linspace(xlo,xhi,15),np.linspace(38,frame['end'],15))
    probes=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]@frame['axes'].T+frame['wrist']
    return depths(scene_for(mesh,mask),probes,frame['axes'][:,2])[2:]


def trial(scan):
    started=time.perf_counter()
    rp=next(p for p in sorted((ROOT/'outputs/models'/f'export_{scan:05d}').glob('*/report.json'),reverse=True)
            if json.loads(p.read_text(encoding='utf-8')).get('mano_hand_processing',{}).get('applied'))
    report=json.loads(rp.read_text(encoding='utf-8'));source=rp.parent/report['artifacts']['ply']['name']
    digest=file_hash(source);raw_path=Path(report['source']);raw_digest=file_hash(raw_path)
    info=json.loads((rp.parent/'body_parts.json').read_text(encoding='utf-8'))
    assert info['mesh']['sha256']==digest
    fit=json.loads((rp.parent/'intermediate/mano/report.json').read_text(encoding='utf-8'))
    original=o3d.io.read_triangle_mesh(str(source));n_original=len(original.vertices)
    raw=np.asarray(o3d.io.read_point_cloud(str(raw_path)).points).copy()
    if report['fingerprint']['unit']=='m':raw*=1000
    raw=raw[np.isfinite(raw).all(1)]
    frame_by={p['part']:p for p in fit['parts']}
    palm_by={p['part']:p for p in report['hand_processing']['palm_volume']['parts']}
    labels=np.full(n_original,-1,int);frames=[]
    for i,name in enumerate(['left_hand','right_hand']):
        labels[info['parts'][name]['vertex_indices']]=i
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
        print(scan,json.dumps(record),flush=True)
        if qa:accepted.append((float(np.median(after_d)),candidate,record))
    selected=min(accepted,key=lambda x:x[0]) if accepted else None
    chosen=selected[1] if selected else reference
    assert np.array_equal(np.asarray(chosen.vertices)[:n_original][labels[:n_original]<0],np.asarray(original.vertices)[labels[:n_original]<0])
    assert np.array_equal(np.asarray(chosen.triangles),faces)
    print('CHECK topology',scan,flush=True)
    assert chosen.is_vertex_manifold()==reference.is_vertex_manifold()
    assert len(chosen.get_non_manifold_edges())==len(reference.get_non_manifold_edges())
    print('RENDER',scan,flush=True)
    out=ROOT/'outputs/experiments/mano_surface'/f'export_{scan:05d}';out.mkdir(parents=True,exist_ok=True)
    baseline=o3d.io.read_triangle_mesh(str(rp.parent/report['artifacts']['base']['name']))
    canvas=Image.new('RGB',(1200,4*425),'white');draw=ImageDraw.Draw(canvas)
    for i,frame in enumerate(frames):
        ids=np.flatnonzero(labels==i);area=v[ids];center=(area.min(0)+area.max(0))/2;span=float(np.ptp(area,axis=0).max()*1.25)
        # Render local crops; distant full-body triangles cannot affect this hand view.
        crops=[]
        for mesh in [baseline,reference,chosen]:
            points=np.asarray(mesh.vertices);tri=np.asarray(mesh.triangles);xyz=points[tri]
            roi=np.all(xyz.max(1)>=area.min(0)-50,axis=1)&np.all(xyz.min(1)<=area.max(0)+50,axis=1)
            crop=copy.deepcopy(mesh);crop.remove_triangles_by_mask(~roi);crop.remove_unreferenced_vertices();crops.append(crop)
        for angle,right in enumerate([frame['axes'][:,0],frame['axes'][:,2]]):
            # Opposite face avoids the adjacent torso on the right hand in scan 4.
            # This affects only the diagnostic camera, never fitting or processing.
            if scan==4 and i==1 and angle==0:right=-right
            row=i*2+angle
            for col,(mesh,title) in enumerate(zip(crops,['Scan reconstruction','MANO reference','MANO + observed surface'])):
                canvas.paste(panel(mesh,center,right,frame['axes'][:,1],span,400),(col*400,row*425+25))
                draw.text((col*400+6,row*425+6),frame['name']+' | '+title,fill='black')
    canvas.save(out/'comparison.jpg',quality=95)
    output=None
    if selected:
        output=out/f'export_{scan:05d}_mano_surface.ply'
        assert o3d.io.write_triangle_mesh(str(output),chosen,write_vertex_normals=True)
        read=o3d.io.read_triangle_mesh(str(output));assert len(read.vertices)==len(chosen.vertices) and len(read.triangles)==len(chosen.triangles)
    result={'source':str(source),'source_sha256':digest,'raw_source':str(raw_path),'raw_sha256':raw_digest,
            'method':'Conforming midpoint subdivision and bounded observation-supported high-pass normal residuals',
            'input_vertices':n_original,'reference_vertices':len(v),'parts':records,'candidates':choices,
            'selected':selected[2] if selected else None,'output':str(output) if output else None,
            'notes':['Thickness measured on common palm rays, not ground-truth anatomy.','Held-out fit measures only raw observations within 3 mm of reference.','Wrist connection, distal fingers and non-hand geometry protected; no synthetic noise.','Original body_parts.json cannot be used with subdivided candidate indices.','Local intersection check only; not full-body print certification.'],
            'seconds':round(time.perf_counter()-started,2)}
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    assert file_hash(source)==digest and file_hash(raw_path)==raw_digest
    print('DONE',scan,result['seconds'],output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scan',type=int,nargs='+',default=[3,4])
    for scan in parser.parse_args().scan:trial(scan)
