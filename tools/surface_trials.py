"""统一表面候选试验，不覆盖 models。根目录：python tools/surface_trials.py --scan 1
--all：所有数据；--method bilateral_light/bilateral_medium/local_plane_5mm：只测一档。
输出 outputs/experiments/surface_quality/输入名/；只保存选中网格、指标和同视角对比。
--region face/hands：仅试验 Meta 标记的脸颈或双手，输出到 part_detail/输入名/部位/。
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,time
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from surface_evaluation import prepare_quality,assess_surface
from body_parts import file_hash




def panel(mesh,center,right,up,span,size=320,colored=False):
    forward=np.cross(up,right);xx,yy=np.meshgrid(np.linspace(-span/2,span/2,size),np.linspace(span/2,-span/2,size))
    origin=center+xx[...,None]*right+yy[...,None]*up-forward*4000
    rays=np.c_[origin.reshape(-1,3),np.tile(forward,(size*size,1))].astype(np.float32)
    scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));hit=scene.cast_rays(o3d.core.Tensor(rays),nthreads=4)
    ok=np.isfinite(hit['t_hit'].numpy());normals=hit['primitive_normals'].numpy()[ok];light=-forward+.3*up+.2*right;light/=np.linalg.norm(light)
    shade=.35+.65*np.abs(normals@light);pix=np.full((size*size,3),.94)
    if colored:
        faces=np.asarray(mesh.triangles)[hit['primitive_ids'].numpy()[ok]]
        uv=hit['primitive_uvs'].numpy()[ok]
        color=np.einsum('ij,ijk->ik',np.c_[1-uv.sum(axis=1),uv],np.asarray(mesh.vertex_colors)[faces])
        pix[ok]=shade[:,None]*color
    else:
        pix[ok]=shade[:,None]*[.75,.8,.87]
    return Image.fromarray(np.rint(pix.reshape(size,size,3)*255).astype(np.uint8))


def trial(parent,methods,target='surface'):
    started=time.perf_counter();paths=sorted(parent.glob('*/report.json'))
    path=next(p for p in reversed(paths) if (p.parent/json.loads(p.read_text(encoding='utf-8'))['artifacts']['ply']['name']).is_file())
    report=json.loads(path.read_text(encoding='utf-8'));source=path.parent/report['artifacts']['ply']['name'];digest=file_hash(source)
    mesh=o3d.io.read_triangle_mesh(str(source));mesh.compute_vertex_normals();v=np.asarray(mesh.vertices)
    info=json.loads((path.parent/'body_parts.json').read_text(encoding='utf-8'));assert info['mesh']['sha256']==digest;parts=info.get('parts',{})
    raw=np.asarray(o3d.io.read_point_cloud(report['source']).points).copy();raw*=1000 if report['fingerprint']['unit']=='m' else 1
    raw=raw[np.isfinite(raw).all(1)]
    # Pilot proposals give a fixed active region shared by every candidate.
    context=prepare_quality(mesh,raw,parts,target);before=context['before'];face=context['face']
    out=(ROOT/'outputs/experiments/surface_quality'/parent.name if target=='surface'
         else ROOT/'outputs/experiments/part_detail'/parent.name/target)
    out.mkdir(parents=True,exist_ok=True)
    records=[];chosen=None;best=-float('inf');meshes=[mesh];names=['baseline']
    for method in methods:
        candidate,detail=assess_surface(mesh,raw,parts,method,context)
        improvement=detail['roughness_decrease_percent']/100;accepted=detail['accepted'];extra=range(detail['new_intersections'])
        records.append(detail);meshes.append(candidate);names.append(method)
        if accepted and improvement>best:chosen=candidate;best=improvement
        print(parent.name,method,'roughness',round(improvement*100,2),'extra',len(extra),'accepted',accepted,flush=True)
    if chosen is not None:o3d.io.write_triangle_mesh(str(out/'selected_mesh.ply'),chosen)
    center=(v.min(0)+v.max(0))/2;axis=np.argmax(np.ptp(v,axis=0));up=np.eye(3)[axis];right=np.eye(3)[(axis+1)%3];span=np.ptp(v,axis=0).max()*1.1
    if face.any():
        head=np.zeros(len(v),bool)
        for name in ('face_neck','hair'):head[parts[name]['vertex_indices']]=True
        center=(v[head].min(0)+v[head].max(0))/2;span=np.ptp(v[head],axis=0).max()*1.15
        direction=np.asarray(mesh.vertex_normals)[face].mean(0);direction-=np.dot(direction,up)*up
        if np.linalg.norm(direction)>.05:right=np.cross(up,direction/np.linalg.norm(direction))
    views=[(center,right,span,'front'),(center,np.cross(up,right),span,'side')]
    if target=='hands':
        views=[]
        for name in ('left_hand','right_hand'):
            ids=parts.get(name,{}).get('vertex_indices',[])
            if len(ids):
                area=v[ids]
                views.append(((area.min(0)+area.max(0))/2,right,np.ptp(area,axis=0).max()*1.3,name))
    canvas=Image.new('RGB',(320*len(meshes),350*len(views)),'white');draw=ImageDraw.Draw(canvas)
    for i,(m,name) in enumerate(zip(meshes,names)):
        for row,(view_center,r,view_span,label) in enumerate(views):
            canvas.paste(panel(m,view_center,r,up,view_span),(i*320,row*350+25))
            draw.text((i*320+5,row*350+5),f'{name} | {label}',fill='black')
    canvas.save(out/'comparison.jpg',quality=94)
    result={'source':str(source),'source_sha256':digest,'region':target,'before':before,'candidates':records,'selected':None if chosen is None else records[np.argmax([r['roughness_decrease_percent'] if r['accepted'] else -1e9 for r in records])]['method'],'elapsed_seconds':round(time.perf_counter()-started,2)}
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');assert file_hash(source)==digest
    print('DONE',parent.name,result['selected'],result['elapsed_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan',type=int,default=1)
    p.add_argument('--all',action='store_true')
    p.add_argument('--region',choices=['surface','face','hands'],default='surface')
    p.add_argument('--method',choices=['bilateral_light','bilateral_medium','local_plane_5mm','normal_relax','local_taubin','regional_mls'])
    a=p.parse_args()
    if a.region!='surface':
        if a.method and a.method!='regional_mls':
            p.error('Face and hand trials currently use regional_mls.')
        methods=['regional_mls']
    else:
        if a.method=='regional_mls':
            p.error('Select --region face or --region hands for regional_mls.')
        methods=[a.method] if a.method else ['bilateral_light','bilateral_medium','local_plane_5mm']
    parents=sorted((ROOT/'outputs/models').glob('export_*')) if a.all else [ROOT/'outputs/models'/f'export_{a.scan:05d}']
    for parent in parents:
        try:
            trial(parent,methods,a.region)
        except ValueError as error:
            if not a.all:
                raise
            print(f'{parent.name} skipped: {error}',flush=True)
