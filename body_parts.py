"""人体部位位置文件：run.py 默认调用，只生成 body_parts.json，不修改模型。
独立补生成：python body_parts.py 模型.ply
  --python 路径：Sapiens GPU Python，默认环境变量 HUMAN_SEG_PYTHON 或本机 test0。
  --axis auto|x|y|z：竖直轴，auto 使用明显最长轴；--up-sign -1：负向朝上。
  --output 路径：自定义 JSON；默认模型同目录 body_parts.json（匹配则复用）。
文件包含 mm 世界坐标的中心/包围盒、0起始顶点编号和源模型 SHA256。
后续读取：ids = load_part_indices('模型目录/body_parts.json', 'left_hand')。
临时渲染图和二维标签自动清理；分割错误写入 JSON，重建模型保持有效。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from tools.sapiens.render_views import rays_for, WIDTH, HEIGHT

ROOT=Path(__file__).resolve().parent
TOOLS=ROOT/'tools'/'sapiens'
DEFAULT_GPU_PYTHON=Path(os.environ.get('HUMAN_SEG_PYTHON',r'D:\Anaconda\conda\envs\test0\python.exe'))
SCHEMA_VERSION=1
PART_NAMES=['background','apparel','face_neck','hair','left_foot','left_hand','left_lower_arm',
            'left_lower_leg','left_shoe','left_sock','left_upper_arm','left_upper_leg',
            'lower_clothing','right_foot','right_hand','right_lower_arm','right_lower_leg',
            'right_shoe','right_sock','right_upper_arm','right_upper_leg','torso','upper_clothing',
            'lower_lip','upper_lip','lower_teeth','upper_teeth','tongue']


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def segmentation_signature(mesh_path, python, axis, up_sign):
    manifest=json.loads((TOOLS/'checkpoint_manifest.json').read_text(encoding='utf-8'))
    sources=[Path(__file__),TOOLS/'infer.py',TOOLS/'render_views.py',TOOLS/'vendor'/'classes_and_palettes.py']
    return {'schema_version':SCHEMA_VERSION,'mesh_sha256':file_hash(mesh_path),
            'checkpoint_sha256':manifest['sha256'],'gpu_python':str(Path(python).resolve()),
            'axis':axis,'up_sign':up_sign,
            'code_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in sources}}


def make_cameras(mesh, source, digest, folder, axis, up_sign):
    vertices=np.asarray(mesh.vertices);extent=np.ptp(vertices,axis=0)
    order=np.argsort(extent)
    if axis=='auto' and extent[order[-1]]<1.3*extent[order[-2]]:
        raise ValueError('Segmentation vertical axis ambiguous; use --seg-axis x/y/z.')
    index=int(order[-1]) if axis=='auto' else 'xyz'.index(axis)
    if extent[index]<=0:raise ValueError('Segmentation axis has zero extent')
    up=np.eye(3)[index]*up_sign
    base=np.eye(3)[[i for i in range(3) if i!=index][-1]]
    center=(vertices.min(0)+vertices.max(0))/2;delta=vertices-center
    cameras=[]
    for yaw in [0,90,180,270]:
        theta=np.deg2rad(yaw);forward=base*np.cos(theta)+np.cross(up,base)*np.sin(theta)
        right=np.cross(forward,up)
        # Fit every model into the same portrait aspect, including wide arm poses.
        height=max(extent[index]*1.1,float(np.ptp(delta@right))*1.1*HEIGHT/WIDTH)
        cameras.append({'input_name':source.stem,'source':str(source),'source_sha256':digest,
                        'yaw':yaw,'target':center.tolist(),'right':right.tolist(),'up':up.tolist(),
                        'forward':forward.tolist(),'height_mm':height,'width_mm':height*WIDTH/HEIGHT,
                        'distance_mm':float(extent.max()*3+1),'image':str(folder/f'view_{yaw}.png')})
    return cameras,'xyz'[index]


def render(mesh, scene, cameras):
    faces=np.asarray(mesh.triangles);colors=np.asarray(mesh.vertex_colors)
    for camera in cameras:
        hit=scene.cast_rays(rays_for(camera),nthreads=4);valid=np.isfinite(hit['t_hit'].numpy())
        ids=hit['primitive_ids'].numpy()[valid].astype(np.int64);uv=hit['primitive_uvs'].numpy()[valid]
        weights=np.maximum(np.c_[1-uv.sum(1),uv],0)
        rgb=np.einsum('ni,nij->nj',weights,colors[faces[ids]])
        light=-np.array(camera['forward'])+.3*np.array(camera['up']);light/=np.linalg.norm(light)
        shade=.75+.25*np.abs(hit['primitive_normals'].numpy()[valid]@light)
        pixels=np.full((WIDTH*HEIGHT,3),.88);pixels[valid]=np.clip(rgb*shade[:,None],0,1)
        Image.fromarray(np.rint(pixels.reshape(HEIGHT,WIDTH,3)*255).astype(np.uint8)).save(camera['image'])


def fuse_labels(mesh, scene, cameras):
    n=len(mesh.vertices);faces=np.asarray(mesh.triangles);k=len(PART_NAMES)
    votes=np.zeros((n,k),dtype=np.float32);support=np.zeros((n,k),dtype=np.uint8)
    for camera in cameras:
        image=Path(camera['image']);labels=np.asarray(Image.open(image.with_name(image.stem+'_labels.png')))
        if labels.shape!=(HEIGHT,WIDTH) or labels.max()>=k:raise ValueError('Unexpected segmentation label image')
        hit=scene.cast_rays(rays_for(camera),nthreads=4);valid=np.isfinite(hit['t_hit'].numpy())
        ids=hit['primitive_ids'].numpy()[valid].astype(np.int64);uv=hit['primitive_uvs'].numpy()[valid]
        weights=np.maximum(np.c_[1-uv.sum(1),uv],0)
        cosine=np.abs(hit['primitive_normals'].numpy()[valid]@np.array(camera['forward']))
        weights*=np.where(cosine>=.2,cosine**2,0)[:,None]
        bins=faces[ids].ravel()*k+np.repeat(labels.ravel()[valid],3)
        current=np.bincount(bins,weights=weights.ravel(),minlength=n*k).reshape(n,k).astype(np.float32)
        totals=current.sum(1)
        support+=((current>1e-5)&(current>=.5*totals[:,None])).astype(np.uint8)
        votes+=current
    winners=votes.argmax(1);total=votes.sum(1);confidence=votes[np.arange(n),winners]/np.maximum(total,1e-20)
    count=support[np.arange(n),winners]
    accepted=(winners>0)&(count>=2)&(confidence>=.65)
    labels=np.where(accepted,winners,-1).astype(np.int16)
    # Keep the previous conservative speck rule for the two hand classes only.
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    specks=[]
    for label in [5,14]:
        selected=labels==label;ids=np.flatnonzero(selected)
        if not len(ids):continue
        mapping=np.full(n,-1);mapping[ids]=np.arange(len(ids));e=edges[np.all(selected[edges],axis=1)]
        graph=coo_matrix((np.ones(len(e)),(mapping[e[:,0]],mapping[e[:,1]])),shape=(len(ids),len(ids))).tocsr()
        _,components=connected_components(graph,directed=False);sizes=np.bincount(components)
        threshold=max(12,int(np.ceil(.03*sizes.max())));remove=ids[sizes[components]<threshold];labels[remove]=-1
        specks.append({'part':PART_NAMES[label],'minimum_component_vertices':threshold,'removed_vertices':len(remove)})
    return labels,confidence,count,specks


def describe_parts(vertices, labels, confidence, support):
    parts={}
    for i,name in enumerate(PART_NAMES[1:],start=1):
        ids=np.flatnonzero(labels==i);points=vertices[ids]
        parts[name]={'class_id':i,'detected':bool(len(ids)),'vertex_count':len(ids),'vertex_indices':ids.tolist(),
                     'center_mm':points.mean(0).tolist() if len(ids) else None,
                     'bbox_mm':{'min':points.min(0).tolist(),'max':points.max(0).tolist()} if len(ids) else None,
                     'vote_agreement_median':float(np.median(confidence[ids])) if len(ids) else None,
                     'minimum_supporting_views':int(support[ids].min()) if len(ids) else None}
    return parts


def write_json(path, data):
    # One compact, atomic sidecar rather than thousands of image/probability artifacts.
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    temp.replace(path)


def ensure_body_parts(mesh_path, *, gpu_python=None, axis='auto', up_sign=1, output=None):
    source=Path(mesh_path).resolve();destination=Path(output).resolve() if output else source.parent/'body_parts.json'
    python=Path(gpu_python or DEFAULT_GPU_PYTHON);started=time.perf_counter();signature=None
    if destination==source or destination.suffix.lower()!='.json':raise ValueError('Position output must be a JSON file distinct from the input mesh')
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        signature=segmentation_signature(source,python,axis,up_sign)
        if destination.is_file():
            try:
                previous=json.loads(destination.read_text(encoding='utf-8'))
                if previous.get('status')=='complete' and previous.get('fingerprint')==signature:
                    print(f'Body-part positions unchanged; reusing: {destination}',flush=True)
                    return {'status':'complete','path':str(destination),'reused':True,'sha256':file_hash(destination)}
            except (OSError,ValueError):pass
        if axis not in ('auto','x','y','z') or up_sign not in (-1,1):raise ValueError('Invalid body-part axis or up direction')
        if not python.is_file():raise FileNotFoundError(f'GPU Python not found: {python}; set --seg-python or HUMAN_SEG_PYTHON')
        mesh=o3d.io.read_triangle_mesh(str(source));vertices=np.asarray(mesh.vertices)
        if not len(vertices) or not mesh.has_triangles() or not mesh.has_vertex_colors() or not np.isfinite(vertices).all():
            raise ValueError('Body-part segmentation requires a finite colored triangle mesh')
        # All scratch files live beside the result and are removed even on failure.
        with tempfile.TemporaryDirectory(prefix='.body_parts_',dir=destination.parent) as temporary:
            folder=Path(temporary);cameras,resolved_axis=make_cameras(mesh,source,signature['mesh_sha256'],folder,axis,up_sign)
            scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            print('[body-part positions: render / Sapiens / map]',flush=True);render(mesh,scene,cameras)
            view_path=folder/'views.json';view_path.write_text(json.dumps(cameras),encoding='utf-8')
            inference=folder/'inference.json'
            subprocess.run([str(python),'-B',str(TOOLS/'infer.py'),'--views',str(view_path),
                            '--report',str(inference),'--labels-only'],check=True,timeout=300)
            model_report=json.loads(inference.read_text(encoding='utf-8'))
            if [name.lower() for name in model_report['classes']]!=PART_NAMES:raise ValueError('Model class map does not match position schema')
            labels,confidence,support,specks=fuse_labels(mesh,scene,cameras)
        if file_hash(source)!=signature['mesh_sha256']:raise RuntimeError('Mesh changed during segmentation; discard positions and retry')
        result={'schema_version':SCHEMA_VERSION,'status':'complete','coordinate_unit':'mm',
                'mesh':{'file':source.name,'sha256':signature['mesh_sha256'],'vertex_count':len(vertices),'triangle_count':len(mesh.triangles)},
                'fingerprint':signature,'method':{'name':'Sapiens-Lite multi-view semantic positions','vertical_axis':resolved_axis,
                'up_sign':up_sign,'views':4,'minimum_supporting_views':2,'minimum_vote_agreement':.65,'hand_speck_filter':specks},
                'class_map':{name:i for i,name in enumerate(PART_NAMES)},'unlabeled_vertex_count':int(np.sum(labels<0)),
                'parts':describe_parts(vertices,labels,confidence,support),'inference_environment':model_report['environment'],
                'seconds':round(time.perf_counter()-started,3),
                'notes':['Semantic regions, not joints or true anatomical ground truth. Clothing can cover limbs.',
                         'Zero-based indices refer only to this exact mesh. Check hash before optimization.',
                         'Unseen/ambiguous vertices remain unlabeled; missing parts have null center/bbox.',
                         'Vote agreement is not calibrated accuracy. Mesh geometry and RGB are unchanged.']}
        write_json(destination,result)
        print(f"Body-part file: {destination}; hand vertices "
              f"{result['parts']['left_hand']['vertex_count']}/{result['parts']['right_hand']['vertex_count']}",flush=True)
        return {'status':'complete','path':str(destination),'sha256':file_hash(destination),'reused':False}
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        result={'schema_version':SCHEMA_VERSION,'status':'failed','fingerprint':signature,'mesh_file':source.name,
                'reason':str(error),'parts':{},'notes':['Reconstruction mesh is preserved. Retry segmentation after resolving this error.']}
        write_json(destination,result);print(f'WARNING: body-part positions unavailable: {error}',flush=True)
        return {'status':'failed','path':str(destination),'reason':str(error),'sha256':file_hash(destination)}


def load_part_indices(position_file, part_name, mesh_path=None):
    """Read a future optimization mask; refuse stale mesh hashes or bad indices."""
    path=Path(position_file);data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('status')!='complete' or data.get('schema_version')!=SCHEMA_VERSION:
        raise ValueError('Body-part positions are incomplete or use an unsupported schema')
    mesh=Path(mesh_path) if mesh_path else path.parent/data['mesh']['file']
    if file_hash(mesh)!=data['mesh']['sha256']:raise ValueError('Mesh hash changed; regenerate body-part positions before optimization')
    part=data['parts'][part_name];ids=np.asarray(part['vertex_indices'],dtype=np.int64)
    if ids.ndim!=1 or len(ids)!=part['vertex_count'] or (len(ids) and (ids.min()<0 or ids.max()>=data['mesh']['vertex_count'])):
        raise ValueError('Invalid body-part vertex indices')
    return ids


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mesh');parser.add_argument('--python')
    parser.add_argument('--axis',choices=['auto','x','y','z'],default='auto');parser.add_argument('--up-sign',type=int,choices=[-1,1],default=1)
    parser.add_argument('--output');args=parser.parse_args()
    status=ensure_body_parts(args.mesh,gpu_python=args.python,axis=args.axis,up_sign=args.up_sign,output=args.output)
    raise SystemExit(0 if status['status']=='complete' else 1)
