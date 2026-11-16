"""恢复“Meta/Sapiens 只识别部位”阶段，不重跑 Poisson，不做手部变形。
项目根目录：python tools/restore_segmentation_stage.py --all 或 --scan 3。
从同次 intermediate 重放拓扑清理和鞋底，另存 *_segmentation_only，旧模型保留。
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import open3d as o3d

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run import file_hash, fingerprint, repair_small_topology_defects
from sole_repair import apply_shoe_plan
from ply_to_obj import convert_ply_to_obj


def restore(parent):
    started=time.perf_counter()
    paths=sorted(parent.glob('*/report.json'),reverse=True)
    for path in paths:
        report=json.loads(path.read_text(encoding='utf-8'))
        if all((path.parent/a['name']).is_file() for a in report.get('artifacts',{}).values()) and report.get('artifacts'):
            break
    else:raise ValueError(f'No complete result: {parent}')
    if report.get('subject',{}).get('kind') in ('object','unknown'):
        print('Skipped generic object; no human hand rollback:',parent.name,flush=True);return
    previous=report['fingerprint'];signature=fingerprint(Path(report['source']),previous['unit'],previous['fast'])
    if previous['source_sha256']!=signature['source_sha256']:raise ValueError('Raw input changed; rebuild rather than reuse this baseline')
    for key in ('sole_axis','up_sign','refine','no_shoe_repair','shoe_axis','appearance'):signature[key]=previous[key]
    signature['hand_refine']=False
    if report.get('hand_rollback') and previous==signature:
        print('Already restored:',path.parent,flush=True);return
    for key in ('reconstruct_human.py','surface_detail.py','mesh_refine.py','sole_repair.py','mesh_appearance.py'):
        if previous['code_sha256'][key]!=signature['code_sha256'][key]:raise ValueError(f'Baseline code changed: {key}')
    if report['refinement_accepted']:raise ValueError('Extra refinement needs a separate exact replay')
    folder=path.parent;source=folder/report['artifacts']['ply']['name'];digest=file_hash(source)
    current=o3d.io.read_triangle_mesh(str(source));base=folder/report['artifacts']['base']['name']
    mesh=o3d.io.read_triangle_mesh(str(base));mesh.scale(.001,center=np.zeros(3))
    if report['base_topology_repair']['applied']:mesh,_=repair_small_topology_defects(mesh,report['base_report'])
    mesh.scale(1000,center=np.zeros(3))
    if report['shoe_repair']['enabled']:mesh,_=apply_shoe_plan(mesh,report['shoe_repair'])
    v=np.asarray(mesh.vertices);old=np.asarray(current.vertices)
    assert v.shape==old.shape and np.array_equal(np.sort(np.asarray(mesh.triangles),axis=1),np.sort(np.asarray(current.triangles),axis=1))
    unchanged=float(np.mean(np.linalg.norm(v-old,axis=1)<1e-6));assert unchanged>.75
    mesh.triangles=current.triangles;mesh.vertex_colors=current.vertex_colors;mesh.compute_vertex_normals()
    out=parent/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_segmentation_only');out.mkdir()
    target=out/source.name
    if not o3d.io.write_triangle_mesh(str(target),mesh):raise OSError('PLY export failed')
    convert_ply_to_obj(target,target.with_suffix('.obj'),vertex_colors=True)
    info=json.loads((folder/'body_parts.json').read_text(encoding='utf-8'));assert info['mesh']['sha256']==digest
    for part in info['parts'].values():
        ids=part['vertex_indices']
        if ids:part['center_mm']=v[ids].mean(0).tolist();part['bbox_mm']={'min':v[ids].min(0).tolist(),'max':v[ids].max(0).tolist()}
    info['mesh']['file']=target.name;info['mesh']['sha256']=file_hash(target);info['fingerprint']['mesh_sha256']=file_hash(target)
    info['geometry_update']={'method':'restore pre-hand geometry; semantic indices transported','source_mesh_sha256':digest,'labels_transported_by_unchanged_vertex_indices':True}
    position=out/'body_parts.json';position.write_text(json.dumps(info,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    (out/'intermediate').mkdir();base_dest=out/'intermediate'/base.name;shutil.copyfile(base,base_dest)
    for p in base.parent.glob('*_report.json'):shutil.copyfile(p,base_dest.parent/p.name)
    report['hand_rollback']={'stage':'Meta/Sapiens positions only, before all hand deformation','prior_model':str(source),'prior_sha256':digest,'unchanged_vertex_fraction':unchanged,'changed_vertices':int(np.sum(np.linalg.norm(v-old,axis=1)>1e-6))}
    report.pop('palm_policy',None);report.pop('finger_restoration',None)
    report['hand_refinement']={'status':'disabled','applied':False,'reason':'Restored segmentation-only stage; all hand geometry processing disabled by default.'}
    report['fingerprint']=signature
    report['artifacts']={key:{'name':str(p.relative_to(out)),'sha256':file_hash(p)} for key,p in [('ply',target),('obj',target.with_suffix('.obj')),('base',base_dest)]}
    report['segmentation'].update(path=str(position),sha256=file_hash(position),labels_transported_after_hand_refinement=False)
    xyz=(v-mesh.get_center())*.001;f=np.asarray(mesh.triangles)
    report['appearance']['exported_centered_signed_flux_m3']=float(np.einsum('ij,ij->i',xyz[f[:,0]],np.cross(xyz[f[:,1]],xyz[f[:,2]])).sum()/6)
    report['total_seconds']=round(time.perf_counter()-started,3)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    assert file_hash(source)==digest
    disk=o3d.io.read_triangle_mesh(str(target));assert np.array_equal(np.asarray(disk.vertices),v)
    print('RESTORED',parent.name,'changed vertices',report['hand_rollback']['changed_vertices'],out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--all',action='store_true');parser.add_argument('--scan',type=int,default=3);args=parser.parse_args()
    parents=sorted((ROOT/'outputs/models').glob('export_*')) if args.all else [ROOT/'outputs/models'/f'export_{args.scan:05d}']
    for parent in parents:
        if args.all and not any(parent.glob('*/report.json')):
            print('Skipped: no completed reconstruction',parent.name,flush=True);continue
        restore(parent)
