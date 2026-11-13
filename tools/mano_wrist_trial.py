"""对最终 MANO 接缝做独立平滑试验；不覆盖正式模型，不改原模板参数。

项目根目录：python tools/mano_wrist_trial.py --scan 0 3
输出 experiments/mano_wrist；只放局部预览、图和指标。
腕部两端渐变固定，指尖和远处身体不移动；相交或翻折时拒绝。
"""
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hand_pipeline import pairs, smoothstep
from tools.surface_trials import panel


def trial(scan):
    found = []
    for p in (ROOT/'outputs/models'/f'export_{scan:05d}').glob('*/report.json'):
        r = json.loads(p.read_text(encoding='utf-8'))
        if r.get('mano_hand_processing', {}).get('applied'):found.append((p, r))
    path, report = max(found, key=lambda x:x[0].parent.name)
    mesh = o3d.io.read_triangle_mesh(str(path.parent/report['artifacts']['ply']['name']))
    xyz = np.asarray(mesh.vertices); faces = np.asarray(mesh.triangles)
    info = json.loads((path.parent/'body_parts.json').read_text(encoding='utf-8'))
    detail = json.loads((path.parent/'intermediate/mano/report.json').read_text(encoding='utf-8'))
    labels = np.full(len(xyz), 'unknown', object)
    for name, part in info['parts'].items():labels[part['vertex_indices']] = name
    edges = np.unique(np.sort(np.r_[faces[:, :2], faces[:, 1:], faces[:, [2, 0]]], axis=1), axis=0)
    adjacency = coo_matrix((np.ones(len(edges)*2), (np.r_[edges[:,0], edges[:,1]], np.r_[edges[:,1], edges[:,0]])), shape=(len(xyz),len(xyz))).tocsr()
    degree = np.maximum(np.asarray(adjacency.sum(1)).ravel(), 1)
    applied = {p['part'] for p in report['mano_hand_processing']['join_attempt']['parts']}
    frames = [p for p in detail['parts'] if p['part'] in applied]
    out = ROOT/'outputs/experiments/mano_wrist'/f'export_{scan:05d}';out.mkdir(parents=True, exist_ok=True)
    canvas = Image.new('RGB', (1400, 760*len(frames)), 'white');draw = ImageDraw.Draw(canvas);records = []
    for k, frame in enumerate(frames):
        name = frame['part'];wrist = np.array(frame['wrist_mm']);axes = np.array(frame['frame_axes']);local = (xyz-wrist) @ axes
        allowed = np.isin(labels, [name, name.replace('hand','lower_arm'), 'unknown'])
        radial = np.linalg.norm(xyz-wrist, axis=1)
        weight = smoothstep((local[:,1]+28)/18)*(1-smoothstep((local[:,1]-12)/25))
        weight *= 1-smoothstep((radial-35)/22);weight[~allowed] = 0
        active = weight>0; old = xyz.copy(); current = old.copy()
        for _ in range(30):
            for amount in (.45, -.47):
                lap = adjacency @ current/degree[:,None]-current
                proposed = current+amount*weight[:,None]*lap
                delta = proposed-old;length = np.linalg.norm(delta, axis=1)
                delta *= np.minimum(1., 6/np.maximum(length,1.e-9))[:,None]
                current = old+delta
        candidate = copy.deepcopy(mesh);candidate.vertices = o3d.utility.Vector3dVector(current);candidate.compute_vertex_normals()
        p = xyz[faces];original_cross = np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        p = current[faces];new_cross = np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        touched = active[faces].any(1)
        bad = int(np.sum(touched & ((np.einsum('ij,ij->i',original_cross,new_cross)<=0) | (np.linalg.norm(new_cross,axis=1)<np.linalg.norm(original_cross,axis=1)*.02))))
        box = np.r_[xyz[active], current[active]];p = xyz[faces]
        region = np.flatnonzero(np.all(p.max(1)>=box.min(0)-10,axis=1)&np.all(p.min(1)<=box.max(0)+10,axis=1))
        intersections = len(pairs(candidate,region,active)-pairs(mesh,region,active)) if not bad else None
        accepted = bad==0 and intersections==0
        before_lap = adjacency @ old/degree[:,None]-old;after_lap = adjacency @ current/degree[:,None]-current
        core = weight>.75
        record = {'part':name,'applied':accepted,'folded_faces':bad,'new_intersections':intersections,'max_move_mm':float(np.linalg.norm(current-old,axis=1).max()),'laplacian_p85_before_mm':float(np.quantile(np.linalg.norm(before_lap[core],axis=1),.85)),'laplacian_p85_after_mm':float(np.quantile(np.linalg.norm(after_lap[core],axis=1),.85))}
        records.append(record)
        assert np.array_equal(current[~active],xyz[~active])
        chosen = np.flatnonzero(radial<100)
        before = mesh.select_by_index(chosen);after = (candidate if accepted else mesh).select_by_index(chosen)
        if accepted:o3d.io.write_triangle_mesh(str(out/(name+'_wrist_preview.ply')),after)
        for angle in range(2):
            right = axes[:,0] if angle==0 else axes[:,2]
            for col in range(4):
                m = copy.deepcopy(before if col%2==0 else after)
                if col>=2:m.paint_uniform_color([.72]*3)
                row=k*2+angle
                canvas.paste(panel(m,wrist+axes[:,1]*10,right,axes[:,1],160,350,colored=col<2),(350*col,380*row+25))
                draw.text((350*col+4,380*row+5),name+(' | Before' if col%2==0 else ' | After accepted'),fill='black')
        print(scan,record,flush=True)
    canvas.save(out/'comparison.jpg',quality=95)
    (out/'report.json').write_text(json.dumps({'source_mesh':str(path.parent/report['artifacts']['ply']['name']),'parts':records,'formal_output_changed':False},ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scan',type=int,nargs='+',default=[0,3])
    for scan in parser.parse_args().scan:trial(scan)
