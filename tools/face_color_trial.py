"""用已配对 RAW 的照片试验脸颈投影着色。python tools/face_color_trial.py --scan 1

复用正式模型及 Meta 标签，在 outputs/experiments/part_detail/ 保存对比、检查报告和通过检查的 PLY。
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from surface_trials import ROOT, panel
from raw_capture import samples_for_stream, sha256


def compare_faces(scan):
    parent = ROOT / 'outputs/models' / f'export_{scan:05d}'
    report_path = next(p for p in sorted(parent.glob('*/report.json'), reverse=True)
                       if json.loads(p.read_text(encoding='utf-8')).get('raw_color', {}).get('status') == 'used')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    source = report_path.parent / report['artifacts']['ply']['name']
    parts = json.loads((report_path.parent/'body_parts.json').read_text(encoding='utf-8'))
    if parts['mesh']['sha256'] != sha256(source):
        raise ValueError('The Meta labels do not match the mesh.')
    mesh = o3d.io.read_triangle_mesh(str(source))
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)
    ids = np.asarray(parts['parts']['face_neck']['vertex_indices'], dtype=int)
    if not len(ids):
        raise ValueError('No face/neck labels to color.')
    mask = np.zeros(len(v), bool)
    mask[ids] = True
    mixed = f[mask[f].any(1) & ~mask[f].all(1)]
    boundary = np.unique(mixed[mask[mixed]])
    taper = np.clip(cKDTree(v[boundary]).query(v[ids])[0]/6, 0, 1) if len(boundary) else np.ones(len(ids))
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    baseline = np.asarray(mesh.vertex_colors)[ids]
    raw = report['raw_color']
    hashes = report['fingerprint']['raw_color_input']['sha256']
    entries = [entry for entry in raw['streams'] if entry['projection_check_passed']]
    views = [samples_for_stream(scene, v[ids], normals[ids], baseline,
                                Path(raw['source']), raw['frame'], entry, hashes)
             for entry in entries]
    weights = np.asarray([w for _, w in views])
    photos = np.asarray([c for c, _ in views])
    weighted = photos * weights[..., None]
    total = weights.sum(axis=0)
    rgb_sum = weighted.sum(axis=0)
    projected = rgb_sum / np.maximum(total[:, None], 1e-9)
    # Compare to an omitted camera using colors estimated from the other cameras.
    other = total - weights
    valid = (weights > 0) & (other > 1e-9) & (taper > 0)
    held_out = (rgb_sum-weighted) / np.maximum(other[..., None], 1e-9)
    if not valid.any():
        raise ValueError('Insufficient overlap for an independent camera comparison.')
    before = np.abs(photos-baseline).mean(axis=2)[valid]
    p50, p90 = np.percentile(before, [50, 90])
    rows = []
    candidates = [mesh]
    for ratio in (1.5, 2.0):
        strength = ratio/(1+ratio)*taper
        prediction = baseline + strength[:, None]*(held_out-baseline)
        after = np.abs(photos-prediction).mean(axis=2)[valid]
        a50, a90 = np.percentile(after, [50, 90])
        accepted = len(before)>=256 and a50<=.95*p50 and a90<=p90+1/255
        candidate = copy.deepcopy(mesh)
        colors = np.asarray(candidate.vertex_colors)
        visible = total>0
        colors[ids[visible]] = baseline[visible] + strength[visible,None]*(projected[visible]-baseline[visible])
        candidates.append(candidate)
        rows.append({'photo_to_original_weight':ratio, 'median_rgb_error':float(a50),
                     'p90_rgb_error':float(a90), 'passes_holdout':bool(accepted)})
    out = ROOT / 'outputs/experiments/part_detail' / parent.name / 'face_color'
    out.mkdir(parents=True, exist_ok=True)
    up = np.eye(3)[np.argmax(np.ptp(v, axis=0))]
    front = normals[ids].mean(axis=0)
    front -= np.dot(front, up)*up
    front /= max(np.linalg.norm(front), 1e-9)
    if np.linalg.norm(front) < .5:
        front = -np.roll(up, -1)
    # Orient the preview toward the camera with the strongest visible contribution.
    best_view = entries[int(np.argmax(weights.sum(axis=1)))]
    direction = np.dot(np.asarray(best_view['transform'])[:3,3]-v[ids].mean(axis=0), front)
    if direction < 0:
        front = -front
    right = np.cross(up, front)
    head = v[ids]
    center = (head.max(axis=0)+head.min(axis=0))/2
    span = np.ptp(head,axis=0).max()*1.25
    canvas = Image.new('RGB',(1200,430),'white')
    draw = ImageDraw.Draw(canvas)
    for i,(candidate,label) in enumerate(zip(candidates,('Current colors','Photo weight 1.5','Photo weight 2.0'))):
        canvas.paste(panel(candidate,center,right,up,span,400,colored=True),(i*400,25))
        draw.text((i*400+8,5),label,fill='black')
    canvas.save(out/'comparison.jpg',quality=95)
    result = {'source':str(source),'label':'face_neck','visible_vertices':int((total>0).sum()),
              'region_vertices':len(ids),'holdout_observations':len(before),
              'baseline_median_rgb_error':float(p50),'baseline_p90_rgb_error':float(p90),
              'candidates':rows,'selected':None}
    output = out/'face_color_mesh.ply'
    accepted = [i for i,row in enumerate(rows) if row['passes_holdout']]
    if accepted:
        best = min(accepted,key=lambda i: rows[i]['median_rgb_error'])
        candidate = candidates[best+1]
        if not o3d.io.write_triangle_mesh(str(output),candidate):
            raise OSError(f'Cannot write {output}')
        saved = o3d.io.read_triangle_mesh(str(output))
        assert np.array_equal(np.asarray(saved.vertices),v) and np.array_equal(np.asarray(saved.triangles),f)
        assert np.array_equal(np.asarray(saved.vertex_colors)[~mask],np.asarray(mesh.vertex_colors)[~mask])
        result['selected'] = {**rows[best],'mesh':str(output)}
    else:
        output.unlink(missing_ok=True)
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan',type=int,required=True)
    compare_faces(parser.parse_args().scan)
