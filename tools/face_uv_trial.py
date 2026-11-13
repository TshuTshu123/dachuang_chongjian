"""用配套 RAW 照片制作脸颈 UV 纹理。示例：python tools/face_uv_trial.py --scan 1 2

读取正式模型与其 Meta 位置文件。输出在 outputs/experiments/uv_detail/，
通过相机交叉检查时一并保存带材质的 OBJ。打开 OBJ 时保留同目录的 MTL 和 PNG。
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.raw_capture import bilinear_rgb, sha256
from tools.face_uv import (
    BODY_SIZE, FACE_SIZE, FACING_POWER, PHOTO_FRACTION, build_face_atlas,
    bake_body_atlas, save_obj, to_rgb8,
)

def compare_views(out, mesh, scene, head, uv, face, textures, reference):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    normal = np.asarray(mesh.vertex_normals)
    colors = [np.asarray(mesh.vertex_colors)]
    labels = ['Current vertex colors']
    if reference.is_file():
        old = o3d.io.read_triangle_mesh(str(reference))
        if (np.array_equal(vertices, np.asarray(old.vertices))
                and np.array_equal(triangles, np.asarray(old.triangles))):
            colors.append(np.asarray(old.vertex_colors))
            labels.append('Previous photo vertex colors')
    labels.append('Photo UV texture')
    up = np.eye(3)[np.argmax(np.ptp(vertices, axis=0))]
    # The two paired captures share the rig coordinate system, with the face toward -Z.
    front = np.array([0., 0., -1.])
    front -= np.dot(front, up)*up
    front /= np.linalg.norm(front)
    center = (head.min(axis=0)+head.max(axis=0))/2
    span = np.ptp(head, axis=0).max()*1.16
    size = 450
    x, y = np.meshgrid(np.linspace(-span/2, span/2, size), np.linspace(span/2, -span/2, size))
    canvas = Image.new('RGB', (size*len(labels), 2*(size+24)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, angle in enumerate(np.deg2rad([0, 32])):
        view = np.cos(angle)*front + np.sin(angle)*np.cross(up, front)
        right = np.cross(up, view)
        origin = center + x[..., None]*right + y[..., None]*up + view*4000
        rays = np.c_[origin.reshape(-1, 3), np.broadcast_to(-view, (size*size, 3))]
        hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)), nthreads=4)
        ok = np.isfinite(hit['t_hit'].numpy())
        triangle = hit['primitive_ids'].numpy()[ok]
        coords = hit['primitive_uvs'].numpy()[ok]
        barycentric = np.c_[1-coords.sum(axis=1), coords]
        corner_ids = triangles[triangle]
        surface_normal = np.einsum('ij,ijk->ik', barycentric, normal[corner_ids])
        surface_normal /= np.maximum(np.linalg.norm(surface_normal, axis=1, keepdims=True), 1e-9)
        shade = .6+.4*np.abs(surface_normal @ view)
        variants = [np.einsum('ij,ijk->ik', barycentric, rgb[corner_ids]) for rgb in colors]
        st = np.einsum('ij,ijk->ik', barycentric, uv[triangle])
        uv_rgb = np.empty_like(variants[0])
        material = face[triangle].astype(int)
        for index, image in enumerate(textures):
            selected = material == index
            u, v = st[selected].T
            uv_rgb[selected] = bilinear_rgb(image, u*image.shape[1]-.5, (1-v)*image.shape[0]-.5)/255
        for col, rgb in enumerate([*variants, uv_rgb]):
            pixels = np.full((size*size, 3), .94)
            pixels[ok] = rgb*shade[:, None]
            top = row*(size+24)
            canvas.paste(Image.fromarray(to_rgb8(pixels.reshape(size, size, 3))), (col*size, top+24))
            draw.text((col*size+8, top+5), labels[col], fill='black')
    canvas.save(out/'comparison.jpg', quality=96, subsampling=0)


def trial(scan):
    start = time.perf_counter()
    name = f'export_{scan:05d}'
    runs = sorted((ROOT/'outputs/models'/name).glob('*/report.json'), reverse=True)
    for path in runs:
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('raw_color', {}).get('status') == 'used' and (path.parent/'body_parts.json').is_file():
            break
    else:
        raise ValueError(f'{name}: no completed paired-RAW model with Meta labels.')
    source = path.parent/report['artifacts']['ply']['name']
    parts = json.loads((path.parent/'body_parts.json').read_text(encoding='utf-8'))
    source_hash = sha256(source)
    if parts['mesh']['sha256'] != source_hash:
        raise ValueError('The Meta labels do not match the source mesh.')
    mesh = o3d.io.read_triangle_mesh(str(source))
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    ids = np.asarray(parts['parts']['face_neck']['vertex_indices'], dtype=int)
    if not len(ids):
        raise ValueError(f'{name}: no Meta face and neck labels.')
    uv, face, face_image, stats = build_face_atlas(
        mesh, ids, report['raw_color'], report['fingerprint']['raw_color_input']['sha256'],
    )
    textures = bake_body_atlas(mesh, uv, face, face_image)
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    out = ROOT/'outputs/experiments/uv_detail'/name
    out.mkdir(parents=True, exist_ok=True)
    head_ids = np.r_[ids, np.asarray(parts['parts'].get('hair', {}).get('vertex_indices', []), dtype=int)]
    reference = ROOT/'outputs/experiments/part_detail'/name/'face_color/face_color_mesh.ply'
    compare_views(out, mesh, scene, v[head_ids], uv, face, textures, reference)
    result = {
        'source': str(source), 'source_sha256': source_hash, 'vertices': len(v), 'triangles': len(f),
        'region': 'face_neck', 'face_triangles': int(face.sum()), 'face_texture_size': FACE_SIZE,
        'body_texture_size': BODY_SIZE, 'photo_fraction_at_full_gain': PHOTO_FRACTION,
        'view_facing_exponent': FACING_POWER, **stats, 'obj': None,
    }
    if stats['passes_holdout']:
        result['obj_max_axis_error_mm'] = save_obj(out, mesh, uv, face, textures)
        result['obj'] = str(out/'face_uv.obj')
    result['elapsed_seconds'] = round(time.perf_counter()-start, 2)
    (out/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=int, required=True, nargs='+')
    for scan in parser.parse_args().scan:
        trial(scan)
