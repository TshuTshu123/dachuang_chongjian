"""Render actual v26 outputs and the previous formal models for visual review."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.export_presentation_images import mesh_scene, mesh_image


def main(scans=None):
    scans = list(range(5)) if scans is None else scans
    folder = ROOT/'outputs/face_v26_validation'
    previous = json.loads((folder/'previous_models.json').read_text('utf-8'))
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 24)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)
    overview = Image.new('RGB', (360*len(scans), 850), 'white')
    draw = ImageDraw.Draw(overview)
    draw.text((15, 10), '脸部修复正式版：上排上一版，下排本次输出（相同视角与光照）', font=font, fill='black')
    records = []
    for slot, scan in enumerate(scans):
        summary = json.loads((folder/f'export_{scan:05d}.json').read_text('utf-8'))
        path = Path(summary['report'])
        report = json.loads(path.read_text('utf-8'))
        current = path.parent/report['artifacts']['ply']['name']
        old = ROOT/previous[str(scan)]['source']
        meshes = [o3d.io.read_triangle_mesh(str(p)) for p in [old, current]]
        for mesh in meshes:
            mesh.compute_vertex_normals()
        face = report['flame_face_refinement']
        fit = path.parent/face.get('fit', 'intermediate/flame/fit.json')
        if fit.is_file():
            nose = np.array(json.loads(fit.read_text('utf-8'))['fit']['nose_initialization_mm'])
        else:
            from face_refine import detect_nose
            parts = json.loads((path.parent/'body_parts.json').read_text('utf-8'))['parts']
            nose = detect_nose(meshes[1], parts)
        center = nose+[10., 0., 40.]
        scenes = [mesh_scene(m) for m in meshes]
        panels = Image.new('RGB', (1440, 1000), 'white')
        d = ImageDraw.Draw(panels)
        captions = ['上一版灰模', '本次正式版灰模', '本次正式版彩色']
        for col, caption in enumerate(captions):
            d.text((480*col+15, 12), f'{scan:05d} {caption}', font=font, fill='black')
        for row, angle in enumerate([0., 28.]):
            a = np.deg2rad(angle)
            right = np.array([0., np.cos(a), np.sin(a)])
            for col in range(3):
                i = min(col, 1)
                picture = mesh_image(meshes[i], scenes[i], center, right, 240., (480, 465), colored=col == 2)
                panels.paste(picture, (col*480, 55+row*465))
        status = 'FLAME 已采用' if face['applied'] else '检查未通过，保留原脸'
        d.text((16, 984), status, font=small, fill='black', anchor='ls')
        review = path.parent/'face_review.jpg'
        panels.save(review, quality=95)
        for row in range(2):
            picture = mesh_image(meshes[row], scenes[row], center, np.array([0., 1., 0.]), 255., (360, 365), colored=False)
            overview.paste(picture, (slot*360, 85+row*380))
        draw.text((slot*360+15, 53), f'{scan:05d}', font=font, fill='black')
        draw.text((slot*360+12, 828), status, font=small, fill='black', anchor='ls')
        records.append({'scan': scan, 'source': str(current), 'previous': str(old), 'review': str(review),
                        'face_applied': face['applied'], 'face_method': face.get('selected', {}).get('radius_scale')})
        print('rendered', scan, review, flush=True)
    suffix = '' if len(scans) == 5 else '_部分'
    target = folder/f'脸部修复_正式版总览{suffix}.jpg'
    overview.save(target, quality=95)
    (folder/f'review_manifest{suffix}.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), 'utf-8')
    print(target, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', nargs='+', type=int)
    main(parser.parse_args().scan)
