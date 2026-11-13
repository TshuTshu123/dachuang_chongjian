"""核验第二路照片标定，并在同一网格比较单路与双路照片融合。

项目根目录运行：python tools/secondary_photo_trial.py --scan 1 2 --calibration-index 路径/采集_joined.xml
--calibration-index 提供完整相机标定；必须与各路已有 JSON 和照片相对变换匹配。
输出 outputs/experiments/secondary_photo/，保存对比图、报告和通过检查的补色 PLY。
"""

import argparse
import copy
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.capture_manifest import camera_sections, field
from tools.raw_capture import samples_for_stream, sha256
from tools.surface_trials import panel


def xml_camera(values):
    rotation = np.eye(4)
    rotation[:3, :3] = np.asarray(values['rotation']).reshape(3, 3)
    return {'width': values['width'], 'height': values['height'],
            'F': [values['fx'], values['fy']], 'C': [values['cx'], values['cy']],
            'distortion': values['dist'], 'translation': values['trans'],
            'rotation': rotation.tolist()}


def calibration_library(path):
    library = {}
    for atom in ET.parse(path).getroot().findall('ATOM'):
        stream = atom.get('STREAMID')
        for data in atom.findall('DATA'):
            description = data.text or ''
            if field(description, 'Type') != 'CameraParams':
                continue
            purpose = field(description, 'Purpose')
            sections = camera_sections(description)
            if 'Color' not in sections:
                continue
            camera = xml_camera(sections['Color'])
            key = stream, purpose
            if key in library and library[key] != camera:
                raise ValueError(f'Calibration changes across frames: {key}')
            library[key] = camera
    return library


def same_camera(a, b):
    return all(np.allclose(a[key], b[key], atol=2e-5, rtol=0)
               for key in ('width', 'height', 'F', 'C', 'distortion', 'translation', 'rotation'))


def second_camera(folder, frame, stream, library):
    directory = folder / stream
    primary = json.loads((directory / f'{frame}_CameraParams_Primary.json').read_text())['color']
    secondary = json.loads((directory / f'{frame}_CameraParams_Secondary.json').read_text())['color']
    if not same_camera(primary, library[stream, 'MV4D']):
        raise ValueError(f'{stream}: primary calibration differs from index.')
    if not same_camera(secondary, library[stream, 'CAMERA_SECONDARY']):
        raise ValueError(f'{stream}: secondary camera identity differs from index.')
    camera = library[stream, 'SECONDARY_COLOR']
    relation = np.loadtxt(directory / f'{frame}_PhotogrammetryInfo.txt').reshape(-1)
    if relation.shape != (12,):
        raise ValueError(f'{stream}: unexpected photograph transform.')
    rotation = np.asarray(primary['rotation'])[:3, :3]
    relative_rotation = relation[:9].reshape(3, 3)
    expected_rotation = relative_rotation @ rotation
    expected_center = relative_rotation @ primary['translation'] + relation[9:]
    if (not np.allclose(expected_rotation, np.asarray(camera['rotation'])[:3, :3], atol=2e-5, rtol=0)
            or not np.allclose(expected_center, camera['translation'], atol=.002, rtol=0)):
        raise ValueError(f'{stream}: photograph transform does not confirm calibration.')
    return camera


def errors_summary(values):
    return {'observations': len(values),
            'median': float(np.median(values)) if len(values) else None,
            'p90': float(np.percentile(values, 90)) if len(values) else None}


def compare(scan, index):
    started = time.perf_counter()
    index_hash = sha256(index)
    library = calibration_library(index)
    parent = ROOT / 'outputs/models' / f'export_{scan:05d}'
    for path in sorted(parent.glob('*/report.json'), reverse=True):
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('raw_color', {}).get('status') == 'used':
            break
    else:
        raise ValueError('No completed model with paired RAW.')
    source = path.parent / report['artifacts']['ply']['name']
    source_hash = sha256(source)
    parts = json.loads((path.parent / 'body_parts.json').read_text(encoding='utf-8'))
    if parts['mesh']['sha256'] != source_hash:
        raise ValueError('Region labels do not match the model.')
    mesh = o3d.io.read_triangle_mesh(str(source))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    xyz = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    baseline = np.asarray(mesh.vertex_colors).copy()
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    raw = report['raw_color']
    folder, frame = Path(raw['source']), raw['frame']
    hashes = dict(report['fingerprint']['raw_color_input']['sha256'])
    entries = [entry for entry in raw['streams'] if entry['projection_check_passed']]
    views = []
    diagnostics = []
    for entry in entries:
        stream = entry['stream']
        camera, reason = None, None
        try:
            camera = second_camera(folder, frame, stream, library)
        except (ValueError, KeyError, OSError) as error:
            reason = str(error)
        for suffix in ('_RGB_02.png', '_CameraParams_Secondary.json', '_PhotogrammetryInfo.txt'):
            file = folder / stream / f'{frame}{suffix}'
            hashes[str(file.relative_to(folder))] = sha256(file)
        samples = []
        cameras = [(1, None)]
        if camera is not None:
            cameras.append((2, camera))
        for number, calibrated in cameras:
            rgb, weight = samples_for_stream(scene, xyz, normals, baseline, folder,
                frame, entry, hashes, bilinear=True, facing_power=6,
                camera=calibrated, photo_number=number)
            samples.append((rgb.astype(np.float32), weight.astype(np.float32)))
        views.append(samples)
        diagnostics.append({'stream': stream, 'calibration_verified': camera is not None, 'reason': reason,
                            'primary_visible_vertices': int((samples[0][1] > 0).sum()),
                            'second_visible_vertices': int((samples[1][1] > 0).sum()) if len(samples) > 1 else 0})
        print(f'{parent.name} stream {stream}: second-camera calibration verified={camera is not None}', flush=True)

    sums, totals, candidates = [], [], []
    for count in (1, 2):
        total = sum(weight for sample in views for _, weight in sample[:count])
        rgb_sum = sum(rgb * weight[:, None] for sample in views for rgb, weight in sample[:count])
        candidate = baseline.copy()
        visible = total > 0
        candidate[visible] += (2 / 3) * (rgb_sum[visible] / total[visible, None] - baseline[visible])
        sums.append(rgb_sum)
        totals.append(total)
        candidates.append(candidate)

    regions = {name: np.asarray(parts['parts'][name]['vertex_indices'], dtype=int)
               for name in ('face_neck', 'left_hand', 'right_hand', 'upper_clothing', 'lower_clothing')}
    metrics, gains = {}, {}
    for name, ids in regions.items():
        triangles = np.asarray(mesh.triangles)
        mask = np.zeros(len(xyz), bool)
        mask[ids] = True
        mixed = triangles[mask[triangles].any(1) & ~mask[triangles].all(1)]
        boundary = np.unique(mixed[mask[mixed]])
        gain = np.minimum(cKDTree(xyz[boundary]).query(xyz[ids])[0] / 6, 1) if len(boundary) else np.ones(len(ids))
        gains[name] = gain
        before, primary_errors, dual_errors = [], [], []
        for sample in views:
            target, weight = sample[0]
            predictions, available = [], []
            for mode, count in enumerate((1, 2)):
                # Exclude both photographs of this stream from the dual-camera prediction.
                excluded_weight = sum(w for _, w in sample[:count])
                excluded_sum = sum(rgb * w[:, None] for rgb, w in sample[:count])
                remaining = totals[mode][ids] - excluded_weight[ids]
                rgb_sum = sums[mode][ids] - excluded_sum[ids]
                available.append(remaining > 1e-5)
                predictions.append(baseline[ids] + gain[:, None] * (2 / 3) *
                    (rgb_sum / np.maximum(remaining[:, None], 1e-9) - baseline[ids]))
            valid = (weight[ids] > 0) & available[0] & available[1]
            truth = target[ids][valid]
            before.extend(np.abs(truth - baseline[ids][valid]).mean(axis=1))
            primary_errors.extend(np.abs(truth - predictions[0][valid]).mean(axis=1))
            dual_errors.extend(np.abs(truth - predictions[1][valid]).mean(axis=1))
        metrics[name] = {'baseline': errors_summary(before), 'primary': errors_summary(primary_errors),
                         'dual': errors_summary(dual_errors),
                         'primary_coverage': int((totals[0][ids] > 0).sum()),
                         'dual_coverage': int((totals[1][ids] > 0).sum()), 'region_vertices': len(ids)}

    selected_colors = baseline.copy()
    selections = {}
    for name, ids in regions.items():
        stats = metrics[name]
        reference = stats['baseline']
        accepted = [mode for mode in ('primary', 'dual')
                    if reference['observations'] >= 256
                    and stats[mode]['median'] <= .95 * reference['median']
                    and stats[mode]['p90'] <= reference['p90']]
        if not accepted:
            continue
        mode = min(accepted, key=lambda mode: stats[mode]['median'])
        index_mode = ('primary', 'dual').index(mode)
        # Fade back to existing colors at semantic boundaries.
        selected_colors[ids] += gains[name][:, None] * (candidates[index_mode][ids] - baseline[ids])
        selections[name] = {'mode': mode, 'median_error_decrease_percent':
                            100 * (1 - stats[mode]['median'] / reference['median'])}

    out = ROOT / 'outputs/experiments/secondary_photo' / parent.name
    out.mkdir(parents=True, exist_ok=True)
    up = np.eye(3)[np.argmax(np.ptp(xyz, axis=0))]
    front = np.array([0., 0., -1.])
    right = np.cross(up, front)
    head = xyz[np.r_[regions['face_neck'], parts['parts']['hair']['vertex_indices']].astype(int)]
    views_to_render = [((xyz.min(0)+xyz.max(0))/2, np.ptp(xyz, axis=0).max()*1.08, 'Full body'),
                       ((head.min(0)+head.max(0))/2, np.ptp(head, axis=0).max()*1.2, 'Face')]
    canvas = Image.new('RGB', (1680, 2*450), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, (colors, label) in enumerate(zip([baseline, *candidates, selected_colors],
                                               ('Current vertex colors', 'Primary photos', 'Both photos', 'Selected regions'))):
        candidate = copy.deepcopy(mesh)
        candidate.vertex_colors = o3d.utility.Vector3dVector(colors)
        for row, (center, span, label_region) in enumerate(views_to_render):
            canvas.paste(panel(candidate, center, right, up, span, size=420, colored=True),
                         (column*420, row*450+25))
            draw.text((column*420+5, row*450+5), f'{label} | {label_region}', fill='black')
    canvas.save(out / 'comparison.jpg', quality=95, subsampling=0)
    if sha256(source) != source_hash or sha256(index) != index_hash:
        raise ValueError('Source changed during experiment.')
    for file, digest in hashes.items():
        if sha256(folder / file) != digest:
            raise ValueError(f'RAW input changed during experiment: {file}')
    result = {'source': str(source), 'source_sha256': source_hash, 'calibration_index': str(index),
              'calibration_sha256': index_hash, 'geometry_changed': False, 'streams': diagnostics,
              'regions': metrics, 'holdout_unit': 'Entire stream, evaluated on identical primary observations',
              'note': 'Photometric consistency on the current geometry, not ground-truth color or geometry accuracy.',
              'selected_regions': selections,
              'elapsed_seconds': round(time.perf_counter()-started, 2)}
    output = out / 'selected_colors.ply'
    if selections:
        candidate = copy.deepcopy(mesh)
        candidate.vertex_colors = o3d.utility.Vector3dVector(selected_colors)
        if not o3d.io.write_triangle_mesh(str(output), candidate):
            raise OSError(f'Failed to export {output}')
        saved = o3d.io.read_triangle_mesh(str(output))
        if (not np.array_equal(np.asarray(saved.vertices), xyz)
                or not np.array_equal(np.asarray(saved.triangles), np.asarray(mesh.triangles))
                or not np.allclose(np.asarray(saved.vertex_colors), selected_colors, atol=.5/255+1e-9, rtol=0)):
            output.unlink(missing_ok=True)
            raise ValueError('PLY roundtrip validation failed.')
        result['candidate'] = str(output)
        result['candidate_sha256'] = sha256(output)
    else:
        output.unlink(missing_ok=True)
    (out / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'scan': scan, 'regions': metrics, 'elapsed_seconds': result['elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--calibration-index', type=Path, required=True)
    args = parser.parse_args()
    for scan in args.scan:
        compare(scan, args.calibration_index.resolve())
