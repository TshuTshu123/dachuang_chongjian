"""检查逐路原始导出、按标定合并，并用其他视角的有效颜色补缺色黑点。
根目录执行：python tools/raw_capture.py "data_raw/00001/raw2"
--frame 00000：选择共同帧；--out 目录：输出位置；--color-radius-mm 8：补色最大距离。
--mesh 网格.ply：在同一网格上生成补色前后对比图（不修改网格）。
只写 outputs，不修改原始数据。输出两份同坐标点云、检查报告和照片总览。
当前读取 mm 制 PLY；相机投影约定会逐路核验，核验失败的路不参与补色。
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
COLOR_RADIUS_MM = 8.0
MIN_NORMAL_COSINE = .7
PAIR_TOLERANCE_MM = .02


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform_points(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def primary_projection(points, camera):
    # Export convention, checked against the existing per-point RGB below.
    rotation = np.asarray(camera['rotation'], dtype=float)[:3, :3]
    local = (points - np.asarray(camera['translation'])) @ rotation
    z = local[:, 2]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        x, y = -local[:, 0] / z, -local[:, 1] / z
        p1, p2, k1, k2, k3 = camera['distortion']
        r2 = x*x + y*y
        radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        xx = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        yy = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
        uv = np.column_stack((camera['F'][0]*xx + camera['C'][0],
                              camera['F'][1]*yy + camera['C'][1]))
    valid = ((z > 0) & np.isfinite(uv).all(axis=1) &
             (uv[:, 0] >= 2) & (uv[:, 0] < camera['width'] - 3) &
             (uv[:, 1] >= 2) & (uv[:, 1] < camera['height'] - 3))
    return uv, valid


def bilinear_rgb(pixels, x, y):
    x = np.clip(x, 0, pixels.shape[1]-1)
    y = np.clip(y, 0, pixels.shape[0]-1)
    x0, y0 = x.astype(int), y.astype(int)
    x1 = np.minimum(x0+1, pixels.shape[1]-1)
    y1 = np.minimum(y0+1, pixels.shape[0]-1)
    dx, dy = (x-x0)[:, None], (y-y0)[:, None]
    top = pixels[y0, x0]*(1-dx) + pixels[y0, x1]*dx
    bottom = pixels[y1, x0]*(1-dx) + pixels[y1, x1]*dx
    return top*(1-dy) + bottom*dy


def samples_for_stream(scene, xyz, normals, baseline, folder, frame, entry, hashes,
                       *, bilinear=False, facing_power=4, camera=None, photo_number=1):
    directory = folder / entry['stream']
    primary = directory / f'{frame}_CameraParams_Primary.json'
    if photo_number not in (1, 2):
        raise ValueError('Expected photo number 1 or 2.')
    if photo_number == 2 and camera is None:
        raise ValueError('The second photograph requires explicitly verified calibration.')
    photo = directory / f'{frame}_RGB_{photo_number:02d}.png'
    ply = directory / f'{frame}.ply'
    transform = directory / f'{frame}_transform.txt'
    for path in (primary, photo, ply, transform):
        if sha256(path) != hashes[str(path.relative_to(folder))]:
            raise ValueError(f'RAW file differs from the verified reconstruction input: {path}')
    if camera is None:
        camera = json.loads(primary.read_text(encoding='utf-8'))['color']
    matrix = np.asarray(entry['transform'])
    local = (xyz - matrix[:3, 3]) @ matrix[:3, :3]
    uv, inside = primary_projection(local, camera)
    position = np.asarray(camera['translation']) @ matrix[:3, :3].T + matrix[:3, 3]
    delta = xyz - position
    depth = np.linalg.norm(delta, axis=1)
    rays = delta / np.maximum(depth[:, None], 1e-9)
    facing = np.abs(np.einsum('ij,ij->i', rays, normals))
    indices = np.flatnonzero(inside & (facing >= .4))
    if not len(indices):
        return np.zeros_like(baseline), np.zeros(len(xyz))
    probes = np.c_[np.broadcast_to(position, (len(indices), 3)), rays[indices]].astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(probes), nthreads=4)['t_hit'].numpy()
    points = np.asarray(o3d.io.read_point_cloud(str(ply)).points).copy()
    support = cKDTree(points).query(local[indices], workers=4)[0]
    indices = indices[(np.abs(hit-depth[indices]) <= 2) & (support <= 5)]
    colors = np.zeros_like(baseline)
    weights = np.zeros(len(xyz))
    if len(indices):
        with Image.open(photo) as image:
            if image.size != (camera['width'], camera['height']):
                raise ValueError('Photograph dimensions do not match its calibration.')
            pixels = np.asarray(image.convert('RGB'))
        xy = uv[indices]
        if bilinear:
            rgb = bilinear_rgb(pixels, *xy.T) / 255
        else:
            x, y = np.rint(xy).astype(int).T
            rgb = pixels[y, x] / 255
        difference = baseline[indices] - rgb
        if len(indices) >= 32:
            rgb = np.clip(rgb + np.clip(np.median(difference, axis=0), -.04, .04), 0, 1)
        colors[indices] = rgb
        weights[indices] = facing[indices]**facing_power / (depth[indices]/1000)**2
    return colors, weights


def make_overview(stream_dirs, frame, destination):
    columns = 6
    rows = 2 * ((len(stream_dirs) + columns - 1) // columns)
    canvas = Image.new('RGB', (columns*300, rows*270), '#222222')
    draw = ImageDraw.Draw(canvas)
    for i, directory in enumerate(stream_dirs):
        for side in (1, 2):
            x, y = (i % columns)*300, ((i // columns)*2 + side-1)*270
            with Image.open(directory / f'{frame}_RGB_{side:02d}.png') as image:
                image.thumbnail((294, 240))
                canvas.paste(image, (x + (300-image.width)//2, y+25))
            draw.text((x+8, y+6), f'stream {directory.name} RGB {side:02d}', fill='white')
    canvas.save(destination, quality=90)


def compare_mesh(mesh_path, out):
    """Compare vertex colors on identical geometry, with fixed cameras and lighting."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError('Comparison requires a triangle mesh')
    original = o3d.io.read_point_cloud(str(out/'aligned_original.ply'))
    repaired = o3d.io.read_point_cloud(str(out/'aligned_color_repaired.ply'))
    points = np.asarray(original.points)
    vertices = np.asarray(mesh.vertices)
    distance, nearest = cKDTree(points).query(vertices, workers=4)
    if np.median(distance) > 5 or np.percentile(distance, 95) > 12:
        raise ValueError('Mesh does not closely match these observations in millimeters')
    before = np.asarray(original.colors)[nearest]
    after = np.asarray(repaired.colors)[nearest]
    faces = np.asarray(mesh.triangles)
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    up = np.eye(3)[np.argmax(np.ptp(vertices, axis=0))]
    # Preview axes only; no model transform or semantic detector is applied.
    axes = [i for i in range(3) if up[i] == 0]
    forward = -np.eye(3)[axes[-1]]
    side = -np.eye(3)[axes[0]]
    center = (vertices.min(axis=0)+vertices.max(axis=0))/2
    span = np.ptp(vertices, axis=0).max()*1.05
    canvas = Image.new('RGB', (1000, 1480), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, outward in enumerate((forward, side)):
        right = np.cross(up, outward)
        width, height = 500, 700
        xx, yy = np.meshgrid(np.linspace(-span*width/height/2, span*width/height/2, width),
                             np.linspace(span/2, -span/2, height))
        origins = center + xx[..., None]*right + yy[..., None]*up + outward*3000
        rays = np.c_[origins.reshape(-1, 3), np.tile(-outward, (width*height, 1))].astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)
        ok = np.isfinite(hit['t_hit'].numpy())
        triangles = faces[hit['primitive_ids'].numpy()[ok]]
        uv = hit['primitive_uvs'].numpy()[ok]
        weights = np.c_[1-uv.sum(axis=1), uv]
        normal = hit['primitive_normals'].numpy()[ok]
        light = outward + .3*up + .2*right
        light /= np.linalg.norm(light)
        shade = .55+.45*np.abs(normal @ light)
        for column, colors in enumerate((before, after)):
            pixels = np.full((width*height, 3), .9)
            pixels[ok] = np.clip(np.einsum('ij,ijk->ik', weights, colors[triangles])*shade[:, None], 0, 1)
            image = Image.fromarray(np.rint(pixels.reshape(height, width, 3)*255).astype(np.uint8))
            y = row*740
            canvas.paste(image, (column*width, y+30))
            draw.text((column*width+10, y+8), ('Original colors' if column == 0 else 'Missing colors repaired') +
                      (' - front' if row == 0 else ' - side'), fill='black')
    canvas.save(out/'color_comparison.jpg', quality=94)
    stats = {'mesh': str(mesh_path.resolve()), 'geometry_identical_in_both_panels': True,
             'vertices': len(vertices), 'triangles': len(faces),
             'nearly_black_vertices_before': int((before.max(axis=1) <= 1/255).sum()),
             'nearly_black_vertices_after': int((after.max(axis=1) <= 1/255).sum()),
             'changed_vertex_colors': int(np.any(before != after, axis=1).sum()),
             'mesh_to_raw_distance_mm_p50_p95': np.percentile(distance, [50, 95]).tolist(),
             'note': 'Color comparison only; darkness of real material and overall exposure are not optimized.'}
    (out/'color_comparison.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2), flush=True)


def capture_frames(folder):
    directories = sorted((p for p in folder.iterdir() if p.is_dir() and p.name.isdigit()),
                         key=lambda p: int(p.name))
    if not directories:
        raise ValueError('No numbered stream directories')
    common = set.intersection(*({p.stem for p in d.glob('*.ply') if p.stem.isdigit()}
                               for d in directories))
    return directories, sorted(common)


def required_paths(directory, frame):
    return [directory / f'{frame}{suffix}' for suffix in
            ('.ply', '_CameraParams_Primary.json', '_transform.txt', '_RGB_01.png')
            ] + [directory / 'CylinderCropParams.cfg']


def raw_color_input(root, stem):
    """Find one capture under a same-name or same-number input directory."""
    match = re.fullmatch(r'export_?(\d+)', stem, flags=re.IGNORECASE)
    parents = [root / stem]
    if match:
        parents.append(root / f'{int(match[1]):05d}')
    captures = []
    for parent in parents:
        if parent.is_dir():
            for path in (parent, *(p for p in parent.iterdir() if p.is_dir())):
                if any(p.parent.name.isdecimal() for p in path.glob('*/*.ply')):
                    captures.append(path)
    if len(captures) != 1:
        reason = 'Multiple RAW captures for this input.' if captures else 'No paired RAW folder.'
        return {'status': 'skipped', 'reason': reason}
    folder = captures[0]
    directories, frames = capture_frames(folder)
    if len(frames) != 1:
        return {'status': 'skipped', 'reason': 'Automatic RAW matching requires one complete frame.',
                'folder': str(folder), 'complete_frames': frames}
    frame = frames[0]
    files = [path for directory in directories for path in required_paths(directory, frame)]
    missing = [str(path.relative_to(folder)) for path in files if not path.is_file()]
    if missing:
        return {'status': 'skipped', 'reason': 'RAW files are missing.', 'folder': str(folder),
                'missing_files': missing}
    return {'status': 'ready', 'folder': str(folder), 'frame': frame,
            'sha256': {str(path.relative_to(folder)): sha256(path) for path in files}}


def load_capture_colors(folder, frame, radius=COLOR_RADIUS_MM, *, inventory=False):
    """Align stream points and fill missing RGB using validated nearby observations."""
    directories, common = capture_frames(folder)
    if not common:
        raise ValueError('No frame has a PLY in every stream')
    frame = frame or sorted(common)[0]
    if frame not in common:
        raise ValueError(f'Incomplete frame {frame}; common frames: {sorted(common)}')
    records, positions, normals, colors, valid_colors, missing_colors = [], [], [], [], [], []
    for directory in directories:
        paths = required_paths(directory, frame)
        ply, primary, transform, image_path, crop = paths
        if inventory:
            paths += [directory/f'{frame}{suffix}' for suffix in
                      ('_CameraParams_Secondary.json', '_PhotogrammetryInfo.txt', '_RGB_02.png')]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise ValueError(f'Missing required files: {missing}')
        if not re.search(r'Units:\s*mm\b', crop.read_text()):
            raise ValueError('Expected explicit millimeter units in CylinderCropParams.cfg')
        matrix = np.loadtxt(transform)
        if matrix.shape != (4, 4):
            raise ValueError(f'Invalid rigid transform in stream {directory.name}')
        rotation = matrix[:3, :3]
        if (not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or
            abs(np.linalg.det(rotation)-1) > 1e-4):
            raise ValueError(f'Invalid rigid transform in stream {directory.name}')
        cloud = o3d.io.read_point_cloud(str(ply))
        if not cloud.has_points() or not cloud.has_colors() or not cloud.has_normals():
            raise ValueError(f'PLY lacks points, colors or normals: {ply}')
        points = np.asarray(cloud.points).copy()
        rgb = np.asarray(cloud.colors).copy()
        normal = np.asarray(cloud.normals).copy()
        if not np.isfinite(points).all() or not np.isfinite(normal).all():
            raise ValueError(f'Non-finite point data: {ply}')
        length = np.linalg.norm(normal, axis=1)
        normal /= np.maximum(length[:, None], 1e-12)
        normal[np.einsum('ij,ij->i', normal, -points) < 0] *= -1
        camera = json.loads(primary.read_text())['color']
        uv, inside = primary_projection(points, camera)
        with Image.open(image_path) as image:
            if image.size != (camera['width'], camera['height']):
                raise ValueError('Image size and calibration disagree')
            pixels = np.asarray(image.convert('RGB'))
        if inventory:
            with Image.open(directory/f'{frame}_RGB_02.png') as image:
                image.verify()
        ids = np.flatnonzero(inside)[::max(1, int(inside.sum())//3000)]
        if not len(ids):
            raise ValueError('No points project inside the primary image')
        xy = np.rint(uv[ids]).astype(int)
        color_error = float(np.median(np.abs(pixels[xy[:, 1], xy[:, 0]]/255-rgb[ids]).mean(axis=1)))
        projection_ok = bool(inside.mean() > .6 and color_error < .06)
        nearly_black = rgb.max(axis=1) <= 1/255
        global_points = transform_points(points, matrix)
        global_normals = normal @ rotation.T
        positions.append(global_points)
        normals.append(global_normals)
        colors.append(rgb)
        # An in-frame black material remains a legitimate color; do not threshold donors by brightness.
        valid_colors.append(inside & projection_ok)
        missing_colors.append(~inside & nearly_black & projection_ok)
        records.append({'stream': directory.name, 'points': len(points),
                        'primary_projection_valid_fraction': float(inside.mean()),
                        'projection_color_median_error_0_1': color_error,
                        'projection_check_passed': projection_ok,
                        'nearly_black_points': int(nearly_black.sum()),
                        'out_of_image_black_points': int((~inside & nearly_black).sum()),
                        'transform': matrix.tolist()})
        if inventory:
            records[-1]['files'] = [{'path': str(p.relative_to(folder)), 'bytes': p.stat().st_size,
                                     'sha256': sha256(p)} for p in paths]
            print(f"stream {directory.name}: {len(points)} points, projection {inside.mean():.1%}, "
                  f"color error {color_error:.4f}, passed={projection_ok}", flush=True)

    points = np.concatenate(positions)
    normal = np.concatenate(normals)
    rgb = np.concatenate(colors)
    reliable = np.concatenate(valid_colors)
    # Repair only missing near-zero colors outside the calibrated image, never in-frame black material.
    targets = np.flatnonzero(np.concatenate(missing_colors))
    donors = np.flatnonzero(reliable)
    repaired = rgb.copy()
    changed = np.zeros(len(points), dtype=bool)
    if len(donors) and len(targets):
        tree = cKDTree(points[donors])
        distance, index = tree.query(points[targets], k=min(16, len(donors)),
                                     distance_upper_bound=radius, workers=4)
        if distance.ndim == 1:
            distance, index = distance[:, None], index[:, None]
        found = np.isfinite(distance)
        safe_index = donors[np.minimum(index, len(donors)-1)]
        agreement = np.einsum('ijk,ik->ij', normal[safe_index], normal[targets])
        eligible = found & (agreement > MIN_NORMAL_COSINE)
        cost = np.where(eligible, distance, np.inf)
        best = cost.argmin(axis=1)
        good = np.isfinite(cost[np.arange(len(targets)), best])
        selected = targets[good]
        repaired[selected] = rgb[safe_index[np.arange(len(targets))[good], best[good]]]
        changed[selected] = True
    starts = np.r_[0, np.cumsum([len(v) for v in positions])]
    for i, record in enumerate(records):
        record['repaired_points'] = int(changed[starts[i]:starts[i+1]].sum())
    report = {'source': str(folder), 'frame': frame, 'complete_frames': common,
              'unit': 'mm', 'stream_count': len(directories), 'total_points': len(points),
              'nearly_black_points': int((rgb.max(axis=1) <= 1/255).sum()),
              'repair_candidates': len(targets), 'repaired_points': int(changed.sum()),
              'repair_max_neighbor_distance_mm': radius,
              'repair_min_normal_cosine': MIN_NORMAL_COSINE, 'geometry_changed': False,
              'streams': records}
    return points, normal, rgb, repaired, reliable, report


def apply_raw_colors(cloud, unit, raw_input):
    """Apply repaired RGB only when every RAW point matches the merged PLY."""
    folder = Path(raw_input['folder'])
    points, _, original, repaired, reliable, report = load_capture_colors(folder, raw_input['frame'])
    if not reliable.any():
        raise ValueError('No RAW stream passed the projection check.')
    merged = np.asarray(cloud.points)
    if unit == 'm':
        merged = merged * 1000
    if len(points) != len(merged) or not cloud.has_colors() or not np.isfinite(merged).all():
        raise ValueError('RAW and input PLY differ in point count or lack usable XYZ/RGB.')
    delta = np.linalg.norm(points - merged, axis=1)
    if delta.max() <= PAIR_TOLERANCE_MM and np.array_equal(original, np.asarray(cloud.colors)):
        order = slice(None)
    else:
        delta, order = cKDTree(points).query(merged, workers=4)
        if (delta.max() > PAIR_TOLERANCE_MM or len(np.unique(order)) != len(points)
                or not np.array_equal(original[order], np.asarray(cloud.colors))):
            raise ValueError('RAW and input PLY do not have matching XYZ/RGB.')
    if any(sha256(folder / name) != digest for name, digest in raw_input['sha256'].items()):
        raise ValueError('RAW files changed during processing.')
    cloud.colors = o3d.utility.Vector3dVector(repaired[order])
    report.update(status='used', pairing_max_distance_mm=float(delta.max()),
                  projection_passed_streams=sum(s['projection_check_passed'] for s in report['streams']))
    return report


def inspect_capture(folder, frame, out, radius):
    points, normal, rgb, repaired, reliable, report = load_capture_colors(
        folder, frame, radius, inventory=True)
    directories, _ = capture_frames(folder)
    frame = report['frame']
    out.mkdir(parents=True, exist_ok=True)
    baseline = o3d.geometry.PointCloud()
    baseline.points = o3d.utility.Vector3dVector(points)
    baseline.normals = o3d.utility.Vector3dVector(normal)
    baseline.colors = o3d.utility.Vector3dVector(rgb)
    if not o3d.io.write_point_cloud(str(out/'aligned_original.ply'), baseline):
        raise IOError('Failed to write aligned point cloud')
    baseline.colors = o3d.utility.Vector3dVector(repaired)
    if not o3d.io.write_point_cloud(str(out/'aligned_color_repaired.ply'), baseline):
        raise IOError('Failed to write repaired point cloud')
    # Verify exact geometry/normals and that valid source colors were untouched after serialization.
    saved = o3d.io.read_point_cloud(str(out/'aligned_color_repaired.ply'))
    assert np.array_equal(np.asarray(saved.points), points)
    assert np.array_equal(np.asarray(saved.normals), normal)
    assert np.all(np.abs(np.asarray(saved.colors)[reliable]-rgb[reliable]) < .5/255 + 1e-9)
    make_overview(directories, frame, out/'images_overview.jpg')
    starts = np.r_[0, np.cumsum([s['points'] for s in report['streams']])]
    for i, record in enumerate(report['streams']):
        indices = np.arange(starts[i], starts[i+1])
        sample = indices[::max(1, len(indices)//1000)]
        other = np.ones(len(points), bool)
        other[indices] = False
        nearest = cKDTree(points[other]).query(points[sample], workers=4)[0]
        record['cross_view_nearest_mm_p50_p90'] = np.percentile(nearest, [50, 90]).tolist()
    all_files = [p for p in folder.rglob('*') if p.is_file()]
    report.update({
              'extensions': dict(Counter(p.suffix.lower() for p in all_files)),
              'projection_convention': 'camera_to_point_frame inverse, negative x/y before pinhole; distortion p1 p2 k1 k2 k3, per-stream photometric check',
              'notes': ['Frame filenames do not prove timestamp synchronization or matching to an existing merged PLY.',
                        'Camera convention validation is photometric, not an independent ground-truth calibration.',
                        'Color repair copies nearby valid observations; it does not create new geometric detail.',
                        'Secondary RGB images are retained for future calibrated texturing, not used by this color repair.']})
    (out/'raw_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('stream_count', 'total_points', 'nearly_black_points',
                                             'repair_candidates', 'repaired_points')}, indent=2), flush=True)
    print('Output:', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--frame', help='Exported frame prefix; default: first common frame')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--mesh', type=Path, help='Generate same-geometry color comparison')
    parser.add_argument('--color-radius-mm', type=float, default=COLOR_RADIUS_MM)
    args = parser.parse_args()
    if not np.isfinite(args.color_radius_mm) or args.color_radius_mm <= 0:
        parser.error('--color-radius-mm must be positive and finite')
    folder = args.capture.resolve()
    destination = args.out or ROOT/'outputs/capture_inputs'/f'{folder.parent.name.replace("(", "").replace(")", "")}_frame{args.frame or "00000"}'
    if destination.resolve().is_relative_to(folder):
        parser.error('Output must be outside the original input directory')
    inspect_capture(folder, args.frame, destination.resolve(), args.color_radius_mm)
    if args.mesh:
        compare_mesh(args.mesh, destination.resolve())
