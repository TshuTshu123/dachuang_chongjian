"""手部密度场、三维拓扑骨架与骨架引导表面试验。

运行：python tools/hand_density_trial.py --scan 0 1 2 3
--worker-python 指向已安装 scikit-image 的 Python，默认使用本机 conda 基础环境。
--voxel-mm 默认 2，控制三维密度网格精度。输出在 outputs/experiments/hand_density/。
默认只保留骨架点、对比图和报告；--keep-meshes 额外保存试验表面。
骨架为密度体的拓扑骨架，不代表已识别手指关节。候选为独立手部，未拼接到全身。
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def skeleton_graph(skeleton):
    nodes = np.argwhere(skeleton)
    lookup = np.full(skeleton.shape, -1, dtype=np.int32)
    lookup[tuple(nodes.T)] = np.arange(len(nodes))
    neighbors = [[] for _ in nodes]
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                offset = np.array([x, y, z])
                if not offset.any():
                    continue
                other = nodes + offset
                valid = np.all((other >= 0) & (other < np.asarray(skeleton.shape)), axis=1)
                ids = np.flatnonzero(valid)
                targets = lookup[tuple(other[valid].T)]
                for a, b in zip(ids[targets >= 0], targets[targets >= 0]):
                    neighbors[a].append(int(b))
    return nodes, neighbors


def prune_spurs(skeleton, voxel, origin):
    skeleton = skeleton.copy()
    removed = 0
    for _ in range(3):
        nodes, neighbors = skeleton_graph(skeleton)
        drop = set()
        for start, adjacent in enumerate(neighbors):
            if len(adjacent) != 1 or origin[1]+nodes[start, 1]*voxel < 10:
                continue
            branch = [start]
            current, previous = adjacent[0], start
            length = float(np.linalg.norm(nodes[current]-nodes[start])*voxel)
            while len(neighbors[current]) == 2 and length <= 8:
                branch.append(current)
                following = next(i for i in neighbors[current] if i != previous)
                length += float(np.linalg.norm(nodes[following]-nodes[current])*voxel)
                previous, current = current, following
            if len(neighbors[current]) >= 3 and length <= 8:
                drop.update(branch)
        if not drop:
            break
        skeleton[tuple(nodes[sorted(drop)].T)] = False
        removed += len(drop)
    return skeleton, removed


def density_volume(points, voxel):
    origin = np.floor(points.min(0)/voxel)*voxel - 4*voxel
    indices = np.rint((points-origin)/voxel).astype(int)
    shape = indices.max(0)+5
    if np.prod(shape) > 4_000_000:
        raise ValueError('Hand density grid exceeds the memory limit.')
    histogram = np.zeros(shape, np.float32)
    np.add.at(histogram, tuple(indices.T), 1)
    density = ndi.gaussian_filter(histogram, sigma=.75)
    threshold = float(np.median(density[tuple(indices.T)])*.22)
    return density, origin, threshold


def volume_mask(density, threshold):
    from skimage.morphology import ball
    mask = ndi.binary_closing(density >= threshold, structure=ball(1))
    mask = ndi.binary_fill_holes(mask)
    components, count = ndi.label(mask)
    if not count:
        raise ValueError('Empty density component.')
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == sizes.argmax()


def fill_thickness(mask, voxel):
    """Close short gaps along the estimated palm thickness, preserving XY gaps."""
    present = mask.any(axis=2)
    lower = mask.argmax(axis=2)
    upper = mask.shape[2]-1-mask[..., ::-1].argmax(axis=2)
    z = np.arange(mask.shape[2])[None, None, :]
    eligible = present & ((upper-lower)*voxel <= 40)
    filled = mask | (eligible[..., None] & (z >= lower[..., None]) & (z <= upper[..., None]))
    return filled


def sphere_volume(skeleton, radius):
    output = np.zeros_like(skeleton)
    for center in np.argwhere(skeleton):
        value = max(float(radius[tuple(center)])*.97, 1.05)
        lower = np.maximum(center-int(np.ceil(value)), 0)
        upper = np.minimum(center+int(np.ceil(value))+1, skeleton.shape)
        grid = np.ogrid[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))]
        inside = sum((axis-coordinate)**2 for axis, coordinate in zip(grid, center)) <= value**2
        output[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))] |= inside
    return output


def density_worker(source, output, voxel):
    from skimage.morphology import skeletonize_3d, skeletonize
    from skimage.measure import marching_cubes
    points = np.load(source)['points']
    density, origin, threshold = density_volume(points, voxel)
    shell = volume_mask(density, threshold)
    mask = fill_thickness(shell, voxel)
    shell_nodes, shell_neighbors = skeleton_graph(skeletonize_3d(shell)>0)
    shell_endpoints = sum(len(neighbors)==1 for neighbors in shell_neighbors)
    raw_skeleton = skeletonize_3d(mask) > 0
    skeleton, removed = prune_spurs(raw_skeleton, voxel, origin)
    radii = ndi.distance_transform_edt(mask)
    guided = sphere_volume(skeleton, radii)
    # A thin topology skeleton cannot represent the broad palm by small spheres.
    # Preserve the density volume in wide XY regions and constrain added material.
    palm_core = ndi.distance_transform_edt(mask.any(axis=2))*voxel >= 10
    guided |= mask & palm_core[..., None]
    guided &= ndi.binary_dilation(mask, iterations=1)
    arrays = {'density': density, 'origin': origin, 'mask': mask,
              'skeleton': np.argwhere(skeleton)*voxel+origin}
    projected = ndi.binary_fill_holes(mask.any(axis=2))
    planar = skeletonize(projected)[..., None]
    planar, _ = prune_spurs(planar, voxel, origin)
    planar_nodes, planar_neighbors = skeleton_graph(planar)
    lower = mask.argmax(axis=2)
    upper = mask.shape[2]-1-mask[..., ::-1].argmax(axis=2)
    lifted = planar_nodes.astype(float)
    lifted[:, 2] = (lower[tuple(planar_nodes[:, :2].T)]+upper[tuple(planar_nodes[:, :2].T)])/2
    arrays['planar_skeleton'] = lifted*voxel+origin
    arrays['planar_degree'] = np.array([len(row) for row in planar_neighbors])
    for prefix, volume in [('density', mask), ('guided', guided)]:
        field = ndi.gaussian_filter(volume.astype(np.float32), .5)
        vertices, triangles, _, _ = marching_cubes(field, .5, spacing=(voxel,)*3)
        arrays[prefix+'_vertices'] = vertices+origin
        arrays[prefix+'_triangles'] = triangles
    stability = []
    for scale in (.75, 1., 1.25):
        alternative = fill_thickness(volume_mask(density, threshold*scale), voxel)
        bones = skeletonize_3d(alternative) > 0
        nodes, neighbors = skeleton_graph(bones)
        endpoints = [i for i, edges in enumerate(neighbors) if len(edges) == 1]
        tips = [i for i in endpoints if nodes[i, 1]*voxel+origin[1] > 10]
        projected_bones = skeletonize(ndi.binary_fill_holes(alternative.any(axis=2)))[..., None]
        projected_bones, _ = prune_spurs(projected_bones, voxel, origin)
        planar_points, planar_graph = skeleton_graph(projected_bones)
        planar_tips = sum(len(edges)==1 and planar_points[i, 1]*voxel+origin[1]>10
                          for i, edges in enumerate(planar_graph))
        stability.append({'density_threshold_scale': scale, 'skeleton_nodes': len(nodes),
                          'non_wrist_endpoints': len(tips), 'projected_non_wrist_endpoints': int(planar_tips)})
    nodes, neighbors = skeleton_graph(skeleton)
    arrays['degree'] = np.array([len(row) for row in neighbors])
    arrays['stats'] = np.array(json.dumps({'voxel_mm': voxel, 'density_threshold': threshold,
        'raw_skeleton_nodes': int(raw_skeleton.sum()), 'pruned_nodes': removed,
        'skeleton_nodes': int(skeleton.sum()), 'density_threshold_stability': stability,
        'surface_shell_endpoints': shell_endpoints,
        'thickness_filled_voxels': int(mask.sum()-shell.sum()),
        'thickness_fill_assumption': 'Single local palm layer with depth span at most 40 mm; folded/contacting fingers may violate this assumption.',
        'palm_preservation_rule': 'Preserve density columns at least 10 mm from the projected boundary.',
        'density_occupied_voxels': int(mask.sum()), 'guided_occupied_voxels': int(guided.sum())}))
    np.savez_compressed(output, **arrays)


def trial(scan, worker_python, voxel, keep_meshes=False):
    import open3d as o3d
    from PIL import Image, ImageDraw
    from tools.hand_pose_trial import hand_frame, point_panel
    from tools.surface_trials import panel
    from surface_evaluation import distances
    from body_parts import file_hash
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    parent = root/'outputs/models'/f'export_{scan:05d}'
    for path in sorted(parent.glob('*/report.json'), reverse=True):
        report = json.loads(path.read_text(encoding='utf-8'))
        ply = path.parent/report['artifacts']['ply']['name']
        if ply.is_file() and (path.parent/'body_parts.json').is_file():
            break
    else:
        raise ValueError('No complete model and region labels.')
    parts = json.loads((path.parent/'body_parts.json').read_text(encoding='utf-8'))
    source_hash = file_hash(ply)
    cloud_hash = file_hash(Path(report['source']))
    if parts['mesh']['sha256'] != source_hash or cloud_hash != report['fingerprint']['source_sha256']:
        raise ValueError('Source or mesh differs from saved labels.')
    mesh = o3d.io.read_triangle_mesh(str(ply))
    vertices = np.asarray(mesh.vertices)
    cloud = o3d.io.read_point_cloud(report['source']).remove_non_finite_points()
    if report['fingerprint']['unit'] == 'm':
        cloud.scale(1000, center=np.zeros(3))
    raw = np.asarray(cloud.points)
    labels = np.full(len(vertices), 'unknown', dtype=object)
    for name, data in parts['parts'].items():
        labels[data['vertex_indices']] = name
    nearest = cKDTree(vertices).query(raw, workers=4)[1]
    out = root/'outputs/experiments/hand_density'/parent.name
    out.mkdir(parents=True, exist_ok=True)
    canvas = Image.new('RGB', (1750, 1520), 'white')
    draw = ImageDraw.Draw(canvas)
    records = []
    for side, name in enumerate(('left_hand', 'right_hand')):
        ids, arm_class, wrist, axes = hand_frame(vertices, parts['parts'], name)
        local_vertices = (vertices-wrist) @ axes
        low = local_vertices[ids].min(0)-[12, 30, 15]
        high = local_vertices[ids].max(0)+[12, 12, 15]
        low[1] = -30
        local_raw = (raw-wrist) @ axes
        selected = (np.all((local_raw >= low) & (local_raw <= high), axis=1)
                    & (cKDTree(vertices[ids]).query(raw, workers=4)[0] < 22)
                    & np.isin(labels[nearest], [name, arm_class, 'unknown']))
        points = local_raw[selected]
        if len(points) < 100:
            records.append({'part': name, 'status': 'insufficient_points'})
            continue
        directory = out/f'.density_work_{uuid4().hex}'
        directory.mkdir()
        input_path, result_path = directory/'input.npz', directory/'result.npz'
        try:
            np.savez(input_path, points=points)
            subprocess.run([str(worker_python), str(Path(__file__).resolve()), '--worker',
                            str(input_path), str(result_path), '--voxel-mm', str(voxel)], check=True)
            with np.load(result_path) as archive:
                data = {key: archive[key].copy() for key in archive.files}
        finally:
            input_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)
            directory.rmdir()
        stats = json.loads(str(data['stats']))
        models = []
        for prefix in ('density', 'guided'):
            local_mesh = o3d.geometry.TriangleMesh()
            world = data[prefix+'_vertices'] @ axes.T + wrist
            local_mesh.vertices = o3d.utility.Vector3dVector(world)
            local_mesh.triangles = o3d.utility.Vector3iVector(data[prefix+'_triangles'])
            color_ids = cKDTree(raw).query(world)[1]
            local_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[color_ids])
            local_mesh.compute_vertex_normals()
            models.append(local_mesh)
        candidate_file = out/f'{name}_density_skeleton.ply'
        if keep_meshes:
            if not o3d.io.write_triangle_mesh(str(candidate_file), models[1]):
                raise OSError(f'Failed to write {candidate_file}')
        else:
            candidate_file.unlink(missing_ok=True)
        skeleton = o3d.geometry.PointCloud()
        bones = data['skeleton'] @ axes.T+wrist
        skeleton.points = o3d.utility.Vector3dVector(bones)
        skeleton.paint_uniform_color([1, .12, .05])
        skeleton_file = out/f'{name}_skeleton.ply'
        o3d.io.write_point_cloud(str(skeleton_file), skeleton)
        lifted_points = data['planar_skeleton'] @ axes.T+wrist
        skeleton.points = o3d.utility.Vector3dVector(lifted_points)
        skeleton.paint_uniform_color([.05, .25, 1])
        planar_file = out/f'{name}_projected_skeleton.ply'
        o3d.io.write_point_cloud(str(planar_file), skeleton)
        baseline_ids = np.flatnonzero(np.all((local_vertices >= low) & (local_vertices <= high), axis=1)
                                     & np.isin(labels, [name, arm_class, 'unknown']))
        baseline = mesh.select_by_index(baseline_ids)
        center = wrist+axes @ ((low+high)/2)
        span = max(high-low)*1.12
        for angle in (0, 1):
            row = side*2+angle
            right, up = (axes[:, 0], axes[:, 1]) if angle == 0 else (axes[:, 2], axes[:, 1])
            raw_picture = point_panel(raw[selected], center, right, up, span)
            canvas.paste(raw_picture, (0, row*380+25))
            density_picture = point_panel(raw[selected], center, right, up, span)
            overlay = ImageDraw.Draw(density_picture)
            uv = np.c_[(lifted_points-center) @ right, (lifted_points-center) @ up]/span
            uv = np.rint((uv+.5)*349).astype(int)
            for (x, y), degree in zip(uv, data['planar_degree']):
                color = '#00bb40' if degree == 1 else '#ef3020'
                overlay.ellipse((x-1, 348-y, x+1, 350-y), fill=color)
            canvas.paste(density_picture, (350, row*380+25))
            for column, model in enumerate((baseline, *models), 2):
                canvas.paste(panel(model, center, right, up, span, 350), (column*350, row*380+25))
            for column, title in enumerate(('Original points', 'Projected skeleton (lifted)', 'Current mesh',
                                             'Density surface', 'Skeleton guided surface')):
                draw.text((column*350+4, row*380+5), f'{name} | {title}', fill='black')
        fit = distances(models[1], raw[selected])
        support = cKDTree(raw[selected]).query(np.asarray(models[1].vertices))[0]
        stats.update(part=name, raw_points=len(points), raw_to_candidate_p95_mm=float(np.quantile(fit, .95)),
                     candidate_to_raw_p95_mm=float(np.quantile(support, .95)),
                     boundary_edges=len(models[1].get_non_manifold_edges(allow_boundary_edges=False)),
                     self_intersecting=models[1].is_self_intersecting(),
                     candidate=str(candidate_file) if keep_meshes else None, skeleton=str(skeleton_file),
                     projected_skeleton=str(planar_file),
                     note='Density topology skeleton, not anatomical bones. Thickness follows voxel occupancy.',
                     candidate_sha256=file_hash(candidate_file) if keep_meshes else None)
        records.append(stats)
        print(parent.name, name, 'skeleton nodes', stats['skeleton_nodes'],
              'endpoints', [s['non_wrist_endpoints'] for s in stats['density_threshold_stability']], flush=True)
    canvas.save(out/'comparison.jpg', quality=95)
    if file_hash(ply) != source_hash or file_hash(Path(report['source'])) != cloud_hash:
        raise ValueError('Input changed during the trial.')
    result = {'source': str(ply), 'source_sha256': source_hash, 'source_cloud_sha256': cloud_hash,
              'parts': records, 'elapsed_seconds': round(time.perf_counter()-started, 2),
              'status': 'local_preview', 'accepted_for_full_body': False}
    (out/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--worker-python', type=Path, default=Path('D:/Anaconda/conda/python.exe'))
    parser.add_argument('--voxel-mm', type=float, default=2.)
    parser.add_argument('--keep-meshes', action='store_true')
    parser.add_argument('--worker', type=Path, nargs=2)
    args = parser.parse_args()
    if not np.isfinite(args.voxel_mm) or not 1 <= args.voxel_mm <= 4:
        parser.error('--voxel-mm must be between 1 and 4.')
    if args.worker:
        density_worker(*args.worker, args.voxel_mm)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        for scan in args.scan:
            trial(scan, args.worker_python, args.voxel_mm, args.keep_meshes)
