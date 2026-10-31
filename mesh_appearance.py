"""外观后处理：run.py 默认调用，也可独立修正已有模型的法线与孤立黑点。
  python mesh_appearance.py 1       处理 models 中 00001 最新完成的模型
  python mesh_appearance.py         处理最近完成的模型；也可直接写网格 PLY 路径
  --normals-only：只修法线；--colors-only：只修颜色；--all：处理各编号最新模型
  --patch-mm 12：黑色小块最大包围尺寸（毫米）；--dark-threshold 0.10：暗点亮度阈值
  --color-region body|all：默认 body，按站立人体比例保护顶部 20% 和底部 8%
  --up-axis auto|x|y|z、--up-sign -1：指定人体竖直方向；auto 按明显最长轴
  --unit m：输入单位为米，默认 mm；--out 目录：指定新的独立输出目录
输出：outputs/appearance/模型名/时间/ 下的 PLY、OBJ、report.json，不覆盖输入。
颜色修补使用网格连接关系，只处理与邻域反差明显的小暗块；真实黑色花纹仍可能误判。
法线按外部射线可见性判断，不靠开口网格的有向体积猜测；证据不足则保留方向。
"""

import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from ply_to_obj import convert_ply_to_obj

ROOT = Path(__file__).resolve().parent
LUMA = np.array([.2126, .7152, .0722])


def select_models(value=None, all_models=False):
    """Only inspect completed run reports; never select intermediate meshes."""
    latest = []
    for folder in sorted((ROOT / 'outputs' / 'models').glob('*')):
        candidates = []
        for report in folder.glob('*/report.json'):
            mesh = report.parent / (folder.name + '_mesh.ply')
            if mesh.is_file():
                candidates.append(mesh)
        if candidates:
            latest.append(max(candidates, key=lambda p: p.parent.name))
    if value is not None:
        if value.isdecimal():
            matches = [p for p in latest if p.stem == f'export_{int(value):05d}_mesh']
            if len(matches) != 1:
                raise ValueError(f'No unique completed model for {value}; run python run.py {value} first.')
            return matches
        path = Path(value).resolve()
        if not path.is_file() or path.suffix.lower() != '.ply':
            raise ValueError('Input must be an existing mesh PLY.')
        return [path]
    if not latest:
        raise ValueError('No completed models found in outputs/models.')
    return latest if all_models else [max(latest, key=lambda p: p.stat().st_mtime_ns)]


def fix_normal_direction(mesh, resolution=48):
    """Orient neighbors, then vote per component using first hits from outside.

    This assumes a sampled outer surface. It is not semantic inside/outside
    recognition for arbitrary open sheets or nested shells. Ambiguous votes
    do not trigger a global flip. Surviving face order/vertices are preserved.
    """
    candidate = copy.deepcopy(mesh)
    if not candidate.orient_triangles():
        mesh.compute_vertex_normals()
        return mesh, {'applied': False, 'reason': 'Cannot orient adjacent faces; winding retained, normals recomputed.'}
    faces = np.asarray(candidate.triangles).copy()
    labels, counts, _ = candidate.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    # Normalize a temporary copy for ray precision with large world offsets.
    ray_mesh = copy.deepcopy(candidate)
    lower, upper = ray_mesh.get_min_bound(), ray_mesh.get_max_bound()
    scale = float(np.max(upper - lower))
    if scale <= 0:
        raise ValueError('Mesh has zero extent.')
    ray_mesh.translate(-(lower + upper) / 2).scale(1 / scale, center=np.zeros(3))
    lower, upper = ray_mesh.get_min_bound(), ray_mesh.get_max_bound()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ray_mesh))
    hits, inward = np.zeros(len(counts), dtype=int), np.zeros(len(counts), dtype=int)
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        a, b = np.meshgrid(*(np.linspace(lower[i], upper[i], resolution) for i in others))
        for sign in (-1, 1):
            origins = np.zeros((a.size, 3))
            origins[:, others[0]], origins[:, others[1]] = a.ravel(), b.ravel()
            origins[:, axis] = upper[axis] + 1 if sign < 0 else lower[axis] - 1
            directions = np.zeros_like(origins)
            directions[:, axis] = sign
            result = scene.cast_rays(o3d.core.Tensor(np.c_[origins, directions].astype(np.float32)))
            valid = np.isfinite(result['t_hit'].numpy())
            face_ids = result['primitive_ids'].numpy()[valid].astype(np.int64)
            dots = np.einsum('ij,ij->i', result['primitive_normals'].numpy()[valid], directions[valid])
            usable = np.abs(dots) > .15  # Ignore grazing hits.
            component_ids = labels[face_ids[usable]]
            hits += np.bincount(component_ids, minlength=len(counts))
            inward += np.bincount(component_ids[dots[usable] > 0], minlength=len(counts))
    components, flipped = [], 0
    for i in range(len(counts)):
        fraction = float(inward[i] / hits[i]) if hits[i] else None
        confident = hits[i] >= 50 and (fraction >= .9 or fraction <= .1)
        flip = bool(confident and fraction >= .9)
        if flip:
            mask = labels == i
            faces[mask] = faces[mask][:, [0, 2, 1]]
            flipped += int(mask.sum())
        components.append({'faces': int(counts[i]), 'visible_hits': int(hits[i]),
                           'inward_fraction_before': fraction, 'confident': bool(confident),
                           'flipped': flip})
    candidate.triangles = o3d.utility.Vector3iVector(faces)
    candidate.compute_vertex_normals()
    return candidate, {'applied': True, 'method': 'adjacent-face orientation and external first-hit votes',
                       'flipped_faces': flipped, 'components': components,
                       'note': 'Visibility heuristic for outer surfaces; not proof of watertightness or correct nested-shell orientation.'}


def repair_dark_patches(mesh, *, patch_mm=12., dark_threshold=.10, color_region='body',
                        up_axis='auto', up_sign=1, unit='mm'):
    """Interpolate small dark islands from their adjacent unchanged colors.

    Use mesh edges, not XYZ neighbors across nearby fingers/limbs. Large dark
    components, weak boundary contrast, and head/shoe bands are left alone.
    """
    if not mesh.has_vertex_colors():
        return {'applied': False, 'reason': 'No vertex RGB in input.'}
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors).copy()
    factor = 1000 if unit == 'm' else 1
    points = (vertices - vertices.mean(0)) * factor
    extent = np.ptp(points, axis=0)
    eligible = np.ones(len(points), dtype=bool)
    region_report = {'region': color_region}
    if color_region == 'body':
        order = np.argsort(extent)
        if up_axis == 'auto' and extent[order[-1]] < 1.3 * extent[order[-2]]:
            return {'applied': False, 'reason': 'Body axis ambiguous; specify --up-axis or use --color-region all.'}
        axis = int(order[-1]) if up_axis == 'auto' else 'xyz'.index(up_axis)
        if extent[axis] <= 0:
            return {'applied': False, 'reason': 'Selected body axis has zero extent.'}
        height = points[:, axis] * up_sign
        fraction = (height - height.min()) / extent[axis]
        eligible = (fraction >= .08) & (fraction <= .80)
        region_report.update(axis='xyz'[axis], up_sign=up_sign,
                             note='Standing-human proportion mask, not semantic body segmentation.')
    light = colors @ LUMA
    dark = (light < dark_threshold) & (colors.max(axis=1) < min(.22, 2 * dark_threshold)) & eligible
    ids = np.flatnonzero(dark)
    if not len(ids):
        return {'applied': False, 'reason': 'No eligible dark patches.', **region_report}
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    weights = 1 / np.maximum(lengths, 1e-6)
    graph = coo_matrix((np.r_[weights, weights], (np.r_[edges[:, 0], edges[:, 1]],
                                                np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(len(points), len(points))).tocsr()
    number, labels = connected_components(graph[ids][:, ids], directed=False)
    sizes = np.bincount(labels, minlength=number)
    order = np.argsort(labels, kind='stable')
    offsets = np.r_[0, np.cumsum(sizes)]
    accepted, patch_count = [], 0
    for label in np.flatnonzero(sizes <= 120):
        patch = ids[order[offsets[label]:offsets[label + 1]]]
        if np.max(np.ptp(points[patch], axis=0)) > patch_mm:
            continue
        neighbors = np.unique(graph[patch].indices)
        boundary = np.setdiff1d(neighbors, patch, assume_unique=True)
        if len(boundary) < 5 or not eligible[boundary].all():
            continue
        # Reject texture edges and mostly dark surroundings; no global exposure boost.
        boundary_light = light[boundary]
        if (np.median(boundary_light) < max(.14, np.median(light[patch]) + .10)
                or np.mean(boundary_light > dark_threshold + .06) < .65):
            continue
        accepted.extend(patch.tolist())
        patch_count += 1
    if not accepted:
        return {'applied': False, 'reason': 'No small, high-contrast dark islands passed checks.',
                'candidate_dark_vertices': len(ids), **region_report}
    targets = np.array(sorted(accepted), dtype=int)
    # Harmonic interpolation: boundary RGB remains fixed, solve only island RGB.
    boundary_colors = colors.copy()
    boundary_colors[targets] = 0
    matrix = diags(np.asarray(graph.sum(axis=1)).ravel()[targets]) - graph[targets][:, targets]
    repaired = spsolve(matrix.tocsc(), graph[targets] @ boundary_colors)
    if not np.isfinite(repaired).all():
        raise RuntimeError('Color interpolation produced nonfinite values.')
    colors[targets] = np.clip(repaired, 0, 1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return {'applied': True, 'method': 'small dark mesh islands with harmonic boundary RGB interpolation',
            'candidate_dark_vertices': len(ids), 'repaired_vertices': len(targets),
            'repaired_patches': patch_count, 'max_patch_extent_mm': patch_mm,
            'dark_luminance_threshold': dark_threshold, **region_report,
            'note': 'Appearance repair, not measured color recovery; small real black markings may be mistaken for artifacts.'}


def apply_appearance(mesh, *, mode='full', patch_mm=12., dark_threshold=.10,
                     color_region='body', up_axis='auto', up_sign=1, unit='mm'):
    """Shared in-memory stage for run.py and standalone use; input untouched."""
    if mode not in ('full', 'normals', 'colors', 'off'):
        raise ValueError('Appearance mode must be full, normals, colors or off.')
    skipped = {'applied': False, 'reason': 'Explicitly disabled.'}
    report = {'enabled': mode != 'off', 'mode': mode,
              'normals': dict(skipped), 'color_repair': dict(skipped),
              'positions_and_face_connectivity_preserved': True}
    if mode == 'off':
        return mesh, report
    candidate = copy.deepcopy(mesh)
    if mode in ('full', 'normals'):
        candidate, report['normals'] = fix_normal_direction(candidate)
    if mode in ('full', 'colors'):
        report['color_repair'] = repair_dark_patches(candidate, patch_mm=patch_mm,
                                                   dark_threshold=dark_threshold,
                                                   color_region=color_region, up_axis=up_axis,
                                                   up_sign=up_sign, unit=unit)
    if not np.array_equal(np.asarray(mesh.vertices), np.asarray(candidate.vertices)):
        raise RuntimeError('Unexpected geometry movement; appearance result rejected.')
    if not np.array_equal(np.sort(np.asarray(mesh.triangles), axis=1),
                          np.sort(np.asarray(candidate.triangles), axis=1)):
        raise RuntimeError('Unexpected face connectivity change; appearance result rejected.')
    return candidate, report


def process_model(source, args):
    source = Path(source).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    mesh = o3d.io.read_triangle_mesh(str(source))
    vertices, faces = np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).copy()
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise ValueError('Input must be a nonempty finite triangle mesh, not raw point cloud.')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid face indices.')
    if mesh.has_vertex_colors() and not np.isfinite(np.asarray(mesh.vertex_colors)).all():
        raise ValueError('Nonfinite RGB in input.')
    mode = 'normals' if args.normals_only else 'colors' if args.colors_only else 'full'
    mesh, appearance = apply_appearance(mesh, mode=mode, patch_mm=args.patch_mm,
                                       dark_threshold=args.dark_threshold, color_region=args.color_region,
                                       up_axis=args.up_axis, up_sign=args.up_sign, unit=args.unit)
    folder = (Path(args.out).resolve() if args.out else ROOT / 'outputs' / 'appearance' / source.stem /
              datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    # Require a fresh directory. Preserve every input, even with explicit --out.
    folder.mkdir(parents=True, exist_ok=False)
    output = folder / (source.stem + '_appearance.ply')
    if not o3d.io.write_triangle_mesh(str(output), mesh, write_vertex_normals=True, write_vertex_colors=True):
        raise OSError(f'Failed to write {output}')
    reread = o3d.io.read_triangle_mesh(str(output))
    if not np.array_equal(vertices, np.asarray(reread.vertices)):
        raise RuntimeError('Export readback changed coordinates.')
    convert_ply_to_obj(output, output.with_suffix('.obj'), vertex_colors=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Input changed while processing; please retry after copying finishes.')
    report = {'source': str(source), 'source_sha256': digest, 'output': str(output),
              'unit': args.unit, **appearance,
              'note': 'Independent appearance variant. Original reconstruction unchanged; no geometry smoothing, exposure boost, or print certification.'}
    (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nSOURCE: {source}\nNormal correction: {appearance['normals']}\nColor repair: {appearance['color_repair']}\nPLY: {output}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', nargs='?', help='Model number or mesh PLY path; default latest completed model.')
    parser.add_argument('--all', action='store_true', help='Process latest completed model of each input.')
    parser.add_argument('--out', help='Fresh output directory, for one input only.')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--normals-only', action='store_true')
    modes.add_argument('--colors-only', action='store_true')
    parser.add_argument('--patch-mm', type=float, default=12.)
    parser.add_argument('--dark-threshold', type=float, default=.10)
    parser.add_argument('--color-region', choices=['body', 'all'], default='body')
    parser.add_argument('--up-axis', choices=['auto', 'x', 'y', 'z'], default='auto')
    parser.add_argument('--up-sign', type=int, choices=[-1, 1], default=1)
    parser.add_argument('--unit', choices=['mm', 'm'], default='mm')
    args = parser.parse_args(argv)
    if args.all and (args.input or args.out):
        parser.error('--all cannot be combined with an input or --out.')
    if not np.isfinite(args.patch_mm) or args.patch_mm <= 0 or not 0 < args.dark_threshold <= .2:
        parser.error('--patch-mm must be positive and --dark-threshold must be in (0, 0.2].')
    try:
        for source in select_models(args.input, args.all):
            process_model(source, args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
