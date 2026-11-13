"""脸颈照片投影和贴图导出，由 run.py 在正式网格及部位标签完成后调用。"""

import shutil
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

from body_parts import file_hash, load_part_indices
from .raw_capture import samples_for_stream

FACE_SIZE = 1024
BODY_SIZE = 2048
PADDING = 4
FACING_POWER = 6
PHOTO_FRACTION = 2 / 3
OBJ_PACKAGE = ('face_uv_0.png', 'face_uv_1.png', 'face_uv.mtl', 'face_uv.obj')


def build_face_atlas(mesh, ids, raw, hashes):
    if not (mesh.has_triangles() and mesh.has_vertex_colors() and mesh.has_vertex_normals()):
        raise ValueError('Texture projection requires a colored mesh with vertex normals.')
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    mask = np.zeros(len(vertices), bool)
    mask[ids] = True
    face = mask[triangles].any(axis=1)
    if not face.any() or face.all():
        raise ValueError('Face and body must each contain triangles.')
    mixed = triangles[face & ~mask[triangles].all(axis=1)]
    boundary = np.unique(mixed[mask[mixed]])
    gain = np.zeros((len(vertices), 1))
    gain[ids, 0] = np.minimum(cKDTree(vertices[boundary]).query(vertices[ids])[0]/6, 1) if len(boundary) else 1
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    uv = np.empty((len(triangles), 3, 2), np.float32)
    uv[face], maps = bake(mesh, face, FACE_SIZE, {
        'albedo': np.asarray(mesh.vertex_colors), 'world_xyz': vertices,
        'direction': np.asarray(mesh.vertex_normals), 'gain': gain,
    }, 1)
    face_image, stats = photo_texture(scene, maps, raw, hashes)
    return uv, face, face_image, stats


def bake_body_atlas(mesh, uv, face, face_image):
    uv[~face], body = bake(mesh, ~face, BODY_SIZE, {'albedo': np.asarray(mesh.vertex_colors)}, 8)
    return [to_rgb8(body['albedo']), face_image]


def bake(mesh, faces, size, attributes, partitions):
    vertices, local_faces = np.unique(np.asarray(mesh.triangles)[faces], return_inverse=True)
    local = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices)[vertices], dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(local_faces.reshape(-1, 3)),
    )
    local.compute_uvatlas(size, gutter=2*PADDING+1, max_stretch=.2,
                          parallel_partitions=partitions, nthreads=4)
    for name, values in attributes.items():
        local.vertex[name] = o3d.core.Tensor(values[vertices], dtype=o3d.core.Dtype.Float32)
    baked = local.bake_vertex_attr_textures(
        size, set(attributes), margin=PADDING, fill=-1, update_material=False,
    )
    return local.triangle.texture_uvs.numpy(), {name: baked[name].numpy() for name in attributes}


def photo_texture(scene, maps, raw, hashes):
    active = (maps['albedo'][..., 0] >= 0) & (maps['gain'][..., 0] > 0)
    baseline = maps['albedo'][active]
    xyz = maps['world_xyz'][active]
    normal = maps['direction'][active]
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    strength = PHOTO_FRACTION * np.clip(maps['gain'][active], 0, 1)
    sample = np.random.default_rng(7).choice(len(xyz), min(16000, len(xyz)), replace=False)
    rgb_sum = np.zeros((len(xyz), 3))
    total = np.zeros(len(xyz))
    views = []
    entries = [entry for entry in raw['streams'] if entry['projection_check_passed']]
    for entry in entries:
        rgb, weight = samples_for_stream(
            scene, xyz, normal, baseline, Path(raw['source']), raw['frame'], entry, hashes,
            bilinear=True, facing_power=FACING_POWER,
        )
        rgb_sum += rgb * weight[:, None]
        total += weight
        views.append((rgb[sample], weight[sample]))

    output = maps['albedo'].copy()
    visible = total > 0
    blended = baseline.copy()
    blended[visible] += strength[visible] * (rgb_sum[visible] / total[visible, None] - baseline[visible])
    output[active] = blended

    before, after = [], []
    for rgb, weight in views:
        other_weight = total[sample] - weight
        test = (weight > 0) & (other_weight > 1e-9)
        indices = sample[test]
        held_out = (rgb_sum[indices] - rgb[test]*weight[test, None]) / other_weight[test, None]
        prediction = baseline[indices] + strength[indices] * (held_out-baseline[indices])
        before.extend(np.abs(rgb[test]-baseline[indices]).mean(axis=1))
        after.extend(np.abs(rgb[test]-prediction).mean(axis=1))
    p50, p90 = np.percentile(before, [50, 90]) if before else (None, None)
    a50, a90 = np.percentile(after, [50, 90]) if after else (None, None)
    stats = {
        'eligible_texels': len(xyz), 'photo_texels': int(visible.sum()),
        'cameras_with_samples': sum(bool(np.any(w)) for _, w in views),
        'holdout_observations': len(before), 'baseline_median_rgb_error': p50,
        'baseline_p90_rgb_error': p90, 'uv_median_rgb_error': a50, 'uv_p90_rgb_error': a90,
        'passes_holdout': bool(len(before) >= 256 and a50 <= .95*p50 and a90 <= p90+1/255),
    }
    return to_rgb8(output), stats


def to_rgb8(colors):
    return np.rint(np.clip(colors, 0, 1)*255).astype(np.uint8)


def publish_obj_package(staging, out):
    # A rename from TemporaryDirectory can retain its private Windows permissions.
    # Create fresh files in the output directory so they inherit its permissions.
    pending = {}
    try:
        for name in OBJ_PACKAGE:
            temp = out/f'.{name}.{uuid4().hex}.tmp'
            with (staging/name).open('rb') as source, temp.open('xb') as target:
                pending[name] = temp
                shutil.copyfileobj(source, target, length=1024*1024)
        for name, temp in pending.items():
            temp.replace(out/name)
    finally:
        for temp in pending.values():
            temp.unlink(missing_ok=True)


def save_obj(out, mesh, uv, face, textures):
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uv.reshape(-1, 2))
    mesh.triangle_material_ids = o3d.utility.IntVector(face.astype(np.int32))
    # Legacy Open3D texture storage starts at the bottom row. OBJ PNGs start at the top.
    mesh.textures = [o3d.geometry.Image(np.ascontiguousarray(image[::-1])) for image in textures]
    with tempfile.TemporaryDirectory(dir=out) as staging:
        staged = Path(staging)/'face_uv.obj'
        if not o3d.io.write_triangle_mesh(str(staged), mesh, write_vertex_colors=False):
            raise OSError(f'Cannot write {staged}')
        reloaded = o3d.io.read_triangle_mesh(str(staged))
        expected = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
        actual = np.asarray(reloaded.vertices)[np.asarray(reloaded.triangles)]
        material = np.asarray(reloaded.triangle_material_ids)
        # The OBJ exporter groups triangles by material. A reader may split UV seam vertices.
        expected = np.concatenate([expected[~face], expected[face]])
        error = float(np.max(np.abs(actual-expected))) if actual.shape == expected.shape else np.inf
        expected_uv = np.concatenate([uv[~face], uv[face]]).reshape(-1, 2)
        actual_uv = np.asarray(reloaded.triangle_uvs)
        body_count = np.count_nonzero(~face)
        if (error > .01 or actual_uv.shape != expected_uv.shape
                or not np.allclose(actual_uv, expected_uv, atol=1e-6, rtol=0)
                or not np.all(material[:body_count] == material[0])
                or not np.all(material[body_count:] == material[-1])
                or material[0] == material[-1]):
            raise ValueError('OBJ read-back differs from the source geometry or materials.')
        for image, name in zip(textures, ('face_uv_0.png', 'face_uv_1.png')):
            with Image.open(staged.with_name(name)) as saved:
                if not np.array_equal(np.asarray(saved), image):
                    raise ValueError(f'Exported texture pixels differ: {name}')
        publish_obj_package(staged.parent, out)
    return error


def export_face_uv(run_dir, report, enabled):
    """Export or reuse the optional texture package for the finalized PLY."""
    previous = report.get('face_uv', {})
    if not enabled:
        return previous or {'status': 'disabled'}
    raw = report.get('raw_color', {})
    if raw.get('status') != 'used':
        return {'status': 'skipped', 'reason': 'No verified paired RAW photos.'}
    segmentation = report.get('segmentation', {})
    if report.get('subject', {}).get('kind') != 'human' or segmentation.get('status') != 'complete':
        return {'status': 'skipped', 'reason': 'Verified human part labels are unavailable.'}
    source = run_dir/report['artifacts']['ply']['name']
    out = run_dir/'textured'
    signature = {
        'mesh_sha256': report['artifacts']['ply']['sha256'],
        'parts_sha256': segmentation['sha256'],
    }
    started = time.perf_counter()
    try:
        if previous.get('input') == signature:
            if previous['status'] == 'rejected':
                return previous
            hashes = previous.get('sha256', {})
            if previous['status'] == 'complete' and set(hashes) == set(OBJ_PACKAGE) and all(
                (out/name).is_file() and file_hash(out/name) == hashes[name] for name in OBJ_PACKAGE
            ):
                return previous
        ids = load_part_indices(run_dir/'body_parts.json', 'face_neck', source)
        if not len(ids):
            return {'status': 'skipped', 'reason': 'No face and neck labels.'}
        print('\n[face UV: paired photos / visibility / color check]', flush=True)
        mesh = o3d.io.read_triangle_mesh(str(source))
        uv, face, face_image, stats = build_face_atlas(
            mesh, ids, raw, report['fingerprint']['raw_color_input']['sha256'],
        )
        result = {'status': 'rejected', 'input': signature, **stats}
        if stats['passes_holdout']:
            textures = bake_body_atlas(mesh, uv, face, face_image)
            out.mkdir(exist_ok=True)
            error = save_obj(out, mesh, uv, face, textures)
            result.update(status='complete', obj='textured/face_uv.obj', obj_max_axis_error_mm=error,
                          sha256={name: file_hash(out/name) for name in OBJ_PACKAGE})
        else:
            print('Face UV skipped: color consistency check did not pass.', flush=True)
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'
        result = {'status': 'failed', 'input': signature, 'reason': reason}
        print(f'WARNING: face UV skipped; reconstruction retained: {reason}', flush=True)
    result['elapsed_seconds'] = round(time.perf_counter()-started, 3)
    return result


