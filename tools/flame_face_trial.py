"""Independent FLAME scan-fitting experiment; never writes outputs/models.

Run with the dachuang Python: python tools/flame_face_trial.py --scan 0
Fits the neutral FLAME shape to scan geometry, then transfers a bounded face
surface prior onto the existing scan vertices. No photo encoder is used.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.mano_fit import ModelLoader, LegacyArray
from tools.export_presentation_images import mesh_image, mesh_scene
from surface_quality import intersection_blocks, intersection_pairs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model():
    folder = ROOT/'tools/flame'
    manifest = json.loads((folder/'download_manifest.json').read_text('utf-8'))
    path = folder/'flame2023_Open.pkl'
    if sha(path) != manifest['model']['sha256']:
        raise ValueError('FLAME model hash mismatch')
    with path.open('rb') as stream:
        data = ModelLoader(stream, encoding='latin1').load()
    for key, value in list(data.items()):
        if isinstance(value, LegacyArray):
            data[key] = value.value()
    vertices = np.asarray(data['v_template'], float)
    unit = 1000. if np.ptp(vertices, axis=0).max() < 5 else 1.
    return vertices*unit, np.asarray(data['shapedirs'], float)*unit, np.asarray(data['f'], int), manifest


def make_mesh(vertices, faces, colors=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    return mesh


def detect_nose(mesh, parts):
    """Geometric initialization only; not an anatomical measurement landmark."""
    v = np.asarray(mesh.vertices)
    face = v[parts['face_neck']['vertex_indices']]
    hair = v[parts['hair']['vertex_indices']]
    top = np.quantile(hair[:, 0], .98)
    middle = np.median(face[:, 1])
    candidates = v[(v[:, 0] > top-200) & (v[:, 0] < top-100) & (abs(v[:, 1]-middle) < 25)]
    if len(candidates) < 20:
        raise ValueError('Insufficient central face surface for initialization')
    return candidates[candidates[:, 2] <= np.quantile(candidates[:, 2], .015)].mean(0)


def fit_template(mesh, parts, template, basis, faces):
    v = np.asarray(mesh.vertices)
    nose = detect_nose(mesh, parts)
    tm = make_mesh(template, faces)
    tip_id = int(np.argmax(template[:, 2]))
    origin = template[tip_id]
    # FLAME: Y up, Z forward. Scan: X up, -Z forward (right-handed).
    frame = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    rel = template-origin
    visible = ((abs(rel[:, 0]) < 78) & (rel[:, 1] > -78) & (rel[:, 1] < 83)
               & (rel[:, 2] > -67) & (np.asarray(tm.vertex_normals)[:, 2] > .05))
    ids = np.flatnonzero(visible)
    ids = ids[np.linspace(0, len(ids)-1, min(1500, len(ids))).astype(int)]
    delta = v-nose
    region = ((delta[:, 0] > -92) & (delta[:, 0] < 100) & (abs(delta[:, 1]) < 100)
              & (delta[:, 2] > -15) & (delta[:, 2] < 110)
              & (np.asarray(mesh.vertex_normals)[:, 2] < -.1))
    # Hair is excluded even where it lies in front of the forehead.
    region[parts['hair']['vertex_indices']] = False
    target = v[region]
    if len(target) < 300 or len(ids) < 300:
        raise ValueError('Insufficient face fitting support')
    tree = cKDTree(target)
    count = min(20, basis.shape[2])
    p = np.zeros(7+count)
    lower = np.r_[[-.5, -.5, -.5], [-25., -25., -25.], np.log(.82), np.full(count, -2.5)]
    upper = np.r_[[.5, .5, .5], [25., 25., 25.], np.log(1.18), np.full(count, 2.5)]

    def evaluate(parameters, selected=None):
        x = template if selected is None else template[selected]
        b = basis[:, :, :count] if selected is None else basis[selected, :, :count]
        local = x-origin + np.einsum('vck,k->vc', b, parameters[7:])
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix() @ frame
        return np.exp(parameters[6])*(local @ rotation.T)+nose+parameters[3:6]

    history = []
    for step in range(18):
        current = evaluate(p, ids)
        distances, nearest = tree.query(current, workers=4)
        keep = distances < (30 if step < 5 else 18)
        if keep.sum() < 300:
            raise ValueError('Template initialization did not overlap the face')
        selected = ids[keep]
        matched = target[nearest[keep]]
        # Shape regularization keeps unsupported structures near the learned mean.
        def residual(parameters):
            return np.r_[(evaluate(parameters, selected)-matched).ravel(),
                         parameters[7:]*3., parameters[:3]*3., parameters[6]*25.,
                         (evaluate(parameters, [tip_id])[0]-nose)*1.5]
        if step < 5:
            def rigid(q):
                return residual(np.r_[q, np.zeros(count)])
            result = least_squares(rigid, p[:7], bounds=(lower[:7], upper[:7]),
                                   loss='soft_l1', f_scale=3., max_nfev=25)
            p[:7] = result.x
        else:
            result = least_squares(residual, p, bounds=(lower, upper),
                                   loss='soft_l1', f_scale=3., max_nfev=20)
            p = result.x
        history.append({'iteration': step, 'median_correspondence_mm': float(np.median(distances)),
                        'p90_correspondence_mm': float(np.quantile(distances, .9))})
    fitted = evaluate(p)
    detail = {'method': 'neutral FLAME shape fitted to 3D scan; no image network',
              'shape_components': count, 'nose_initialization_mm': nose.tolist(),
              'geometric_initialization_is_measurement_landmark': False,
              'rotation_vector': p[:3].tolist(), 'translation_mm': p[3:6].tolist(),
              'scale': float(np.exp(p[6])), 'shape_coefficients': p[7:].tolist(), 'history': history}
    return fitted, nose, detail


def fuse(mesh, parts, fitted, model_faces, nose, strength, cap):
    """Transfer the model depth field on the existing scan topology."""
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    model = make_mesh(fitted, model_faces)
    scene = mesh_scene(model)
    # Same image-plane coordinates avoid sideways sliding of scan landmarks.
    origins = v.copy()
    origins[:, 2] = min(fitted[:, 2].min(), v[:, 2].min())-100
    rays = np.c_[origins, np.tile([0., 0., 1.], (len(v), 1))].astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)
    distance = hit['t_hit'].numpy()
    ok = np.isfinite(distance)
    z = origins[:, 2]+np.where(ok, distance, 0.)
    d = v-nose
    inside = ((d[:, 0] > -78) & (d[:, 0] < 92) & (abs(d[:, 1]) < 84)
              & (d[:, 2] > -20) & (d[:, 2] < 85) & ok
              & (np.asarray(mesh.vertex_normals)[:, 2] < -.12) & (abs(z-v[:, 2]) < 20))
    # The neck, hair and back of the head stay exactly at original coordinates.
    inside[parts['hair']['vertex_indices']] = False
    edges = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(v[edges[:, 0]]-v[edges[:, 1]], axis=1)
    graph = coo_matrix((np.r_[lengths, lengths],
                       (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(len(v), len(v))).tocsr()
    boundary = np.unique(edges[inside[edges[:, 0]] != inside[edges[:, 1]]])
    margin = dijkstra(graph, directed=False, indices=boundary, min_only=True, limit=24.)
    w = np.clip(margin/18., 0., 1.)
    w = w*w*(3-2*w)*inside*strength
    step = np.clip(z-v[:, 2], -cap, cap)*w
    adjacency = coo_matrix((np.ones(len(edges)*2),
                           (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                          shape=(len(v), len(v))).tocsr()
    degree = np.maximum(np.asarray(adjacency.sum(1)).ravel(), 1.)
    for _ in range(5):
        step = (.7*step+.3*(adjacency@step)/degree)*inside
    candidate = copy.deepcopy(mesh)
    cv = v.copy()
    cv[:, 2] += step
    candidate.vertices = o3d.utility.Vector3dVector(cv)
    candidate.compute_vertex_normals()
    before = np.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]])
    retracted = np.zeros(len(v), bool)
    blocks = intersection_blocks(mesh, inside)
    base_pairs = intersection_pairs(mesh, blocks)
    # Retract only neighborhoods where the scan's existing folds would invert.
    # Attenuation rings keep the correction from introducing a sharp crease.
    for _ in range(18):
        after = np.cross(cv[f[:, 1]]-cv[f[:, 0]], cv[f[:, 2]]-cv[f[:, 0]])
        bad = np.einsum('ij,ij->i', before, after) <= 0
        candidate.vertices = o3d.utility.Vector3dVector(cv)
        added = intersection_pairs(candidate, blocks)-base_pairs
        if not np.any(bad) and not added:
            break
        halo = np.zeros(len(v), bool)
        halo[np.unique(f[bad])] = True
        if added:
            halo[np.unique(f[np.asarray(list(added), dtype=int).ravel()])] = True
        attenuation = np.ones(len(v))
        for value in [0., .2, .4, .7, .9]:
            attenuation[halo] = np.minimum(attenuation[halo], value)
            halo |= (adjacency @ halo.astype(float)) > 0
        retracted |= attenuation < 1
        step *= attenuation
        cv = v.copy()
        cv[:, 2] += step
    candidate.vertices = o3d.utility.Vector3dVector(cv)
    candidate.compute_vertex_normals()
    after = np.cross(cv[f[:, 1]]-cv[f[:, 0]], cv[f[:, 2]]-cv[f[:, 0]])
    flips = int(np.sum(np.einsum('ij,ij->i', before, after) <= 0))
    added = intersection_pairs(candidate, blocks)-base_pairs
    detail = {'strength': strength, 'cap_mm': cap, 'changed_vertices': int(np.count_nonzero(step)),
              'max_displacement_mm': float(abs(step).max()),
              'p95_displacement_mm': float(np.quantile(abs(step[step != 0]), .95)) if np.any(step) else 0.,
              'flipped_faces': flips, 'new_intersections': len(added),
              'locally_retracted_vertices': int(retracted.sum()),
              'protected_vertices_exact': bool(np.array_equal(cv[~inside], v[~inside])),
              'connectivity_unchanged': bool(np.array_equal(np.asarray(candidate.triangles), f)),
              'geometry_checks_passed': not flips and not added,
              'is_measurement_accuracy_validation': False}
    return candidate, detail


def comparison(meshes, labels, nose, path):
    width, height = 480, 460
    canvas = Image.new('RGB', (width*len(meshes), height*2+60), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 23)
    center = nose + [10., 0., 40.]
    for col, (mesh, label) in enumerate(zip(meshes, labels)):
        scene = mesh_scene(mesh)
        draw.text((col*width+16, 14), label, font=font, fill='black')
        for row, angle in enumerate([0., 28.]):
            a = np.deg2rad(angle)
            right = np.array([0., np.cos(a), np.sin(a)])
            pic = mesh_image(mesh, scene, center, right, 240., (width, height), colored=False)
            canvas.paste(pic, (col*width, 60+row*height))
    canvas.save(path, quality=95)


def trial(scan):
    started = time.perf_counter()
    name = f'export_{scan:05d}'
    labels_path = sorted((ROOT/'outputs/models'/name).glob('*/body_parts.json'))[-1]
    labels = json.loads(labels_path.read_text('utf-8'))
    source = labels_path.parent/labels['mesh']['file']
    digest = sha(source)
    if digest != labels['mesh']['sha256']:
        raise ValueError('Semantic labels belong to a different mesh')
    mesh = o3d.io.read_triangle_mesh(str(source))
    mesh.compute_vertex_normals()
    template, basis, faces, manifest = load_model()
    fitted, nose, fit_detail = fit_template(mesh, labels['parts'], template, basis, faces)
    out = ROOT/'outputs/experiments/face_flame'/name
    out.mkdir(parents=True, exist_ok=True)
    fitted_mesh = make_mesh(fitted, faces)
    o3d.io.write_triangle_mesh(str(out/'fitted_flame.ply'), fitted_mesh)
    variants = [mesh]
    captions = ['原扫描模型']
    results = []
    for strength, cap, caption in [(.55, 6., '局部融合 55%'), (.85, 10., '局部融合 85%')]:
        candidate, checks = fuse(mesh, labels['parts'], fitted, faces, nose, strength, cap)
        checks['file'] = f'face_blend_{int(strength*100)}.ply'
        # Diagnostic candidate only: a failed geometry check is never published.
        if checks['geometry_checks_passed']:
            target = out/checks['file']
            if not o3d.io.write_triangle_mesh(str(target), candidate):
                raise IOError('Candidate mesh export failed')
            readback = o3d.io.read_triangle_mesh(str(target))
            checks['export_exact'] = bool(np.array_equal(np.asarray(readback.vertices), np.asarray(candidate.vertices))
                                          and np.array_equal(np.asarray(readback.triangles), np.asarray(mesh.triangles)))
            checks['sha256'] = sha(target)
            variants.append(candidate)
            captions.append(caption)
        results.append(checks)
    comparison(variants, captions, nose, out/'comparison.jpg')
    report = {'status': 'experiment_only', 'scan': scan, 'source': str(source), 'source_sha256': digest,
              'model_sha256': manifest['model']['sha256'], 'fit': fit_detail, 'candidates': results,
              'source_unchanged': sha(source) == digest,
              'notes': ['Template-imputed face shape; not recovered measurement truth.',
                        'Neutral FLAME shape fitting only; no DECA/EMOCA encoder or wrinkle weights.',
                        'Head, neck and manual measurement landmarks must be reviewed before production use.'],
              'seconds': time.perf_counter()-started}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps({'scan': scan, 'output': str(out), 'candidates': results,
                      'seconds': report['seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', nargs='+', type=int, default=[0])
    args = parser.parse_args()
    for number in args.scan:
        trial(number)
