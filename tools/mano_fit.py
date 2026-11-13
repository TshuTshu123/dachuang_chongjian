"""MANO 模板手实验：骨架初始化、稳健点云拟合，输出独立手部预览。

项目根目录：python tools/mano_fit.py --scan 3
--scan 输入编号；模型在 tools/mano；无需安装旧 chumpy 或更改 conda。
输出 outputs/experiments/mano_fit/export_00003，不覆盖正式模型。
使用授权的 MANO 模型；不是 HaMeR/WiLoR 图像网络推理。
"""
import argparse
import copy
import hashlib
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hand_centerlines import hand_frame
from tools.surface_trials import panel


class LegacyArray:
    """Decode stored Ch/Select arrays without executing the old Chumpy package."""
    def __setstate__(self, state):
        self.__dict__.update(state)

    def value(self):
        if 'x' in self.__dict__:
            return self.x
        return self.a.value().ravel()[self.idxs].reshape(self.preferred_shape)


class ModelLoader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in [('chumpy.ch', 'Ch'), ('chumpy.reordering', 'Select')]:
            return LegacyArray
        return super().find_class(module, name)


class Mano:
    def __init__(self, side):
        self.path = ROOT/'tools/mano'/f'MANO_{side.upper()}.pkl'
        manifest = json.loads((self.path.parent/'download_manifest.json').read_text())
        expected = next(p['sha256'] for p in manifest['models'] if p['name'] == self.path.name)
        assert hashlib.sha256(self.path.read_bytes()).hexdigest() == expected
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            with self.path.open('rb') as stream:
                self.m = ModelLoader(stream, encoding='latin1').load()
        for key, value in list(self.m.items()):
            if isinstance(value, LegacyArray):
                self.m[key] = value.value()
        self.faces = np.asarray(self.m['f'], int)
        self.regressor = self.m['J_regressor'].toarray()
        self.parents = self.m['kintree_table'][0].astype(int)
        self.parents[0] = -1

    def evaluate(self, coefficients=None, beta=None, mean=0.):
        coefficients = np.zeros(6) if coefficients is None else coefficients
        beta = np.zeros(3) if beta is None else beta
        shaped = self.m['v_template'] + np.einsum('vck,k->vc', self.m['shapedirs'][:, :, :len(beta)], beta)
        joints = self.regressor @ shaped
        pose = np.r_[np.zeros(3), mean*self.m['hands_mean']+coefficients @ self.m['hands_components'][:len(coefficients)]]
        rotations = Rotation.from_rotvec(pose.reshape(16, 3)).as_matrix()
        posed = shaped+np.einsum('vck,k->vc', self.m['posedirs'], (rotations[1:]-np.eye(3)).ravel())
        global_transforms = []
        for j in range(16):
            transform = np.eye(4); transform[:3, :3] = rotations[j]
            transform[:3, 3] = joints[j] if j == 0 else joints[j]-joints[self.parents[j]]
            global_transforms.append(transform if j == 0 else global_transforms[self.parents[j]] @ transform)
        transforms = np.array(global_transforms)
        posed_joints = transforms[:, :3, 3].copy()
        transforms[:, :3, 3] -= np.einsum('jkl,jl->jk', transforms[:, :3, :3], joints)
        blended = np.einsum('vj,jkl->vkl', self.m['weights'], transforms)
        vertices = np.einsum('vkl,vl->vk', blended, np.c_[posed, np.ones(len(posed))])[:, :3]
        return (vertices-joints[0])*1000, (posed_joints-joints[0])*1000


def mesh_for(vertices, faces):
    result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    result.compute_vertex_normals()
    return result


def fit_hand(model, vertices, raw, parts, frame):
    ids, arm, wrist, axes = hand_frame(vertices, parts, frame['part'])
    local = (vertices-wrist) @ axes
    points = (raw-wrist) @ axes
    lo, hi = local[ids].min(0)-12, local[ids].max(0)+12
    selection = np.all((points >= lo) & (points <= hi), axis=1) & (points[:, 1] >= -5)
    points = points[selection]
    points = points[np.unique(np.floor(points/2.5).astype(int), axis=0, return_index=True)[1]]
    if len(points) < 100:
        raise ValueError('Too few hand observations')
    points = points[np.linspace(0, len(points)-1, min(1800, len(points))).astype(int)]
    tree = cKDTree(points)
    rest, joints = model.evaluate()
    _, eig = np.linalg.eigh(np.cov(rest.T)); normal = eig[:, 0]
    forward = rest.mean(0); forward -= (forward @ normal)*normal; forward /= np.linalg.norm(forward)
    canonical = np.column_stack((np.cross(forward, normal), forward, normal))
    reference = rest @ canonical
    ratio = np.quantile(local[ids, 1], .98)/np.quantile(reference[:, 1], .98)
    branch_points = [((np.asarray(b['world_path_mm'])-wrist) @ axes) for b in frame['branches'] if b['status'] == 'supported_candidate']
    lines = cKDTree(np.concatenate(branch_points)) if branch_points else None
    records = []; best = None
    for flip in (1., -1.):
        basis = canonical @ np.diag([flip, 1., flip])
        def decode(x):
            v, j = model.evaluate(x[7:13], x[13:16], x[16])
            rotation = Rotation.from_rotvec(x[:3]).as_matrix()
            scale = ratio*np.exp(x[6])
            return ((v @ basis) @ rotation.T)*scale+x[3:6], ((j @ basis) @ rotation.T)*scale+x[3:6]
        x = np.zeros(17); x[16] = .3
        # Bounded wrist, size, shape and pose prevent fitting the thick scan blob.
        lower = np.r_[[-.7]*3, [-6, -8, -22], -.25, [-2.]*6, [-1.5]*3, 0.]
        upper = np.r_[[.7]*3, [6, 8, 22], .25, [2.]*6, [1.5]*3, 1.]
        def residual(x):
            v, j = decode(x)
            d, nearest = tree.query(v)
            errors = v-points[nearest]
            # Reject outward points instead of expanding the template to include them.
            errors *= np.minimum(1., 10/np.maximum(d, 1.e-6))[:, None]
            extra = [x[3:6]*1.5, x[6:7]*30, x[7:13]*2, x[13:16]*3]
            if lines is not None:
                dd, nn = lines.query(j[1:])
                extra.append(((j[1:]-lines.data[nn])*np.minimum(1., 15/np.maximum(dd, 1.e-6))[:, None]*.35).ravel())
            return np.r_[errors.ravel(), *extra]
        result = least_squares(residual, x, bounds=(lower, upper), loss='soft_l1', f_scale=3., max_nfev=70, diff_step=1.e-4)
        fitted, posed_joints = decode(result.x)
        d = tree.query(fitted)[0]
        mesh = mesh_for(fitted, model.faces)
        intersections = len(np.asarray(mesh.get_self_intersecting_triangles()))
        pose_retraction = 1.
        if intersections:
            # Retract toward the neutral pose instead of publishing intersecting fingers.
            for pose_retraction in (.75, .5, .25, 0.):
                safe = result.x.copy(); safe[7:13] *= pose_retraction; safe[16] *= pose_retraction
                safe_v, safe_j = decode(safe)
                if len(mesh_for(safe_v, model.faces).get_self_intersecting_triangles()) == 0:
                    fitted, posed_joints = safe_v, safe_j; result.x = safe
                    intersections = 0; d = tree.query(fitted)[0]; break
        score = float(np.quantile(d, .75)+intersections*.1)
        record = {'flip': flip, 'surface_to_observation_p75_mm': float(np.quantile(d, .75)), 'self_intersections': intersections, 'evaluations': result.nfev, 'score': score, 'pose_retained_fraction': pose_retraction}
        records.append(record)
        if intersections == 0 and (best is None or score < best[0]):
            best = (score, fitted, posed_joints, result.x)
    if best is None:
        raise ValueError('All template candidates self-intersect; no hand published')
    _, fitted, posed_joints, parameters = best
    world = fitted @ axes.T+wrist
    return mesh_for(world, model.faces), {'part': frame['part'], 'observations': len(points), 'candidates': records, 'parameters': parameters.tolist(), 'joints_mm': (posed_joints @ axes.T+wrist).tolist(), 'wrist_mm': wrist.tolist(), 'frame_axes': axes.tolist(), 'scale_initial': float(ratio)}


def trial(scan, report_path=None, output_folder=None, part_names=None):
    if report_path is None:
        base = json.loads((ROOT/'outputs/experiments/hand_branches'/f'export_{scan:05d}'/'report.json').read_text())
    else:
        report_path = Path(report_path)
        upstream = json.loads(report_path.read_text(encoding='utf-8'))
        base = {'source_mesh': str(report_path.parent/upstream['artifacts']['ply']['name']), 'source_cloud': upstream['source']}
    source = Path(base['source_mesh']); before = o3d.io.read_triangle_mesh(str(source)); before.compute_vertex_normals()
    vertices = np.asarray(before.vertices)
    positions_path = source.parent/'body_parts.json' if report_path is None else report_path.parent/'body_parts.json'
    info = json.loads(positions_path.read_text(encoding='utf-8'))
    frames_path = (ROOT/'outputs/experiments/hand_auto'/f'export_{scan:05d}'/'hand_centerlines.json') if report_path is None else report_path.parent/'hand_centerlines.json'
    frames = json.loads(frames_path.read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() in (frames['source_mesh_sha256'], frames.get('output_mesh_sha256'))
    cloud = o3d.io.read_point_cloud(base['source_cloud']); raw = np.asarray(cloud.points).copy()
    if report_path is not None and upstream['fingerprint']['unit'] == 'm':raw *= 1000
    out = ROOT/'outputs/experiments/mano_fit'/f'export_{scan:05d}' if output_folder is None else Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    canvas = Image.new('RGB', (1400, 1520), 'white'); draw = ImageDraw.Draw(canvas); records = []
    for side, frame in enumerate(frames['parts']):
        if part_names is not None and frame['part'] not in part_names:continue
        name = frame['part']; model = Mano(name.split('_')[0])
        fitted, record = fit_hand(model, vertices, raw, info['parts'], frame)
        ids, arm, wrist, axes = hand_frame(vertices, info['parts'], name)
        local = (vertices-wrist) @ axes
        lo, hi = local[ids].min(0)-[12, 25, 15], local[ids].max(0)+12
        chosen = np.flatnonzero(np.all((local >= lo) & (local <= hi), axis=1) & (cKDTree(vertices[ids]).query(vertices)[0] < 28))
        old = before.select_by_index(chosen)
        # Vertex colours are transferred for a preview; this is not photo texturing.
        nearest = cKDTree(vertices).query(np.asarray(fitted.vertices))[1]
        fitted.vertex_colors = o3d.utility.Vector3dVector(np.asarray(before.vertex_colors)[nearest])
        o3d.io.write_triangle_mesh(str(out/(name+'_template.ply')), fitted)
        center = wrist+axes @ ((lo+hi)/2); span = float(max(hi-lo)*1.18)
        for angle in (0, 1):
            right, up = (axes[:, 0], axes[:, 1]) if angle == 0 else (axes[:, 2], axes[:, 1])
            for column in range(4):
                mesh = copy.deepcopy(old if column % 2 == 0 else fitted)
                if column >= 2: mesh.paint_uniform_color([.72]*3)
                row = side*2+angle
                canvas.paste(panel(mesh, center, right, up, span, 350, colored=column < 2), (column*350, row*380+25))
                draw.text((column*350+4, row*380+5), name+(' | Original' if column % 2 == 0 else ' | MANO fitted'), fill='black')
        records.append(record); print(name, record['candidates'], flush=True)
    canvas.save(out/'comparison.jpg', quality=95)
    report = {'source_mesh': str(source), 'source_positions': str(positions_path), 'source_cloud': base['source_cloud'], 'method': 'MANO parametric template fit; not image-network inference', 'parts': records, 'body_replaced': False, 'wrist_stitched': False}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=int, default=3)
    trial(parser.parse_args().scan)
