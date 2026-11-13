"""比较 Meta 局部点云经多视角校正前后的 Poisson 网格。

python tools/multiview_geometry_trial.py --scan 1
加 --relaxed 对比 6 mm 邻域和每视角至少两个支持点。
两组重建使用相同参数和相同留出点。结果写入 outputs/experiments/multiview_geometry/。
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from surface_trials import ROOT, panel
from body_parts import file_hash
from surface_evaluation import metrics
from surface_quality import regions


PARTS = ('face_neck', 'left_hand', 'right_hand')
RADIUS_MM = 4.
MIN_NORMAL_COSINE = .75
MAX_MOVE_MM = .8


def capture_streams(report, points):
    raw = report['raw_color']
    folder = Path(raw['source'])
    hashes = report['fingerprint']['raw_color_input']['sha256']
    normals, streams, xyz = [], [], []
    for index, item in enumerate(raw['streams']):
        prefix = f"{item['stream']}/{raw['frame']}"
        for name in (f'{prefix}.ply', f'{prefix}_transform.txt'):
            if file_hash(folder / name) != hashes[str(Path(name))]:
                raise ValueError(f'RAW file changed since pairing: {name}')
        cloud = o3d.io.read_point_cloud(str(folder / f'{prefix}.ply'))
        local = np.asarray(cloud.points)
        normal = np.asarray(cloud.normals).copy()
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
        normal[np.einsum('ij,ij->i', normal, local) > 0] *= -1
        transform = np.asarray(item['transform'])
        xyz.append(local @ transform[:3, :3].T + transform[:3, 3])
        normals.append(normal @ transform[:3, :3].T)
        streams.append(np.full(len(local), index))
    aligned = np.concatenate(xyz)
    if aligned.shape != points.shape or np.max(np.linalg.norm(aligned - points, axis=1)) > .02:
        raise ValueError('This trial requires RAW points in merged PLY order.')
    return np.concatenate(normals), np.concatenate(streams)


def consensus_offsets(points, normals, streams, labels, train, influence, radius, neighbors):
    displacement = np.zeros(len(points))
    support_counts = np.zeros(len(points), dtype=int)
    for part in range(1, len(PARTS)+1):
        target = np.flatnonzero(train & (labels == part) & (influence > .1))
        if not len(target):
            continue
        votes = []
        for stream in np.unique(streams):
            donors = np.flatnonzero(train & (labels == part) & (streams == stream))
            if len(donors) < neighbors:
                continue
            distance, index = cKDTree(points[donors]).query(
                points[target], k=min(len(donors), 8), distance_upper_bound=radius, workers=4)
            index = donors[np.minimum(index, len(donors)-1)]
            dot = np.einsum('ijk,ik->ij', normals[index], normals[target])
            height = np.einsum('ijk,ijk->ij', points[index]-points[target,None], normals[index])
            height /= np.maximum(dot, MIN_NORMAL_COSINE)
            ok = ((streams[target,None] != stream) & np.isfinite(distance) &
                  (dot >= MIN_NORMAL_COSINE) & (np.abs(height) <= 2.))
            weights = np.where(ok, 1/(1+distance**2), 0.)
            vote = np.sum(weights*height, axis=1) / np.maximum(weights.sum(axis=1), 1e-12)
            votes.append(np.ma.array(vote, mask=ok.sum(axis=1)<neighbors))
        if not votes:
            continue
        votes = np.ma.stack(votes)
        count = votes.count(axis=0)
        support_counts[target] = count
        median = np.ma.median(votes,axis=0).filled(0)
        spread = (votes.max(axis=0)-votes.min(axis=0)).filled(np.inf)
        valid = (count>=2)&(spread<=1.5)&(np.abs(median)>=.15)
        displacement[target[valid]] = np.clip(.75*median[valid],-MAX_MOVE_MM,MAX_MOVE_MM)*influence[target[valid]]
    return displacement, support_counts


def reconstruct(path, cloud, out, settings):
    if not o3d.io.write_point_cloud(str(path),cloud):
        raise OSError(f'Cannot write {path}')
    options = ('voxel','depth','faces','normal_radius','normal_max_nn','normal_method',
               'point_denoise_iterations','point_denoise_max_move_mm','point_components',
               'point_cluster_radius_mm','outlier_neighbors','outlier_std','orient_k',
               'poisson_scale','trim_quantile','max_extrapolation_mm','smooth_iterations')
    command = [sys.executable,'-B',str(ROOT/'reconstruct_human.py'),str(path),
               '--out',str(out),'--input-unit','mm','--output-unit','mm','--no-vis',
               '--skip-self-intersections','--threads','4','--qa-samples','1000']
    for name in options:
        command.extend([f"--{name.replace('_','-')}",str(settings[name])])
    subprocess.run(command,check=True)
    return (o3d.io.read_triangle_mesh(str(out/f'{path.stem}_mesh.ply')),
            json.loads((out/f'{path.stem}_report.json').read_text(encoding='utf-8')))


def compare_meshes(before, after, reference, masks, out):
    vertices = np.asarray(reference.vertices)
    up = np.eye(3)[np.ptp(vertices,axis=0).argmax()]
    front = np.asarray(reference.vertex_normals)[masks[0]].mean(axis=0)
    front -= np.dot(front,up)*up
    if np.linalg.norm(front)<1e-8:
        front = -np.roll(up,-1)
    front /= np.linalg.norm(front)
    # The paired human captures have the body front toward negative rig Z.
    if front[2]>0:
        front = -front
    right = np.cross(up,front)
    views = [(0,right,'face front'),(0,np.cross(up,right),'face side'),
             (1,right,'left hand'),(2,right,'right hand')]
    canvas = Image.new('RGB',(720,390*len(views)),'white')
    draw = ImageDraw.Draw(canvas)
    for row,(part,axis,name) in enumerate(views):
        v = vertices[masks[part]]
        center = (v.min(axis=0)+v.max(axis=0))/2
        span = np.ptp(v,axis=0).max()*1.35
        for col,mesh in enumerate((before,after)):
            draw.text((col*360+8,row*390+6),f"{('Baseline','View consensus')[col]} | {name}",fill='black')
            canvas.paste(panel(mesh,center,axis,up,span,360),(col*360,row*390+25))
    canvas.save(out/'comparison.jpg',quality=94)


def run_trial(scan, relaxed=False):
    parent = ROOT/'outputs/models'/f'export_{scan:05d}'
    report_path = next(path for path in sorted(parent.glob('*/report.json'),reverse=True)
                       if json.loads(path.read_text(encoding='utf-8')).get('raw_color',{}).get('status')=='used')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    model = report_path.parent/report['artifacts']['ply']['name']
    sidecar = json.loads((report_path.parent/'body_parts.json').read_text(encoding='utf-8'))
    parts = sidecar['parts']
    if sidecar['mesh']['sha256'] != file_hash(model) or file_hash(report['source'])!=report['fingerprint']['source_sha256']:
        raise ValueError('The reference mesh or input PLY differs from the verified reconstruction.')
    reference = o3d.io.read_triangle_mesh(str(model))
    ref_xyz = np.asarray(reference.vertices)
    cloud = o3d.io.read_point_cloud(report['source'])
    points = np.asarray(cloud.points)
    normals,streams = capture_streams(report,points)
    tree = cKDTree(ref_xyz)
    distance,nearest = tree.query(points,workers=4)
    vertex_labels = np.zeros(len(ref_xyz),dtype=int)
    for i,name in enumerate(PARTS,1):
        vertex_labels[parts[name]['vertex_indices']] = i
    masks = [vertex_labels==i for i in range(1,len(PARTS)+1)]
    point_labels = np.where(distance<=5,vertex_labels[nearest],0)
    region_weight = np.maximum(regions(reference,parts,'face')[0],regions(reference,parts,'hands')[0])
    influence = region_weight[nearest]
    rng = np.random.default_rng(17)
    held_out = (point_labels>0)&(rng.random(len(points))<.2)
    train = ~held_out
    radius, neighbors = (6.,2) if relaxed else (RADIUS_MM,3)
    shifts,support = consensus_offsets(points,normals,streams,point_labels,train,influence,radius,neighbors)
    moved = np.flatnonzero(shifts)
    if not len(moved):
        raise ValueError('No points passed the view agreement checks.')
    print(f'Scan {scan}: {len(moved)} adjusted points, {held_out.sum()} withheld points.',flush=True)
    source = cloud.select_by_index(np.flatnonzero(train))
    candidate = o3d.geometry.PointCloud(source)
    candidate.points = o3d.utility.Vector3dVector((points+shifts[:,None]*normals)[train])
    out = ROOT/'outputs/experiments/multiview_geometry'/parent.name
    if relaxed:
        out = out/'relaxed'
    out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='multiview-',dir=out) as scratch:
        work = Path(scratch)
        baseline,base_report = reconstruct(work/'baseline.ply',source,work/'base',report['base_report']['settings'])
        improved,updated_report = reconstruct(work/'consensus.ply',candidate,work/'candidate',report['base_report']['settings'])
    result = {'reference':str(model),'holdout':'20% of RAW points assigned to the target regions; excluded from both reconstructions and the consensus votes.',
              'radius_mm':radius,'minimum_points_per_view':neighbors,'minimum_normal_cosine':MIN_NORMAL_COSINE,
              'maximum_observed_movement_mm':float(np.abs(shifts).max()),'regions':{}}
    for i,name in enumerate(PARTS,1):
        samples = points[held_out&(point_labels==i)]
        values = []
        for mesh in (baseline,improved):
            mapped = tree.query(np.asarray(mesh.vertices),workers=4)[1]
            region = vertex_labels[mapped]==i
            values.append(metrics(mesh,points,samples,None,region,region))
        result['regions'][name] = {'held_out_points':len(samples),
                                   'moved_points':int(np.sum((shifts!=0)&(point_labels==i))),
                                   'points_with_two_other_views':int(np.sum((support>=2)&(point_labels==i))),
                                   'baseline':values[0],'candidate':values[1]}
    keys = ('vertices','triangles','boundary_edges','edges_with_more_than_two_faces','orientation_succeeded')
    result['topology'] = {name:{key:r[key] for key in keys} for name,r in
                          (('baseline',base_report),('candidate',updated_report))}
    changes = [r['candidate']['raw_to_mesh_p95_mm']-r['baseline']['raw_to_mesh_p95_mm']
               for r in result['regions'].values()]
    result['passes_distance_gate'] = min(changes)<=-.10 and max(changes)<=.10
    result['passes_roughness_gate'] = all(
        r['candidate']['normal_angle_p90_deg'] <= 1.05*r['baseline']['normal_angle_p90_deg']
        for r in result['regions'].values())
    result['passes_topology_gate'] = (updated_report['orientation_succeeded']
        and updated_report['boundary_edges'] <= base_report['boundary_edges']
        and updated_report['edges_with_more_than_two_faces'] <= base_report['edges_with_more_than_two_faces'])
    result['selected'] = None
    selected = out/'selected_mesh.ply'
    if all(result[key] for key in ('passes_distance_gate','passes_roughness_gate','passes_topology_gate')):
        if not o3d.io.write_triangle_mesh(str(selected),improved):
            raise OSError(f'Cannot write {selected}')
        result['selected'] = str(selected)
    else:
        selected.unlink(missing_ok=True)
    compare_meshes(baseline,improved,reference,masks,out)
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan',type=int,required=True)
    parser.add_argument('--relaxed',action='store_true',help='Test a 6 mm radius and two supporting points per view.')
    args=parser.parse_args()
    run_trial(args.scan,args.relaxed)
