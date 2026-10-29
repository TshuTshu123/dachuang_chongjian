"""本轮诊断用：同视角灰色头部对比与原始观测支撑统计，不修改任何模型。"""
from pathlib import Path
import json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
for number in (1, 3):
    stem = f'export_{number:05d}'
    old_auto = sorted((ROOT / 'outputs/models' / stem).glob('20261029*/*_mesh.ply'))[0]
    paths = [('history', ROOT / 'outputs/history/scan' / f'{stem}_mesh.ply'),
             ('old auto', old_auto),
             ('scan baseline', OUT / f'baseline_{number}' / f'{stem}_mesh.ply'),
             ('candidate', OUT / f'candidate_{number}' / f'{stem}_mesh.ply')]
    paths = [(name, path) for name, path in paths if path.exists()]
    source = o3d.io.read_point_cloud(str(ROOT / 'data' / f'{stem}.ply'))
    tree = cKDTree(np.asarray(source.points))
    reference = o3d.io.read_triangle_mesh(str(paths[0][1]))
    top = np.asarray(reference.vertices)[:, 0].max()
    reference_points = np.asarray(reference.vertices)
    reference_head = reference_points[reference_points[:, 0] > top - 320]
    fig, axes = plt.subplots(2, len(paths), figsize=(4 * len(paths), 8), squeeze=False)
    metrics = {}
    for col, (name, path) in enumerate(paths):
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        d = tree.query(vertices, workers=4)[0]
        head = vertices[:, 0] >= top - 320
        body = (vertices[:, 0] < top - 320) & (vertices[:, 0] > reference_points[:, 0].min() + 200)
        metrics[name] = {'path': str(path), 'vertices': len(vertices), 'faces': len(faces),
                         'head_vertex_raw_p95_mm': float(np.quantile(d[head], .95)),
                         'body_vertex_raw_p95_mm': float(np.quantile(d[body], .95)),
                         'body_vertices_over_10mm_fraction': float(np.mean(d[body] > 10))}
        selected = vertices[faces].max(axis=1)[:, 0] > top - 320
        tri = vertices[faces[selected]]
        for row, (horizontal, depth, sign, view) in enumerate([(1, 2, -1, 'front -Z'), (2, 1, 1, 'side +Y')]):
            ax = axes[row, col]
            order = np.argsort(sign * tri[:, :, depth].mean(axis=1))
            normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
            light = np.zeros(3); light[0] = .4; light[horizontal] = -.3; light[depth] = sign
            light /= np.linalg.norm(light)
            intensity = .25 + .7 * np.abs(normals @ light)
            colors = np.repeat(intensity[:, None], 3, axis=1)
            polygons = tri[:, :, [horizontal, 0]]
            ax.add_collection(PolyCollection(polygons[order], facecolors=colors[order], edgecolors='none', antialiased=False))
            center = (reference_head[:, horizontal].min() + reference_head[:, horizontal].max()) / 2
            ax.set_xlim(center - 170, center + 170)
            ax.set_ylim(top - 320, top + 15)
            ax.set_aspect('equal'); ax.set_facecolor('#edf0f5'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{name} | {view}', fontsize=12)
    fig.suptitle(f'{stem}: identical orthographic views, flat gray shading (not accuracy)', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / f'{stem}_comparison.png', dpi=150)
    plt.close(fig)
    (OUT / f'{stem}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(stem, json.dumps(metrics), flush=True)
