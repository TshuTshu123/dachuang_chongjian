"""本批数据局部诊断：原始观测与网格支撑分布，仅输出对比图与统计。"""
from pathlib import Path
import json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from surface_detail import select_main_point_component

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
cloud = o3d.io.read_point_cloud(str(ROOT / 'data/export_00001.ply'))
cloud, _, _ = select_main_point_component(cloud, 12, proxy_voxel=4)
raw = np.asarray(cloud.points)
rgb = np.asarray(cloud.colors)
mesh = o3d.io.read_triangle_mesh(str(ROOT / 'outputs/models/export_00001/20261029_214717_112915/export_00001_mesh.ply'))
v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
d = cKDTree(raw).query(v, workers=4)[0]
regions = [('feet', [-500, -160, 900], [-230, 150, 1250]),
           ('hand Y negative', [160, -270, 900], [440, -140, 1250]),
           ('hand Y positive', [160, 140, 900], [440, 280, 1250])]
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
stats = {}
for row, (name, low, high) in enumerate(regions):
    low, high = np.array(low), np.array(high)
    inside = lambda p: np.all((p >= low) & (p <= high), axis=1)
    keep = inside(raw); vertex_keep = inside(v)
    tri = v[f[np.all(vertex_keep[f], axis=1)]]
    tri_d = d[f[np.all(vertex_keep[f], axis=1)]].max(axis=1)
    stats[name] = {'raw_points': int(keep.sum()), 'mesh_vertices': int(vertex_keep.sum()),
                   'nearest_raw_distance_mm': np.quantile(d[vertex_keep], [.5, .95, 1]).tolist(),
                   'raw_min_x_mm': float(raw[keep, 0].min()), 'mesh_min_x_mm': float(v[vertex_keep, 0].min())}
    for col, (horizontal, vertical, depth) in enumerate([(1, 0, 2), (2, 0, 1)]):
        ax = axes[row, col * 2]
        order = np.argsort(raw[keep, depth])
        ax.scatter(raw[keep, horizontal][order], raw[keep, vertical][order], c=rgb[keep][order], s=2)
        ax.set_title(f'{name}: raw {"YX" if col == 0 else "ZX"}')
        ax = axes[row, col * 2 + 1]
        colors = np.tile([.6, .65, .7], (len(tri), 1))
        colors[tri_d > 6] = [1., .3, .1]
        order = np.argsort(tri[:, :, depth].mean(1))
        ax.add_collection(PolyCollection(tri[:, :, [horizontal, vertical]][order], facecolors=colors[order], edgecolors='none'))
        ax.set_title(f'mesh: orange > 6 mm from raw')
        for a in axes[row, col * 2:col * 2 + 2]:
            a.set_xlim(low[horizontal], high[horizontal]); a.set_ylim(low[vertical], high[vertical]); a.set_aspect('equal'); a.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUT / 'raw_mesh_support.png', dpi=160)
(OUT / 'region_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
print(json.dumps(stats, indent=2))
