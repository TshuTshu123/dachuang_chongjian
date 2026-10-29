"""本轮候选的相同视角局部灰色比较图；不改任何模型。"""
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
paths = [ROOT / 'outputs/models/export_00001/20261029_214717_112915/export_00001_mesh.ply',
         OUT / 'export_00001_feet_hands_candidate.ply']
# Horizontal / vertical / depth and viewing sign in the original mm frame.
views = [('shoe side', [-480, -160, 900], [-230, 150, 1250], 2, 0, 1, -1),
         ('shoe underside', [-480, -160, 900], [-300, 150, 1250], 1, 2, 0, -1),
         ('hand Y negative', [220, -250, 970], [380, -155, 1130], 2, 0, 1, -1),
         ('hand Y positive', [240, 150, 960], [400, 250, 1110], 1, 0, 2, -1)]
fig, axes = plt.subplots(4, 2, figsize=(10, 14))
for col, path in enumerate(paths):
    mesh = o3d.io.read_triangle_mesh(str(path))
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    for row, (name, low, high, horizontal, vertical, depth, sign) in enumerate(views):
        low, high = np.array(low), np.array(high)
        c = v[f].mean(1)
        selected = np.all((c >= low) & (c <= high), axis=1)
        tri = v[f[selected]]
        normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
        light = np.zeros(3); light[depth] = sign; light[vertical] = .4; light[horizontal] = -.3
        light /= np.linalg.norm(light)
        shade = .25 + .7 * np.abs(normal @ light)
        order = np.argsort(sign * tri[:, :, depth].mean(1))
        ax = axes[row, col]
        ax.add_collection(PolyCollection(tri[:, :, [horizontal, vertical]][order],
                                         facecolors=np.repeat(shade[:, None], 3, axis=1)[order], edgecolors='none', antialiased=False))
        ax.set_xlim(low[horizontal], high[horizontal]); ax.set_ylim(low[vertical], high[vertical]); ax.set_aspect('equal')
        ax.set_facecolor('#edf0f5'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{name}: {"baseline" if col == 0 else "local candidate"}')
fig.suptitle('export_00001 local trial: identical gray views, not measured accuracy')
fig.tight_layout()
fig.savefig(OUT / 'local_candidate_comparison.png', dpi=160)
