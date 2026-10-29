"""本轮诊断：只翻回指定旧输出的面朝向及法线，不平滑、不移动顶点、不改颜色。"""
from pathlib import Path
import json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'outputs/models/export_00001/20261029_212248_041159/export_00001_mesh.ply'
OUT = Path(__file__).resolve().parent / 'export_00001_normal_only.ply'
mesh = o3d.io.read_triangle_mesh(str(SOURCE))
vertices = np.asarray(mesh.vertices).copy()
colors = np.asarray(mesh.vertex_colors).copy()
triangles = np.asarray(mesh.triangles).copy()
normals = np.asarray(mesh.vertex_normals).copy()
mesh.triangles = o3d.utility.Vector3iVector(triangles[:, [0, 2, 1]].copy())
mesh.vertex_normals = o3d.utility.Vector3dVector(-normals)
assert o3d.io.write_triangle_mesh(str(OUT), mesh)
check = o3d.io.read_triangle_mesh(str(OUT))
np.testing.assert_array_equal(np.asarray(check.vertices), vertices)
np.testing.assert_array_equal(np.asarray(check.vertex_colors), colors)
np.testing.assert_array_equal(np.asarray(check.triangles), triangles[:, [0, 2, 1]])
np.testing.assert_array_equal(np.asarray(check.vertex_normals), -normals)
history = o3d.io.read_triangle_mesh(str(ROOT / 'outputs/history/scan/export_00001_mesh.ply'))
distance, nearest = cKDTree(np.asarray(history.vertices)).query(vertices, workers=4)
head = vertices[:, 0] > np.asarray(history.vertices)[:, 0].max() - 320
dot = np.einsum('ij,ij->i', -normals, np.asarray(history.vertex_normals)[nearest])
report = {'source': str(SOURCE), 'output': str(OUT), 'changes': 'triangle winding and normal signs only',
          'vertex_positions_bitwise_identical': True, 'vertex_colors_bitwise_identical': True,
          'faces': len(triangles), 'head_nearest_history_p95_mm': float(np.quantile(distance[head], .95)),
          'head_normal_dot_history_median': float(np.median(dot[head])),
          'note': 'Matches history global orientation for display comparison, not a geometric detail enhancement.'}
(OUT.parent / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
