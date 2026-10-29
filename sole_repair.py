"""已验证的鞋底截断补面工具，由 run.py 调用，不单独执行。
输入网格单位 mm，保持切口上方几何和整体朝向，不处理手掌。
0001 使用已确认配置；新数据默认尝试按主体范围估计底面，属于待验证测试功能。
底面为近似平面，保留原有两脚连接，不保证真实底纹或可打印性。
"""
import numpy as np
import open3d as o3d
import copy

APPROVED_SOLES = {
    "318b37496e22b0b2d0b622f57fd24bd85858fc2466c64cd0986ded533ebafed9": -377.9881286621094,
}


def shoe_plan(base_report, source_hash, enabled=True, axis='auto', up_sign=1):
    """Use filtered-subject bounds, never scene/background raw extrema."""
    if not enabled:
        return {'enabled': False, 'reason': 'Explicitly disabled.'}
    if axis not in ('auto', 'x', 'y', 'z') or up_sign not in (-1, 1):
        raise ValueError('Invalid shoe axis/direction')
    extent = np.asarray(base_report['extent_m']) * 1000
    origin = np.asarray(base_report['meter_coordinate_origin']) * 1000
    if extent.shape != (3,) or origin.shape != (3,) or not np.isfinite(np.r_[extent, origin]).all():
        raise ValueError('Invalid subject bounds for shoe estimation')
    order = np.argsort(extent)
    if axis == 'auto' and extent[order[-1]] < 1.3 * extent[order[-2]]:
        return {'enabled': False, 'reason': 'Vertical axis ambiguous; specify --shoe-axis x/y/z or disable repair.'}
    index = int(order[-1]) if axis == 'auto' else 'xyz'.index(axis)
    approved = source_hash in APPROVED_SOLES and index == 0 and up_sign == 1
    floor = APPROVED_SOLES[source_hash] if approved else float(up_sign * origin[index] - extent[index] / 2 - 3)
    return {'enabled': True, 'axis': 'xyz'[index], 'up_sign': up_sign, 'floor_up_mm': floor,
            'profile': 'user_approved_export_00001' if approved else 'new_scan_experimental',
            'experimental': not approved, 'margin_mm': 3.,
            'assumption': 'Standing subject, positive-up by default. Not semantic foot detection.'}


def apply_shoe_plan(mesh, plan):
    """Transform only for clipping, restore original axes/winding afterward."""
    if not plan['enabled']:
        return mesh, dict(plan)
    axis = 'xyz'.index(plan['axis'])
    perm = [axis] + [i for i in range(3) if i != axis]
    points = np.asarray(mesh.vertices)[:, perm].copy()
    points[:, 0] *= plan['up_sign']
    floor = plan['floor_up_mm']
    fraction = float(np.mean(points[:, 0] < floor))
    if fraction == 0 or fraction > .08:
        return mesh, {**plan, 'enabled': False, 'reason': 'No excess below plane, or cut exceeds 8% of vertices.',
                      'proposed_cut_fraction': fraction}
    transformed = copy.deepcopy(mesh)
    transformed.vertices = o3d.utility.Vector3dVector(points)
    repaired, result = cut_and_cap(transformed, floor)
    restored = np.asarray(repaired.vertices).copy()
    restored[:, 0] *= plan['up_sign']
    repaired.vertices = o3d.utility.Vector3dVector(restored[:, np.argsort(perm)])
    repaired.compute_vertex_normals()
    return repaired, {**result, **plan, 'proposed_cut_fraction': fraction}

def signed2(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def triangulate(loop, vertices):
    """Ear clipping, retaining collinear boundary vertices to avoid T junctions."""
    poly = np.asarray(vertices)[loop, 1:3]
    area2 = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    direction = 1 if area2 > 0 else -1
    remaining = list(range(len(loop)))
    result = []
    while len(remaining) > 3:
        found = False
        for k, b in enumerate(remaining):
            a, c = remaining[k-1], remaining[(k+1) % len(remaining)]
            if direction * signed2(poly[a], poly[b], poly[c]) <= 1e-9:
                continue
            others = [i for i in remaining if i not in (a, b, c)]
            p = poly[others]
            ab = direction * signed2(poly[a], poly[b], p.T)
            bc = direction * signed2(poly[b], poly[c], p.T)
            ca = direction * signed2(poly[c], poly[a], p.T)
            if np.any((ab >= -1e-9) & (bc >= -1e-9) & (ca >= -1e-9)):
                continue
            result.append([loop[a], loop[b], loop[c]])
            remaining.pop(k)
            found = True
            break
        if not found:
            raise RuntimeError('Cut boundary cannot be triangulated safely; no final candidate written.')
    if abs(signed2(*poly[remaining])) <= 1e-9:
        raise RuntimeError('Degenerate final cap triangle')
    result.append([loop[i] for i in remaining])
    return result


def cut_and_cap(mesh, floor):
    v = np.asarray(mesh.vertices).copy()
    rgb = np.asarray(mesh.vertex_colors).copy()
    faces = np.asarray(mesh.triangles)
    vertices, colors = v.tolist(), rgb.tolist()
    inside = v[:, 0] >= floor
    all_in, all_out = inside[faces].all(1), (~inside[faces]).all(1)
    result = faces[all_in].tolist()
    cache = {}

    def intersection(a, b):
        edge = tuple(sorted((int(a), int(b))))
        if edge not in cache:
            ratio = (floor - v[a, 0]) / (v[b, 0] - v[a, 0])
            point = v[a] + ratio * (v[b] - v[a]); point[0] = floor
            cache[edge] = len(vertices)
            vertices.append(point.tolist())
            colors.append((rgb[a] + ratio * (rgb[b] - rgb[a])).tolist())
        return cache[edge]

    for tri in faces[~all_in & ~all_out]:
        polygon = []
        for a, b in zip(tri, np.roll(tri, -1)):
            if inside[a]:
                polygon.append(int(a))
            if inside[a] != inside[b]:
                polygon.append(intersection(a, b))
        for k in range(1, len(polygon) - 1):
            result.append([polygon[0], polygon[k], polygon[k+1]])
    f = np.asarray(result, dtype=np.int32)
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse] == 1]
    vertices_array = np.asarray(vertices)
    cut_edges = boundary[np.all(np.abs(vertices_array[boundary, 0] - floor) < 1e-7, axis=1)]
    if len(cut_edges) > 2000:
        raise RuntimeError('Cut contour exceeds 2000 edges; skipping expensive/complex cap')
    next_vertex = {int(a): int(b) for a, b in cut_edges}
    if len(next_vertex) != len(cut_edges) or len(set(next_vertex.values())) != len(cut_edges):
        raise RuntimeError('Cut boundary is not a set of simple oriented loops')
    loops, caps = [], []
    while next_vertex:
        start = next(iter(next_vertex))
        loop = [start]
        current = next_vertex.pop(start)
        while current != start:
            loop.append(current)
            if current not in next_vertex:
                raise RuntimeError('Open cut contour')
            current = next_vertex.pop(current)
        if len(loop) < 3:
            raise RuntimeError('Invalid cut loop')
        loops.append(loop)
        caps.extend(triangulate(loop[::-1], vertices))
    if len(loops) not in (1, 2):
        raise RuntimeError(f'Unexpected number of cut contours: {len(loops)}')
    output = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices_array),
                                      o3d.utility.Vector3iVector(np.asarray(result + caps)))
    output.vertex_colors = o3d.utility.Vector3dVector(colors)
    output.remove_unreferenced_vertices()
    output.compute_vertex_normals()
    return output, {'floor_x_mm': floor, 'removed_original_vertices': int((~inside).sum()),
                     'new_cut_vertices': len(cache), 'closed_shoe_contours': len(loops), 'cap_triangles': len(caps),
                     'note': 'Planar inferred caps, not measured soles or print certification. A shared contour retains existing foot connectivity; feet are not separated.'}
