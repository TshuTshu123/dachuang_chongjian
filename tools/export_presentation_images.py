"""从真实 0000 点云和既有模型导出答辩配图，中文图注，不修改几何或 PPT。

python tools/export_presentation_images.py --out "输出目录"
点云预处理复用正式模型报告的参数；原始、处理后使用同一正交相机和点大小。
科学数据渲染：不补造表面、纹理或细节，不使用生成式图像。
"""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from body_parts import file_hash
from surface_detail import denoise_local_planes, select_main_point_component

UP = np.array([1., 0., 0.])
FRONT = np.array([0., 1., 0.])
FONT = Path('C:/Windows/Fonts/msyh.ttc')
BOLD = Path('C:/Windows/Fonts/msyhbd.ttc')


def mesh_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def hand_crop(mesh, points):
    """Isolate the actual hand surface; omit neighboring head/body fragments."""
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    inside = np.all((v >= points.min(0)-10) & (v <= points.max(0)+10), axis=1)
    cropped = copy.deepcopy(mesh)
    cropped.remove_triangles_by_mask(~np.all(inside[f], axis=1))
    labels, counts, _ = cropped.cluster_connected_triangles()
    cropped.remove_triangles_by_mask(np.asarray(labels) != int(np.argmax(counts)))
    cropped.remove_unreferenced_vertices()
    cropped.compute_vertex_normals()
    return cropped


def mesh_image(mesh, scene, center, right, height, size, colored=True, transparent=False):
    width, rows = size
    scale = 1.5
    w, h = int(width*scale), int(rows*scale)
    forward = np.cross(UP, right)
    light = -forward + .35*UP + .15*right
    light /= np.linalg.norm(light)
    xx = np.linspace(-height*w/h/2, height*w/h/2, w)
    yy = np.linspace(height/2, -height/2, h)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)
    colors = np.asarray(mesh.vertex_colors)
    pixels = np.full((h, w, 4), 255, np.uint8)
    if transparent:
        pixels[:, :, 3] = 0
    for start in range(0, h, 128):
        stop = min(h, start+128)
        x, y = np.meshgrid(xx, yy[start:stop])
        origins = center + x[..., None]*right + y[..., None]*UP - forward*4000
        rays = np.c_[origins.reshape(-1, 3), np.broadcast_to(forward, (x.size, 3))].astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)
        valid = np.isfinite(hit['t_hit'].numpy())
        ids = hit['primitive_ids'].numpy()[valid].astype(int)
        uv = hit['primitive_uvs'].numpy()[valid]
        bary = np.c_[1-uv.sum(1), uv]
        normal = np.einsum('ni,nij->nj', bary, normals[faces[ids]])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        lighting = np.abs(normal @ light)
        if colored:
            rgb = np.einsum('ni,nij->nj', bary, colors[faces[ids]]) * (.78+.22*lighting[:, None])
        else:
            rgb = (.42+.58*lighting[:, None]) * np.array([.73, .77, .82])
        part = pixels[start:stop].reshape(-1, 4)
        part[valid, :3] = np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8)
        part[valid, 3] = 255
    return Image.fromarray(pixels).resize(size, Image.Resampling.LANCZOS)


def point_image(cloud, center, right, height, size, radius=.65):
    width, rows = size
    ss = 2
    w, h = width*ss, rows*ss
    forward = np.cross(UP, right)
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    local = points-center
    x = np.rint((local@right/(height*w/h)+.5)*(w-1)).astype(int)
    y = np.rint((.5-local@UP/height)*(h-1)).astype(int)
    depth = (local@forward).astype(np.float32)
    r = radius*ss
    reach = int(np.ceil(r))
    offsets = [(a, b) for a in range(-reach, reach+1) for b in range(-reach, reach+1) if a*a+b*b <= r*r]
    zbuffer = np.full(w*h, np.inf, np.float32)
    rgb = np.full((w*h, 3), 255, np.uint8)
    projected = []
    for dx, dy in offsets:
        a, b = x+dx, y+dy
        valid = (a>=0)&(a<w)&(b>=0)&(b<h)
        ids = np.flatnonzero(valid)
        pixel = b[ids]*w+a[ids]
        np.minimum.at(zbuffer, pixel, depth[ids])
        projected.append((ids, pixel))
    values = np.rint(np.clip(colors, 0, 1)*255).astype(np.uint8)
    for ids, pixel in projected:
        visible = depth[ids] == zbuffer[pixel]
        rgb[pixel[visible]] = values[ids[visible]]
    return Image.fromarray(rgb.reshape(h, w, 3)).resize(size, Image.Resampling.LANCZOS)


def centered(draw, center_x, y, text, font, color='#252525'):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((int(center_x-(box[2]-box[0])/2), y), text, font=font, fill=color)


def figure(images, labels, path, details=None, size=(2400, 1360)):
    w, h = size
    canvas = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(canvas)
    col = w/len(images)
    label_font = ImageFont.truetype(str(BOLD), 48)
    small = ImageFont.truetype(str(FONT), 33)
    for index, (im, label) in enumerate(zip(images, labels)):
        panel_w, panel_h = int(col)-44, h-185
        scaled = im.copy()
        scaled.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
        canvas.paste(scaled, (int(col*index+(col-scaled.width)/2), 10+(panel_h-scaled.height)//2), scaled if scaled.mode=='RGBA' else None)
        centered(draw, col*(index+.5), h-150, label, label_font)
        if details:
            centered(draw, col*(index+.5), h-77, details[index], small, '#666666')
    canvas.save(path, dpi=(200, 200))


def export(out):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    single = out/'单张无字图'
    single.mkdir(exist_ok=True)
    rp = next(p for p in sorted((ROOT/'outputs/models/export_00000').glob('*/report.json'), reverse=True)
              if (p.parent/'export_00000_mesh.ply').is_file() and (p.parent/'body_parts.json').is_file())
    report = json.loads(rp.read_text(encoding='utf-8'))
    model = rp.parent/report['artifacts']['ply']['name']
    raw_path = Path(report['source'])
    raw_hash, model_hash = file_hash(raw_path), file_hash(model)
    assert raw_hash == report['fingerprint']['source_sha256']
    assert model_hash == report['artifacts']['ply']['sha256']
    info = json.loads((rp.parent/'body_parts.json').read_text(encoding='utf-8'))
    assert info['mesh']['sha256'] == model_hash
    mesh = o3d.io.read_triangle_mesh(str(model))
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    raw = o3d.io.read_point_cloud(str(raw_path))
    raw_count = len(raw.points)
    source = copy.deepcopy(raw)
    source.remove_non_finite_points()
    source.remove_duplicated_points()
    source.scale(.001 if report['fingerprint']['unit']=='mm' else 1., center=np.zeros(3))
    settings = report['base_report']['settings']
    pcd = source.voxel_down_sample(settings['voxel'])
    voxel_count = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=settings['outlier_neighbors'], std_ratio=settings['outlier_std'])
    sor_count = len(pcd.points)
    if settings['point_components']=='largest':
        pcd, _, component = select_main_point_component(pcd, settings['point_cluster_radius_mm']*.001)
    else:
        component = {'removed_points': 0}
    origin = (pcd.get_min_bound()+pcd.get_max_bound())/2
    pcd.translate(-origin)
    before = copy.deepcopy(pcd)
    xyz, denoise = denoise_local_planes(np.asarray(pcd.points), settings['normal_max_nn'], settings['normal_radius'],
                                        4, settings['point_denoise_iterations'], settings['point_denoise_max_move_mm']*.001)
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.translate(origin).scale(1000, center=np.zeros(3))
    before.translate(origin).scale(1000, center=np.zeros(3))
    if report['fingerprint']['unit']=='m':
        raw.scale(1000, center=np.zeros(3))
    counts = {'input_points': raw_count, 'voxel_points': voxel_count, 'sor_retained_points': sor_count, 'filtered_points': len(pcd.points)}
    for key, value in counts.items():
        assert value == report['base_report'][key], (key, value, report['base_report'][key])
    print('Point processing verified', counts, flush=True)
    scene = mesh_scene(mesh)
    center = (v.min(0)+v.max(0))/2
    height = np.ptp(v, axis=0)[0]*1.07
    pair_size = (1120, 1170)
    full = [point_image(c, center, FRONT, height, pair_size) for c in (raw, pcd)]
    figure(full, ['原始点云', '去噪点云'], out/'01_原始点云与去噪点云_全身对比.png',
           [f'{raw_count:,} 个点', f'{len(pcd.points):,} 个点'])
    for im, label in zip(full, ('原始点云_正面', '去噪点云_正面')):
        im.save(single/(label+'.png'), dpi=(200, 200))
    head_ids = np.r_[info['parts']['face_neck']['vertex_indices'], info['parts']['hair']['vertex_indices']]
    head = v[head_ids]
    hc = (head.min(0)+head.max(0))/2
    hh = np.ptp(head, axis=0).max()*1.15
    zoom = [point_image(c, hc, FRONT, hh, pair_size, radius=1.25) for c in (raw, pcd)]
    figure(zoom, ['原始点云（头颈局部）', '去噪点云（头颈局部）'], out/'02_原始点云与去噪点云_局部对比.png')
    # Controlled denoising-only comparison: identical selected points before/after position fitting.
    controlled = [point_image(c, hc, FRONT, hh, pair_size, radius=1.25) for c in (before, pcd)]
    figure(controlled, ['局部平面去噪前', '局部平面去噪后'], out/'03_局部平面去噪_同点数对比.png',
           [f'主体点云 {len(pcd.points):,} 个点', '点数不变，累计位移不超过 1 毫米'])
    views = [(FRONT, '正面'), (np.array([0., 0., 1.]), '侧面'), (-FRONT, '背面')]
    for colored, name, number in ((True, '彩色模型', '04'), (False, '灰色模型', '05')):
        images = []
        for right, label in views:
            im = mesh_image(mesh, scene, center, right, height, (760, 1170), colored, transparent=True)
            im.save(single/(name+'_'+label+'.png'), dpi=(200, 200))
            images.append(im)
        figure(images, [f'{name}（{label}）' for _, label in views], out/(number+'_'+name+'_三视图.png'))
    closeups = [mesh_image(mesh, scene, hc, FRONT, hh, (760, 1170), False, transparent=True)]
    closeups[0].save(single/'头颈灰模_无字.png', dpi=(200, 200))
    for name, label in (('left_hand', '左手'), ('right_hand', '右手')):
        q = v[info['parts'][name]['vertex_indices']]
        c = (q.min(0)+q.max(0))/2
        span = np.ptp(q, axis=0).max()*1.3
        local_mesh = hand_crop(mesh, q)
        im = mesh_image(local_mesh, mesh_scene(local_mesh), c, FRONT, span, (760, 1170), False, transparent=True)
        im.save(single/(label+'灰模_无字.png'), dpi=(200, 200))
        closeups.append(im)
    figure(closeups, ['头颈部细节', '左手细节', '右手细节'], out/'06_重建模型_局部细节.png')
    # Data-backed semantic visualization; unknown vertices remain neutral gray.
    labels_mesh = copy.deepcopy(mesh)
    colors = np.full_like(v, .65)
    palette = {'hair': [.36,.12,.31], 'face_neck': [.75,.76,.81], 'upper_clothing': [.17,.58,.38],
               'lower_clothing': [.3,.28,.7], 'left_hand': [.2,.78,.3], 'right_hand': [.88,.25,.65],
               'left_shoe': [.85,.66,.2], 'right_shoe': [.85,.66,.2]}
    for name, rgb in palette.items():
        colors[info['parts'].get(name, {}).get('vertex_indices', [])] = rgb
    labels_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    labels_image = mesh_image(labels_mesh, scene, center, FRONT, height, (760, 1170), True, transparent=True)
    hands_mesh = copy.deepcopy(mesh)
    colors = np.asarray(mesh.vertex_colors).copy()
    for name in ('left_hand', 'right_hand'):
        colors[info['parts'][name]['vertex_indices']] = palette[name]
    hands_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    hands_image = mesh_image(hands_mesh, scene, center, FRONT, height, (760, 1170), True, transparent=True)
    original_image = mesh_image(mesh, scene, center, FRONT, height, (760, 1170), True, transparent=True)
    figure([original_image, labels_image, hands_image], ['彩色重建模型', '三维部位标注', '左右手区域标注'], out/'07_模型与三维部位标注.png')
    labels_image.save(single/'三维部位标注_正面.png', dpi=(200, 200))
    hands_image.save(single/'左右手区域标注_正面.png', dpi=(200, 200))
    assert file_hash(raw_path)==raw_hash and file_hash(model)==model_hash
    manifest = {'input_id':'export_00000', 'model':str(model), 'model_sha256':model_hash,
                'model_pipeline_version':report['fingerprint']['version'], 'raw':str(raw_path), 'raw_sha256':raw_hash,
                'counts':counts, 'point_denoise':denoise, 'component_selection':component,
                'render':'Orthographic CPU rendering of original coordinates and colors; interpolated vertex-normal lighting for meshes; identical camera and point size for each point pair.',
                'geometry_changed_for_presentation':False, 'labels':'Chinese below images; transparent uncaptained model PNGs also supplied.',
                'three_dimensional_labels_note':'3D transported labels, not 2D segmentation predictions. Unknown regions remain gray.',
                'png_files':[str(p.relative_to(out)) for p in sorted(out.rglob('*.png'))],
                'elapsed_seconds':round(time.perf_counter()-start,2)}
    (out/'来源与参数.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'配图使用说明.txt').write_text('0000 展示配图\n\n01：原始与完整预处理后点云，可替换 PPT 第 4 页图片。两边相机、尺度、点大小相同。\n02：同一头颈区域的原始与去噪点云。\n03：已完成降采样、离群点清理、主体提取后的同点数对照，只观察局部平面去噪。\n04、05：现有 0000 模型的彩色与灰模正面、侧面、背面。\n06：头颈与手部局部细节。\n07：三维部位与双手区域标注；如果替换原来的二维预测图，标题应改为“三维部位标注”。\n单张无字图：可独立插入 PPT；模型图背景透明。\n\n所有带字图均使用中文图下标注，主图 2400×1360。保留当前 0000 的既有模型几何，未为展示加平滑或生成缺失细节。\n点数减少同时包含降采样、离群点清理和主体提取，不能全部算作去噪效果；局部平面去噪不减少点数。\n本次未修改原始点云或正式模型。\n',encoding='utf-8')
    print('DONE',str(out),len(manifest['png_files']),'PNGs',manifest['elapsed_seconds'],'seconds',flush=True)


if __name__=='__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/presentation_assets/export_00000/deliverables')
    export(parser.parse_args().out)
