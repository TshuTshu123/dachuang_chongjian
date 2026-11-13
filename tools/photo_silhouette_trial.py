"""用已标定的主相机照片核对脸颈和手部轮廓。

python tools/photo_silhouette_trial.py --scan 1 2
调用现有 Meta 分割环境。输出叠加图和报告，符合检查的剪裁候选另存 PLY。
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from body_parts import DEFAULT_GPU_PYTHON, file_hash
from raw_capture import primary_projection
from reconstruct_human import geometry_report

SIZE = (768,1024)
PARTS = ('face_neck','left_hand','right_hand')
STREAMS = {'1','2','3','5','6','7','8','11','12','14','16','24'}
MIN_PIXEL_MARGIN = 6.
MIN_VIEW_ANGLE = 20.
RAW_GAP_MM = 5.


def rotate_pixels(points,width,height,turn):
    x,y = points.T
    return np.column_stack(((x,y),(y,width-1-x),(width-1-x,height-1-y),(height-1-y,x))[turn])


def project(points,view):
    uv,inside = primary_projection((points-view['offset'])@view['rotation'],view['camera'])
    uv = rotate_pixels(uv,*view['image_size'],view['turn'])
    uv = (uv-view['crop'][:2]+.5)*view['scale']-.5+view['padding']
    left,top,right,bottom = view['content']
    inside &= ((uv[:,0]>=left+10)&(uv[:,0]<right-10)&
               (uv[:,1]>=top+10)&(uv[:,1]<bottom-10))
    return uv,inside


def prepare_view(capture,entry,folder):
    stream = entry['stream']
    raw = capture['report']['raw_color']
    base = Path(raw['source'])/stream
    frame = raw['frame']
    ply,params,photo = [base/f'{frame}{suffix}' for suffix in
                        ('.ply','_CameraParams_Primary.json','_RGB_01.png')]
    hashes = capture['report']['fingerprint']['raw_color_input']['sha256']
    for path in (ply,params,photo):
        if file_hash(path)!=hashes[str(path.relative_to(raw['source']))]:
            raise ValueError(f'RAW file changed: {path}')
    camera = json.loads(params.read_text(encoding='utf-8'))['color']
    matrix = np.asarray(entry['transform'])
    R,t = matrix[:3,:3],matrix[:3,3]
    local = np.asarray(o3d.io.read_point_cloud(str(ply)).points).copy()
    world = local@R.T+t
    center = np.median(world,axis=0)
    test,_ = primary_projection((np.vstack((center,center+100*capture['up']))-t)@R,camera)
    if not np.isfinite(test).all() or np.linalg.norm(test[1]-test[0])<40:
        return None
    with Image.open(photo) as src:
        photo_image = src.convert('RGB')
    width,height = photo_image.size
    turn = int(np.argmin([np.diff(rotate_pixels(test,width,height,k),axis=0)[0,1]
                          for k in range(4)]))
    if turn:
        photo_image = photo_image.transpose((None,Image.Transpose.ROTATE_90,Image.Transpose.ROTATE_180,
                                             Image.Transpose.ROTATE_270)[turn])
    image_size = (width,height)
    raw_uv,seen = primary_projection(local,camera)
    if seen.sum()<100:
        return None
    xy = rotate_pixels(raw_uv[seen],*image_size,turn)
    lower,upper = np.percentile(xy,[.5,99.5],axis=0)
    margin = np.maximum(40,.13*(upper-lower))
    crop = np.r_[np.maximum(0,np.floor(lower-margin)),
                 np.minimum(photo_image.size,np.ceil(upper+margin))].astype(int)
    span = crop[2:]-crop[:2]
    if (span<=0).any():
        return None
    resized = np.rint(span*min(np.asarray(SIZE)/span)).astype(int)
    padding = (np.asarray(SIZE)-resized)//2
    canvas = Image.new('RGB',SIZE,(128,128,128))
    canvas.paste(photo_image.crop(tuple(crop)).resize(tuple(resized),Image.Resampling.BILINEAR),tuple(padding))
    image_path = folder/f"scan{capture['scan']:05d}_{stream}.png"
    canvas.save(image_path)
    nearest_distance,nearest = capture['tree'].query(world,workers=4)
    return {'stream':stream,'camera':camera,'rotation':R,'offset':t,'turn':turn,
            'image_size':image_size,'crop':crop,'scale':resized/span,'padding':padding,
            'content':np.r_[padding,padding+resized],'image':image_path,
            'camera_center':np.asarray(camera['translation'])@R.T+t,
            'raw_points':world,'raw_labels':np.where(nearest_distance<=6,capture['labels'][nearest],0)}


def prepare(scan,folder):
    parent = ROOT/'outputs/models'/f'export_{scan:05d}'
    path = next(p for p in sorted(parent.glob('*/report.json'),reverse=True)
                if json.loads(p.read_text(encoding='utf-8')).get('raw_color',{}).get('status')=='used')
    report = json.loads(path.read_text(encoding='utf-8'))
    model = path.parent/report['artifacts']['ply']['name']
    parts = json.loads((path.parent/'body_parts.json').read_text(encoding='utf-8'))
    if file_hash(model)!=parts['mesh']['sha256'] or file_hash(report['source'])!=report['fingerprint']['source_sha256']:
        raise ValueError('Model or observations differ from the verified inputs.')
    mesh = o3d.io.read_triangle_mesh(str(model))
    v = np.asarray(mesh.vertices)
    labels = np.zeros(len(v),dtype=int)
    for i,name in enumerate(PARTS,1):
        labels[parts['parts'][name]['vertex_indices']] = i
    up = np.eye(3)[np.ptp(v,axis=0).argmax()]
    capture = {'scan':scan,'report':report,'model':model,'mesh':mesh,'vertices':v,
               'labels':labels,'up':up,'tree':cKDTree(v)}
    capture['views'] = [view for entry in report['raw_color']['streams']
                        if entry['stream'] in STREAMS and entry['projection_check_passed']
                        if (view:=prepare_view(capture,entry,folder)) is not None]
    return capture


def silhouette(scene,view):
    """Ray-cast a half-resolution contour in the same cropped photo coordinates."""
    xx,yy = np.meshgrid(np.arange(0,SIZE[0],2),np.arange(0,SIZE[1],2))
    uv = np.c_[xx.ravel(),yy.ravel()]
    left,top,right,bottom = view['content']
    content = (uv[:,0]>=left)&(uv[:,0]<right)&(uv[:,1]>=top)&(uv[:,1]<bottom)
    uv = (uv[content]+.5-view['padding'])/view['scale']-.5+view['crop'][:2]
    w,h = view['image_size']
    turn = view['turn']
    uv = rotate_pixels(uv,*( (h,w) if turn%2 else (w,h)),(-turn)%4)
    camera = view['camera']
    xd,yd = ((uv-np.asarray(camera['C']))/camera['F']).T
    x,y = xd.copy(),yd.copy()
    p1,p2,k1,k2,k3 = camera['distortion']
    for _ in range(8):
        r2 = x*x+y*y
        radial = np.maximum(1+k1*r2+k2*r2*r2+k3*r2*r2*r2,.2)
        x,y = ((xd-2*p1*x*y-p2*(r2+2*x*x))/radial,
               (yd-p1*(r2+2*y*y)-2*p2*x*y)/radial)
    local = np.c_[-x,-y,np.ones(len(x))]@np.asarray(camera['rotation'])[:3,:3].T
    direction = local@view['rotation'].T
    ray = np.c_[np.broadcast_to(view['camera_center'],direction.shape),direction].astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(ray),nthreads=4)['t_hit'].numpy()
    valid = np.zeros(xx.size,dtype=bool)
    valid[content] = np.isfinite(hit)
    valid = valid.reshape(xx.shape)
    return valid & ~binary_erosion(valid)


def check_views(capture):
    vertices,labels = capture['vertices'],capture['labels']
    target = np.flatnonzero(labels>0)
    records,votes = [],[]
    for view in capture['views']:
        image = view['image']
        seg = np.asarray(Image.open(image.with_name(image.stem+'_labels.png')))
        foreground = seg!=0
        x0,y0,x1,y1 = view['content']
        area = float(foreground[y0:y1,x0:x1].mean())
        metric = distance_transform_edt(~foreground)
        xy,inside = project(view['raw_points'],view)
        ix = np.rint(xy[inside]).astype(int)
        raw_error = metric[ix[:,1],ix[:,0]]
        coverage = float(np.mean(raw_error<=MIN_PIXEL_MARGIN)) if len(raw_error) else 0.
        reliable = len(raw_error)>=100 and coverage>=.96 and .06<=area<=.92
        background = np.zeros(len(target),dtype=bool)
        per_part = {}
        uv,visible = project(vertices[target],view)
        for i,name in enumerate(PARTS,1):
            support = raw_error[view['raw_labels'][inside]==i]
            accuracy = float(np.mean(support<=MIN_PIXEL_MARGIN)) if len(support) else 0.
            eligible = reliable and len(support)>=40 and accuracy>=.96
            margin = max(MIN_PIXEL_MARGIN,float(np.percentile(support,99))+2) if eligible else None
            ids = np.flatnonzero(visible&(labels[target]==i))
            if eligible:
                pixel = np.rint(uv[ids]).astype(int)
                background[ids] = metric[pixel[:,1],pixel[:,0]]>margin
            per_part[name] = {'raw_samples':len(support),'raw_foreground_fraction':accuracy,
                              'eligible':eligible,'pixel_margin':margin,
                              'background_vertices':int(background[ids].sum())}
        records.append({'stream':view['stream'],'raw_foreground_fraction':coverage,
                        'foreground_fraction':float(area),'reliable':reliable,'parts':per_part})
        votes.append(background)
        view['foreground'] = foreground
        view['targets_uv'] = uv
    agreed = np.zeros(len(target),dtype=bool)
    for i,view in enumerate(capture['views']):
        for j,other in enumerate(capture['views'][i+1:],i+1):
            ids = np.flatnonzero(votes[i]&votes[j])
            a = view['camera_center']-vertices[target[ids]]
            b = other['camera_center']-vertices[target[ids]]
            cosine = np.einsum('ij,ij->i',a,b)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-9)
            agreed[ids[cosine<np.cos(np.deg2rad(MIN_VIEW_ANGLE))]] = True
    cloud = o3d.io.read_point_cloud(capture['report']['source'])
    gap = cKDTree(np.asarray(cloud.points)).query(vertices[target],workers=4)[0]
    return records,target,agreed,agreed&(gap>RAW_GAP_MM),votes


def review(capture):
    records,target,agreed,candidate,votes = check_views(capture)
    mesh = capture['mesh']
    labels = capture['labels']
    flag = np.zeros(len(labels),bool)
    flag[target[candidate]] = True
    out = ROOT/'outputs/experiments/photo_silhouette'/f"export_{capture['scan']:05d}"
    out.mkdir(parents=True,exist_ok=True)
    parts = {}
    for i,name in enumerate(PARTS,1):
        mask = labels[target]==i
        parts[name] = {'vertices':int(mask.sum()),'background_in_distinct_views':int((mask&agreed).sum()),
                       'also_no_raw_within_5mm':int((mask&candidate).sum())}
    remove = flag[np.asarray(mesh.triangles)].all(axis=1)
    result = {'mesh':str(capture['model']),'view_count':len(records),'minimum_view_angle_deg':MIN_VIEW_ANGLE,
              'raw_gap_mm':RAW_GAP_MM,'regions':parts,'views':records,
              'fully_flagged_triangles':int(remove.sum()),'selected':None}
    selected = out/'trim_candidate.ply'
    if remove.any():
        trial = copy.deepcopy(mesh)
        trial.remove_triangles_by_mask(remove)
        trial.remove_unreferenced_vertices()
        keys = ('boundary_edges','edges_with_more_than_two_faces','connected_components')
        before = geometry_report(copy.deepcopy(mesh),check_intersections=False)
        after = geometry_report(copy.deepcopy(trial),check_intersections=False)
        result['topology'] = {key:[before[key],after[key]] for key in keys}
        if after['orientation_succeeded'] and all(after[key]<=before[key] for key in keys):
            if not o3d.io.write_triangle_mesh(str(selected),trial):
                raise OSError(f'Cannot write {selected}')
            result['selected'] = str(selected)
    if result['selected'] is None:
        selected.unlink(missing_ok=True)
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    order = sorted(range(len(records)),key=lambda i:(int((candidate&votes[i]).sum()),
                  records[i]['reliable'],sum(p['eligible'] for p in records[i]['parts'].values())),reverse=True)
    ids = order[:6]
    canvas = Image.new('RGB',(384*3,544*((len(ids)+2)//3)),'white')
    draw = ImageDraw.Draw(canvas)
    for col,k in enumerate(ids):
        view = capture['views'][k]
        pixels = np.array(Image.open(view['image']).resize((384,512),Image.Resampling.BILINEAR))
        mask = view['foreground'][::2,::2]
        pixels[mask&~binary_erosion(mask)] = [40,240,70]
        pixels[silhouette(scene,view)] = [50,205,255]
        panel = Image.fromarray(pixels)
        mark = ImageDraw.Draw(panel)
        for x,y in view['targets_uv'][votes[k]&candidate]/2:
            mark.ellipse((x-2,y-2,x+2,y+2),fill=(255,55,45))
        x,y = (col%3)*384,(col//3)*544
        draw.text((x+6,y+6),f"View {view['stream']} | green Meta | cyan mesh | red candidate",fill='black')
        canvas.paste(panel,(x,y+24))
    canvas.save(out/'overlay.jpg',quality=92)
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(capture['scan'],json.dumps({k:result[k] for k in ('view_count','regions','fully_flagged_triangles','selected')}),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan',type=int,nargs='+',required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        captures = [prepare(scan,folder) for scan in args.scan]
        views = [{'image':str(view['image']),'input_name':str(capture['scan']),'yaw':int(view['stream'])}
                 for capture in captures for view in capture['views']]
        manifest = folder/'views.json'
        manifest.write_text(json.dumps(views),encoding='utf-8')
        subprocess.run([str(DEFAULT_GPU_PYTHON),'-B',str(ROOT/'tools/sapiens/infer.py'),
                        '--views',str(manifest),'--report',str(folder/'infer.json'),'--labels-only'],
                       check=True,timeout=300)
        for capture in captures:
            review(capture)
