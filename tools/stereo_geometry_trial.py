"""检查配套主相机照片能为 Meta 脸颈和手部提供多少立体测量点。

python tools/stereo_geometry_trial.py --scan 1
主环境读取网格；立体匹配使用 --opencv-python（默认项目的 Sapiens 环境）。
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
PARTS = ('face_neck','left_hand','right_hand')
IMAGE_SCALE = .65
MAX_REPROJECTION_PX = 2.
MAX_RAW_DISTANCE_MM = 8.


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def camera_model(item, camera):
    transform = np.asarray(item['transform'])
    rotation = np.asarray(camera['rotation'])[:3,:3]
    center = np.asarray(camera['translation'])
    world_center = transform[:3,:3] @ center + transform[:3,3]
    axes = np.diag([-1.,-1.,1.]) @ rotation.T @ transform[:3,:3].T
    extrinsic = np.c_[axes,-axes @ world_center]
    intrinsic = np.eye(3)
    intrinsic[[0,1],[0,1]] = camera['F']
    intrinsic[:2,2] = camera['C']
    p1,p2,k1,k2,k3 = camera['distortion']
    return extrinsic,intrinsic,np.array([k1,k2,p1,p2,k3]),world_center


def load_features(cv, raw_report, hashes, vertices, labels):
    folder = Path(raw_report['source'])
    sift = cv.SIFT_create(4000,contrastThreshold=.018)
    views = []
    for item in raw_report['streams']:
        if not item['projection_check_passed']:
            continue
        prefix = folder/item['stream']/raw_report['frame']
        ply,primary,photo = [prefix.with_name(prefix.name+suffix) for suffix in
                             ('.ply','_CameraParams_Primary.json','_RGB_01.png')]
        for path in (ply,primary,photo):
            if digest(path) != hashes[str(path.relative_to(folder))]:
                raise ValueError(f'Capture input changed: {path}')
        camera = json.loads(primary.read_text(encoding='utf-8'))['color']
        extrinsic,K,dist,center = camera_model(item,camera)
        with ply.open() as stream:
            header = next(i for i,line in enumerate(stream) if line.strip()=='end_header')
        local = np.loadtxt(ply,skiprows=header+1,usecols=(0,1,2))
        transform = np.asarray(item['transform'])
        raw = local@transform[:3,:3].T+transform[:3,3]
        supported = cKDTree(raw).query(vertices,workers=4)[0] <= 5.
        xyz = vertices[supported]
        if len(xyz)<20:
            continue
        rotation = cv.Rodrigues(extrinsic[:,:3])[0]
        uv = cv.projectPoints(xyz,rotation,extrinsic[:,3],K,dist)[0].reshape(-1,2)
        image = cv.imdecode(np.fromfile(photo,dtype=np.uint8),cv.IMREAD_GRAYSCALE)
        image = cv.resize(image,None,fx=IMAGE_SCALE,fy=IMAGE_SCALE,interpolation=cv.INTER_AREA)
        pixel = np.rint(uv*IMAGE_SCALE).astype(int)
        inside = ((pixel[:,0]>=0)&(pixel[:,0]<image.shape[1])&
                  (pixel[:,1]>=0)&(pixel[:,1]<image.shape[0])&
                  ((xyz@extrinsic[2,:3]+extrinsic[2,3])>0))
        if inside.sum()<20:
            continue
        mask = np.zeros_like(image)
        mask[pixel[inside,1],pixel[inside,0]] = 255
        mask = cv.dilate(mask,np.ones((13,13),dtype=np.uint8))
        keypoints,descriptors = sift.detectAndCompute(image,mask)
        if descriptors is None or len(descriptors)<2:
            continue
        pixels = np.asarray([point.pt for point in keypoints])/IMAGE_SCALE
        distance,nearest = cKDTree(uv[inside]).query(pixels)
        label = np.where(distance<10,labels[supported][inside][nearest],0)
        good = label>0
        if good.sum()<2:
            continue
        views.append({'id':item['stream'],'extrinsic':extrinsic,'K':K,'dist':dist,'center':center,
                      'pixels':pixels[good],'labels':label[good],'descriptors':descriptors[good],
                      'image':image,'supported':supported,
                      'undistorted':cv.undistortPoints(pixels[good],K,dist).reshape(-1,2)})
    return views


def match_pair(cv, first, second):
    matcher = cv.BFMatcher(cv.NORM_L2)
    def matches(left,right):
        return {a.queryIdx:a.trainIdx for a,b in matcher.knnMatch(left,right,k=2) if a.distance<.77*b.distance}
    forward = matches(first['descriptors'],second['descriptors'])
    reverse = matches(second['descriptors'],first['descriptors'])
    pairs = [(a,b) for a,b in forward.items() if reverse.get(b)==a and first['labels'][a]==second['labels'][b]]
    if not pairs:
        return np.empty((0,3)),np.empty(0,dtype=int),0
    a,b = np.asarray(pairs).T
    h = cv.triangulatePoints(first['extrinsic'],second['extrinsic'],
                             first['undistorted'][a].T,second['undistorted'][b].T)
    finite = np.abs(h[3])>1e-8
    xyz = (h[:3,finite]/h[3,finite]).T
    a,b = a[finite],b[finite]
    if not len(xyz):
        return xyz,np.empty(0,dtype=int),len(pairs)
    ok = np.ones(len(xyz),dtype=bool)
    for view,index in ((first,a),(second,b)):
        E = view['extrinsic']
        rotation = cv.Rodrigues(E[:,:3])[0]
        pixel = cv.projectPoints(xyz,rotation,E[:,3],view['K'],view['dist'])[0].reshape(-1,2)
        ok &= ((xyz@E[2,:3]+E[2,3]>0)&
               (np.linalg.norm(pixel-view['pixels'][index],axis=1)<=MAX_REPROJECTION_PX))
    da,db = xyz-first['center'],xyz-second['center']
    cosine = np.einsum('ij,ij->i',da,db)/np.maximum(np.linalg.norm(da,axis=1)*np.linalg.norm(db,axis=1),1e-12)
    ok &= (cosine>np.cos(np.deg2rad(70))) & (cosine<np.cos(np.deg2rad(8)))
    return xyz[ok],first['labels'][a[ok]],len(pairs)


def patch_depth(cv, views, vertices, normals, labels, raw_trees):
    """Search along mesh normals using textured patches and three-view agreement."""
    _, sample = np.unique(np.floor(vertices/3),axis=0,return_index=True)
    v, n = vertices[sample], normals[sample]
    direction = np.eye(3)[np.abs(n).argmin(axis=1)]
    tangent = np.cross(n,direction)
    tangent /= np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-9)
    other = np.cross(n,tangent)
    gx,gy = np.meshgrid(np.arange(-2,3),np.arange(-2,3))
    patch = gx.ravel()[None,:,None]*tangent[:,None,:] + gy.ravel()[None,:,None]*other[:,None,:]
    shifts = np.arange(-3,4)
    candidates = v[:,None,None,:]+shifts[None,:,None,None]*n[:,None,None,:]+patch[:,None,:,:]
    aligned = []
    for view in views:
        E = view['extrinsic']
        rays = view['center']-v
        rays /= np.maximum(np.linalg.norm(rays,axis=1)[:,None],1e-9)
        facing = np.einsum('ij,ij->i',rays,n)
        allowed = view['supported'][sample] & (np.abs(facing)>=.55)
        ids = np.flatnonzero(allowed)
        data = np.zeros(candidates.shape[:3],dtype=np.float32)
        if len(ids):
            points = candidates[ids].reshape(-1,3)
            uv = cv.projectPoints(points,cv.Rodrigues(E[:,:3])[0],E[:,3],view['K'],view['dist'])[0]
            uv = (uv.reshape(len(ids),-1,2)*IMAGE_SCALE).astype(np.float32)
            image = view['image']
            inside = ((uv[:,:,0]>=1)&(uv[:,:,0]<image.shape[1]-2)&
                      (uv[:,:,1]>=1)&(uv[:,:,1]<image.shape[0]-2)).all(axis=1)
            pixels = cv.remap(image,uv[:,:,0].copy(),uv[:,:,1].copy(),cv.INTER_LINEAR).reshape(len(ids),len(shifts),-1)
            pixels = pixels.astype(np.float32)
            pixels -= pixels.mean(axis=2,keepdims=True)
            size = np.linalg.norm(pixels,axis=2,keepdims=True)
            allowed[ids] = inside & (size[:,:,0].min(axis=1)>=22)
            data[ids] = pixels/np.maximum(size,1e-9)
        aligned.append((data,allowed,rays,facing))
    votes = np.zeros((len(v),len(shifts)),dtype=np.uint16)
    eligible_pairs = np.zeros(len(v),dtype=np.uint16)
    camera_votes = np.zeros((len(views),len(v),len(shifts)),dtype=bool)
    for i,(left,a,ray1,f1) in enumerate(aligned):
        for j in range(i+1,len(aligned)):
            right,b,ray2,f2 = aligned[j]
            angle = np.einsum('ij,ij->i',ray1,ray2)
            ids = np.flatnonzero(a&b&(f1*f2>0)&(angle<np.cos(np.deg2rad(10)))&
                                 (angle>np.cos(np.deg2rad(55))))
            if not len(ids):
                continue
            eligible_pairs[ids] += 1
            scores = np.einsum('ijk,ijk->ij',left[ids],right[ids])
            best = scores.argmax(axis=1)
            row = np.arange(len(ids))
            high = scores[row,best]
            margin = high-np.partition(scores,-2,axis=1)[:,-2]
            good = ((best>0)&(best<len(shifts)-1)&(best!=3)&(high>=.8)&
                    (high-scores[:,3]>=.06)&(margin>=.012))
            ii,bb = ids[good],best[good]
            votes[ii,bb] += 1
            camera_votes[i,ii,bb] = True
            camera_votes[j,ii,bb] = True
    best = votes.argmax(axis=1)
    row = np.arange(len(v))
    support = votes[row,best]
    valid = ((support>=2)&(support*2>votes.sum(axis=1))&
             (camera_votes.sum(axis=0)[row,best]>=3))
    result = {}
    for i,name in enumerate(PARTS,1):
        region = labels[sample]==i
        used = valid&region
        proposed = v[used]+shifts[best[used],None]*n[used]
        before = raw_trees[i].query(v[used])[0]
        after = raw_trees[i].query(proposed)[0]
        result[name] = {'sampled_vertices':int(region.sum()),
                        'vertices_with_testable_pair':int((region&(eligible_pairs>0)).sum()),
                        'vertices_with_strong_pair':int((region&(votes.sum(axis=1)>0)).sum()),
                        'three_view_candidates':int(used.sum()),
                        'nearest_raw_median_before_mm':float(np.median(before)) if len(before) else None,
                        'nearest_raw_median_after_mm':float(np.median(after)) if len(after) else None}
    return result


def evaluate(npz, report_path, out):
    import cv2 as cv
    cv.setNumThreads(4)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    arrays = np.load(npz)
    vertices,labels = arrays['vertices'],arrays['labels']
    views = load_features(cv,report['raw_color'],report['fingerprint']['raw_color_input']['sha256'],
                          vertices,labels)
    counts = {name:{'mesh_vertices':int((labels==i).sum()),'points':[],'pairs':0}
              for i,name in enumerate(PARTS,1)}
    raw_trees = {i:cKDTree(arrays[name]) for i,name in enumerate(PARTS,1)}
    matched = triangulated = 0
    for i,first in enumerate(views):
        for second in views[i+1:]:
            xyz,parts,count = match_pair(cv,first,second)
            matched += count
            triangulated += len(xyz)
            for k,name in enumerate(PARTS,1):
                points = xyz[parts==k]
                if not len(points):
                    continue
                d = raw_trees[k].query(points,workers=4)[0]
                supported = points[d<=MAX_RAW_DISTANCE_MM]
                if len(supported):
                    counts[name]['pairs'] += 1
                    counts[name]['points'].append(supported)
    for i,(name,stats) in enumerate(counts.items(),1):
        groups = stats.pop('points')
        xyz = np.concatenate(groups) if groups else np.empty((0,3))
        stats['matched_points_near_raw'] = len(xyz)
        stats['occupied_3mm_cells'] = len(np.unique(np.floor(xyz/3),axis=0))
        stats['mesh_vertices_within_4mm_percent'] = (float(np.mean(cKDTree(xyz).query(vertices[labels==i])[0]<=4)*100)
                                                     if len(xyz) else 0.)
    result = {'source':str(report_path),'camera_views':len(views),'mutual_feature_matches':matched,
              'passed_stereo_geometry':triangulated,'maximum_reprojection_px':MAX_REPROJECTION_PX,
              'maximum_raw_distance_mm':MAX_RAW_DISTANCE_MM,'regions':counts,
              'patch_depth_search':patch_depth(cv,views,vertices,arrays['normals'],labels,raw_trees)}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


def prepare(scan, python):
    import open3d as o3d
    parent = ROOT/'outputs/models'/f'export_{scan:05d}'
    report_path = next(p for p in sorted(parent.glob('*/report.json'),reverse=True)
                       if json.loads(p.read_text(encoding='utf-8')).get('raw_color',{}).get('status')=='used')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    parts = json.loads((report_path.parent/'body_parts.json').read_text(encoding='utf-8'))
    model = report_path.parent/report['artifacts']['ply']['name']
    source = Path(report['source'])
    if digest(model)!=parts['mesh']['sha256'] or digest(source)!=report['fingerprint']['source_sha256']:
        raise ValueError('Model or point cloud differs from the verified capture.')
    mesh = o3d.io.read_triangle_mesh(str(model))
    v = np.asarray(mesh.vertices)
    labels = np.zeros(len(v),dtype=int)
    for i,name in enumerate(PARTS,1):
        labels[parts['parts'][name]['vertex_indices']] = i
    raw = np.asarray(o3d.io.read_point_cloud(str(source)).points).copy()
    d,n = cKDTree(v).query(raw,workers=4)
    out = ROOT/'outputs/experiments/stereo_geometry'/parent.name
    out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as work:
        npz = Path(work)/'regions.npz'
        np.savez(npz,vertices=v[labels>0],normals=np.asarray(mesh.vertex_normals)[labels>0],labels=labels[labels>0],
                 **{name:raw[(labels[n]==i)&(d<=5)] for i,name in enumerate(PARTS,1)})
        subprocess.run([str(python),'-B',__file__,'--worker',str(npz),str(report_path),
                        str(out/'report.json')],check=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan',type=int)
    parser.add_argument('--opencv-python',type=Path)
    parser.add_argument('--worker',type=Path,nargs=3,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        evaluate(*args.worker)
    elif args.scan is not None:
        sys.path.insert(0,str(ROOT))
        from body_parts import DEFAULT_GPU_PYTHON
        prepare(args.scan,args.opencv_python or DEFAULT_GPU_PYTHON)
    else:
        parser.error('--scan is required.')
