"""腕部切除与曲面连接：核心函数由正式 MANO 流程调用，CLI 只写独立试验。

项目根目录：python tools/wrist_cut_trial.py --scan 0 3 --keep-full
沿现有腕部坐标切除 -20 到 35 mm 腕带；保留前臂端与手掌端，重建中间连接。
默认只保存局部预览、比较和报告；--keep-full 另存完整 PLY。
--contour：只处理腕口到掌侧 35 mm，匹配真实边界和保留侧切向。
正式流程使用该真实轮廓模式，前臂和远端手指坐标保留。
边界或相交检查失败时回退。
"""
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.mano_join import boundary_loops, bridge_loops
from tools.surface_trials import panel
from hand_pipeline import pairs, smoothstep


def clip_keep(v, c, faces, wrist, axes, level, below, region):
    """Clip only the local region, sharing each interpolated edge vertex."""
    y=((v-wrist)@axes)[:,1]-level
    y[np.abs(y)<1.e-7]=0
    inside=y<=0 if below else y>=0
    candidates=region & ~np.all(inside[faces],axis=1)
    kept=faces[~candidates];added=[];verts=[];colors=[];cache={}
    for tri in faces[candidates]:
        poly=[]
        for a,b in zip(tri,np.roll(tri,-1)):
            if inside[a]:poly.append(int(a))
            if (y[a]<0<y[b]) or (y[b]<0<y[a]):
                edge=tuple(sorted((int(a),int(b))))
                if edge not in cache:
                    t=y[a]/(y[a]-y[b]);cache[edge]=len(v)+len(verts)
                    verts.append(v[a]+t*(v[b]-v[a]));colors.append(c[a]+t*(c[b]-c[a]))
                poly.append(cache[edge])
        for k in range(1,len(poly)-1):added.append([poly[0],poly[k],poly[k+1]])
    if verts:v=np.r_[v,np.array(verts)];c=np.r_[c,np.array(colors)]
    return v,c,np.r_[kept,np.array(added,int).reshape(-1,3)]


def contour_section(v, faces, loop, wrist, axes, level, below, count=64):
    """按真实边界采样截面和保留侧切向，避免用椭圆硬接不规则腕口。"""
    local=(v-wrist)@axes
    xy=local[loop][:,[0,2]]
    area=np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))
    if area<0:loop=loop[::-1];xy=xy[::-1]
    center=xy.mean(0);angle=np.arctan2(xy[:,1]-center[1],xy[:,0]-center[0])
    shift=np.argmin(np.abs(angle));loop=np.roll(loop,-shift);xy=np.roll(xy,-shift,axis=0)
    edge=np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1)
    arc=np.r_[0.,np.cumsum(edge)];query=np.linspace(0,arc[-1],count,endpoint=False)
    section=np.c_[*[np.interp(query,arc,np.r_[xy[:,k],xy[0,k]]) for k in range(2)]]
    slopes=[]
    for index in loop:
        neighbors=np.unique(faces[np.any(faces==index,axis=1)])
        dy=local[neighbors,1]-level
        valid=(dy<-.2)&(dy>-18) if below else (dy>.2)&(dy<18)
        if valid.any():
            delta=local[neighbors[valid]][:,[0,2]]-local[index,[0,2]]
            slope=np.sum(delta*dy[valid,None],axis=0)/np.sum(dy[valid]**2)
        else:slope=np.zeros(2)
        slopes.append(np.clip(slope,-1.5,1.5))
    slopes=np.asarray(slopes)
    tangent=np.c_[*[np.interp(query,arc,np.r_[slopes[:,k],slopes[0,k]]) for k in range(2)]]
    # Suppress scan triangle noise in the derivative, without moving the boundary.
    for _ in range(5):tangent=.5*tangent+.25*(np.roll(tangent,1,axis=0)+np.roll(tangent,-1,axis=0))
    return section,tangent


def cut_wrist(mesh, wrist, axes, low_level=-20., high_level=35., contour_match=False):
    original=np.asarray(mesh.vertices).copy();v=original.copy();c=np.asarray(mesh.vertex_colors).copy();f=np.asarray(mesh.triangles)
    local=(v-wrist)@axes;fc=local[f].mean(1)
    # A local prism excludes other limbs. End cuts are in arm and proximal palm.
    local_region=(np.abs(fc[:,0])<80)&(np.abs(fc[:,2])<85)&(fc[:,1]>-50)&(fc[:,1]<80)
    selected=f[local_region];outside=f[~local_region]
    v1,c1,low=clip_keep(v,c,selected,wrist,axes,low_level,True,np.ones(len(selected),bool))
    # Reuse original indices for the other retained side, independent edge cache.
    v2,c2,high=clip_keep(v1,c1,selected,wrist,axes,high_level,False,np.ones(len(selected),bool))
    v,c=v2,c2;f=np.r_[outside,low,high]
    loops=boundary_loops(f);lower=[];upper=[]
    for loop in loops:
        p=(v[loop]-wrist)@axes
        if np.linalg.norm(p[:,[0,2]].mean(0))>80:continue
        if np.max(np.abs(p[:,1]-low_level))<1.e-5:lower.append(loop)
        if np.max(np.abs(p[:,1]-high_level))<1.e-5:upper.append(loop)
    if len(lower)!=1 or len(upper)!=1:raise ValueError('Cuts do not produce one wrist loop per end')
    lower,upper=lower[0],upper[0]
    if contour_match:
        section0,tangent0=contour_section(v,f,lower,wrist,axes,low_level,True)
        section1,tangent1=contour_section(v,f,upper,wrist,axes,high_level,False)
        # Both sections have a common clockwise direction and phase; select the
        # small residual shift that gives the least twisted trajectories.
        shifts=range(-8,9)
        shift=min(shifts,key=lambda k:np.mean(np.linalg.norm(section0-np.roll(section1,k,axis=0),axis=1)))
        section1=np.roll(section1,shift,axis=0);tangent1=np.roll(tangent1,shift,axis=0)
    endpoints=[]
    for loop in [lower,upper]:
        xy=((v[loop]-wrist)@axes)[:,[0,2]]
        # Sample segment midpoints by arc weight, avoiding bias from dense scan vertices.
        length=np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1);weights=length/length.sum()
        center=((xy+np.roll(xy,-1,axis=0))/2*weights[:,None]).sum(0)
        covariance=np.einsum('n,ni,nj->ij',weights,xy-center,xy-center)
        values,basis=np.linalg.eigh(covariance)
        shape=(basis*np.sqrt(np.maximum(values,1.)*2))@basis.T
        endpoints.append((center,shape))
    theta=np.linspace(0,2*np.pi,64,endpoint=False);unit=np.c_[np.cos(theta),np.sin(theta)]
    previous=lower;bridge=[];new_ids=[]
    for y in np.linspace(low_level,high_level,14)[1:-1]:
        t=(y-low_level)/(high_level-low_level);blend=smoothstep(t)
        center=(1-blend)*endpoints[0][0]+blend*endpoints[1][0]
        shape=(1-blend)*endpoints[0][1]+blend*endpoints[1][1]
        xy=unit@shape.T+center
        if contour_match:
            length=high_level-low_level
            xy=(2*t**3-3*t*t+1)*section0+(t**3-2*t*t+t)*length*tangent0+(-2*t**3+3*t*t)*section1+(t**3-t*t)*length*tangent1
        xyz=np.c_[xy[:,0],np.full(64,y),xy[:,1]]@axes.T+wrist
        ids=np.arange(len(v),len(v)+64);new_ids.extend(ids)
        color=np.asarray(mesh.vertex_colors)[cKDTree(original).query(xyz)[1]]
        v=np.r_[v,xyz];c=np.r_[c,color]
        triangles,_=bridge_loops(v,previous,ids);bridge.extend(triangles);previous=ids
    triangles,_=bridge_loops(v,previous,upper);bridge.extend(triangles)
    f=np.r_[f,np.asarray(bridge,int)]
    candidate=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(f));candidate.vertex_colors=o3d.utility.Vector3dVector(c)
    candidate.orient_triangles();candidate.compute_vertex_normals()
    # No retained original vertex moves; the twisted band has been replaced.
    assert np.array_equal(v[:len(original)],original)
    before_open=sum(map(len,boundary_loops(np.asarray(mesh.triangles))))
    after_open=sum(map(len,boundary_loops(f)))
    if after_open>before_open or len(candidate.get_non_manifold_edges()) or len(candidate.get_non_manifold_vertices()):raise ValueError('New opening or nonmanifold seam')
    active=np.arange(len(v))>=len(original);box=v[np.unique(np.r_[lower,upper,new_ids])]
    p=v[f];roi=np.flatnonzero(np.all(p.max(1)>=box.min(0)-10,axis=1)&np.all(p.min(1)<=box.max(0)+10,axis=1))
    intersections=len(pairs(candidate,roi,active))
    if intersections:raise ValueError(f'{intersections} local intersections in replacement')
    return candidate,{'applied':True,'cut_levels_mm':[low_level,high_level],'loft_rings':12,'profile':'boundary_contours_with_retained_side_tangents' if contour_match else 'elliptical_sections','new_open_edges':0,'local_intersections':0,'retained_original_vertices_unchanged':True}


def trial(scan, keep_full=False, contour=False):
    entries=[]
    for p in (ROOT/'outputs/models'/f'export_{scan:05d}').glob('*/report.json'):
        r=json.loads(p.read_text(encoding='utf-8'))
        if r.get('mano_hand_processing',{}).get('applied'):entries.append((p,r))
    path,report=max(entries,key=lambda x:x[0].parent.name)
    source=path.parent/report['artifacts']['ply']['name'];original=o3d.io.read_triangle_mesh(str(source));selected=copy.deepcopy(original)
    frames=json.loads((path.parent/'intermediate/mano/report.json').read_text(encoding='utf-8'))['parts']
    names={p['part'] for p in report['mano_hand_processing']['join_attempt']['parts']};frames=[f for f in frames if f['part'] in names]
    out=ROOT/'outputs/experiments'/('wrist_contour' if contour else 'wrist_cut')/f'export_{scan:05d}';out.mkdir(parents=True,exist_ok=True)
    canvas=Image.new('RGB',(1400,760*len(frames)),'white');draw=ImageDraw.Draw(canvas);records=[]
    for k,frame in enumerate(frames):
        wrist=np.array(frame['wrist_mm']);axes=np.array(frame['frame_axes']);name=frame['part']
        try:selected,result=cut_wrist(selected,wrist,axes,low_level=0. if contour else -20.,contour_match=contour)
        except ValueError as e:result={'applied':False,'reason':str(e)}
        result['part']=name;records.append(result);print(scan,result,flush=True)
        def crop(mesh):return mesh.select_by_index(np.flatnonzero(np.linalg.norm(np.asarray(mesh.vertices)-wrist,axis=1)<110))
        before,after=crop(original),crop(selected)
        if result['applied']:o3d.io.write_triangle_mesh(str(out/(name+'_wrist_preview.ply')),after)
        for angle in range(2):
            right=axes[:,0] if angle==0 else axes[:,2]
            for col in range(4):
                m=copy.deepcopy(before if col%2==0 else after)
                if col>=2:m.paint_uniform_color([.72]*3)
                row=k*2+angle
                canvas.paste(panel(m,wrist+axes[:,1]*10,right,axes[:,1],160,350,colored=col<2),(col*350,row*380+25))
                draw.text((col*350+4,row*380+5),name+(' | Before' if col%2==0 else ' | After accepted'),fill='black')
    canvas.save(out/'comparison.jpg',quality=95)
    full=None
    if keep_full and any(r['applied'] for r in records):
        full=f'export_{scan:05d}_wrist_cut.ply'
        selected.remove_unreferenced_vertices();selected.compute_vertex_normals()
        if not o3d.io.write_triangle_mesh(str(out/full),selected):raise OSError('Full preview export failed')
    (out/'report.json').write_text(json.dumps({'source_mesh':str(source),'parts':records,'formal_output_changed':False,'full_preview':full},ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scan',type=int,nargs='+',default=[0,3]);parser.add_argument('--keep-full',action='store_true');parser.add_argument('--contour',action='store_true',help='仅掌侧，使用真实边界与切向连续过渡')
    args=parser.parse_args()
    for scan in args.scan:trial(scan,args.keep_full,args.contour)
