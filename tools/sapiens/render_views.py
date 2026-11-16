"""渲染现有网格供 Sapiens 试验，不移动模型、不修改真实颜色。
根目录：python tools/sapiens/render_views.py。使用 dachuang 环境的 Open3D。
四组模型共用四个正交视角和768x1024画幅，输出 outputs/segmentation/sapiens/。
只保存小体积图片及相机/输入哈希；三维映射需要时可重建同一组射线。
"""
from pathlib import Path
import hashlib,json
import numpy as np,open3d as o3d
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/segmentation/sapiens'
WIDTH,HEIGHT=768,1024


def rays_for(camera):
    target=np.array(camera['target']);right=np.array(camera['right']);up=np.array(camera['up']);forward=np.array(camera['forward'])
    xx,yy=np.meshgrid(np.linspace(-camera['width_mm']/2,camera['width_mm']/2,WIDTH),
                       np.linspace(camera['height_mm']/2,-camera['height_mm']/2,HEIGHT))
    origin=target+xx[...,None]*right+yy[...,None]*up-forward*camera['distance_mm']
    return o3d.core.Tensor(np.concatenate([origin,np.broadcast_to(forward,origin.shape)],axis=2).reshape(-1,6).astype(np.float32))


def main():
    records=[]
    for folder in sorted((ROOT/'outputs/models').glob('*')):
        sources=[p.parent/(folder.name+'_mesh.ply') for p in folder.glob('*/report.json')]
        sources=[p for p in sources if p.is_file()]
        if not sources:continue
        source=max(sources,key=lambda p:p.parent.name);digest=hashlib.sha256(source.read_bytes()).hexdigest()
        mesh=o3d.io.read_triangle_mesh(str(source));vertices=np.asarray(mesh.vertices);triangles=np.asarray(mesh.triangles);colors=np.asarray(mesh.vertex_colors)
        if len(colors)!=len(vertices):raise ValueError('Colored mesh required')
        lower,upper=vertices.min(0),vertices.max(0);center=(lower+upper)/2;span=(upper[0]-lower[0])*1.1
        scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        target=OUT/folder.name;target.mkdir(parents=True,exist_ok=True)
        for yaw in [0,90,180,270]:
            theta=np.deg2rad(yaw);forward=np.array([0,-np.sin(theta),np.cos(theta)])
            right=np.cross(forward,[1,0,0]);up=np.cross(right,forward)
            camera={'source':str(source),'source_sha256':digest,'input_name':folder.name,'yaw':yaw,
                    'target':center.tolist(),'right':right.tolist(),'up':up.tolist(),'forward':forward.tolist(),
                    'width_mm':span*WIDTH/HEIGHT,'height_mm':span,'distance_mm':4000,'image_size':[WIDTH,HEIGHT],
                    'render':'Orthographic vertex RGB with mild two-sided shading; no exposure correction.'}
            hit=scene.cast_rays(rays_for(camera),nthreads=4);valid=np.isfinite(hit['t_hit'].numpy())
            face_ids=hit['primitive_ids'].numpy()[valid].astype(np.int64);uv=hit['primitive_uvs'].numpy()[valid]
            weights=np.c_[1-uv.sum(1),uv];rgb=np.einsum('ni,nij->nj',weights,colors[triangles[face_ids]])
            normal=hit['primitive_normals'].numpy()[valid];light=-forward+.3*up;light/=np.linalg.norm(light)
            shade=.75+.25*np.abs(normal@light);pixels=np.full((WIDTH*HEIGHT,3),.88);pixels[valid]=np.clip(rgb*shade[:,None],0,1)
            image=target/f'view_{yaw}.png';Image.fromarray(np.rint(pixels.reshape(HEIGHT,WIDTH,3)*255).astype(np.uint8)).save(image)
            camera['image']=str(image);camera['rendered_pixels']=int(valid.sum());camera['image_sha256']=hashlib.sha256(image.read_bytes()).hexdigest();records.append(camera)
            print(folder.name,yaw,flush=True)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    (OUT/'views.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

if __name__=='__main__':main()
