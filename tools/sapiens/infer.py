"""独立调用 Sapiens-Lite 官方 TorchScript 分割权重，默认一次一张图。
根目录：D:/Anaconda/conda/envs/test0/python.exe -B tools/sapiens/infer.py
--limit 1：先测一张；默认处理全部渲染图。GPU float32，官方均值/标准差。
只输出标签PNG、小报告及对照图，不保存巨大的逐像素概率数组，不修改重建结果。
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from PIL import Image
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'outputs/segmentation/sapiens'
sys.path.insert(0,str(HERE/'vendor'))
from classes_and_palettes import GOLIATH_CLASSES,GOLIATH_PALETTE


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--limit',type=int)
    parser.add_argument('--views',type=Path,default=OUT/'views.json',help='View manifest for one reconstruction or standalone trial.')
    parser.add_argument('--report',type=Path,help='Custom inference report path.')
    parser.add_argument('--labels-only',action='store_true',help='Save labels only; no comparison images (run.py temporary stage).')
    args=parser.parse_args()
    manifest=json.loads((HERE/'checkpoint_manifest.json').read_text());path=HERE/'checkpoints'/manifest['file']
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['sha256']
    if not torch.cuda.is_available():raise RuntimeError('GPU environment required for this trial')
    torch.set_num_threads(4);device='cuda';torch.cuda.reset_peak_memory_stats()
    print('Loading official TorchScript...',flush=True)
    model=torch.jit.load(str(path),map_location='cpu').eval().to(device)
    mean=torch.tensor([123.5,116.5,103.5],device=device).view(1,3,1,1)
    std=torch.tensor([58.5,57.,57.5],device=device).view(1,3,1,1)
    views=json.loads(args.views.read_text(encoding='utf-8'));records=[]
    for view in (views[:args.limit] if args.limit else views):
        started=time.perf_counter();image_path=ROOT/view['image'];image=Image.open(image_path).convert('RGB');rgb=np.asarray(image)
        assert image.size==(768,1024)
        tensor=torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).to(device=device,dtype=torch.float32)
        tensor=(tensor-mean)/std
        with torch.inference_mode():
            result=model(tensor)
            if isinstance(result,(tuple,list)):result=result[0]
            if result.ndim==3:result=result.unsqueeze(0)
            if result.shape[-2:]!=(1024,768):result=torch.nn.functional.interpolate(result,size=(1024,768),mode='bilinear',align_corners=False)
            labels=result.argmax(1)[0].to('cpu').numpy().astype(np.uint8)
        if labels.max()>=len(GOLIATH_CLASSES):raise RuntimeError('Unexpected class indices')
        output=image_path.parent/(image_path.stem+'_labels.png');Image.fromarray(labels).save(output)
        if not args.labels_only:
            colored=np.array(GOLIATH_PALETTE,dtype=np.uint8)[labels]
            overlay=np.rint(rgb*.55+colored*.45).astype(np.uint8)
            hand=np.full_like(rgb,180);hand[labels==5]=[0,220,0];hand[labels==14]=[255,0,220]
            panel=np.concatenate([rgb,overlay,hand],axis=1);Image.fromarray(panel).save(image_path.parent/(image_path.stem+'_comparison.jpg'),quality=90)
        record={'input_name':view['input_name'],'yaw':view['yaw'],'image':str(image_path),
                'class_counts':{name:int(np.sum(labels==i)) for i,name in enumerate(GOLIATH_CLASSES)},
                'seconds':round(time.perf_counter()-started,2),'peak_cuda_MiB':round(torch.cuda.max_memory_allocated()/1024**2,1)}
        records.append(record)
        print(f"Sapiens view {view['yaw']}: left/right hand pixels "
              f"{record['class_counts']['Left_Hand']}/{record['class_counts']['Right_Hand']}" if args.labels_only else json.dumps(record),flush=True)
        del tensor,result;torch.cuda.empty_cache()
    report={'environment':{'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)},
            'checkpoint':manifest,'classes':list(GOLIATH_CLASSES),'views':records,
            'note':'Rendered-image predictions, not anatomical ground truth; class coverage alone is not accuracy.'}
    report_path=args.report or OUT/('inference_probe.json' if args.limit else 'inference.json')
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
