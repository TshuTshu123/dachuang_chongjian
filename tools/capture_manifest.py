"""只读采集 XML 索引，提取数据层、逐路标定和帧时间；不加载厂商 DLL。
根目录：python tools/capture_manifest.py "采集文件夹" [--frame 0] [--out 输出目录]
输出 capture_manifest.json 与 calibration.json，不解码、不改写原始 MVX。
变换矩阵方向、畸变系数次序、平移单位保留原描述，等待导出数据核对。
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
META={'StreamInfo','CameraParams','PhotogrammetryInfo','Transform','CylinderCrop'}


def field(text,name):
    hit=re.search(r'(?:^|,\s*)'+re.escape(name)+r'=([^,]+)',text)
    return hit.group(1).strip() if hit else None


def camera_sections(text):
    result={}
    for name,body in re.findall(r'(Color|IR)=\{([^}]+)\}',text):
        values={}
        for item in body.split(','):
            key,value=item.strip().split('=',1)
            try:
                nums=[float(x) for x in value.split()]
                values[key]=int(nums[0]) if key in ('width','height') else nums[0] if len(nums)==1 else nums
            except ValueError:values[key]=value
        result[name]=values
    return result


def inspect(folder,frame_index,out):
    inputs=sorted(folder.glob('*_joined.xml'))
    if len(inputs)!=1:raise ValueError('Expected one *_joined.xml index in the capture folder')
    source=inputs[0];tree=ET.parse(source);frames=[e for e in tree.getroot() if e.attrib.get('TYPE')=='Frame']
    streams={};counts=Counter();calibrations={};last_meta={};changes=[]
    for atom in frames:
        stream=atom.attrib['STREAMID'];index=int(atom.attrib['FRAME']);timestamp=atom.attrib.get('TS')
        record=streams.setdefault(stream,{'frames':[],'layers':set(),'selected_frame':None})
        record['frames'].append({'index':index,'timestamp':timestamp})
        if index==frame_index:record['selected_frame']={'index':index,'timestamp':timestamp}
        for data in atom:
            text=(data.text or '').strip();purpose=field(text,'Purpose');kind=field(text,'Type')
            key=f'{purpose}/{kind}';counts[key]+=1;record['layers'].add(key)
            if kind in META:
                metadata={'purpose':purpose,'type':kind,'raw_description':text}
                if kind=='CameraParams':metadata['cameras']=camera_sections(text)
                calibrations.setdefault(stream,{}).setdefault(key,metadata)
                lookup=(stream,key)
                if lookup in last_meta and last_meta[lookup]!=text:
                    changes.append({'stream':stream,'frame':index,'timestamp':timestamp,'layer':key,'metadata':metadata})
                last_meta[lookup]=text
    complete=[]
    frame_ids=sorted(set(int(a.attrib['FRAME']) for a in frames))
    for index in frame_ids:
        if all(any(f['index']==index for f in s['frames']) for s in streams.values()):complete.append(index)
    for stream,record in streams.items():
        record['layers']=sorted(record['layers']);record['frame_count']=len(record['frames'])
        remote=folder/'remote'/f'd_{folder.name}_{stream}.xml'
        if remote.exists() and record['selected_frame']:
            timestamp=record['selected_frame']['timestamp']
            matches=[a.attrib['FRAME'] for a in ET.parse(remote).getroot() if a.attrib.get('TYPE')=='Frame' and a.attrib.get('TS')==timestamp]
            record['selected_frame']['matching_remote_frames']=matches
            record['remote_index_file']=str(remote)
    if frame_index not in complete:raise ValueError(f'Selected frame is incomplete; available complete frames: {complete}')
    out.mkdir(parents=True,exist_ok=True)
    calibration={'capture':folder.name,'source_index':str(source),'streams':calibrations,'metadata_changes':changes,
                 'coordinate_conventions':'Original descriptions retained; do not assume camera-to-world direction, units or distortion coefficient order.'}
    (out/'calibration.json').write_text(json.dumps(calibration,ensure_ascii=False,indent=2),encoding='utf-8')
    result={'capture':folder.name,'source_index':str(source),'source_index_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'joined_mvx':str(source.with_suffix('.mvx')),'stream_count':len(streams),'complete_frames':complete,
            'selected_frame':frame_index,'layer_counts':dict(counts),'streams':streams,
            'explicit_depth_layer_present':any('depth' in key.split('/')[-1].lower() for key in counts),
            'notes':['This is an index inventory, not a validation or decompression of the MVX payload.',
                     'Joined and remote frame indices can differ; use the recorded timestamps for alignment.',
                     'Export one synchronized joined frame with per-stream points, normals and images through the authorized server software.']}
    (out/'capture_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'streams':len(streams),'complete_frame_count':len(complete),'selected_frame':frame_index,
                      'explicit_depth_layer_present':result['explicit_depth_layer_present'],'metadata_changes':len(changes),'output':str(out)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('--frame',type=int,default=0);p.add_argument('--out',type=Path);a=p.parse_args()
    folder=a.capture.resolve();inspect(folder,a.frame,(a.out or ROOT/'outputs/capture_inputs'/folder.name).resolve())
