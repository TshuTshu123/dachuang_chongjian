"""MANO 成对启用门槛：比较双手候选，明显过厚且候选改善才一起替换。

由 run.py 的 MANO 步骤读取刚完成的掌/腕截面诊断。缺少可靠截面时跳过。
原掌厚超过腕厚 25% 且至少 8 mm，候选同截面又减少至少 8 mm，才判异常。
任一侧满足时成对替换；不按编号选择，也不因候选与原手不同就直接替换。
这是外观异常启发式，不是解剖或精度判断。
"""


def choose_hands(report):
    records=[];selected=[]
    for part in report.get('hand_processing',{}).get('palm_volume',{}).get('parts',[]):
        name=part['part'];palm=part.get('after_palm_p85_mm') if part.get('applied') else part.get('before_palm_p85_mm')
        wrist=part.get('wrist_reference_mm')
        valid=palm is not None and wrist is not None and part.get('valid_palm_rays',0)>=80
        abnormal=bool(valid and palm>wrist*1.25 and palm-wrist>8.)
        records.append({'part':name,'palm_p85_mm':palm,'wrist_reference_mm':wrist,'reference_valid':valid,'selected_for_mano':abnormal,'reason':'Clear excess palm thickness' if abnormal else ('Within thickness threshold' if valid else 'Insufficient reliable thickness reference')})
        if abnormal:selected.append(name)
    return selected,{'method':'per_hand_palm_vs_wrist_thickness','ratio_threshold':1.25,'minimum_excess_mm':8.,'parts':records,'uses_scan_id':False}


def compare_pair(report, mesh, info, candidate_folder):
    from pathlib import Path
    import numpy as np
    import open3d as o3d
    from hand_centerlines import hand_frame
    from hand_pipeline import scene_for, depths
    _, evidence=choose_hands(report)
    source=np.asarray(mesh.vertices);comparisons=[]
    diagnostics={p['part']:p for p in report.get('hand_processing',{}).get('palm_volume',{}).get('parts',[])}
    abnormal={p['part']:p for p in evidence['parts']}
    for name in ['left_hand','right_hand']:
        record={'part':name,'source_abnormally_thick':abnormal.get(name,{}).get('selected_for_mano',False),'large_improving_difference':False}
        ids,arm,wrist,axes=hand_frame(source,info['parts'],name)
        local=(source-wrist)@axes
        end=diagnostics.get(name,{}).get('palm_end_mm',np.quantile(local[ids,1],.98)*.5)
        palm_ids=ids[(local[ids,1]>10)&(local[ids,1]<end)]
        if len(palm_ids)<30:
            comparisons.append({**record,'reason':'Insufficient palm region'});continue
        low,high=np.quantile(local[palm_ids,0],[.15,.85]);xx,yy=np.meshgrid(np.linspace(low,high,15),np.linspace(12,end,15))
        probes=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]@axes.T+wrist
        region=np.all(local>=local[ids].min(0)-15,axis=1)&np.all(local<=local[ids].max(0)+15,axis=1)
        candidate=o3d.io.read_triangle_mesh(str(Path(candidate_folder)/(name+'_template.ply')))
        _,_,before,ok_before=depths(scene_for(mesh,region),probes,axes[:,2])
        _,_,after,ok_after=depths(scene_for(candidate,np.ones(len(candidate.vertices),bool)),probes,axes[:,2])
        common=ok_before&ok_after;count=int(common.sum())
        if count<40:
            comparisons.append({**record,'reason':'Too few corresponding template palm sections','common_sections':count});continue
        a,b=float(np.quantile(before[common],.85)),float(np.quantile(after[common],.85));gain=a-b
        record.update(original_palm_p85_mm=a,template_palm_p85_mm=b,thickness_decrease_mm=gain,common_sections=count,
                      large_improving_difference=bool(record['source_abnormally_thick'] and gain>=8.),
                      reason='Compared thickness on common palm rays; not ground-truth accuracy')
        comparisons.append(record)
    use=any(p['large_improving_difference'] for p in comparisons)
    return ['left_hand','right_hand'] if use else [],{'method':'paired_template_comparison','pair_replacement':use,'source_thickness_evidence':evidence,'candidate_comparisons':comparisons,'minimum_thickness_gain_mm':8.,'uses_scan_id':False}
