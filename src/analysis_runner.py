from pathlib import Path
import time
import math
import os
import pandas as pd
import numpy as np
import yaml
from .common import write_json,read_json,sha256,digest,environment,seed_all,timestamp
from .data_audit import load_records,choose_split
from .degradation import prepare,degrade,network_input,transform_box
from .discrepancy import discrepancy,to_input_map
from .region_pooling import pool_box,pool_background,exclusion_mask,box_slice
from .statistics import export_tables
from .qualitative import render,save_views

OBJECT_FIELDS=['run_id','image_id','object_id','class_id','original_w','original_h','original_size',
 'input_w','input_h','input_size','size_bin','checkpoint_id','ratio','stage','metric','score',
 'valid_sample_count','valid_sample_area','native_cell_count','native_cell_area','original_bbox','input_bbox']
BG_FIELDS=['run_id','image_id','score','sum','sum_squares','valid_sample_count']


def combinations(phase,selection=None):
    selection=selection or {}
    if phase=='A': return [(r,'C3','cosine') for r in (.75,.5,.25)]
    if 'ratio' not in selection: raise ValueError('Select degradation after A first')
    if phase=='B': return [(selection['ratio'],s,'cosine') for s in ('C3','C4','C5')]
    if 'stage' not in selection: raise ValueError('Select feature stage after B first')
    if phase=='C': return [(selection['ratio'],selection['stage'],m) for m in ('cosine','l1','l2')]
    if phase=='final':
        if 'metric' not in selection: raise ValueError('Select metric after C first')
        return [(selection['ratio'],selection['stage'],selection['metric'])]
    raise ValueError(phase)


def analyze_image(extractor,record,config,conditions):
    preprocessing=config['preprocess']
    backend=preprocessing.get('resize_backend','torch')
    if preprocessing.get('channel_order','RGB')!='RGB' or preprocessing.get('interpolation','bilinear')!='bilinear' or preprocessing.get('align_corners',False):
        raise ValueError('Supported preprocessing: RGB, bilinear, align_corners=False')
    expected_antialias = backend == 'torch'
    if backend not in ('torch','opencv') or preprocessing.get('antialias',True) != expected_antialias:
        raise ValueError('Torch resize requires antialias=True; detector-compatible OpenCV resize requires antialias=False')
    if preprocessing.get('short_edge') is not None and preprocessing['short_edge']<=0:
        raise ValueError('Short edge must be positive')
    if preprocessing['long_edge']<=0 or preprocessing['pad_divisor']<=0:
        raise ValueError('Image size and padding divisor must be positive')
    if hasattr(extractor,'cfg'):
        pre=extractor.cfg.model.data_preprocessor
        if list(pre.mean)!=preprocessing['mean'] or list(pre.std)!=preprocessing['std'] or pre.pad_size_divisor!=preprocessing['pad_divisor'] or not pre.get('bgr_to_rgb',False):
            raise ValueError('Analysis normalization/channel order/padding must match detector config')
    policy=config.get('degradation',{})
    for key,value in dict(downsample='bilinear',upsample='bilinear',antialias=True,align_corners=False,rounding='half_up',dtype='float32').items():
        if key in policy and policy[key]!=value: raise ValueError(f'Unsupported degradation {key}: {policy[key]}')
    original,meta=prepare(record['path'],preprocessing)
    ori=extractor(network_input(original,meta,config['preprocess']))
    ignored=[transform_box(o['bbox'],meta) for o in record['ignored']]
    boxes=[transform_box(o['bbox'],meta) for o in record['objects']]
    allowed=np.zeros(meta['network_hw'],dtype=bool)
    h,w=meta['resized_hw'];allowed[:h,:w]=True
    allowed &= ~exclusion_mask(allowed.shape,ignored)
    outputs={}
    for ratio in sorted({c[0] for c in conditions}):
        deg,degmeta=degrade(original,ratio)
        features=extractor(network_input(deg,meta,config['preprocess']))
        for r,stage,metric in conditions:
            if r!=ratio: continue
            raw,valid=discrepancy(ori[stage],features[stage],metric,config['epsilon'])
            up,_=to_input_map(raw,valid,meta['network_hw'])
            values=up[0].cpu().numpy()
            objects=[]
            for o,box in zip(record['objects'],boxes):
                ow,oh=o['bbox'][2:];iw,ih=box[2:]
                pooled=pool_box(values,box,allowed)
                stride=extractor.mapping[stage]['stride']
                _,native_weights=box_slice([v/stride for v in box],tuple(ori[stage].shape[-2:]))
                objects.append(dict(image_id=record['image_id'],object_id=o['object_id'],class_id=o['class_id'],
                    original_w=ow,original_h=oh,original_size=math.sqrt(ow*oh),input_w=iw,input_h=ih,
                    input_size=math.sqrt(iw*ih),size_bin=o['size_bin'],original_bbox=o['bbox'],input_bbox=box,
                    native_cell_count=int((native_weights>0).sum()),native_cell_area=iw*ih/(extractor.mapping[stage]['stride']**2),**pooled))
            bg=pool_background(values,meta['resized_hw'],boxes,ignored) if record['gt_available'] else None
            outputs[(r,stage,metric)]=dict(objects=objects,background=bg,values=values,original=original,degraded=deg,
                geometry=meta,degradation=degmeta,invalid_feature_ratio=float((~valid).float().mean()),
                module_mapping=dict(extractor.mapping),gt_available=record['gt_available'])
    return outputs


def run_analysis(config,extractor,checkpoint,conditions,split=None,limit=None,progress=None,stop=None):
    seed_all(config['seed'])
    requested=split or config['split']
    actual,fallback=choose_split(config['dataset_root'],requested,config['format'])
    records=load_records(config['dataset_root'],actual,config['format'])
    if limit: records=records[:limit]
    # Content-based identity prevents stale reuse when images, labels or preprocessing change.
    data_identity=[(r['image_id'],sha256(r['path']),sha256(r['annotation_path'])) for r in records]
    identity_config={k:v for k,v in config.items() if k!='selection_snapshot'}
    common=dict(resolved_detector_sha256=digest(extractor.cfg.to_dict()) if hasattr(extractor,'cfg') else None,config=identity_config,checkpoint_sha256=sha256(checkpoint),split=actual,requested_split=requested,
                split_fallback_reason=fallback,dataset_fingerprint=digest(data_identity),limit=limit,
                purpose=config['purpose'],schema_version=1,code_fingerprint=digest({p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}))
    directories={};todo=[]
    for condition in conditions:
        r,s,m=condition
        identity=dict(**common,ratio=r,stage=s,metric=m)
        run_id=digest(identity)[:20]
        out=Path(config['runs_dir'])/run_id
        directories[condition]=out
        if read_json(out/'state.json',{}).get('status')=='DONE': continue
        out.mkdir(parents=True,exist_ok=True)
        (out/'shards').mkdir(exist_ok=True)
        (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        manifest=dict(identity,run_id=run_id,created_at=timestamp(),environment=environment(),
            selection_snapshot=config.get('selection_snapshot',{}),checkpoint_path=str(Path(checkpoint).resolve()),ignore_regions_preserved=config['format']=='visdrone',
            bg_rule='valid input pixels outside touched GT/ignore pixels',
            object_rule='fractional area mean, instances equally weighted',
            zero_norm_rule='invalid if either norm <= epsilon; strict valid interpolation footprint',
            size_bins='underflow (0,2), VT [2,8), T [8,16), S [16,32), M [32,64), L [64,inf)',
            reproducibility='seeded; deterministic algorithms warn_only; any warnings remain in run.log',
            training_split=extractor.provenance.get('training_split','unknown') if hasattr(extractor,'provenance') else 'unknown')
        training_split=manifest['training_split']
        manifest['checkpoint_selection_split']=getattr(extractor,'provenance',{}).get('selection_split','unknown')
        manifest['independent_holdout']=False if manifest['checkpoint_selection_split']==actual else 'not_established'
        manifest['analysis_split_in_training']='unknown' if training_split=='unknown' else (actual in training_split)
        write_json(out/'manifest.json',manifest)
        write_json(out/'state.json',dict(status='RUNNING',done=0,total=len(records)))
        todo.append(condition)
    start=time.monotonic()
    try:
        for index,record in enumerate(records):
            if stop and stop(): raise InterruptedError('Stopped by user; completed image shards retained')
            needed=[c for c in todo if not (directories[c]/'shards'/f'{index:07d}.json').exists()]
            outputs=analyze_image(extractor,record,config,needed) if needed else {}
            for condition,output in outputs.items():
                out=directories[condition];r,s,m=condition
                context=dict(run_id=out.name,checkpoint_id=common['checkpoint_sha256'][:16],ratio=r,stage=s,metric=m)
                objects=[dict(**context,**o) for o in output['objects']]
                bg=dict(run_id=out.name,image_id=record['image_id'],**output['background'])
                shard=dict(objects=objects,background=bg,geometry=output['geometry'],degradation=output['degradation'],
                           invalid_feature_ratio=output['invalid_feature_ratio'],image_id=record['image_id'],excluded_annotations=record['invalid'])
                if index<config.get('qualitative_limit',12):
                    folder=out/'qualitative'/record['image_id']
                    save_views(render(output['original'],output['degraded'],output['values'],objects,
                                      output['geometry']['resized_hw'],config['heatmap_max'][m]),folder)
                    write_json(folder/'scores.json',shard)
                    np.savez_compressed(folder/'map.npz',discrepancy=output['values'])
                write_json(out/'shards'/f'{index:07d}.json',shard)
                manifest=read_json(out/'manifest.json');manifest['module_mapping']=output['module_mapping']
                write_json(out/'manifest.json',manifest)
            elapsed=time.monotonic()-start
            status=dict(status='RUNNING',done=index+1,total=len(records),elapsed_seconds=elapsed,
                        eta_seconds=elapsed/(index+1)*(len(records)-index-1),current_image=record['image_id'])
            for c in todo: write_json(directories[c]/'state.json',status)
            if progress: progress(status)
        for condition in todo:
            out=directories[condition]
            shards=[read_json(p) for p in sorted((out/'shards').glob('*.json'))]
            objects=pd.DataFrame([o for shard in shards for o in shard['objects']],columns=OBJECT_FIELDS)
            bg=pd.DataFrame([s['background'] for s in shards],columns=BG_FIELDS)
            export_tables(objects,bg,out,config['seed'])
            write_json(out/'state.json',dict(status='DONE',done=len(records),total=len(records),finished_at=timestamp()))
            (out/'run.log').write_text(f'Completed {len(records)} images. See manifest.json and image shards.\n')
    except BaseException as e:
        for condition in todo:
            write_json(directories[condition]/'state.json',dict(status='STOPPED' if isinstance(e,InterruptedError) else 'FAILED',error=str(e)))
        raise
    return [str(directories[c]) for c in conditions]
