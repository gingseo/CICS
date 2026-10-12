from pathlib import Path
import torch
from .common import write_json,read_json,sha256,timestamp,environment,digest
from .data_audit import CLASSES
from .feature_extractor import build_detector,load_strict


def validate(config,checkpoint,provenance_path,output,device='cuda:0',evaluate=True):
    provenance=read_json(provenance_path)
    required=('source_url','license','code_commit','training_split','classes','preprocessing')
    if not provenance or any(k not in provenance or not provenance[k] for k in required) or any('REPLACE' in str(v) for v in provenance.values()):
        raise ValueError(f'Provenance requires {required}')
    if provenance['classes']!=list(CLASSES): raise ValueError('Class order must match VisDrone 10 classes')
    report=dict(status='RUNNING',checkpoint_sha256=sha256(checkpoint),config_sha256=sha256(config),
                checkpoint=str(Path(checkpoint).resolve()),provenance=provenance,classes=list(CLASSES),
                created_at=timestamp(),environment=environment(),
                evaluation_conditions='provided detector config test pipeline; distinct from analysis tensor preprocessing')
    write_json(output,report)
    try:
        model,cfg=build_detector(config,'cpu')
        report['resolved_config_sha256']=digest(cfg.to_dict())
        mismatches=load_strict(model,checkpoint)
        report.update(mismatches)
        write_json(output,report)
        if any(mismatches.values()): raise ValueError('State dict mismatch; see validation report')
        if evaluate:
            from mmengine.runner import Runner
            # Model already loaded strictly: do not let Runner silently reload mismatched weights.
            cfg.load_from=None
            cfg.resume=False
            cfg.work_dir=str(Path(output).parent/'evaluation')
            cfg.custom_hooks=[]
            runner=Runner.from_cfg(cfg)
            runner.model.load_state_dict(model.state_dict(),strict=True)
            runner.model.to(device)
            del model
            result=runner.test()
            if not result or 'coco/bbox_mAP' not in result:
                raise RuntimeError('Evaluation did not return bbox mAP')
            report.update(status='DONE',metrics=result,evaluation_config=cfg.pretty_text)
        else:
            report.update(status='STRUCTURE_ONLY',metrics=None)
        report['finished_at']=timestamp()
        write_json(output,report)
    except BaseException as e:
        report.update(status='FAILED',error=str(e));write_json(output,report);raise
    return report
