from pathlib import Path
import time
import subprocess
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from .common import write_json,read_json,timestamp,sha256


@HOOKS.register_module()
class CICSMonitorHook(Hook):
    priority='LOWEST'
    def before_train(self,runner):
        self.start=time.monotonic()
        self.start_iter=runner.iter

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        if runner.iter % 20: return
        losses={k:float(v.detach().mean()) if isinstance(v,torch.Tensor) else float(v)
                for k,v in (outputs or {}).items() if 'loss' in k and isinstance(v,(float,int,torch.Tensor))}
        elapsed=time.monotonic()-self.start
        result=dict(status='RUNNING',epoch=runner.epoch+1,total_epochs=runner.max_epochs,
                    iteration=runner.iter,losses=losses,learning_rate=runner.optim_wrapper.get_lr(),
                    elapsed_seconds=elapsed,eta_seconds=elapsed/max(1,runner.iter-self.start_iter+1)*(runner.max_iters-runner.iter-1))
        try:
            p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2)
            values=[float(x.strip()) for x in p.stdout.strip().splitlines()[0].split(',')] if p.returncode==0 else []
            result['gpu']=dict(zip(['utilization_percent','used_mib','total_mib'],values))
        except (OSError,subprocess.TimeoutExpired,ValueError,IndexError): result['gpu']=None
        old=read_json(Path(runner.work_dir)/'training_status.json',{})
        result['validation']=old.get('validation')
        write_json(Path(runner.work_dir)/'training_status.json',result)
        import json
        with (Path(runner.work_dir)/'training_scalars.jsonl').open('a') as f:
            f.write(json.dumps(result)+'\n')

    def after_val_epoch(self,runner,metrics=None):
        path=Path(runner.work_dir)/'training_status.json'
        s=read_json(path,{})
        # EpochBasedTrainLoop increments epoch before calling validation.
        s.update(validation=metrics,last_validated_epoch=runner.epoch)
        write_json(path,s)
        with (Path(runner.work_dir)/'training_history.jsonl').open('a') as f:
            import json
            f.write(json.dumps(dict(epoch=runner.epoch,metrics=metrics))+'\n')

    def after_train_epoch(self,runner):
        write_json(Path(runner.work_dir)/'completed_epoch.json',dict(epoch=runner.epoch+1,total_epochs=runner.max_epochs))
        path=Path(runner.work_dir)/'provenance.json'
        provenance=read_json(path,{})
        provenance['completed_epochs']=runner.epoch+1
        write_json(path,provenance)


def train(config,work_dir,resume=False,amp=False,pretrained=None):
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    register_all_modules()
    cfg=Config.fromfile(config)
    cfg.work_dir=str(Path(work_dir).resolve())
    cfg.resume=bool(resume and (Path(work_dir)/'last_checkpoint').exists())
    if cfg.resume:
        cfg.load_from=None
    elif pretrained: cfg.load_from=str(Path(pretrained).resolve())
    elif not cfg.resume:
        cached=Path(__file__).resolve().parents[1]/'data/pretrained/faster_rcnn_r50_fpn_coco.pth'
        if cached.exists(): cfg.load_from=str(cached)
    if cfg.load_from or cfg.resume: cfg.model.backbone.init_cfg=None
    if amp:
        cfg.optim_wrapper.type='AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale='dynamic'
    Path(work_dir).mkdir(parents=True,exist_ok=True)
    cfg.dump(str(Path(work_dir)/'resolved_config.py'))
    write_json(Path(work_dir)/'training_status.json',dict(status='RUNNING',started_at=timestamp(),resumed=cfg.resume,restarted_before_first_checkpoint=resume and not cfg.resume))
    from .data_audit import CLASSES
    write_json(Path(work_dir)/'provenance.json',dict(source_url='local fine-tuning',license='local experiment',
        code_commit='mmdet 3.3.0',training_split=['train'],classes=list(CLASSES),preprocessing='resolved_config.py',
        evaluation_protocol='COCO bbox mAP; YOLO export lacks original ignore regions',
        training_annotations_sha256=sha256(cfg.train_dataloader.dataset.ann_file),
        validation_annotations_sha256=sha256(cfg.val_dataloader.dataset.ann_file),
        completed_epochs=read_json(Path(work_dir)/'completed_epoch.json',{}).get('epoch',0),
        initialization_checkpoint=cfg.load_from,
        initialization_sha256=sha256(cfg.load_from) if cfg.load_from and Path(cfg.load_from).is_file() else None,
        requested_epochs=cfg.train_cfg.max_epochs,
        config_path=str(Path(config).resolve()),
        resolved_config_sha256=sha256(Path(work_dir)/'resolved_config.py')))
    try:
        runner=Runner.from_cfg(cfg)
        from .checkpoint_io import checkpoint_globals
        with checkpoint_globals():
            runner.train()
        status=read_json(Path(work_dir)/'training_status.json',{})
        status.update(status='DONE',completed_epochs=runner.epoch,all_60_epochs_complete=runner.epoch>=60,
                      all_requested_epochs_complete=runner.epoch>=runner.max_epochs)
        write_json(Path(work_dir)/'training_status.json',status)
    except BaseException as e:
        status=read_json(Path(work_dir)/'training_status.json',{})
        status.update(status='STOPPED' if isinstance(e,InterruptedError) else 'FAILED',error=str(e))
        write_json(Path(work_dir)/'training_status.json',status)
        raise
