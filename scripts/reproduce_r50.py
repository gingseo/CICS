"""Train on train; select retained epochs by test AP; report test-selected results."""
import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.job_manager import JobManager
from src.common import write_json, read_json, timestamp, sha256
from src.test_selection import retain_completed, select_best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.chdir(PROJECT)
    root = PROJECT / 'runs/reproduction_r50_ms20'
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = JobManager()
        old = read_json(root/'pipeline.json', {})
        manifest = dict(status='RUNNING', started_at=old.get('started_at',timestamp()),
                        selection_changed_at=timestamp(), training_split='train', selection_split='test',
                        checkpoint_selection='highest test coco/bbox_mAP among retained epoch checkpoints',
                        final_split='test', independent_holdout=False,
                        protocol='test-selected; local YOLO-export COCO bbox; not official VisDrone challenge evaluation',
                        detector_config='configs/detector_r50_visdrone_ms20.py', jobs=[])
        def save():
            write_json(root/'pipeline.json',manifest)
        def capture():
            paths=retain_completed(root)
            manifest['candidate_epochs']=[int(p.stem.split('_')[-1]) for p in paths]
            completed=read_json(root/'completed_epoch.json',{}).get('epoch',0)
            manifest['unavailable_completed_epochs']=sorted(set(range(1,completed+1))-set(manifest['candidate_epochs']))
        def run(spec, capture_training=False):
            job=manager.launch(spec,resume=args.resume)
            manifest['jobs'].append(dict(job_id=job['job_id'],kind=spec['kind'],spec=spec))
            save()
            print(json.dumps(job),flush=True)
            while True:
                if capture_training:
                    capture();save()
                current=next(j for j in manager.list() if j['job_id']==job['job_id'])
                if current['status']=='DONE':return current
                if current['status'] not in ('PENDING','RUNNING'):
                    raise RuntimeError(f"Job {current['job_id']}: {current['status']}: {current.get('error')}")
                time.sleep(10)
        try:
            capture();save()
            job=run(dict(kind='training',detector_config=manifest['detector_config'],
                         work_dir=str(root),amp=False,pretrained=None),capture_training=True)
            subprocess.run([sys.executable,'scripts/plot_training_curves.py','--log',
                            str(Path(job['folder'])/'run.log'),'--output-dir',str(root/'curves')],check=True)
            from mmengine.config import Config
            capture();save()
            provenance=read_json(root/'provenance.json')
            provenance.update(selection_split='test',final_split='test',independent_holdout=False,
                              checkpoint_selection=manifest['checkpoint_selection'],
                              candidate_epochs=manifest['candidate_epochs'],
                              unavailable_completed_epochs=manifest['unavailable_completed_epochs'])
            provenance_path=root/'provenance_test_selected.json'
            write_json(provenance_path,provenance)
            def evaluate(checkpoint, folder):
                folder.mkdir(parents=True,exist_ok=True)
                cfg=Config.fromfile(manifest['detector_config'])
                cfg.test_dataloader.dataset.ann_file=str(PROJECT/'data/test.json')
                cfg.test_dataloader.dataset.data_prefix.img='images/test/'
                cfg.test_evaluator.ann_file=str(PROJECT/'data/test.json')
                config_path=folder/'config.py';cfg.dump(str(config_path))
                output=folder/'report.json'
                run(dict(kind='validation',detector_config=str(config_path),checkpoint=str(checkpoint),
                         provenance=str(provenance_path),output=str(output)))
                result=read_json(output)
                result.update(selection_split='test',evaluation_split='test',independent_holdout=False)
                write_json(output,result)
                return result
            results=[]
            for checkpoint in retain_completed(root):
                epoch=int(checkpoint.stem.split('_')[-1])
                report=evaluate(checkpoint,root/'test_selection'/f'epoch_{epoch}')
                results.append(dict(epoch=epoch,checkpoint=str(checkpoint),metrics=report['metrics']))
                manifest['test_candidate_results']=results
                manifest['best_test_so_far']=select_best(results)
                save()
            best=select_best(results)
            checkpoint=Path(best['checkpoint'])
            manifest.update(selected_epoch=best['epoch'],checkpoint=str(checkpoint),
                            checkpoint_sha256=sha256(checkpoint))
            save()
            final=evaluate(checkpoint,root/'eval_test')
            manifest['test_metrics']=final['metrics']
            keys=list(best['metrics'])
            with (root/'test_selection/epoch_metrics.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['epoch']+keys)
                writer.writeheader()
                writer.writerows(dict(epoch=r['epoch'],**r['metrics']) for r in results)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
            for key,label in [('coco/bbox_mAP','AP'),('coco/bbox_mAP_s','AP small'),
                              ('coco/bbox_mAP_m','AP medium'),('coco/bbox_mAP_l','AP large')]:
                ax.plot([r['epoch'] for r in results],[100*r['metrics'][key] for r in results],marker='.',label=label)
            ax.axvline(best['epoch'],color='gray',linestyle='--',label=f"Selected epoch {best['epoch']}")
            ax.set(xlabel='Epoch (retained checkpoints only)',ylabel='AP (%)',
                   title='Test-selected performance | selection and reporting use the same test set')
            ax.legend();ax.grid(alpha=.2)
            for ext in ['png','pdf','svg']:fig.savefig(root/'test_selection'/f'curves.{ext}',dpi=180)
            plt.close(fig)
            manifest.update(status='DONE',finished_at=timestamp());save()
            print(json.dumps(manifest,indent=2),flush=True)
        except BaseException as exc:
            manifest.update(status='FAILED',error=str(exc),finished_at=timestamp());save()
            raise


if __name__=='__main__':main()
