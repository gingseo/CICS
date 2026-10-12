"""Durable process jobs with a single GPU worker, atomic state and advisory locks."""
from pathlib import Path
from contextlib import contextmanager
import fcntl
import os
import signal
import subprocess
import sys
from .common import read_json,write_json,digest,timestamp

PROJECT=Path(__file__).resolve().parents[1]


def process_identity(pid):
    try:
        # /proc starttime prevents PID reuse from targeting an unrelated process.
        raw=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return raw[19] if raw[0]!='Z' else None
    except (FileNotFoundError,ProcessLookupError): return None


class JobManager:
    def __init__(self,root='runs/jobs'):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)

    def list(self):
        result=[]
        for p in sorted(self.root.glob('*/job.json')):
            s=read_json(p)
            if s['status'] in ('PENDING','RUNNING') and s.get('pid') and process_identity(s['pid'])!=s.get('process_identity'):
                s.update(status='FAILED',error='Worker exited without completion; logs and shards retained')
                write_json(p,s)
            result.append(s)
        return result

    def launch(self,spec,resume=False):
        job_id=digest(spec)[:20];folder=self.root/job_id
        with (self.root/'.launch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            jobs=self.list()
            old=next((j for j in jobs if j['job_id']==job_id),None)
            if old and old['status'] in ('PENDING','RUNNING','DONE'): return old
            if old and not resume: raise ValueError('Job failed/stopped. Explicitly resume it to retain progress.')
            if any(j['status'] in ('PENDING','RUNNING') for j in jobs):
                raise RuntimeError('Another worker is active; stop it or wait before launching')
            folder.mkdir(exist_ok=True)
            if (folder/'STOP').exists(): (folder/'STOP').unlink()
            write_json(folder/'spec.json',spec)
            state=dict(job_id=job_id,status='PENDING',kind=spec['kind'],created_at=timestamp(),folder=str(folder),resume=resume)
            write_json(folder/'job.json',state)
            with (folder/'run.log').open('a') as log:
                proc=subprocess.Popen([sys.executable,'-m','src.worker',str(folder)],cwd=PROJECT,
                                      stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                      env=dict(os.environ,PYTHONHASHSEED='128',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
            state.update(pid=proc.pid,process_identity=process_identity(proc.pid))
            write_json(folder/'job.json',state)
            return state

    def stop(self,job_id):
        folder=self.root/job_id
        if folder.parent!=self.root: raise ValueError('Invalid job ID')
        state=read_json(folder/'job.json')
        if not state or state['status'] not in ('RUNNING','PENDING'): return
        (folder/'STOP').touch()
        if state.get('pid') and process_identity(state['pid'])==state.get('process_identity'):
            # Analysis observes STOP between images; training SIGTERM preserves completed checkpoints.
            if state['kind']!='analysis': os.killpg(state['pid'],signal.SIGTERM)


def save_selection(path,phase,value,reason,run_ids):
    if not reason.strip() or not run_ids: raise ValueError('Selection reason and evidence run IDs required')
    key={'A':'ratio','B':'stage','C':'metric'}[phase]
    valid={'A':(.75,.5,.25),'B':('C3','C4','C5'),'C':('cosine','l1','l2')}
    if value not in valid[phase]: raise ValueError('Invalid selection')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        selection=read_json(path,{'history':[]})
        if phase=='B' and 'ratio' not in selection: raise ValueError('Select A first')
        if phase=='C' and 'stage' not in selection: raise ValueError('Select B first')
        manifests=[]
        for run in run_ids:
            folder=Path(run)
            m=read_json(folder/'manifest.json')
            if not m or read_json(folder/'state.json',{}).get('status')!='DONE':
                raise ValueError('Selection evidence must reference completed analysis runs')
            manifests.append(m)
        scopes={(m['checkpoint_sha256'],m['dataset_fingerprint'],m['split']) for m in manifests}
        if len(scopes)!=1: raise ValueError('Selection evidence must share checkpoint and dataset')
        scope=list(next(iter(scopes)))
        if phase!='A' and selection.get('scope')!=scope:
            raise ValueError('Upstream selection uses a different checkpoint or dataset')
        if phase=='A' and {m['ratio'] for m in manifests}!={.75,.5,.25}:
            raise ValueError('Complete degradation comparison required')
        if phase=='B' and ({m['stage'] for m in manifests}!={'C3','C4','C5'} or any(m['ratio']!=selection['ratio'] for m in manifests)):
            raise ValueError('Feature comparison must use the selected degradation')
        if phase=='C' and ({m['metric'] for m in manifests}!={'cosine','l1','l2'} or any(m['ratio']!=selection['ratio'] or m['stage']!=selection['stage'] for m in manifests)):
            raise ValueError('Metric comparison must use the selected degradation and stage')
        selection['scope']=scope
        # Changing an upstream decision invalidates downstream decisions, retains their history.
        for downstream in {'A':['stage','metric'],'B':['metric'],'C':[]}[phase]: selection.pop(downstream,None)
        selection[key]=value
        selection['history'].append(dict(phase=phase,value=value,reason=reason,run_ids=run_ids,selected_at=timestamp()))
        write_json(path,selection)
    return selection


@contextmanager
def gpu_lease(root='runs/jobs'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.gpu.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError('GPU worker/viewer is busy') from e
        yield
