from pathlib import Path
import sys
import time
import signal
import traceback
import fcntl
import yaml
from .common import read_json,write_json,timestamp,sha256


def main(folder):
    folder=Path(folder)
    # Parent writes process identity while holding the launch lock.
    with (folder.parent/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
    state=read_json(folder/'job.json');spec=read_json(folder/'spec.json')
    state.update(status='RUNNING',started_at=timestamp());write_json(folder/'job.json',state)
    def interrupted(signum,frame): raise InterruptedError('Worker stopped; completed checkpoints retained')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        with (folder.parent/'.gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if spec['kind']=='analysis':
                from .analysis_runner import run_analysis,combinations
                from .feature_extractor import FeatureExtractor
                config=yaml.safe_load(Path(spec['analysis_config']).read_text())
                config['purpose']='exploratory_test_selected' if config.get('checkpoint_selection_split')=='test' else ('final_report' if spec['phase']=='final' else 'selection')
                config['detector_config_sha256']=sha256(spec['detector_config'])
                selection=spec.get('selection',{})
                if spec['phase'] in ('B','C') and selection.get('scope', [None])[0]!=sha256(spec['checkpoint']):
                    raise ValueError('Selection checkpoint differs; choose conditions for this checkpoint first')
                conditions=combinations(spec['phase'],selection)
                extractor=FeatureExtractor(spec['detector_config'],spec['checkpoint'],spec.get('device','cuda:0'),spec['validation'])
                extractor.provenance=read_json(spec['validation'])['provenance']
                config['checkpoint_provenance']=extractor.provenance
                if extractor.provenance.get('purpose')=='integration_test_only': config['purpose']='integration_test_only'
                config['selection_snapshot']=selection
                def progress(value):
                    state['progress']=value;write_json(folder/'job.json',state)
                try:
                    state['runs']=run_analysis(config,extractor,spec['checkpoint'],conditions,
                        split=spec.get('split','val'),limit=spec.get('limit'),progress=progress,stop=lambda:(folder/'STOP').exists())
                finally: extractor.close()
                state['phase_status']='DONE' if spec['phase']=='final' else 'WAITING_FOR_SELECTION'
            elif spec['kind']=='training':
                from .train import train
                train(spec['detector_config'],spec['work_dir'],resume=state.get('resume',False),amp=spec.get('amp',False),pretrained=spec.get('pretrained'))
            elif spec['kind']=='validation':
                from .checkpoint_validation import validate
                validate(spec['detector_config'],spec['checkpoint'],spec['provenance'],spec['output'],spec.get('device','cuda:0'))
            else: raise ValueError('Unknown job kind')
        state.update(status='DONE',finished_at=timestamp())
        if 'progress' in state: state['progress']['status']='DONE'
    except BaseException as e:
        traceback.print_exc()
        state.update(status='STOPPED' if isinstance(e,InterruptedError) else 'FAILED',error=str(e),finished_at=timestamp())
    write_json(folder/'job.json',state)

if __name__=='__main__': main(sys.argv[1])
