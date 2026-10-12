import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from src.job_manager import JobManager
p=argparse.ArgumentParser()
p.add_argument('--detector-config',default='configs/detector.py')
p.add_argument('--work-dir',default='runs/training')
p.add_argument('--pretrained')
p.add_argument('--amp',action='store_true')
p.add_argument('--resume',action='store_true')
a=p.parse_args();spec=vars(a).copy();resume=spec.pop('resume');spec['kind']='training'
print(JobManager().launch(spec,resume=resume))
