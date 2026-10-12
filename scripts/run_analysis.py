import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from src.job_manager import JobManager
from src.common import read_json
p=argparse.ArgumentParser()
p.add_argument('--analysis-config',default='configs/analysis.yaml')
p.add_argument('--detector-config',default='configs/detector.py')
p.add_argument('--checkpoint',required=True)
p.add_argument('--validation',required=True)
p.add_argument('--phase',choices=['A','B','C','final'],default='A')
p.add_argument('--selection',default='runs/selection.json')
p.add_argument('--split',choices=['val','test'],default='val')
p.add_argument('--limit',type=int)
p.add_argument('--device',default='cuda:0')
p.add_argument('--resume',action='store_true')
a=p.parse_args();spec=vars(a).copy();resume=spec.pop('resume')
spec['selection']=read_json(spec['selection'],{})
spec['kind']='analysis'
print(JobManager().launch(spec,resume=resume))
