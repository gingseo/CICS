import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from src.job_manager import save_selection
p=argparse.ArgumentParser()
p.add_argument('--phase',choices=['A','B','C'],required=True)
p.add_argument('--value',required=True)
p.add_argument('--reason',required=True)
p.add_argument('--runs',nargs='+',required=True)
p.add_argument('--output',default='runs/selection.json')
a=p.parse_args()
print(save_selection(a.output,a.phase,float(a.value) if a.phase=='A' else a.value,a.reason,a.runs))
