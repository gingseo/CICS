import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from src.checkpoint_validation import validate
p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/detector.py')
p.add_argument('--checkpoint',required=True)
p.add_argument('--provenance',required=True)
p.add_argument('--output',default='runs/checkpoint_validation.json')
p.add_argument('--device',default='cuda:0')
p.add_argument('--structure-only',action='store_true')
a=p.parse_args()
print(validate(a.config,a.checkpoint,a.provenance,a.output,a.device,not a.structure_only))
