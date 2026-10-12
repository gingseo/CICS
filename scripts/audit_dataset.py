import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from src.data_audit import audit,convert_coco
p=argparse.ArgumentParser()
p.add_argument('--root',default='/mnt/icelab_4tb/datasets/VisDrone')
p.add_argument('--format',choices=['yolo','visdrone'],default='yolo')
p.add_argument('--convert',action='store_true')
p.add_argument('--exclude-invalid',action='store_true')
a=p.parse_args()
r=audit(a.root,a.format)
for split,stats in r['splits'].items():
 print(split,stats['n_images'],stats['n_objects'],len(stats['invalid']),'invalid')
 if a.convert: print(convert_coco(a.root,split,a.format,f'data/{split}.json',exclude_invalid=a.exclude_invalid))
