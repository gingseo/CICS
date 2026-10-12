import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.common import environment,write_json
r=environment()
try:
 from mmcv.ops import nms,roi_align
 a=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]],device='cuda')
 nms(a,torch.tensor([.9,.8],device='cuda'),.5)
 x=torch.rand(1,3,16,16,device='cuda',requires_grad=True)
 rois=torch.tensor([[0.,0.,0.,8.,8.]],device='cuda')
 roi_align(x,rois,(3,3),1.,0,'avg',True).sum().backward()
 r['custom_ops']='PASS CUDA nms + roi_align forward/backward'
except Exception as e:
 r['custom_ops']='FAILED: '+repr(e)
write_json('reports/environment.json',r)
print(r)

if r['custom_ops'].startswith('FAILED'): sys.exit(1)
