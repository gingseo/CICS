from pathlib import Path
import copy
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.analysis_runner import run_analysis,combinations,analyze_image
from src.common import read_json
from src.data_audit import load_records


class FixtureExtractor:
    """Deterministic synthetic features for pipeline correctness, never a detector result."""
    def __init__(self): self.calls=0;self.mapping={}
    def __call__(self,x):
        self.calls+=1
        self.mapping={s:dict(stride=k,module='TEST_ONLY',channels=3,shape=[1,3,x.shape[-2]//k,x.shape[-1]//k]) for s,k in [('C3',8),('C4',16),('C5',32)]}
        return {s:F.avg_pool2d(x,k)+5 for s,k in [('C3',8),('C4',16),('C5',32)]}


def test_pipeline_reuse_and_artifacts(tmp_path):
    root=tmp_path/'dataset';(root/'images'/'val').mkdir(parents=True);(root/'labels'/'val').mkdir(parents=True)
    arr=np.random.default_rng(128).integers(0,255,(50,80,3),dtype=np.uint8)
    Image.fromarray(arr).save(root/'images'/'val'/'one.png')
    (root/'labels'/'val'/'one.txt').write_text('0 .5 .5 .01 .01\n3 .5 .5 .4 .4\n')
    checkpoint=tmp_path/'fixture.bin';checkpoint.write_bytes(b'TEST ONLY')
    cfg=dict(seed=128,dataset_root=str(root),format='yolo',split='val',purpose='test_fixture',runs_dir=str(tmp_path/'runs'),
        preprocess=dict(long_edge=64,pad_divisor=32,mean=[0,0,0],std=[255,255,255]),epsilon=1e-8,
        qualitative_limit=1,heatmap_max=dict(cosine=2,l1=1,l2=1))
    extractor=FixtureExtractor()
    runs=run_analysis(cfg,extractor,checkpoint,combinations('A'))
    assert extractor.calls==4  # one original + three degraded forwards
    for run in runs:
        p=Path(run)
        assert read_json(p/'state.json')['status']=='DONE'
        assert (p/'figures'/'boxplot.pdf').exists()
        assert (p/'qualitative'/'one'/'gt.png').exists()
        assert read_json(p/'manifest.json')['module_mapping']['C3']['module']=='TEST_ONLY'
    previous=extractor.calls
    assert run_analysis(cfg,extractor,checkpoint,combinations('A'))==runs
    assert extractor.calls==previous
    cfg['selection_snapshot']={'ratio':.5}
    stage_runs=run_analysis(cfg,extractor,checkpoint,combinations('B',{'ratio':.5}))
    assert stage_runs[0]==runs[1]
    assert extractor.calls==previous+2  # same pair used for C4 and C5


def test_unlabeled_viewer_has_no_background_statistics(tmp_path):
    Image.new('RGB',(50,50),'white').save(tmp_path/'a.png')
    record=load_records(tmp_path,'val',image_dir=tmp_path)[0]
    cfg=dict(preprocess=dict(long_edge=64,pad_divisor=32,mean=[0]*3,std=[255]*3),epsilon=1e-8)
    out=analyze_image(FixtureExtractor(),record,cfg,[(.5,'C3','cosine')])[(.5,'C3','cosine')]
    assert out['background'] is None and not out['gt_available'] and out['objects']==[]


def test_resume_keeps_completed_images(tmp_path):
    import pytest
    root=tmp_path/'dataset';(root/'images'/'val').mkdir(parents=True);(root/'labels'/'val').mkdir(parents=True)
    for i in range(2):
        Image.new('RGB',(50,50),(i*20+30,40,60)).save(root/'images'/'val'/f'{i}.png')
        (root/'labels'/'val'/f'{i}.txt').write_text('0 .5 .5 .1 .1\n')
    cp=tmp_path/'fixture';cp.write_bytes(b'fixture')
    cfg=dict(seed=128,dataset_root=str(root),format='yolo',split='val',purpose='test_fixture',runs_dir=str(tmp_path/'runs'),
        preprocess=dict(long_edge=64,pad_divisor=32,mean=[0,0,0],std=[255,255,255]),epsilon=1e-8,
        qualitative_limit=0,heatmap_max=dict(cosine=2,l1=1,l2=1))
    e=FixtureExtractor();stop=[False]
    with pytest.raises(InterruptedError):
        run_analysis(cfg,e,cp,[(.5,'C3','l1')],progress=lambda _:stop.__setitem__(0,True),stop=lambda:stop[0])
    assert e.calls==2
    runs=run_analysis(cfg,e,cp,[(.5,'C3','l1')])
    assert e.calls==4
    assert read_json(Path(runs[0])/'state.json')['status']=='DONE'
