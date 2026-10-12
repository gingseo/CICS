import numpy as np
from PIL import Image
import torch
from src.degradation import prepare,degrade,network_input,transform_box
from src.region_pooling import pool_box,pool_background,exclusion_mask
from src.data_audit import size_bin


def test_resize_padding_and_box_geometry(tmp_path):
    p=tmp_path/'x.png';Image.new('RGB',(100,51),'white').save(p)
    cfg=dict(long_edge=64,pad_divisor=32,mean=[0,0,0],std=[1,1,1])
    valid,meta=prepare(p,cfg)
    assert meta['resized_hw']==[33,64] and meta['network_hw']==[64,64]
    assert transform_box([0,0,100,51],meta)==[0,0,64,33]
    deg,info=degrade(valid,.5)
    assert info['intermediate_hw']==[17,32]
    assert torch.allclose(deg,valid,atol=1e-4)
    x=network_input(deg,meta,cfg)
    assert x.shape==(1,3,64,64) and x[:,:,33:,:].count_nonzero()==0


def test_tiny_fractional_box_has_defined_score():
    values=np.array([[2.,8.],[4.,6.]])
    pooled=pool_box(values,[.9,.1,.2,.2])
    assert np.isclose(pooled['score'],5)
    assert np.isclose(pooled['valid_sample_area'],.04)
    assert pooled['valid_sample_count']==2


def test_union_ignore_padding_exclusion():
    d=np.arange(16).reshape(4,4).astype(float)
    # Valid 3x3; overlapping boxes remove (0,0),(0,1); ignore removes (1,1).
    bg=pool_background(d,(3,3),[[0,0,1.5,1],[1,0,1,1]],[[1,1,1,1]])
    assert bg['valid_sample_count']==6
    assert bg['sum']==sum([2,4,6,8,9,10])
    allowed=~exclusion_mask(d.shape,[[0,0,1,1]])
    assert pool_box(d,[0,0,.1,.1],allowed)['score'] is None


def test_original_size_boundary_bins():
    expected={1:'underflow',2:'VT',8:'T',16:'S',32:'M',64:'L',0:'invalid'}
    for s,name in expected.items(): assert size_bin(s,s)==name


def test_highres_preprocessing_matches_detector_resize(tmp_path):
    import mmcv
    from mmdet.datasets.transforms import Resize
    rng=np.random.default_rng(128)
    cfg=dict(long_edge=1333,short_edge=800,resize_backend='opencv',antialias=False,
             pad_divisor=32,mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375])
    for width,height in [(1360,765),(1400,1050),(765,1360),(600,600)]:
        rgb=rng.integers(0,256,(height,width,3),dtype=np.uint8)
        p=tmp_path/f'{width}_{height}.png';Image.fromarray(rgb).save(p)
        x,meta=prepare(p,cfg)
        reference=Resize(scale=(1333,800),keep_ratio=True)(dict(img=rgb.copy(),img_shape=rgb.shape[:2]))
        np.testing.assert_array_equal(x[0].permute(1,2,0).numpy(),reference['img'])
        assert max(meta['resized_hw'])<=1333 and min(meta['resized_hw'])<=800
        np.testing.assert_allclose(meta['scale_factor'],reference['scale_factor'])
        assert network_input(x,meta,cfg).shape[-2:]==tuple(meta['network_hw'])
