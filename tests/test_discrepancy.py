import torch
import pytest
from src.discrepancy import discrepancy,to_input_map


def test_same_input_and_channel_axis():
    a=torch.tensor([[[[1.,2.]],[[2.,1.]]]])
    d,v=discrepancy(a,a)
    assert d.shape==(1,1,2)
    assert v.all() and torch.max(d)<1e-6
    opposite,_=discrepancy(a,-a)
    assert torch.allclose(opposite,torch.full_like(opposite,2.))


def test_constant_shift_l1_rms():
    a=torch.ones(1,4,2,3)
    for m in ('l1','l2'):
        d,v=discrepancy(a,a+3,m)
        assert torch.allclose(d,torch.full_like(d,3))
        assert v.all()


def test_zero_vector_invalid_only_for_cosine():
    a=torch.zeros(1,3,2,2)
    d,v=discrepancy(a,a)
    assert not v.any() and d.isnan().all()
    assert torch.equal(discrepancy(a,a,'l1')[0],torch.zeros(1,2,2))


def test_invalid_upsampling_is_not_invented_information():
    d=torch.tensor([[[1.,float('nan')],[1.,1.]]])
    valid=torch.isfinite(d)
    up,mask=to_input_map(d,valid,(8,8))
    assert not mask.all() and torch.all(up[mask]==1)
    assert up[~mask].isnan().all()
