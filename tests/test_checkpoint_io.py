import pytest
import torch
from src.checkpoint_io import load_checkpoint,history_getattr


def test_mmengine_metadata_weights_only_roundtrip(tmp_path):
    mmengine=pytest.importorskip('mmengine')
    from mmengine.logging import HistoryBuffer
    p=tmp_path/'checkpoint.pth'
    torch.save(dict(state_dict={'a':torch.ones(3)},message_hub={'loss':HistoryBuffer([1.,2.],[1,1])}),p)
    restored=load_checkpoint(p)
    assert torch.equal(restored['state_dict']['a'],torch.ones(3))
    assert restored['message_hub']['loss'].mean()==1.5


def test_metadata_getattr_is_restricted():
    pytest.importorskip('mmengine')
    from mmengine.logging import HistoryBuffer
    with pytest.raises(ValueError): history_getattr(HistoryBuffer,'__class__')
    with pytest.raises(ValueError): history_getattr(torch,'load')
