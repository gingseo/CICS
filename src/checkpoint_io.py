"""Restricted MMEngine checkpoint metadata support for PyTorch >= 2.6."""
from contextlib import contextmanager
import numpy as np
import torch


def history_getattr(owner,name):
    from mmengine.logging import HistoryBuffer
    if owner is not HistoryBuffer or name not in ('min','max','current','mean'):
        raise ValueError('Unsupported checkpoint metadata attribute')
    return getattr(HistoryBuffer,name)


@contextmanager
def checkpoint_globals():
    from mmengine.logging import HistoryBuffer
    from numpy._core.multiarray import _reconstruct,scalar
    allowed=[HistoryBuffer, np.ndarray, np.dtype, _reconstruct, scalar,
             (history_getattr,'builtins.getattr'),torch.torch_version.TorchVersion,
             type(np.dtype('float64')),type(np.dtype('float32')),type(np.dtype('int64')),
             type(np.dtype('int32')),type(np.dtype('uint32'))]
    with torch.serialization.safe_globals(allowed):
        yield


def load_checkpoint(path):
    with checkpoint_globals():
        return torch.load(path,map_location='cpu',weights_only=True)
