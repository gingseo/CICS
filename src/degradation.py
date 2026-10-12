import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def round_half_up(x):
    return max(1, int(math.floor(x + 0.5)))


def prepare(path, config):
    with Image.open(path) as im:
        a = np.array(im.convert('RGB'), copy=True)
    h,w = a.shape[:2]
    scale = config['long_edge'] / max(h,w)
    if config.get('short_edge') is not None:
        scale = min(scale, config['short_edge'] / min(h,w))
    nh,nw = round_half_up(h*scale),round_half_up(w*scale)
    backend = config.get('resize_backend', 'torch')
    if backend == 'opencv':
        import mmcv
        a = mmcv.imresize(a, (nw,nh), interpolation='bilinear', backend='cv2')
        x = torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()
    elif backend == 'torch':
        x = torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()
        x = F.interpolate(x, (nh,nw),mode='bilinear',align_corners=False,antialias=True)
    else:
        raise ValueError(f'Unsupported resize backend: {backend}')
    div = config['pad_divisor']
    ph,pw = math.ceil(nh/div)*div,math.ceil(nw/div)*div
    meta = dict(resize_backend=backend,original_hw=[h,w],resized_hw=[nh,nw],network_hw=[ph,pw],
                scale_factor=[nw/w,nh/h],padding_offset=[0,0],padding=[0,pw-nw,0,ph-nh])
    return x,meta


def degrade(valid_image, ratio):
    if not 0 < ratio <= 1:
        raise ValueError('ratio must be in (0,1]')
    h,w = valid_image.shape[-2:]
    mid = (round_half_up(h*ratio),round_half_up(w*ratio))
    low = F.interpolate(valid_image.float(),mid,mode='bilinear',align_corners=False,antialias=True)
    result = F.interpolate(low,(h,w),mode='bilinear',align_corners=False)
    return result,dict(ratio=ratio,intermediate_hw=list(mid),valid_hw=[h,w])


def network_input(valid, meta, config):
    mean = torch.tensor(config['mean'],device=valid.device).view(1,3,1,1)
    std = torch.tensor(config['std'],device=valid.device).view(1,3,1,1)
    return F.pad((valid-mean)/std, meta['padding'],value=0)


def transform_box(box,meta):
    x,y,w,h = box
    sx,sy = meta['scale_factor']
    return [x*sx,y*sy,w*sx,h*sy]
