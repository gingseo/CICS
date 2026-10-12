import math
import numpy as np


def box_slice(box,shape):
    x,y,w,h = box
    height,width = shape
    x0,x1 = max(0,x),min(width,x+w)
    y0,y1 = max(0,y),min(height,y+h)
    if x1 <= x0 or y1 <= y0:
        return (slice(0,0),slice(0,0)),np.empty((0,0))
    xs,ys = np.arange(math.floor(x0),math.ceil(x1)),np.arange(math.floor(y0),math.ceil(y1))
    wx = np.maximum(0,np.minimum(xs+1,x1)-np.maximum(xs,x0))
    wy = np.maximum(0,np.minimum(ys+1,y1)-np.maximum(ys,y0))
    return (slice(ys[0],ys[-1]+1),slice(xs[0],xs[-1]+1)),wy[:,None]*wx[None,:]


def exclusion_mask(shape,boxes):
    mask = np.zeros(shape,dtype=bool)
    for box in boxes:
        sl,w = box_slice(box,shape)
        mask[sl] |= w>0
    return mask


def pool_box(values,box,allowed=None):
    sl,weights = box_slice(box,values.shape)
    v = values[sl]
    valid = np.isfinite(v)
    if allowed is not None:
        valid &= allowed[sl]
    weights = weights*valid
    area = float(weights.sum())
    return dict(score=float((np.nan_to_num(v)*weights).sum()/area) if area else None,
                valid_sample_count=int((weights>0).sum()),valid_sample_area=area)


def pool_background(values,valid_hw,boxes,ignored=()):
    h,w = valid_hw
    mask = np.zeros(values.shape,dtype=bool)
    mask[:h,:w] = True
    mask &= ~exclusion_mask(values.shape,list(boxes)+list(ignored))
    mask &= np.isfinite(values)
    v = values[mask].astype(np.float64)
    return dict(score=float(v.mean()) if len(v) else None,sum=float(v.sum()),
                sum_squares=float(np.square(v).sum()),valid_sample_count=len(v))
