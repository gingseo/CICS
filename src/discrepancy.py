import torch
import torch.nn.functional as F


def discrepancy(a,b,metric='cosine',epsilon=1e-8):
    if a.shape != b.shape or a.ndim != 4:
        raise ValueError('features must be matching N,C,H,W tensors')
    a,b = a.float(),b.float()
    valid = torch.isfinite(a).all(1) & torch.isfinite(b).all(1)
    if metric == 'cosine':
        na,nb = a.norm(dim=1),b.norm(dim=1)
        valid &= (na>epsilon)&(nb>epsilon)
        d = 1-((a/na.clamp_min(epsilon).unsqueeze(1))*(b/nb.clamp_min(epsilon).unsqueeze(1))).sum(1)
        d = d.clamp(0,2)
    elif metric == 'l1':
        d = (a-b).abs().mean(1)
    elif metric == 'l2':
        d = (a-b).square().mean(1).sqrt()
    else:
        raise ValueError(metric)
    return d.masked_fill(~valid,float('nan')),valid


def to_input_map(d,valid,network_hw):
    # Conservative validity: every contributing interpolation sample must be valid.
    weights = F.interpolate(valid[:,None].float(),network_hw,mode='bilinear',align_corners=False)[:,0]
    values = F.interpolate(torch.nan_to_num(d)[:,None],network_hw,mode='bilinear',align_corners=False)[:,0]
    mask = weights >= 1-1e-6
    return values.masked_fill(~mask,float('nan')),mask
