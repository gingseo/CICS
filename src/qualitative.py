from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from matplotlib import colormaps


def rgb(tensor):
    return np.uint8(tensor[0].detach().cpu().permute(1,2,0).numpy().clip(0,255))


def render(original,degraded,values,objects,valid_hw,vmax=2.0):
    h,w=valid_hw
    original=Image.fromarray(rgb(original))
    degraded=Image.fromarray(rgb(degraded))
    v=values[:h,:w]
    heat=np.uint8(colormaps['inferno'](np.nan_to_num(v)/max(vmax,1e-12))[...,:3]*255)
    heat[~np.isfinite(v)]=[128,128,128]
    heat=Image.fromarray(heat)
    overlay=Image.blend(original,heat,.45)
    gt=original.copy();draw=ImageDraw.Draw(gt)
    for o in objects:
        x,y,bw,bh=o['input_bbox']
        draw.rectangle((x,y,x+bw,y+bh),outline='#00ff88',width=1)
        s=o['score'];text=f"{o['size_bin']} | D={s:.3f}" if s is not None else f"{o['size_bin']} | D=NA"
        draw.text((max(0,x),max(0,y-10)),text,fill='#00ff88',stroke_width=1,stroke_fill='black')
    return dict(original=original,degraded=degraded,heatmap=heat,overlay=overlay,gt=gt)


def save_views(views,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for name,im in views.items(): im.save(folder/f'{name}.png')
