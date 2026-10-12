"""Small-object image browsing, size-colored GT, and per-image summaries."""
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import colormaps

SIZE_COLORS={'VT':'#FF4D6D','T':'#FFB000','S':'#00BFC4','M':'#4C78FF','L':'#B279FF','underflow':'#BDBDBD'}
REGIONS=['BG','VT','T','S','M','L']


def tiny_dominant_records(records):
    selected=[]
    for record in records:
        counts=Counter(o['size_bin'] for o in record['objects'])
        total=sum(counts.values());tiny=counts['VT']+counts['T']
        if total and tiny/total>.5:
            selected.append((record,dict(total=total,VT=counts['VT'],T=counts['T'],tiny=tiny,fraction=tiny/total)))
    return sorted(selected,key=lambda pair:(-pair[1]['tiny'],-pair[1]['fraction'],pair[0]['image_id']))


def draw_gt(image,objects,box_key='bbox'):
    image=image.copy();draw=ImageDraw.Draw(image)
    for obj in objects:
        x,y,w,h=obj[box_key]
        draw.rectangle((x,y,x+w,y+h),outline=SIZE_COLORS.get(obj['size_bin'],'white'),width=2)
    return image


def summary_rows(objects,background):
    rows=[dict(region='BG',count=None,mean=background.get('score') if background else None)]
    for region in REGIONS[1:]:
        group=[o for o in objects if o['size_bin']==region]
        scores=[o['score'] for o in group if o.get('score') is not None and np.isfinite(o['score'])]
        rows.append(dict(region=region,count=len(group),mean=float(np.mean(scores)) if scores else None))
    return rows


def color_scale(vmax):
    ramp=np.linspace(0,1,480)[None,:]
    rgb=np.uint8(colormaps['inferno'](ramp)[...,:3]*255)
    return Image.fromarray(np.repeat(rgb,16,axis=0))
