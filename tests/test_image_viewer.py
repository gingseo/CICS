import numpy as np
from PIL import Image
from src.image_viewer import tiny_dominant_records,summary_rows,draw_gt,SIZE_COLORS


def record(name,bins):
    return dict(image_id=name,objects=[dict(size_bin=b,bbox=[0,0,1,1]) for b in bins])


def test_filter_is_strict_majority_and_does_not_change_records():
    records=[record('half',['VT','L']),record('dense',['VT','T','T','L']),
             record('tiny',['VT']),record('empty',[]),record('underflow',['underflow','L'])]
    selected=tiny_dominant_records(records)
    assert [r['image_id'] for r,_ in selected]==['dense','tiny']
    assert selected[0][1]['fraction']==.75
    assert len(records)==5 and len(records[1]['objects'])==4


def test_summary_uses_object_means_and_current_image_background():
    objects=[dict(size_bin='VT',score=.2),dict(size_bin='VT',score=.6),dict(size_bin='T',score=None)]
    rows={r['region']:r for r in summary_rows(objects,{'score':.1})}
    assert rows['BG']==dict(region='BG',count=None,mean=.1)
    assert rows['VT']['count']==2 and np.isclose(rows['VT']['mean'],.4)
    assert rows['T']['count']==1 and rows['T']['mean'] is None
    assert rows['L']['count']==0 and rows['L']['mean'] is None


def test_gt_uses_distinct_size_colors():
    from PIL import ImageColor
    assert len({SIZE_COLORS[k] for k in ['VT','T','S','M','L']})==5
    source=Image.new('RGB',(20,20),'black')
    out=draw_gt(source,[dict(size_bin='VT',bbox=[2,2,10,10])])
    assert out.getpixel((2,2))==ImageColor.getrgb(SIZE_COLORS['VT'])
    assert source.getpixel((2,2))==(0,0,0)
