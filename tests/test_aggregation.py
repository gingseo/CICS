import numpy as np
import pandas as pd
from src.statistics import summarize
from src.analysis_runner import OBJECT_FIELDS,BG_FIELDS


def test_object_weighting_and_global_bg_are_different():
    objects=pd.DataFrame([dict(size_bin='VT',score=1,original_size=3,input_size=2,image_id='a'),
                          dict(size_bin='VT',score=9,original_size=7,input_size=4,image_id='a')])
    bg=pd.DataFrame([dict(score=2,sum=2,valid_sample_count=1),dict(score=8,sum=72,valid_sample_count=9)])
    summary,stats=summarize(objects,bg)
    assert summary.set_index('region').loc['VT','mean']==5
    assert summary.set_index('region').loc['BG','mean']==5
    assert stats['global_bg']['mean']==7.4
    empty=summary.set_index('region').loc['T']
    assert empty['count']==0 and np.isnan(empty['mean'])


def test_empty_annotations_do_not_invent_zero_bins():
    summary,stats=summarize(pd.DataFrame(columns=OBJECT_FIELDS),pd.DataFrame(columns=BG_FIELDS))
    assert summary['count'].sum()==0
    assert summary['mean'].isna().all()
    assert stats['global_bg']['mean'] is None
