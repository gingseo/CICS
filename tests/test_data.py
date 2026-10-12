from PIL import Image
import pytest
from src.data_audit import load_records,choose_split,convert_coco
from src.common import read_json


def make(root,split,label=None):
    p=root/'images'/split;p.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',(100,100)).save(p/'x.png')
    if label is not None:
        p=root/'labels'/split;p.mkdir(parents=True,exist_ok=True);(p/'x.txt').write_text(label)


def test_unlabeled_vs_empty_gt(tmp_path):
    make(tmp_path,'test');make(tmp_path,'val','')
    assert not load_records(tmp_path,'test')[0]['gt_available']
    assert load_records(tmp_path,'val')[0]['gt_available']
    split,reason=choose_split(tmp_path)
    assert split=='val' and reason


def test_invalid_explicit_conversion_and_ids(tmp_path):
    make(tmp_path,'train','0 .5 .5 .1 .1\n3 .5 .5 .1 0\n')
    with pytest.raises(ValueError): convert_coco(tmp_path,'train','yolo',tmp_path/'a.json')
    assert convert_coco(tmp_path,'train','yolo',tmp_path/'a.json',True)==(1,1)
    assert len(read_json(tmp_path/'a.json.excluded.json'))==1
    assert read_json(tmp_path/'a.json.source_ids.json')[0]['source_id']=='x:1'


def test_original_visdrone_ignores(tmp_path):
    p=tmp_path/'VisDrone2019-DET-val';(p/'images').mkdir(parents=True);(p/'annotations').mkdir()
    Image.new('RGB',(100,100)).save(p/'images'/'x.png')
    (p/'annotations'/'x.txt').write_text('0,0,10,10,0,0,0,0\n20,20,10,10,1,4,0,0\n40,40,10,10,1,11,0,0\n')
    r=load_records(tmp_path,'val','visdrone')[0]
    assert len(r['objects'])==1 and len(r['ignored'])==2
