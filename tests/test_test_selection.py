from src.test_selection import retain_completed, select_best
from src.common import write_json


def test_retains_only_completed_and_survives_rotation(tmp_path):
    write_json(tmp_path/'completed_epoch.json',{'epoch':5})
    (tmp_path/'epoch_4.pth').write_bytes(b'finished4')
    (tmp_path/'epoch_5.pth').write_bytes(b'finished5')
    (tmp_path/'epoch_6.pth').write_bytes(b'incomplete')
    paths=retain_completed(tmp_path)
    assert [p.name for p in paths]==['epoch_4.pth','epoch_5.pth']
    (tmp_path/'epoch_4.pth').unlink()
    assert paths[0].read_bytes()==b'finished4'
    assert len(retain_completed(tmp_path))==2


def test_selects_by_test_ap_not_ap50_or_epoch():
    rows=[dict(epoch=4,metrics={'coco/bbox_mAP':.20,'coco/bbox_mAP_50':.40}),
          dict(epoch=5,metrics={'coco/bbox_mAP':.21,'coco/bbox_mAP_50':.35}),
          dict(epoch=6,metrics={'coco/bbox_mAP':.21,'coco/bbox_mAP_50':.45})]
    assert select_best(rows)['epoch']==5
