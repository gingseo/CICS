"""Two-image integration check; outputs are NOT experimental evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
from src.common import read_json,write_json

root=Path(__file__).resolve().parents[1]
out=root/'reports'/'detector_smoke';out.mkdir(exist_ok=True)
for split in ('train','val'):
    data=read_json(root/'data'/f'{split}.json')
    data['images']=data['images'][:2]
    ids={i['id'] for i in data['images']}
    data['annotations']=[a for a in data['annotations'] if a['image_id'] in ids]
    write_json(out/f'{split}.json',data)
cfg=Config.fromfile(root/'configs/detector.py')
cfg.work_dir=str(out)
cfg.train_dataloader.dataset.ann_file=str(out/'train.json')
cfg.val_dataloader.dataset.ann_file=str(out/'val.json')
cfg.test_dataloader=cfg.val_dataloader
cfg.val_evaluator.ann_file=str(out/'val.json')
cfg.test_evaluator=cfg.val_evaluator
for loader in [cfg.train_dataloader,cfg.val_dataloader,cfg.test_dataloader]:
    loader.num_workers=0;loader.persistent_workers=False
cfg.train_cfg.max_epochs=1
cfg.param_scheduler=[dict(type='MultiStepLR',begin=0,end=1,by_epoch=True,milestones=[],gamma=.1)]
cfg.load_from=str(root/'data/pretrained/faster_rcnn_r50_fpn_coco.pth')
cfg.model.backbone.init_cfg=None
cfg.dump(str(out/'smoke_config.py'))
register_all_modules()
runner=Runner.from_cfg(cfg)
runner.train()
write_json(out/'result.json',dict(status='PASS',purpose='integration test only; not 60-epoch training or scientific result',epochs=runner.epoch))
