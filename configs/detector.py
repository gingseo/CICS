# Standard MMDetection baseline; source commit recorded in compatibility_report.md.
_base_ = ['./base_faster_rcnn.py']

root = '/mnt/icelab_4tb/datasets/VisDrone'
project = '{{fileDirname}}/..'
classes = ('pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor')
model = dict(roi_head=dict(bbox_head=dict(num_classes=10)),
             test_cfg=dict(rcnn=dict(max_per_img=500)))
train_pipeline = [dict(type='LoadImageFromFile'),dict(type='LoadAnnotations',with_bbox=True),
                  dict(type='Resize',scale=(640,640),keep_ratio=True),dict(type='RandomFlip',prob=0.5),
                  dict(type='PackDetInputs')]
test_pipeline = [dict(type='LoadImageFromFile'),dict(type='Resize',scale=(640,640),keep_ratio=True),
                 dict(type='LoadAnnotations',with_bbox=True),
                 dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))]
train_dataloader = dict(batch_size=2,num_workers=4,persistent_workers=True,
    sampler=dict(type='DefaultSampler',shuffle=True),batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='CocoDataset',data_root=root,metainfo=dict(classes=classes),
        ann_file=project+'/data/train.json',data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=True,min_size=1),pipeline=train_pipeline))
val_dataloader = dict(batch_size=1,num_workers=2,persistent_workers=True,drop_last=False,
    sampler=dict(type='DefaultSampler',shuffle=False),dataset=dict(type='CocoDataset',data_root=root,
        metainfo=dict(classes=classes),ann_file=project+'/data/val.json',
        data_prefix=dict(img='images/val/'),test_mode=True,pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric',ann_file=project+'/data/val.json',metric='bbox',
                     proposal_nums=(100,300,500))
test_evaluator = val_evaluator
train_cfg = dict(type='EpochBasedTrainLoop',max_epochs=60,val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# 0.02 at batch 16 -> 0.0025 at batch 2; 12-epoch milestones 8/11 scaled by 5.
optim_wrapper = dict(type='OptimWrapper',optimizer=dict(type='SGD',lr=0.0025,momentum=0.9,weight_decay=0.0001))
param_scheduler = [dict(type='LinearLR',start_factor=0.001,by_epoch=False,begin=0,end=500),
                   dict(type='MultiStepLR',begin=0,end=60,by_epoch=True,milestones=[40,55],gamma=0.1)]
auto_scale_lr = dict(enable=False,base_batch_size=16)
default_scope = 'mmdet'
default_hooks = dict(timer=dict(type='IterTimerHook'),logger=dict(type='LoggerHook',interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),sampler_seed=dict(type='DistSamplerSeedHook'),
    checkpoint=dict(type='CheckpointHook',interval=1,max_keep_ckpts=2,save_last=True,
                    save_best='coco/bbox_mAP',rule='greater'))
custom_imports = dict(imports=['src.train'],allow_failed_imports=False)
custom_hooks = [dict(type='CICSMonitorHook')]
env_cfg = dict(cudnn_benchmark=False,mp_cfg=dict(mp_start_method='fork',opencv_num_threads=0),dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer',vis_backends=vis_backends,name='visualizer')
log_processor = dict(type='LogProcessor',window_size=20,by_epoch=True)
log_level = 'INFO'
randomness = dict(seed=128,deterministic=True)
# Full COCO detector initialization. Classification/regression tensors reinitialized for 10 classes.
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
resume = False
