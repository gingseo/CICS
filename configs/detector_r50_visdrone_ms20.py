# R50-FPN baseline adaptation, not an exact reproduction of a paper checkpoint.
# Public recipe reference (ATSS, MMDetection 2.x):
# https://github.com/hani-yu/MoE-RelationNet-/blob/b964ca39bf09b1ebd44528a515127b1b2dc38800/configs/atss_r50_visdrone_2x.py
# Keep standard Faster R-CNN architecture and local COCO initialization.
_base_ = ['./detector.py']
train_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=[(1333,640),(1333,1280)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5), dict(type='PackDetInputs')]
test_pipeline = [
    dict(type='LoadImageFromFile'), dict(type='Resize',scale=(1333,800),keep_ratio=True),
    dict(type='LoadAnnotations',with_bbox=True),
    dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
train_cfg = dict(max_epochs=20)
# Single GPU, batch 2: retain standard Faster R-CNN LR 0.02 * 2 / 16.
# The paper code uses LR .01 on 8 GPUs, ImageNet initialization, and ATSS;
# those differences are recorded explicitly rather than claiming exact reproduction.
optim_wrapper = dict(clip_grad=dict(max_norm=35,norm_type=2))
param_scheduler = [dict(type='LinearLR',start_factor=0.001,by_epoch=False,begin=0,end=500),
                   dict(type='MultiStepLR',begin=0,end=20,by_epoch=True,milestones=[16,19],gamma=0.1)]
