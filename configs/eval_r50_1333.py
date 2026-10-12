# Controlled resolution ablation: same checkpoint, same val GT, same evaluator.
_base_ = ['./detector.py']
test_pipeline = [dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333,800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))]
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
