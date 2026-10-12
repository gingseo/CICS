from pathlib import Path
import torch
from .common import sha256, read_json, digest
from .data_audit import CLASSES

STAGES = {'C3': ('layer2',512,8), 'C4': ('layer3',1024,16), 'C5': ('layer4',2048,32)}


def build_detector(config, device='cuda:0'):
    try:
        from mmengine.config import Config
        from mmdet.utils import register_all_modules
        from mmdet.registry import MODELS
        from mmcv.ops import nms, roi_align
    except (ImportError, OSError) as e:
        raise RuntimeError('MMDetection compiled ops unavailable. Run scripts/check_environment.py; see README.') from e
    register_all_modules()
    cfg = Config.fromfile(config)
    if cfg.model.type != 'FasterRCNN' or cfg.model.backbone.type != 'ResNet' or cfg.model.backbone.depth != 50 or cfg.model.neck.type != 'FPN':
        raise ValueError('Expected standard FasterRCNN + ResNet50 + FPN')
    if cfg.model.roi_head.bbox_head.num_classes != 10:
        raise ValueError('Expected 10 VisDrone foreground classes')
    cfg.model.backbone.init_cfg = None
    model = MODELS.build(cfg.model).to(device).eval()
    return model,cfg


def load_strict(model, checkpoint):
    from .checkpoint_io import load_checkpoint
    payload = load_checkpoint(checkpoint)
    state = payload.get('state_dict',payload)
    own = model.state_dict()
    missing = sorted(set(own)-set(state))
    unexpected = sorted(set(state)-set(own))
    mismatched = {k:dict(expected=list(own[k].shape),actual=list(state[k].shape))
                  for k in own.keys() & state.keys() if own[k].shape != state[k].shape}
    report = dict(missing_keys=missing,unexpected_keys=unexpected,shape_mismatches=mismatched)
    if not missing and not unexpected and not mismatched:
        model.load_state_dict(state,strict=True)
    return report


class FeatureExtractor:
    def __init__(self,config,checkpoint,device='cuda:0',validation=None):
        self.device = device
        self.checkpoint_hash = sha256(checkpoint)
        if validation:
            v = read_json(validation)
            if not v or v.get('status') != 'DONE' or v.get('checkpoint_sha256') != self.checkpoint_hash:
                raise ValueError('Validation must be DONE and match checkpoint hash')
            if v.get('config_sha256') != sha256(config):
                raise ValueError('Validation config does not match detector config')
            if v.get('classes') != list(CLASSES):
                raise ValueError('Validation class order mismatch')
        self.model,self.cfg = build_detector(config,device)
        if validation and v.get('resolved_config_sha256')!=digest(self.cfg.to_dict()):
            raise ValueError('Resolved detector config (including base config) changed since validation')
        report = load_strict(self.model,checkpoint)
        if any(report.values()):
            raise ValueError(f'Checkpoint mismatch: {report}')
        self.model.requires_grad_(False)
        self.mapping = {}
        self._features = {}
        self._handles = []
        for stage,(name,channels,stride) in STAGES.items():
            module = getattr(self.model.backbone,name)
            def hook(_,inputs,output,stage=stage,name=name,channels=channels,stride=stride):
                if output.shape[1] != channels:
                    raise ValueError(f'{stage}: unexpected channels {output.shape}')
                self._features[stage] = output
                self.mapping[stage] = dict(module='backbone.'+name,channels=channels,stride=stride,shape=list(output.shape))
            self._handles.append(module.register_forward_hook(hook))

    @torch.inference_mode()
    def __call__(self,x):
        self.model.eval()
        self._features = {}
        self.model.backbone(x.to(self.device))
        return dict(self._features)

    def close(self):
        for handle in self._handles:
            handle.remove()
