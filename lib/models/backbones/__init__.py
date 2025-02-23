import warnings
import sys
sys.path.append("/data/yangjian/POEM/lib/models/backbones")

from ...utils.builder import BACKBONE, build_from_cfg
from .resnet import build_resnet
from .mobilenet import build_mobilenet
from ...utils.config import CN


def build_backbone(cfg, **kwargs):
    return build_from_cfg(cfg, BACKBONE, **kwargs)


def create_backbone(cfg: CN):
    warnings.warn(
        "the old version of `create_backbone` is deprecated, please use  `build_backbone` with registry instead")
    if 'resnet' in cfg.TYPE:
        _cfg = cfg.clone()
        _cfg.defrost()
        _cfg.TYPE = _cfg.TYPE.replace('resnet', 'ResNet')
        return build_backbone(_cfg)
    elif 'mobile' in cfg.TYPE:
        _cfg = cfg.clone()
        _cfg.defrost()
        _cfg.TYPE = _cfg.TYPE.replace('mobilenet', 'MobileNet')
        return build_backbone(_cfg)
    else:
        raise NotImplementedError(f"create_backbone for {cfg.TYPE} is not supported")
