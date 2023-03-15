# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
from .ours1 import ours
from .ours_swin_attn import ours_swin_attn
from .ours_cswin import ours_cswin
from .ours_pooling import ours_pooling
from .cswin_replace_window import CswinReplaceWindow
from .ours_convnext import ours_convnext
from .o_poolingformer import PoolFormer
from .ours_pvt import ours_pvt
from .o_pvt import pvt_small
from .ours_lastglobal import ours_lastglobal
from .ablation_stem import ablation_stem
from .ablation_lepe import ablation_lepe
from .ablation_cpe import ablation_cpe
from .ablation_dwmlp import ablation_dwmlp
from .test_flops import test_flops
__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet', 'ours', 'ours_swin_attn', 'ours_cswin', 'ours_pooling', 'CswinReplaceWindow',
    'ours_convnext', 'PoolFormer', 'ours_pvt', 'pvt_small', 'ours_lastglobal', 'ablation_stem', 'ablation_lepe','ablation_dwmlp',
    'ablation_cpe', 'test_flops'
]
