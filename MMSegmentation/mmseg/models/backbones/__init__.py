# Copyright (c) OpenMMLab. All rights reserved.
from .beit import BEiT
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .cgnet import CGNet
from .erfnet import ERFNet
from .fast_scnn import FastSCNN
from .hrnet import HRNet
from .icnet import ICNet
from .mae import MAE
from .mit import MixVisionTransformer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
from .stdc import STDCContextPathNet, STDCNet
from .swin import SwinTransformer
from .timm_backbone import TIMMBackbone
from .twins import PCPVT, SVT
from .unet import UNet
from .vit import VisionTransformer
from .ours1 import ours
from .ours_swin_attn import ours_swin_attn
from .ours_cswin import ours_cswin
from .ours_pooling import ours_pooling
from .ours_convnext import ours_convnext
from .o_poolingformer import PoolFormer
from .o_pvt import pvt_small
from .ours_pvt import ours_pvt
from .ours_last_global import ours_last_global
from .ablation_stem import ablation_stem
from .ablation_lepe import ablation_lepe
from .ablation_cpe import ablation_cpe
from .ablation_dwmlp import ablation_dwmlp
from .test_flops import test_flops
__all__ = [
    'ResNet', 'ResNetV1c', 'ResNetV1d', 'ResNeXt', 'HRNet', 'FastSCNN',
    'ResNeSt', 'MobileNetV2', 'UNet', 'CGNet', 'MobileNetV3',
    'VisionTransformer', 'SwinTransformer', 'MixVisionTransformer',
    'BiSeNetV1', 'BiSeNetV2', 'ICNet', 'TIMMBackbone', 'ERFNet', 'PCPVT',
    'SVT', 'STDCNet', 'STDCContextPathNet', 'BEiT', 'MAE', 'ours', 'ours_swin_attn', 'ours_cswin',
    'ours_pooling', 'ours_convnext', 'PoolFormer', 'pvt_small', 'ours_pvt', 'ours_last_global','ablation_stem', 'ablation_lepe','ablation_dwmlp',
    'ablation_cpe', 'test_flops'
]
