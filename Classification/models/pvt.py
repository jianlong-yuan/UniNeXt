# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
from einops import rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np
import time


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'DilatedFormer_224': _cfg(),
    'DilatedFormer_384': _cfg(
        crop_pct=1.0
    ),

}


def local_group(x, H, W, ws, ds):
    '''  
    ws: window size 
    ds: dilated size
    x: BNC
    return: BG ds*ds ws*ws C
    '''
    B, _, C = x.shape
    pad_right, pad_bottom, pad_opt, pad_opt_d = 0, 0, False, False

    if H % ws != 0 or W % ws != 0:
        pad_opt =True
        # reshape (B, N, C) -> (B, H, W, C)
        x = x.view(B, H, W, C)
        # padding right & below
        pad_right = ws - W % ws
        pad_bottom = ws - H % ws
        x = F.pad(x, (0, 0, 0, pad_right, 0, pad_bottom))
        H = H + pad_bottom
        W = W + pad_right
        N = H * W
        # reshape (B, H, W, C) -> (B, N, C)
        x = x.view(B, N, C)
    Gh = H//ws
    Gw = W//ws
    x = x.view(B, Gh, ws, Gw, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B*Gh*Gw, ws*ws, C)
    # pad
    # Hd, Wd, pad_right_d, pad_bottom_d = ws, ws, 0, 0
    # if ws % ds != 0:
    #     pad_opt_d =True
    #     # padding right & below
    #     pad_right_d = ds - ws % ds
    #     pad_bottom_d = ds - ws % ds
    #     x = F.pad(x, (0, 0, 0, pad_right_d, 0, pad_bottom_d))
    #     Hd = ws + pad_bottom_d
    #     Wd = ws + pad_right_d

    # kh, kw = Hd // ds, Wd // ds
    # x = x.view(B*Gh*Gw, kh, ds, kw, ds, C).permute(0, 2, 4, 1, 3, 5).contiguous().view(B*Gh*Gw, ds*ds, kh*kw, C)
    
    return x, H, W, pad_right, pad_bottom, pad_opt



def img2group(x, H, W, ws, ds, num_head):
    '''
    x: B, H*W, C
    return : (B G) head  N C
    '''
    # After group x：B G N C
    x, H, W, pad_right, pad_bottom, pad_opt = local_group(x, H, W, ws, ds)
    B, N, C =x.shape
    x = x.view(B, N, num_head, C//num_head).permute(0, 2, 1, 3).contiguous()

    return x, H, W, pad_right, pad_bottom, pad_opt


def group2image(x, H, W, pad_right, pad_bottom, pad_opt, ws):
    # Input x: (BG G) Head n C
    # Output x: B N C

    BG, Head, n, C = x.shape
    Gh, Gw = H//ws, W//ws
    Gn = Gh * Gw
    nb1 = BG//Gn
    x = x.view(nb1, Gh, Gw, Head, ws, ws, C).permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(nb1, -1, Head*C)
    
    if pad_opt:
        x = x.view(nb1, H, W, Head*C)
        x = x[:, :(H - pad_bottom), :(W - pad_right), :].contiguous()  # remove
        x = x.view(nb1, -1, hc)

    return x


# dwconv MLP
class Mlp(nn.Module):
    def __init__(self,
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.norm_act = nn.Sequential(
            nn.LayerNorm(hidden_features),
            nn.GELU()
        )

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        B, N, C = x.shape
        x1 = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x1 = self.dwconv(x1)
        x1 = x1.view(B, C, -1).permute(0, 2, 1).contiguous()
        x1 = self.norm_act(x1)
        x = x + x1
        x = self.fc2(x)
        x = self.drop(x)
        return x

  
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        
        self.lepe = nn.Conv2d(in_channels=dim, 
                              out_channels=dim, 
                              kernel_size=3,
                              stride=1, 
                              padding=1, 
                              groups=dim, 
                              bias=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        
        # lepe
        lepe = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        lepe = self.lepe(lepe)
        lepe = lepe.view(B, C, -1).permute(0, 2, 1).contiguous()
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + lepe
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x



class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x

class DilatedFormer_Windows(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=96, 
                 depth=[2,2,6,2], 
                 ws = [7,7,7,7], 
                 wd=[7,7,7,7],
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 hybrid_backbone=None, 
                 norm_layer=nn.LayerNorm, 
                 use_chk=False,
                 pretrained_cfg=None,
                 bn_tf=None):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads=num_heads
        #------------- stem -----------------------
        stem_out = embed_dim//2
        self.stem1 = nn.Conv2d(in_chans, stem_out, 3, 2, 1)
        self.norm_act1 = nn.Sequential(
            nn.LayerNorm(stem_out),
            nn.GELU()
        )
        self.stem2 = nn.Conv2d(stem_out, stem_out, 3, 1, 1)
        self.norm_act2 = nn.Sequential(
            nn.LayerNorm(stem_out),
            nn.GELU()
        )
        self.stem3 = nn.Conv2d(stem_out, stem_out, 3, 1, 1)
        self.norm_act3 = nn.Sequential(
            nn.LayerNorm(stem_out),
            nn.GELU()
        )
        #----------------------------------------
        
        
        self.merge0 = Merge_Block(stem_out, embed_dim)
        sr_ratios = [8, 4, 2, 1]
        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            Block(
                dim=curr_dim, 
                num_heads=heads[0], 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depth[0])])
        
        self.cpe1 = nn.ModuleList([
            nn.Conv2d(curr_dim, curr_dim, 3, 1, 1, groups=curr_dim)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        self.stage2 = nn.ModuleList(
            [Block(
                dim=curr_dim, 
                num_heads=heads[1], 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1])+i],
                norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depth[1])])
        
        self.cpe2 = nn.ModuleList([
            nn.Conv2d(curr_dim, curr_dim, 3, 1, 1, groups=curr_dim)
            for i in range(depth[1])])
        
        self.merge2 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        temp_stage3 = []
        temp_stage3.extend(
            [Block(
                dim=curr_dim, 
                num_heads=heads[2], 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2])+i],
                norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depth[2])])
        self.cpe3 = nn.ModuleList([
            nn.Conv2d(curr_dim, curr_dim, 3, 1, 1, groups=curr_dim)
            for i in range(depth[2])])
        self.stage3 = nn.ModuleList(temp_stage3)
        
        self.merge3 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        self.stage4 = nn.ModuleList(
            [Block(
                dim=curr_dim, 
                num_heads=heads[3], 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1])+i],
                norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depth[-1])])
        self.cpe4 = nn.ModuleList([
            nn.Conv2d(curr_dim, curr_dim, 3, 1, 1, groups=curr_dim)
            for i in range(depth[-1])])
        
        self.norm = norm_layer(curr_dim)
        # Classifier head
        self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes, global_pool=''):
        if self.num_classes != num_classes:
            print ('reset head to', num_classes)
            self.num_classes = num_classes
            self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head = self.head.cuda()
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x):
        B, _, H, W = x.shape
        H0, W0, H1, W1, H2, W2, H3, W3, H4, W4 = H//2, W//2, H//4, W//4, H//8, W//8, H//16, W//16, H//32, W//32
        # stem
        x = self.stem1(x)
        c1 = x.size(1)
        x = x.view(B, c1, -1).permute(0, 2, 1).contiguous()
        x = self.norm_act1(x)
        x = x.permute(0, 2, 1).contiguous().view(B, c1, H0, W0)
        x = self.stem2(x)
        c2 = x.size(1)
        x = x.view(B, c2, -1).permute(0, 2, 1).contiguous()
        x = self.norm_act2(x)
        x = x.permute(0, 2, 1).contiguous().view(B, c2, H0, W0)
        x = self.stem3(x)
        c3 = x.size(1)
        x = x.view(B, c3, -1).permute(0, 2, 1).contiguous()
        x = self.norm_act3(x)

        x = self.merge0(x)
        C = x.shape[2]
        
        for blk, cpe in zip(self.stage1, self.cpe1):
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
                pe = cpe(x.transpose(1, 2).reshape(B, C, H1, W1))
                pe = pe.view(B, C, -1).transpose(1, 2)
                x = x + pe
            else:
                x = blk(x, H1, W1)
                pe = cpe(x.transpose(1, 2).reshape(B, C, H1, W1))
                pe = pe.view(B, C, -1).transpose(1, 2)
                x = x + pe
        
        for pre, blocks, cpe, H_i, W_i in zip([self.merge1, self.merge2, self.merge3],
                                              [self.stage2, self.stage3, self.stage4],
                                              [self.cpe2, self.cpe3, self.cpe4],
                                              [H2, H3, H4],
                                              [W2, W3, W4]):
            x = pre(x)
            C = x.shape[2]
            for blk, cpe_layer in zip(blocks, cpe):
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                    pe = cpe_layer(x.transpose(1, 2).reshape(B, C, H_i, W_i))
                    pe = pe.view(B, C, -1).transpose(1, 2)
                    x = x + pe
                else:
                    x = blk(x, H_i, W_i)
                    pe = cpe_layer(x.transpose(1, 2).reshape(B, C, H_i, W_i))
                    pe = pe.view(B, C, -1).transpose(1, 2)
                    x = x + pe

        x = self.norm(x)
        return torch.mean(x, dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

### 224 models

@register_model
def pvt(pretrained=False, **kwargs):
#dim   56 28 14 7
#ws    7  7  7  7
#wd    3  3  3  3
    model = DilatedFormer_Windows(patch_size=4, embed_dim=64, depth=[2,2,18,2],
        ws = [7,7,7,7], wd=[3,3,3,3], num_heads=[2,4,8,16], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['DilatedFormer_224']
    return model



