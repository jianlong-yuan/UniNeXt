# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import BACKBONES
from mmcv.runner import BaseModule, _load_checkpoint
from mmcv.cnn import build_norm_layer
from ...utils import get_root_logger




import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.utils.checkpoint as checkpoint
import numpy as np
import time





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

  
class WindowMSA(nn.Module):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.
    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
        #                 num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        # Wh, Ww = self.window_size
        # rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        # rel_position_index = rel_index_coords + rel_index_coords.T
        # rel_position_index = rel_position_index.flip(1).contiguous()
        # self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        # self.proj = nn.Linear(embed_dims, embed_dims)
        # self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)
        self.lepe = nn.Conv2d(in_channels=embed_dims, 
                              out_channels=embed_dims, 
                              kernel_size=3,
                              stride=1, 
                              padding=1, 
                              groups=embed_dims, 
                              bias=True)

    # def init_weights(self):
    #     trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        B_new, h, _, vc = v.shape
        lepe = v.permute(0, 1, 3, 2).contiguous().view(B_new, h*vc, self.window_size[0], self.window_size[1])
        lepe = self.lepe(lepe)
        lepe = lepe.view(B_new, h*vc, -1).permute(0, 2, 1).contiguous()

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[
        #     self.relative_position_index.view(-1)].view(
        #         self.window_size[0] * self.window_size[1],
        #         self.window_size[0] * self.window_size[1],
        #         -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(
        #     2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + lepe
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(nn.Module):
    """Shifted Window Multihead Self-Attention Module.
    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 ):
        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        # self.drop = build_dropout(dropout_layer)

    def forward(self, query, H, W):
        B, L, C = query.shape
        # H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows















class DilatedBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ws=7,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 shift=False,
                 ):
        super().__init__()
        self.mlp_ratio = mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.attns = nn.ModuleList()
        # RC
        self.attns.append(
            ShiftWindowMSA(
            embed_dims=dim,
            num_heads=num_heads,
            window_size=ws,
            shift_size=ws // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop,
            proj_drop_rate=drop)
        )
        # if qkv_bias:
        #     self.q_bias = nn.Parameter(torch.zeros(dim))
        #     self.v_bias = nn.Parameter(torch.zeros(dim))
        # else:
        #     self.q_bias = None
            # self.v_bias = None
        # norm_cfg = dict(type='LN', requires_grad=True)
        self.norm1 = nn.LayerNorm(dim)
        # self.norm1 = norm_layer(dim)
        # self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        # qkv_bias = None
        # if self.q_bias is not None:
        #     qkv_bias = torch.cat(
        #         (self.q_bias,
        #          torch.zeros_like(self.v_bias,
        #                           requires_grad=False), self.v_bias))

        # qkv = F.linear(input=img, weight=self.qkv.weight, bias=qkv_bias)
        # qkv = qkv.reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        attened_x = self.attns[0](img, H, W)

        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x




class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x, H, W):
        B, new_HW, C = x.shape
        # H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x



@BACKBONES.register_module()
class ours_swin_attn(BaseModule):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 in_chans=3, 
                 embed_dim=64, 
                 depth=[2,2,18,2], 
                 ws = [7,7,7,7], 
                 wd=[7,7,7,7],
                 num_heads=[2,4,8,16], 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 hybrid_backbone=None, 
                 norm_layer=nn.LayerNorm, 
                 use_chk=False,
                 init_cfg=None):
        super(ours_swin_attn, self).__init__(init_cfg=init_cfg)
        self.use_chk = use_chk
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
        self.num_features=[]
        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            DilatedBlock(
                dim=curr_dim, 
                num_heads=heads[0], 
                ws=ws[0], 
                # ds=wd[0], 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[i])
            for i in range(depth[0])])
        self.num_features.append(curr_dim)
        self.cpe1 = nn.ModuleList([
            nn.Conv2d(curr_dim, curr_dim, 3, 1, 1, groups=curr_dim)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        self.stage2 = nn.ModuleList(
            [DilatedBlock(
                dim=curr_dim, 
                num_heads=heads[1], 
                ws=ws[1], 
                # ds=wd[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1])+i])
            for i in range(depth[1])])
        self.num_features.append(curr_dim)
        self.cpe2 = nn.ModuleList([
            nn.Conv2d(curr_dim, curr_dim, 3, 1, 1, groups=curr_dim)
            for i in range(depth[1])])
        
        self.merge2 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        temp_stage3 = []
        temp_stage3.extend(
            [DilatedBlock(
                dim=curr_dim, 
                num_heads=heads[2], 
                ws=ws[2], 
                # ds=wd[2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2])+i])
            for i in range(depth[2])])
        self.num_features.append(curr_dim)
        self.cpe3 = nn.ModuleList([
            nn.Conv2d(curr_dim, curr_dim, 3, 1, 1, groups=curr_dim)
            for i in range(depth[2])])
        self.stage3 = nn.ModuleList(temp_stage3)
        
        self.merge3 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        self.stage4 = nn.ModuleList(
            [DilatedBlock(
                dim=curr_dim, 
                num_heads=heads[3], 
                ws=ws[3], 
                # ds=wd[3],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1])+i])
            for i in range(depth[-1])])

        self.num_features.append(curr_dim)
        self.cpe4 = nn.ModuleList([
            nn.Conv2d(curr_dim, curr_dim, 3, 1, 1, groups=curr_dim)
            for i in range(depth[-1])])

        # trunc_normal_(self.head.weight, std=0.02)
        
        out_indices = [0, 1, 2, 3]
        for i in out_indices:
            layer = build_norm_layer(dict(type='LN', requires_grad=True), self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)
        
    def init_weights(self):
    
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            # resize?
            self.load_state_dict(checkpoint, False)
        elif self.init_cfg is not None:
            super(ours_swin_attn, self).init_weights()
        else:
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
            self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def forward(self, x):
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

        x = self.merge0(x, H0, W0)
        C = x.shape[2]
        out = []
        norm_layer0 = getattr(self, f'norm{0}')
        norm_layer1 = getattr(self, f'norm{1}')
        norm_layer2 = getattr(self, f'norm{2}')
        norm_layer3 = getattr(self, f'norm{3}')
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
        out.append(norm_layer0(x).permute(0,2,1).contiguous().view(B, C, H1, W1))
        
        for pre, blocks, cpe, ln, H_i, W_i in zip([self.merge1, self.merge2, self.merge3],
                                              [self.stage2, self.stage3, self.stage4],
                                              [self.cpe2, self.cpe3, self.cpe4],
                                              [norm_layer1, norm_layer2, norm_layer3],
                                              [H2, H3, H4],
                                              [W2, W3, W4]):
            x = pre(x, H_i*2, W_i*2)
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
            out.append(ln(x).permute(0,2,1).contiguous().view(B, C, H_i, W_i))
        
        return out

