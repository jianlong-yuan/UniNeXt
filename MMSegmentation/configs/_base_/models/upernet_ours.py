# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    # pretrained=None,
    backbone=dict(
        type='ours',
        # in_chans=3, 
        # embed_dim=64, 
        # depth=[2,2,18,2], 
        # ws = [7,7,7,7], 
        # wd=[7,7,7,7],
        # num_heads=12, 
        # mlp_ratio=4., 
        # qkv_bias=True, 
        # qk_scale=None, 
        # drop_rate=0., 
        # attn_drop_rate=0.,
        # drop_path_rate=0.2, 
        # hybrid_backbone=None, 
        # norm_layer=nn.LayerNorm, 
        # use_chk=False,
        # init_cfg=None
        ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
