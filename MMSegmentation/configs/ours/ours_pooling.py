_base_ = [
    '../_base_/models/upernet_ours.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    # pretrained='/mnt/workspace/linfangjian.lfj/DilatedFormer/output/train/20221019-103300-DiT_T_swin-224/model_best.pth.tar',
    backbone=dict(
         type='ours_pooling',
         init_cfg=dict(type='Pretrained', checkpoint='/mnt/workspace/linfangjian.lfj/mmsegmentation/pretrain/pooling_82_43.pth'),
    #     embed_dim=64, 
    #     depth=[2,2,18,2], 
    #     num_heads=[2, 4, 8, 16],
    #     drop_path_rate=0.2
    ),
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=150),
    auxiliary_head=dict(in_channels=256, num_classes=150))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.),
            'decode_head': dict(lr_mult=1.),
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
