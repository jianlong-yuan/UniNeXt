_base_ = [
    '../_base_/models/upernet_ours.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    # pretrained='/mnt/workspace/linfangjian.lfj/DilatedFormer/output/train/20221019-103300-DiT_T_swin-224/model_best.pth.tar',
    backbone=dict(
         type='ours_last_global',
         init_cfg=dict(type='Pretrained', checkpoint='/mnt/workspace/linfangjian.lfj/mmsegmentation/pretrain/UniNeXt_B.pth'),
         ws = [11,11,11,11], 
         embed_dim=128, 
         num_heads=[4,8,16,32],
         drop_path_rate=0.4
    ),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=150),
    auxiliary_head=dict(in_channels=512, num_classes=150))

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

checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)