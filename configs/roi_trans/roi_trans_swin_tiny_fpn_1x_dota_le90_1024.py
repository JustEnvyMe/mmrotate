_base_ = ['./roi_trans_r50_fpn_1x_dota_le90.py']

# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='~/.cache/torch/hub/checkpoints/'
                       'swin_tiny_patch4_window7_224.pth')),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5)
)

# dataset settings
dataset_type = 'DOTADataset'
data_root = '../../datasets/DOTA/splited/DOTA-v1.0_1024/'  # 6.120
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/labelTxt',
        img_prefix=data_root + 'train/images/',
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/labelTxt',
        img_prefix=data_root + 'val/images',
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/images/',  # specify the images dir
        img_prefix=data_root + 'test/images',
    ))

# schedule settings
evaluation = dict(interval=4, metric='mAP')
checkpoint_config = dict(interval=4)
# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
runner = dict(max_epochs=60)

# runtime settings
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
