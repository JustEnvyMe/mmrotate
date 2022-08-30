# data_root = '/media/alex/AC6A2BDB6A2BA0D6/alex_dataset/DOTA/DOTA-v1.0/split_ms_dota1_0/'
data_root = '../dataset/DOTA-v1.0/split_ms_dota1_0/'
img_norm_cfg = dict(    
mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    # dict(type='RandomFlip', flip_ratio=0.5),
    # dict(
    #     type='AutoAugment',
    #     policies=[[{
    #         'type':
    #         'Resize',
    #         'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #                       (608, 1333), (640, 1333), (672, 1333), (704, 1333),
    #                       (736, 1333), (768, 1333), (800, 1333)],
    #         'multiscale_mode':
    #         'value',
    #         'keep_ratio':
    #         True
    #     }],
    #               [{
    #                   'type': 'Resize',
    #                   'img_scale': [(400, 1333), (500, 1333), (600, 1333)],
    #                   'multiscale_mode': 'value',
    #                   'keep_ratio': True
    #               }, {
    #                   'type': 'RandomCrop',
    #                   'crop_type': 'absolute_range',
    #                   'crop_size': (384, 600),
    #                   'allow_negative_crop': True
    #               }, {
    #                   'type':
    #                   'Resize',
    #                   'img_scale': [(480, 1333), (512, 1333), (544, 1333),
    #                                 (576, 1333), (608, 1333), (640, 1333),
    #                                 (672, 1333), (704, 1333), (736, 1333),
    #                                 (768, 1333), (800, 1333)],
    #                   'multiscale_mode':
    #                   'value',
    #                   'override':
    #                   True,
    #                   'keep_ratio':
    #                   True
    #               }]]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='DOTADataset',
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(1024, 1024)),
            dict(
                type='RRandomFlip',
                flip_ratio=[0.25, 0.25, 0.25],
                direction=['horizontal', 'vertical', 'diagonal'],
                version='le90'),
            # dict(
            #     type='AutoAugment',
            #     policies=[[{
            #         'type':
            #         'Resize',
            #         'img_scale': [(480, 1333), (512, 1333), (544, 1333),
            #                       (576, 1333), (608, 1333), (640, 1333),
            #                       (672, 1333), (704, 1333), (736, 1333),
            #                       (768, 1333), (800, 1333)],
            #         'multiscale_mode':
            #         'value',
            #         'keep_ratio':
            #         True
            #     }],
            #               [{
            #                   'type': 'Resize',
            #                   'img_scale': [(400, 1333), (500, 1333),
            #                                 (600, 1333)],
            #                   'multiscale_mode': 'value',
            #                   'keep_ratio': True
            #               }, {
            #                   'type': 'RandomCrop',
            #                   'crop_type': 'absolute_range',
            #                   'crop_size': (384, 600),
            #                   'allow_negative_crop': True
            #               }, {
            #                   'type':
            #                   'Resize',
            #                   'img_scale': [(480, 1333), (512, 1333),
            #                                 (544, 1333), (576, 1333),
            #                                 (608, 1333), (640, 1333),
            #                                 (672, 1333), (704, 1333),
            #                                 (736, 1333), (768, 1333),
            #                                 (800, 1333)],
            #                   'multiscale_mode':
            #                   'value',
            #                   'override':
            #                   True,
            #                   'keep_ratio':
            #                   True
            #               }]]),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=1),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='DOTADataset',
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=1),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='le90'),
    test=dict(
        type='DOTADataset',
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=1),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='le90'))
evaluation = dict(save_best='auto', interval=1, metric='mAP')
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[100])
runner = dict(type='EpochBasedRunner', max_epochs=150)
checkpoint_config = dict(interval=4, create_symlink=False)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
model = dict(
    type='DETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_head=dict(
        type='RotatedDETRHead',
        num_classes=15,
        in_channels=2048,
        num_query=300,
        transformer=dict(
            type='Transformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),

        # no iou_loss
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(
            type='GDLoss_v1',
            loss_type='kld',
            fun='log1p',
            tau=1.0,
            loss_weight=2.0),
    ),
    # set loss_bbox weight larger than loss_cls for paying more attention to regression
    train_cfg=dict(
        assigner=dict(
            type='ObbHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=1.0),
            reg_cost=dict(type='KLDLossCost', weight=2.0, fun='log1p', tau=1.0),
        )),
    test_cfg=dict(max_per_img=100))
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)

# work_dir = './work_dirs/detr_r50_8x2_150e_coco'
auto_resume = False
gpu_ids = [0]
