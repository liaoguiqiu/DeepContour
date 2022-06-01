norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VIT_MLA',
        model_name='deit_base_distilled_path16_384',
        img_size=480,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=19,
        drop_rate=0.0,
        norm_cfg=dict(type='BN', requires_grad=True),
        pos_embed_interp=True,
        align_corners=False,
        mla_channels=256,
        mla_index=(2, 5, 8, 11)),
    decode_head=dict(
        type='VIT_MLAHead',
        in_channels=1024,
        channels=512,
        img_size=480,
        mla_channels=256,
        mlahead_channels=128,
        num_classes=3,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=0,
            img_size=480,
            num_classes=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            norm_cfg=dict(type='BN', requires_grad=True)),
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=1,
            img_size=480,
            num_classes=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            norm_cfg=dict(type='BN', requires_grad=True)),
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=2,
            img_size=480,
            num_classes=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            norm_cfg=dict(type='BN', requires_grad=True)),
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=3,
            img_size=480,
            num_classes=3,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            norm_cfg=dict(type='BN', requires_grad=True))
    ])
train_cfg = dict()
test_cfg = dict(mode='slide', crop_size=(480, 480), stride=(320, 320))
dataset_type = 'PascalContextDataset'
data_root = 'data/VOCdevkit/VOC2010/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (520, 520)
crop_size = (480, 480)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(520, 520), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(480, 480), cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(480, 480), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(520, 520),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, dataset='pascal_context'),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='PascalContextDataset',
        data_root='data/VOCdevkit/VOC2010/',
        img_dir='JPEGImages',
        ann_dir='SegmentationClassContext',
        split='ImageSets/SegmentationContext/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(520, 520), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(480, 480), cat_max_ratio=0.75),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(480, 480), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='PascalContextDataset',
        data_root='data/VOCdevkit/VOC2010/',
        img_dir='JPEGImages',
        ann_dir='SegmentationClassContext',
        split='ImageSets/SegmentationContext/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(520, 520),
                flip=False,
                transforms=[
                    dict(
                        type='Resize',
                        keep_ratio=True,
                        dataset='pascal_context'),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='PascalContextDataset',
        data_root='data/VOCdevkit/VOC2010/',
        img_dir='JPEGImages',
        ann_dir='SegmentationClassContext',
        split='ImageSets/SegmentationContext/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(520, 520),
                flip=False,
                transforms=[
                    dict(
                        type='Resize',
                        keep_ratio=True,
                        dataset='pascal_context'),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
total_iters = 80000
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU')
find_unused_parameters = True
work_dir = './work_dirs/SETR_MLA_DeiT_480x480_80k_pascal_context_bs_16'
gpu_ids = range(0, 1)
