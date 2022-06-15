_base_ = [
    '../_base_/models/setr_mla.py',
    '../_base_/datasets/pascal_context.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    backbone=dict(img_size=256, pos_embed_interp=True, drop_rate=0.,
                  mla_channels=256, mla_index=(5, 11, 17, 23),
                    norm_cfg = dict(type='BN', requires_grad=True)
                  ),
    decode_head=dict(img_size=256, mla_channels=256,
                     mlahead_channels=128, num_classes=1,
                     norm_cfg = dict(type='BN', requires_grad=True)),
    auxiliary_head=[
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=0,
            img_size=256,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        norm_cfg = dict(type='BN', requires_grad=True)),
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=1,
            img_size=256,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        norm_cfg = dict(type='BN', requires_grad=True)),
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=2,
            img_size=256,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        norm_cfg = dict(type='BN', requires_grad=True)),
        dict(
            type='VIT_MLA_AUXIHead',
            in_channels=256,
            channels=512,
            in_index=3,
            img_size=256,
            num_classes=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        norm_cfg = dict(type='BN', requires_grad=True)),
    ])

optimizer = dict(lr=0.001, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )

test_cfg = dict(mode='slide', crop_size=(256, 256), stride=(170, 170))
find_unused_parameters = True
data = dict(samples_per_gpu=1)
