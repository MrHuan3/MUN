checkpoint_file = '/PATH/TO/PRETRAINED/CONVNEXT-V2.pth'
crop_size = (
    512,
    512,
)
custom_imports = dict(allow_failed_imports=False, imports='mmpretrain.models')
data_preprocessor = dict(
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=1,
    size=(
        512,
        512,
    ),
    type='SegDataPreProcessor')
data_root = '/PATH/TO/YOUR/DATASET'
dataset_type = 'Mydata'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=500, type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        arch='base',
        drop_path_rate=0.4,
        gap_before_final_norm=False,
        init_cfg=dict(
            checkpoint=
            '/PATH/TO/PRETRAINED/CONVNEXT-V2.pth',
            prefix='backbone.',
            type='Pretrained'),
        layer_scale_init_value=0.0,
        out_indices=[
            0,
            1,
            2,
            3,
        ],
        type='mmpretrain.ConvNeXt',
        use_grn=True),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        pad_val=0,
        seg_pad_val=1,
        size=(
            512,
            512,
        ),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=128,
        dropout_ratio=0.1,
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            alpha0=1.0,
            alpha1=1.0,
            lambda_iou=1.5,
            loss_weight=1.0,
            type='IoULoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=2,
        type='NUHead'),
    pretrained=None,
    test_cfg=dict(crop_size=(
        512,
        512,
    ), mode='slide', stride=(
        341,
        341,
    )),
    train_cfg=dict(),
    type='NPPEncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(decay_rate=0.9, decay_type='stage_wise', num_layers=12),
    type='AmpOptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=87770,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/validation', seg_map_path='ann_dir/validation'),
        data_root='/PATH/TO/YOUR/DATASET',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='NPPTest'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='NPPResize'),
            dict(size_divisor=32, type='NPPResizeToMultiple'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='NPPPackSegInputs'),
        ],
        type='Mydata'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='NPPTest'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='NPPResize'),
    dict(size_divisor=32, type='NPPResizeToMultiple'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='NPPPackSegInputs'),
]
train_cfg = dict(max_iters=87770, type='IterBasedTrainLoop', val_interval=8777)
train_dataloader = dict(
    batch_size=7,
    dataset=dict(
        dataset=dict(
            data_prefix=dict(
                img_path='img_dir/training', seg_map_path='ann_dir/training'),
            data_root='/PATH/TO/YOUR/DATASET',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(reduce_zero_label=False, type='LoadAnnotations'),
                dict(type='NPPTrain'),
                dict(keep_ratio=False, scale=(
                    512,
                    512,
                ), type='NPPResize'),
                dict(
                    cat_max_ratio=0.75,
                    crop_size=(
                        512,
                        512,
                    ),
                    type='NPPRandomCrop'),
                dict(prob=0.5, type='NPPRandomFlip'),
                dict(type='PhotoMetricDistortion'),
                dict(type='NPPPackSegInputs'),
            ],
            type='Mydata'),
        times=50,
        type='RepeatDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='NPPTrain'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='NPPResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='NPPRandomCrop'),
    dict(prob=0.5, type='NPPRandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='NPPPackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/validation', seg_map_path='ann_dir/validation'),
        data_root='/PATH/TO/YOUR/DATASET',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='NPPValidation'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='NPPResize'),
            dict(size_divisor=32, type='NPPResizeToMultiple'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='NPPPackSegInputs'),
        ],
        type='Mydata'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/output'
