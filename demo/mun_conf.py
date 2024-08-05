model = dict(
    backbone=dict(
        arch='base',
        drop_path_rate=0.4,
        gap_before_final_norm=False,
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
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=2,
        type='NUHead'),
    test_cfg=dict(crop_size=(
        512,
        512,
    ), mode='slide', stride=(
        341,
        341,
    )),
    type='NPPEncoderDecoder')
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
