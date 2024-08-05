import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch import Tensor

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead

from mmdet.models.layers.csp_layer import \
    DarknetBottleneck as MMDET_DarknetBottleneck
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.utils import digit_version
from typing import Sequence
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule)
from mmengine.model import BaseModule


if digit_version(torch.__version__) >= digit_version('1.7.0'):
    MODELS.register_module(module=nn.SiLU, name='SiLU')
else:

    class SiLU(nn.Module):
        """Sigmoid Weighted Liner Unit."""

        def __init__(self, inplace=True):
            super().__init__()

        def forward(self, inputs) -> Tensor:
            return inputs * torch.sigmoid(inputs)

    MODELS.register_module(module=SiLU, name='SiLU')

class DarknetBottleneck(MMDET_DarknetBottleneck):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 kernel_size: Sequence[int] = (1, 3),
                 padding: Sequence[int] = (0, 1),
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels, out_channels, init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        assert isinstance(kernel_size, Sequence) and len(kernel_size) == 2

        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size[0],
            padding=padding[0],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            kernel_size[1],
            stride=1,
            padding=padding[1],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels
        
class CSPLayerWithTwoConv(BaseModule):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            2 * self.mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            (2 + num_blocks) * self.mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.ModuleList(
            DarknetBottleneck(
                self.mid_channels,
                self.mid_channels,
                expansion=1,
                kernel_size=(3, 3),
                padding=(1, 1),
                add_identity=add_identity,
                use_depthwise=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks))

    def forward(self, x: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        return self.final_conv(torch.cat(x_main, 1))


@MODELS.register_module()
class NUHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # tdbu Module
        self.lateral_convs = nn.ModuleList()
        # self.lateral_convs2 = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.nu_convs = nn.ModuleList()

        for in_channels in self.in_channels:
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            n_conv = ConvModule(
                self.channels * 2,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.nu_convs.append(n_conv)

        for _ in self.in_channels[:-1]:
            u_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.up_convs.append(u_conv)
        self.top_bottleneck = CSPLayerWithTwoConv(in_channels=128, out_channels=128)
        self.bottom_bottleneck = CSPLayerWithTwoConv(in_channels=128, out_channels=128)

        self.nu_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        inputs = self._transform_inputs(inputs)
        laterals_top_down = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals_bottom_up = [layer.clone() for layer in laterals_top_down]

        laterals_top_down[-1] = self.top_bottleneck(laterals_top_down[-1])

        used_backbone_levels = len(laterals_top_down)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals_top_down[i - 1].shape[2:]
            laterals_top_down[i - 1] = laterals_top_down[i - 1] + resize(
                laterals_top_down[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        laterals_bottom_up[0] = self.bottom_bottleneck(laterals_bottom_up[0])

        for i in range(0, used_backbone_levels - 1):
            laterals_bottom_up[i + 1] = laterals_bottom_up[i + 1] + self.up_convs[i](laterals_bottom_up[i])

        laterals = [torch.cat(
            [laterals_top_down[i], laterals_bottom_up[i]], dim=1)
            for i in range(used_backbone_levels)
        ]

        outputs = [
            self.nu_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]

        for i in range(used_backbone_levels - 1, 0, -1):
            outputs[i] = resize(
                outputs[i],
                size=outputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        nu_outs = torch.cat(outputs, dim=1)
        feats = self.nu_bottleneck(nu_outs)
        return feats

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output