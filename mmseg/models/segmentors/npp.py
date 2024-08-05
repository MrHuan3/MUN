# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor
import torch

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
import copy
from .mypro.imgnorm import ImgNorm
from .mypro.noise import Noise
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
import os

class Fuse(BaseModule):
    def __init__(self, dim):
        super().__init__()
        self.pool1 = nn.MaxPool2d(1)
        self.pool5 = nn.MaxPool2d(5, 1, 2)
        self.pool11 = nn.MaxPool2d(11, 1, 5)
        self.pool15 = nn.MaxPool2d(15, 1, 7)
        self.dim_conv1 = ConvModule(
            dim,
            1,
            1,
            conv_cfg=None,
            norm_cfg=dict(requires_grad=True, type='BN'),
            act_cfg=dict(type='GELU'))
        self.dim_conv5 = ConvModule(
            dim,
            1,
            1,
            conv_cfg=None,
            norm_cfg=dict(requires_grad=True, type='BN'),
            act_cfg=dict(type='GELU'))
        self.dim_conv11 = ConvModule(
            dim,
            1,
            1,
            conv_cfg=None,
            norm_cfg=dict(requires_grad=True, type='BN'),
            act_cfg=dict(type='GELU'))
        self.dim_conv15 = ConvModule(
            dim,
            1,
            1,
            conv_cfg=None,
            norm_cfg=dict(requires_grad=True, type='BN'),
            act_cfg=dict(type='GELU'))
        self.dim_reduce1 = ConvModule(
            dim,
            dim // 4,
            1,
            conv_cfg=None,
            norm_cfg=dict(requires_grad=True, type='BN'),
            act_cfg=dict(type='GELU'))
        self.dim_reduce5 = ConvModule(
            dim,
            dim // 4,
            1,
            conv_cfg=None,
            norm_cfg=dict(requires_grad=True, type='BN'),
            act_cfg=dict(type='GELU'))
        self.dim_reduce11 = ConvModule(
            dim,
            dim // 4,
            1,
            conv_cfg=None,
            norm_cfg=dict(requires_grad=True, type='BN'),
            act_cfg=dict(type='GELU'))
        self.dim_reduce15 = ConvModule(
            dim,
            dim // 4,
            1,
            conv_cfg=None,
            norm_cfg=dict(requires_grad=True, type='BN'),
            act_cfg=dict(type='GELU'))

    def forward(self, x, y):
        pool1 = self.pool1(x)
        pool5 = self.pool5(x)
        pool11 = self.pool11(x)
        pool15 = self.pool15(x)
        dim_pool1 = self.dim_conv1(pool1)
        dim_pool5 = self.dim_conv5(pool5)
        dim_pool11 = self.dim_conv11(pool11)
        dim_pool15 = self.dim_conv15(pool15)
        mul1 = torch.matmul(dim_pool1, y)
        mul5 = torch.matmul(dim_pool5, y)
        mul11 = torch.matmul(dim_pool11, y)
        mul15 = torch.matmul(dim_pool15, y)
        query0 = self.dim_reduce1(mul1)
        query1 = self.dim_reduce5(mul5)
        query2 = self.dim_reduce11(mul11)
        query3 = self.dim_reduce15(mul15)
        final = torch.cat((query0, query1, query2, query3), dim=1)
        return final



@MODELS.register_module()
class NPPEncoderDecoder(BaseSegmentor):
    
    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        NPP_backbone = copy.deepcopy(backbone)
        NPP_backbone.init_cfg = dict()
        self.NPP_backbone = MODELS.build(NPP_backbone)

        self.img_norm = ImgNorm()
        self.noise = Noise()
        
        self.R2N_0 = Fuse(128)
        self.R2N_1 = Fuse(256)
        self.R2N_2 = Fuse(512)
        self.R2N_3 = Fuse(1024)

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)


    def extract_feat(self, inputs: Tensor) -> List[Tensor]:

        inputs_bgr = inputs[:,0:3,:,:].clone()
        inputs_npp = inputs[:,3:6,:,:].clone().contiguous()

        inputs_rgb = torch.Tensor(inputs.shape[0], inputs.shape[1] // 2, 
                                  inputs.shape[2], inputs.shape[3]).to(inputs.device)
        inputs_rgb[:,0,:,:] = inputs_bgr[:,2,:,:]
        inputs_rgb[:,1,:,:] = inputs_bgr[:,1,:,:]
        inputs_rgb[:,2,:,:] = inputs_bgr[:,0,:,:]
        inputs_rgb = inputs_rgb.contiguous()
        
        if inputs.shape[1] == 7:
            gt = inputs[:,6,:,:].unsqueeze(1)
            inputs_rgb = self.noise(inputs_rgb, gt)
        
        inputs_rgb = self.img_norm(inputs_rgb)
        
        x = self.backbone(inputs_rgb)
        
        y = self.NPP_backbone(inputs_npp)
        
        fuse = [self.R2N_0(x[0], y[0] * 0.1), 
                  self.R2N_1(x[1], y[1] * 0.1), 
                  self.R2N_2(x[2], y[2] * 0.1), 
                  self.R2N_3(x[3], y[3] * 0.1)]
        
        out = []
        for i in range(len(x)):
            out.append(torch.cat([x[i], fuse[i]], dim=1))
        out = tuple(out)
        
        return out

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)
        
        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        # return seg_logit
        if self.out_channels == 1:
            output = F.sigmoid(seg_logit)
        else:
            output = F.softmax(seg_logit, dim=1)
        
        return output


    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
