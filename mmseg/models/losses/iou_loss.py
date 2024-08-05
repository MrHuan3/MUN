import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss
import math
import os


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
   
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    if (avg_factor is None) and reduction == 'mean':
        if class_weight is None:
            if avg_non_ignore:
                avg_factor = label.numel() - (label
                                              == ignore_index).sum().item()
            else:
                avg_factor = label.numel()

        else:
            label_weights = torch.stack([class_weight[cls] for cls in label
                                         ]).to(device=class_weight.device)

            if avg_non_ignore:
                label_weights[label == ignore_index] = 0
            avg_factor = label_weights.sum()

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@MODELS.register_module()
class IoULoss(nn.Module):
   
    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_iou',
                 avg_non_ignore=False, 
                 lambda_iou=1.0, 
                 alpha0=1.0, 
                 alpha1=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        self.cls_criterion = cross_entropy
        self._loss_name = loss_name
        self.lambda_iou = lambda_iou
        self.alpha0=alpha0
        self.alpha1=alpha1
        

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        probability = F.softmax(cls_score, dim=1)

        pro = probability.clone().detach()
        lab = label.clone().detach().to(torch.uint8)

        matrix = pro[:,1,:,:] > pro[:,0,:,:]
        matrix = matrix.to(torch.uint8)

        matrix_I = lab & matrix
        matrix_I = matrix_I.to(torch.uint8)

        matrix_U = lab | matrix
        matrix_U = matrix_U.to(torch.uint8)

        i = torch.sum(matrix_I).item()
        u = torch.sum(matrix_U).item()

        IoU = 1.0 * i / u

        iou_loss = (1.0 - IoU) * self.lambda_iou

        beta = math.exp(iou_loss)

        cls_wei = []
        
        cls_wei.append(1.0 * self.alpha0)
        cls_wei.append(beta * self.alpha1)
        class_weight = cls_score.new_tensor(cls_wei)

        loss_cls_same = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            avg_non_ignore=self.avg_non_ignore,
            ignore_index=ignore_index,
            **kwargs)
        return loss_cls_same
        

    @property
    def loss_name(self):
        return self._loss_name
