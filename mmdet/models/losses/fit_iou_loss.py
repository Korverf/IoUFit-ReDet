import torch.nn as nn
from mmdet.core import weighted_iou_loss
import torch.nn.functional as F
import torch
from ..registry import LOSSES


def fit_iou_loss(iou_pred, linear=False, weight=None, reduction='mean', avg_factor=None):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        iou_pred (torch.Tensor): Predicted IOUs,  shape (n).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    if linear:
        loss = 1 - iou_pred
    else:
        loss = -iou_pred.log()

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        reduction_enum = F._Reduction.get_enum(reduction)
        # none: 0, elementwise_mean:1, sum: 2
        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            if weight is not None:
                loss = torch.sum(loss * weight)[None] / avg_factor
            else:
                loss = torch.sum(loss)[None] / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


@LOSSES.register_module
class FitIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(FitIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                iou_pred,
                weight=None,
                linear=False,
                avg_factor=None,
                reduce=True):
        if weight is not None and not torch.any(weight > 0):
            return (iou_pred * weight).sum()  # 0
        if reduce:
            reduction = self.reduction
        else:
            reduction = 'none'
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            weight = weight.mean(-1)
            assert weight.shape == iou_pred.shape

        if avg_factor is None:
            avg_factor = torch.sum(weight > 0).float().item() + 1e-6
        loss = self.loss_weight * fit_iou_loss(
            iou_pred,
            weight=weight,
            linear=linear,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss
