import torch
import torch.nn as nn
#from mmdet.ops import box_iou_rotated_differentiable

from ..registry import LOSSES
#from .utils import weighted_loss
from .piou.pixel_weights import Pious
from mmdet.ops import obb_overlaps


def template_w_pixels(width):
  x = torch.tensor(torch.arange(-100, width + 100))
  grid_x = x.float() + 0.5
  return grid_x


@LOSSES.register_module
class PIoULoss(nn.Module):

    def __init__(self, linear=False, eps=1e-6, reduction='mean', loss_weight=1.0, img_size=1024):
        super(PIoULoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.PIoU = Pious(k=10, is_hard=False)
        self.template = template_w_pixels(img_size)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs
                ):
        # if weight is not None and not torch.any(weight > 0):
        #     return (pred * weight).sum()  # 0
        # assert reduction_override in (None, 'none', 'mean', 'sum')
        # reduction = (
        #     reduction_override if reduction_override else self.reduction)
        # if (weight is not None) and (not torch.any(weight > 0)) and (
        #         reduction != 'none'):
        #     return (pred * weight).sum()  # 0
        # if weight is not None and weight.dim() > 1:
        #     # TODO: remove this in the future
        #     # reduce the weight of shape (n, 4) to (n,) to match the
        #     # iou_loss of shape (n,)
        #     assert weight.shape == pred.shape
        #     weight = weight.mean(-1)
        loss = self.loss_weight * self.piou_loss(pred, target)
        return loss
    
    def piou_loss(self, pred, target):
        """IoU loss.

        Computing the IoU loss between a set of predicted bboxes and target bboxes.
        The loss is calculated as negative log of IoU.

        Args:
            pred (Tensor): Predicted bboxes of format (x, y, w, h, a),
                shape (n, 5).
            target (Tensor): Corresponding gt bboxes, shape (n, 5).
            linear (bool):  If True, use linear scale of loss instead of
                log scale. Default: False.
            eps (float): Eps to avoid log(0).

        Return:
            Tensor: Loss tensor.
        """
        pious = self.PIoU(pred, target, self.template.cuda(pred.get_device())).clamp(min=0.1, max=1.0)
        # caculate IOU as gt
        #IoU_targets = obb_overlaps(pred, target.detach(), is_aligned=True).squeeze(1).clamp(min=1e-6, max=1) #未归一化的5参数框
        pious = -1.0 * torch.log(pious)# -2
        loss = torch.sum(pious)
        loss = loss / (pred.size(0) + 1e-9)
        
        return loss
