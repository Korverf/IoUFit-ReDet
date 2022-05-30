from mmdet.models.utils import IOUfitModule
import torch
from ..registry import HEADS
from .convfc_rbbox_head import ConvFCBBoxHeadRbbox
from mmdet.core import (delta2dbbox_v2, delta2dbbox, delta2dbbox_v3, \
    hbb2obb_v2, RotBox2Polys_torch, polygonToRotRectangle_batch, accuracy)
from mmdet.core.visualization import draw_poly_detections
from mmdet.ops import obb_overlaps
import cv2


@HEADS.register_module
class IOUFitHeadRbbox(ConvFCBBoxHeadRbbox):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        super(IOUFitHeadRbbox, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        self.IOUfit = IOUfitModule(in_features=16, hidden_features=16)
        #self.mse_loss = nn.MSELoss(reduction='mean')
        #self.loss_fit_bbox_weight = loss_fit_bbox_weight
        #self.loss_fit_iou_weight = loss_fit_iou_weight
        #self.loss_aux_weight = loss_aux_weight

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             bbox_targets_decode,
             reduce=True):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['rbbox_loss_cls'] = self.loss_cls(
                    cls_score, labels, label_weights, avg_factor=avg_factor, reduce=reduce)
                losses['rbbox_acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if pos_inds.any():
                if rois.size(1) == 5:
                    obbs = hbb2obb_v2(rois[:, 1:])
                    if self.with_module:
                        bbox_pred_decode = delta2dbbox(obbs, bbox_pred, self.target_means, self.target_stds)
                        bbox_targets_decode_new = delta2dbbox(obbs, bbox_targets, self.target_means, self.target_stds)
                    else:
                        bbox_pred_decode = delta2dbbox_v3(obbs, bbox_pred, self.target_means, self.target_stds)
                        bbox_targets_decode_new = delta2dbbox_v3(obbs, bbox_targets, self.target_means,
                                                                 self.target_stds)
                elif rois.size(1) == 6:
                    obbs = rois[:, 1:]
                    bbox_pred_decode = delta2dbbox_v2(obbs, bbox_pred, self.target_means, self.target_stds)
                    bbox_targets_decode_new = delta2dbbox_v2(obbs, bbox_targets, self.target_means, self.target_stds)

                #pos_target = bbox_targets[pos_inds.type(torch.bool)]
                pos_target_decode = bbox_targets_decode_new[pos_inds.type(torch.bool)] #
                if self.reg_class_agnostic:
                    #pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 5)[pos_inds]
                    pos_rbbox_pred_decode = bbox_pred_decode.view(
                        bbox_pred.size(0), 5)[pos_inds]
                else:
                    #pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 5)[
                        #pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
                    pos_rbbox_pred_decode = bbox_pred_decode.view(
                        bbox_pred.size(0), -1,
                        5)[pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]

                #torch.set_printoptions(sci_mode=False, precision=6)
                #caculate IOU as gt
                # IoU_targets = obb_overlaps(pos_rbbox_pred_decode, pos_target_decode.detach(), is_aligned=True).squeeze(1)\
                #    .clamp(min=1e-6, max=1) #未归一化的5参数框

                # change rbbox to polys and normalize
                #TODO: change the normalize factor according to input image size

                # pos_poly_pred_decode = RotBox2Polys_torch(pos_rbbox_pred_decode) / 1024  # 归一化后的poly
                # pos_poly_target_decode = RotBox2Polys_torch(pos_target_decode) / 1024
                # pos_poly_pred_decode = (RotBox2Polys_torch(pos_rbbox_pred_decode) - 512) / 512 #归一化后的poly
                # pos_poly_target_decode = (RotBox2Polys_torch(pos_target_decode) - 512) / 512
                #for HSRC2016:
                
                pos_poly_pred_decode = RotBox2Polys_torch(pos_rbbox_pred_decode)
                pos_poly_target_decode = RotBox2Polys_torch(pos_target_decode)
                # polys1_draw = pos_poly_pred_decode
                # polys2_draw = pos_poly_target_decode

                pos_poly_pred_decode = (pos_poly_pred_decode - 400) / 400
                pos_poly_target_decode = (pos_poly_target_decode - 400) / 400

                # pos_bbox_pred_decode_new = pos_poly_pred_decode.detach().cpu().numpy()
                # pos_bbox_pred_decode_new = polygonToRotRectangle_batch(pos_bbox_pred_decode_new)
                # pos_bbox_pred_decode_new = pos_bbox_pred_decode_new * 512 + 512
                # use MLP to fit the rotate IOU
                iou_fit_value = self.IOUfit(pos_poly_pred_decode, pos_poly_target_decode.detach()) #0~1
                iou_fit_value = iou_fit_value[:, 0].clamp(min=1e-6, max=1)

                # draw_ind = IoU_targets > 0.6
                # draw_condition = IoU_targets[IoU_targets > 0.6]
                # if len(draw_condition) > 0:
                #     polys1_draw = polys1_draw[draw_ind].detach().cpu().numpy()  # 取前2个可视化
                #     polys2_draw = polys2_draw[draw_ind].detach().cpu().numpy()
                #     img = draw_poly_detections(polys1_draw, showStart=False, colormap=(0, 255, 0))
                #     img = draw_poly_detections(polys2_draw, img=img, showStart=False, colormap=(0, 0, 255))
                #     cv2.imwrite('/home/yyw/yyf/projects/ReDet/vis/vis.jpg', img)

                losses_bbox = self.loss_bbox(
                    iou_fit_value,
                    linear=False,
                    avg_factor=bbox_targets.size(0),
                    reduce=reduce
                )
                # pos_poly_pred_decode = pos_poly_pred_decode.detach().cpu().numpy()
                # pos_poly_target_decode = pos_poly_target_decode.detach().cpu().numpy()
                # polys1_draw = pos_poly_pred_decode
                # polys2_draw = pos_poly_target_decode
                # polys1_draw[:, 0::2] = pos_poly_pred_decode[:, 0::2] * 400 + 400
                # polys1_draw[:, 1::2] = pos_poly_pred_decode[:, 1::2] * 256 + 256
                # polys2_draw[:, 0::2] = pos_poly_target_decode[:, 0::2] * 400 + 400
                # polys2_draw[:, 1::2] = pos_poly_target_decode[:, 1::2] * 256 + 256
            #     rbboxes11 = torch.tensor([[421., 256., 100., 50., 0.]]).cuda()
            #     #[459.983215, 250.793396, 407.882141, 133.106705,  -1.567452]
            #     rbboxes22 = torch.tensor([[471., 256., 100., 50., 0.]]).cuda()
            #     polys1_draw, polys2_draw = RotBox2Polys_torch(rbboxes11), RotBox2Polys_torch(rbboxes22)
            #     polys1 = (polys1_draw - 512) / 512
            #     polys2 = (polys2_draw - 512) / 512
            #     iou_fit_value1 = self.IOUfit(polys1, polys2)
            #     iou_fit_value1 = iou_fit_value1[:, 0].clamp(min=1e-6, max=1)
            #     IoU_targets1 = obb_overlaps(rbboxes11, rbboxes22.detach(), is_aligned=True).squeeze(
            # 1).clamp(min=1e-6)
            #     img = draw_poly_detections(pos_poly_pred_decode, showStart=False, colormap=(0, 255, 0))
            #     img = draw_poly_detections(pos_poly_target_decode, img=img, showStart=False, colormap=(0, 0, 255))
            #     cv2.imwrite('/home/yyw/yyf/projects/ReDet/vis/vis.jpg', img)
            #     #torch.set_printoptions(sci_mode=False, precision=6)
            #     print('fit IOU: ', iou_fit_value)
            #     print('IOU targets: ', IoU_targets)
            #     print('loss:', losses_bbox)
                losses['rbbox_loss_bbox'] = losses_bbox

        return losses