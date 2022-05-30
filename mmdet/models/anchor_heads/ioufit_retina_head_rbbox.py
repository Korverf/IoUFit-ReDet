import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

# from .anchor_head import AnchorHead
from .anchor_head_rbbox import AnchorHeadRbbox
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule
from mmdet.core import (anchor_target_rbbox, multi_apply, RotBox2Polys_torch, delta2dbbox, delta2dbbox_v3, \
    hbb2obb_v2, images_to_levels)
import torch
from mmdet.models.utils import IOUfitModule
import copy
from mmdet.core.visualization import draw_poly_detections
from mmdet.ops import obb_overlaps
import cv2


@HEADS.register_module
class IOUFitRetinaHeadRbbox(AnchorHeadRbbox):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(IOUFitRetinaHeadRbbox, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)
        self.IOUfit = IOUfitModule(in_features=16, hidden_features=16)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_masks,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        anchor_list_copy = copy.deepcopy(anchor_list)
        cls_reg_targets = anchor_target_rbbox(
            anchor_list_copy,
            valid_flag_list,
            gt_bboxes,
            gt_masks,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            with_module=self.with_module,
            reg_decoded_bbox=self.reg_decoded_bbox,
            hbb_trans=self.hbb_trans)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        losses_bbox_new = []
        for loss in losses_bbox:
            if isinstance(loss, torch.Tensor):
                losses_bbox_new.append(loss)
        return dict(rbbox_loss_cls=losses_cls, rbbox_loss_bbox=losses_bbox_new)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        #bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        loss_bbox = 0.
        pos_inds = labels > 0
        if pos_inds.any():
            anchors = anchors.reshape(-1, 4)
            obbs = hbb2obb_v2(anchors)
            if self.with_module:
                bbox_pred_decode = delta2dbbox(obbs, bbox_pred, self.target_means, self.target_stds)
                #bbox_targets_decode_new = delta2dbbox(obbs, bbox_targets, self.target_means, self.target_stds)
            else:
                bbox_pred_decode = delta2dbbox_v3(obbs, bbox_pred, self.target_means, self.target_stds)
                # bbox_targets_decode_new = delta2dbbox_v3(obbs, bbox_targets, self.target_means,
                #                                              self.target_stds)

            pos_target_decode = bbox_targets[pos_inds.type(torch.bool)]  #
            #if self.reg_class_agnostic:
                # pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 5)[pos_inds]
            pos_rbbox_pred_decode = bbox_pred_decode.view(
                    bbox_pred.size(0), 5)[pos_inds]
            # else:
            #     # pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 5)[
            #     # pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
            #     pos_rbbox_pred_decode = bbox_pred_decode.view(
            #         bbox_pred.size(0), -1,
            #         5)[pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]
            #torch.set_printoptions(sci_mode=False, precision=6)
            # caculate IOU as gt
            IoU_targets = obb_overlaps(pos_rbbox_pred_decode, pos_target_decode.detach(), is_aligned=True).squeeze(1)\
               .clamp(min=1e-6, max=1) #未归一化的5参数框
            pos_poly_pred_decode = RotBox2Polys_torch(pos_rbbox_pred_decode)
            pos_poly_target_decode = RotBox2Polys_torch(pos_target_decode)
            # polys1_draw = pos_poly_pred_decode[:2].detach().cpu().numpy() #取前2个可视化
            # polys2_draw = pos_poly_target_decode[:2].detach().cpu().numpy()
            pos_poly_pred_decode = (pos_poly_pred_decode - 400) / 400
            pos_poly_target_decode = (pos_poly_target_decode - 400) / 400
            iou_fit_value = self.IOUfit(pos_poly_pred_decode, pos_poly_target_decode)  # 0~1
            iou_fit_value = iou_fit_value[:, 0].clamp(min=1e-6, max=1)

            # img = draw_poly_detections(polys1_draw, showStart=False, colormap=(0, 255, 0))
            # img = draw_poly_detections(polys2_draw, img=img, showStart=False, colormap=(0, 0, 255))
            # cv2.imwrite('/home/yyw/yyf/projects/ReDet/vis/vis.jpg', img)
            # print('fit IOU: ', iou_fit_value)
            # print('IOU targets: ', IoU_targets)
            loss_bbox = self.loss_bbox(
                iou_fit_value,
                linear=False,
                avg_factor=num_total_samples,
                reduce=True
            )
        # loss_bbox = self.loss_bbox(
        #     bbox_pred,
        #     bbox_targets,
        #     bbox_weights,
        #     avg_factor=num_total_samples)


        return loss_cls, loss_bbox
