from .single_stage_rbbox import SingleStageDetectorRbbox
from ..registry import DETECTORS


@DETECTORS.register_module
class IOUFitRetinaNetRbbox(SingleStageDetectorRbbox):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head=None,
                 rbbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(IOUFitRetinaNetRbbox, self).__init__(backbone, neck, bbox_head, rbbox_head,
                                             train_cfg, test_cfg, pretrained)
    def forward_train(self,
                img,
                img_metas,
                gt_bboxes,
                gt_masks,
                gt_labels,
                gt_bboxes_ignore=None):
        # print('in single stage rbbox')
        # import pdb
        # pdb.set_trace()
        x = self.extract_feat(img)

        losses = dict()

        if self.with_bbox:
            bbox_outs = self.bbox_head(x)
            bbox_loss_inputs = bbox_outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
            # TODO: make if flexible to add the bbox_head
            bbox_losses = self.bbox_head.loss(
                *bbox_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(bbox_losses)
        if self.with_rbbox:

            rbbox_outs = self.rbbox_head(x)
            rbbox_loss_inputs = rbbox_outs + (gt_bboxes, gt_masks, gt_labels, img_metas, self.train_cfg)
            rbbox_losses = self.rbbox_head.loss(
                *rbbox_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rbbox_losses)
        return losses

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            rbbox_ex_anchors = hbb2obb_v2(anchors)
            if self.with_module:
                bboxes = delta2dbbox(rbbox_ex_anchors, bbox_pred, self.target_means,
                                     self.target_stds, img_shape)
            else:
                bboxes = delta2dbbox_v3(rbbox_ex_anchors, bbox_pred, self.target_means,
                                     self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            # mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[:, :4] /= mlvl_bboxes[:, :4].new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        # det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
        #                                         cfg.score_thr, cfg.nms,
        #                                         cfg.max_per_img)
        det_bboxes, det_labels = multiclass_nms_rbbox(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
