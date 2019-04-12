import torch
import torch.nn as nn

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from mmdet.core import (assign_and_sample, bbox2roi, bbox2result, multi_apply, weighted_cross_entropy, weighted_smoothl1)


class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 with_8_coo=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if rpn_head is not None:
            self.rpn_head = builder.build_rpn_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_bbox_head(bbox_head)

        if mask_head is not None:
            self.mask_roi_extractor = builder.build_roi_extractor(
                mask_roi_extractor)
            self.mask_head = builder.build_mask_head(mask_head)
        self.with_8_coo = with_8_coo
        if self.with_8_coo:
            self.coo_num = 8
        else:
            self.coo_num = 4    
        self.level = bbox_head['level']
        self.merge_mode = bbox_head['merge_mode']
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_roi_extractor.init_weights()
            self.mask_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    # def forward_train(self,
    #                   img,
    #                   img_meta,
    #                   gt_bboxes,
    #                   gt_bboxes_ignore,
    #                   gt_labels,
    #                   gt_masks=None,
    #                   proposals=None):
    def forward_train(self,
                    img,
                    img_meta,
                    gt_bboxes,
                    gt_bboxes_8_coo,
                    gt_labels,
                    gt_bboxes_ignore=None,
                    gt_masks=None,
                    proposals=None):            
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(len(gt_bboxes))]
        x = self.extract_feat(img)
        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)

            rpn_loss_inputs = rpn_outs + (gt_bboxes, gt_bboxes_8_coo, img_meta, self.coo_num,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)
            proposal_inputs = rpn_outs + (img_meta, self.coo_num, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_proposals(*proposal_inputs)
        else:
            proposal_list = proposals

        if self.train_cfg.rpn.with_gt_bboxes:
            for i in range(len(gt_bboxes)):
                gt_bboxes_trans = []
                for j in self.train_cfg.rpn.gt_bboxes_scale:
                    x_center = (gt_bboxes[i][:, 0:1]+gt_bboxes[i][:, 2:3])/2
                    y_center = (gt_bboxes[i][:, 1:2]+gt_bboxes[i][:, 3:4])/2
                    left = torch.clamp(((gt_bboxes[i][:, 0:1] - x_center) * j + x_center), min=0)
                    right = torch.clamp(((gt_bboxes[i][:, 2:3] - x_center) * j + x_center), max=img_meta[i]['img_shape'][1])
                    top = torch.clamp(((gt_bboxes[i][:, 1:2] - y_center) * j + y_center), min=0)
                    bottom = torch.clamp(((gt_bboxes[i][:, 3:4] - y_center) * j + y_center), max=img_meta[i]['img_shape'][0])
                    trans_gt_bboxes = torch.cat([left, top, right, bottom], 1)
                    gt_bboxes_trans.append(trans_gt_bboxes)
 
                gt_bboxes_trans = torch.cat(gt_bboxes_trans, 0)
                n = gt_bboxes_trans.shape[0]
                gt_bboxes_trans = torch.cat([gt_bboxes_trans, torch.ones([n, 1], device=gt_bboxes[i].device)], 1)
                proposal_list[i] = torch.cat([proposal_list[i], gt_bboxes_trans], 0)

        # assign gts and sample proposals
        _gt_bboxes_8_coo = [None for i in range(len(proposal_list))]
        if self.with_bbox or self.with_mask:
            assign_results, sampling_results = multi_apply(
                assign_and_sample,
                proposal_list,
                gt_bboxes,
                _gt_bboxes_8_coo,
                gt_bboxes_ignore,
                gt_labels,
                cfg=self.train_cfg.rcnn)
          
        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])

            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois, self.level, self.merge_mode)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            if self.train_cfg.rcnn.with_hard_example_mining:
                pred_label = torch.argmax(cls_score, 1)
                labels, label_weights, bbox_gt, bbox_weights = bbox_targets
                ind = pred_label != labels
                if torch.sum(ind).item() != 0:
                    x_stop_grad = [feature.data for feature in x]
                    bbox_feats_stop_grad = self.bbox_roi_extractor(
                    x_stop_grad[:self.bbox_roi_extractor.num_inputs], rois, self.level, self.merge_mode)  
                    cls_score_stop_grad, bbox_pred_stop_grad = self.bbox_head(bbox_feats_stop_grad)
                    cls_score_stop_grad, labels, label_weights = cls_score_stop_grad[ind], labels[ind], label_weights[ind]
                    num = cls_score.shape[0]
                    loss_bbox['loss_cls'] = loss_bbox['loss_cls'] + weighted_cross_entropy(
                           cls_score_stop_grad, labels, label_weights, avg_factor=num)
                    if self.train_cfg.rcnn.with_reg:
                        bbox_pred_stop_grad, bbox_gt, bbox_weights = bbox_pred_stop_grad[ind], bbox_gt[ind], bbox_weights[ind]
                        loss_bbox['loss_reg'] = loss_bbox['loss_reg'] + weighted_smoothl1(
                            bbox_pred_stop_grad,
                            bbox_gt,
                            bbox_weights,
                            avg_factor=num)
                            
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], pos_rois)
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.coo_num, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
def test_fun(x):
    return x+1