# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

from mmrotate.core.bbox.iou_calculators import rbbox_overlaps

from .builder import MATCH_COST


@MATCH_COST.register_module()
class BBoxL1Cost:
    """BBoxL1Cost.

     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., box_format='xyxy'):
        self.weight = weight
        assert box_format in ['xyxy', 'xywh', 'xywha'], "box_format {} is not supported.".format(box_format)
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif self.box_format == 'xyxy':
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        elif self.box_format == 'xywha':
            # no need to transform
            pass
        else:
            pass

        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class FocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         alpha (int | float, optional): focal_loss alpha
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12
         binary_input (bool, optional): Whether the input is binary,
            default False.

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import FocalLossCost
         >>> import torch
         >>> self = FocalLossCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self,
                 weight=1.,
                 alpha=0.25,
                 gamma=2,
                 eps=1e-12,
                 binary_input=False):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def _focal_loss_cost(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_query, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
                1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
                1 - cls_pred).pow(self.gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight

    def _mask_focal_loss_cost(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classfication logits
                in shape (num_query, d1, ..., dn), dtype=torch.float32.
            gt_labels (Tensor): Ground truth in shape (num_gt, d1, ..., dn),
                dtype=torch.long. Labels should be binary.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_query, num_gt).
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
                1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
                1 - cls_pred).pow(self.gamma)

        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
                   torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost / n * self.weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classfication logits.
            gt_labels (Tensor)): Labels.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_query, num_gt).
        """
        if self.binary_input:
            return self._mask_focal_loss_cost(cls_pred, gt_labels)
        else:
            return self._focal_loss_cost(cls_pred, gt_labels)


@MATCH_COST.register_module()
class ClassificationCost:
    """ClsSoftmaxCost.

     Args:
         weight (int | float, optional): loss_weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import \
         ... ClassificationCost
         >>> import torch
         >>> self = ClassificationCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3430, -0.3525, -0.3045],
                [-0.3077, -0.2931, -0.3992],
                [-0.3664, -0.3455, -0.2881],
                [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_query, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class IoUCost:
    """IoUCost.

     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode='giou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        overlaps = bbox_overlaps(
            bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight


@MATCH_COST.register_module()
class RotatedIoUCost:
    """RotatedIoUCost.

     Args:
         mode (str): Loss scaling mode, including "linear", "square", and "log".
                Default: 'log'
         weight (int | float, optional): loss weight

    """

    def __init__(self, mode='iou', weight=1.):
        self.weight = weight
        self.mode = mode

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with unnormalized coordinates
                (x, y, h, w, a). Shape (num_query, 5).
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x, y, h, w, a). Shape (num_gt, 5).

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        # iou_cost = rotated_iou_loss(bboxes, gt_bboxes, mode=self.mode)
        overlaps = rbbox_overlaps(bbox_pred, gt_bboxes, mode=self.mode, is_aligned=False)
        iou_cost = -overlaps
        return iou_cost * self.weight


@MATCH_COST.register_module()
class DiceCost:
    """Cost of mask assignments based on dice losses.

    Args:
        weight (int | float, optional): loss_weight. Defaults to 1.
        pred_act (bool, optional): Whether to apply sigmoid to mask_pred.
            Defaults to False.
        eps (float, optional): default 1e-12.
        naive_dice (bool, optional): If True, use the naive dice loss
            in which the power of the number in the denominator is
            the first power. If Flase, use the second power that
            is adopted by K-Net and SOLO.
            Defaults to True.
    """

    def __init__(self, weight=1., pred_act=False, eps=1e-3, naive_dice=True):
        self.weight = weight
        self.pred_act = pred_act
        self.eps = eps
        self.naive_dice = naive_dice

    def binary_mask_dice_loss(self, mask_preds, gt_masks):
        """
        Args:
            mask_preds (Tensor): Mask prediction in shape (num_query, *).
            gt_masks (Tensor): Ground truth in shape (num_gt, *)
                store 0 or 1, 0 for negative class and 1 for
                positive class.

        Returns:
            Tensor: Dice cost matrix in shape (num_query, num_gt).
        """
        mask_preds = mask_preds.flatten(1)
        gt_masks = gt_masks.flatten(1).float()
        numerator = 2 * torch.einsum('nc,mc->nm', mask_preds, gt_masks)
        if self.naive_dice:
            denominator = mask_preds.sum(-1)[:, None] + \
                          gt_masks.sum(-1)[None, :]
        else:
            denominator = mask_preds.pow(2).sum(1)[:, None] + \
                          gt_masks.pow(2).sum(1)[None, :]
        loss = 1 - (numerator + self.eps) / (denominator + self.eps)
        return loss

    def __call__(self, mask_preds, gt_masks):
        """
        Args:
            mask_preds (Tensor): Mask prediction logits in shape (num_query, *)
            gt_masks (Tensor): Ground truth in shape (num_gt, *)

        Returns:
            Tensor: Dice cost matrix with weight in shape (num_query, num_gt).
        """
        if self.pred_act:
            mask_preds = mask_preds.sigmoid()
        dice_cost = self.binary_mask_dice_loss(mask_preds, gt_masks)
        return dice_cost * self.weight


@MATCH_COST.register_module()
class CrossEntropyLossCost:
    """CrossEntropyLossCost.

    Args:
        weight (int | float, optional): loss weight. Defaults to 1.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to True.
    Examples:
         >>> from mmdet.core.bbox.match_costs import CrossEntropyLossCost
         >>> import torch
         >>> bce = CrossEntropyLossCost(use_sigmoid=True)
         >>> cls_pred = torch.tensor([[7.6, 1.2], [-1.3, 10]])
         >>> gt_labels = torch.tensor([[1, 1], [1, 0]])
         >>> print(bce(cls_pred, gt_labels))
    """

    def __init__(self, weight=1., use_sigmoid=True):
        assert use_sigmoid, 'use_sigmoid = False is not supported yet.'
        self.weight = weight
        self.use_sigmoid = use_sigmoid

    def _binary_cross_entropy(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): The prediction with shape (num_query, 1, *) or
                (num_query, *).
            gt_labels (Tensor): The learning label of prediction with
                shape (num_gt, *).

        Returns:
            Tensor: Cross entropy cost matrix in shape (num_query, num_gt).
        """
        cls_pred = cls_pred.flatten(1).float()
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        pos = F.binary_cross_entropy_with_logits(
            cls_pred, torch.ones_like(cls_pred), reduction='none')
        neg = F.binary_cross_entropy_with_logits(
            cls_pred, torch.zeros_like(cls_pred), reduction='none')
        cls_cost = torch.einsum('nc,mc->nm', pos, gt_labels) + \
                   torch.einsum('nc,mc->nm', neg, 1 - gt_labels)
        cls_cost = cls_cost / n

        return cls_cost

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits.
            gt_labels (Tensor): Labels.

        Returns:
            Tensor: Cross entropy cost matrix with weight in
                shape (num_query, num_gt).
        """
        if self.use_sigmoid:
            cls_cost = self._binary_cross_entropy(cls_pred, gt_labels)
        else:
            raise NotImplementedError

        return cls_cost * self.weight


@MATCH_COST.register_module()
class KLDLossCost:

    def __init__(self, weight=1., fun='log1p', tau=1.0):
        self.weight = weight
        self.fun = fun
        self.tau = tau

    def xy_wh_r_2_xy_sigma(self, xywhr):
        """Convert oriented bounding box to 2-D Gaussian distribution.

        Args:
            xywhr (torch.Tensor): rbboxes with shape (N, 5).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = xywhr[..., 4]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma

    def __call__(self, pred, target):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2, x3, y3, x4, y4). Shape (num_query, 8).
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2, x3, y3, x4, y4). Shape (num_gt, 8).

        Returns:
            torch.Tensor: iou_cost value with weight （num_query, num_gt）
        """
        pred = self.xy_wh_r_2_xy_sigma(pred)
        target = self.xy_wh_r_2_xy_sigma(target)

        mu_p, sigma_p = pred
        mu_t, sigma_t = target

        mu_p = mu_p.reshape(-1, 2)
        mu_t = mu_t.reshape(-1, 2)
        sigma_p = sigma_p.reshape(-1, 2, 2)
        sigma_t = sigma_t.reshape(-1, 2, 2)

        # todo mu_p (100, 2)  mu_t(8, 2)
        # todo not sure the result is right
        num_query = mu_p.shape[0]
        num_gt = mu_t.shape[0]
        assert num_query >= num_gt, "number of ground truth bbox large than number of query"

        delta = (mu_p[:, None, :] - mu_t[None, :, :]) \
            .reshape(num_query, num_gt, 2) \
            .unsqueeze(-1)  # (m, n, 2, 1)
        sigma_t_inv = torch.inverse(sigma_t)  # (n, 2, 2)

        term1 = delta.transpose(-1, -2).matmul(sigma_t_inv[None, :, :, :]).matmul(delta) \
            .reshape(mu_p.shape[0], mu_t.shape[0])
        term2 = torch.diagonal(
            sigma_t_inv[None, :, :, :].matmul(sigma_p[:, None, :, :]),
            dim1=-2, dim2=-1
        ).sum(dim=-1, keepdim=True).squeeze(-1) \
                + torch.log(torch.det(sigma_t_inv[None, :, :, :]) / torch.det(sigma_p[:, None, :, :]))

        dis = term1 + term2 - 2
        kl_dis = dis.clamp(min=1e-6)
        # The 1 is a constant that doesn't change the matching, so omitted.
        if self.fun == 'sqrt':
            kl_loss = - 1 / (self.tau + torch.sqrt(kl_dis))
        else:
            kl_loss = - 1 / (self.tau + torch.log1p(kl_dis))

        return kl_loss * self.weight
