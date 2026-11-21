# P2PNet + uncertainty branch, adapted from P2PNet / Consistent-Point for
# "Calibrating Uncertainty for Semi-Supervised Crowd Counting" (ICCV 2023).
#
# Model (paper Sec. 3.3):
#   VGG-16_bn front 13 layers  ->  shared feature extractor (1/2 res, 128 ch)
#        |-- counting branch       (P2PNet: FPN + regression/classification)
#        |-- uncertainty branch    (convs + sigmoid -> per-patch uncertainty)
#
# The uncertainty branch outputs ONE sigmoid value per patch_size x patch_size
# region of the network input (paper granularity: 64, so a 128x128 training
# crop yields a 2x2 uncertainty map; C = 64*sqrt(2) is the unmatched penalty
# used by the ASM surrogate).
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .backbone import build_backbone
from .matcher import build_matcher_crowd

PATCH_SIZE_DEFAULT = 64  # uncertainty unit (paper Table 8 optimum)


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.output(out)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 2)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=2,
                 feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes,
                                kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.output(out)
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, _ = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


def generate_anchor_points(stride=16, row=2, line=2):
    row_step = stride / row
    line_step = stride / line
    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    anchor_points = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
    return anchor_points


def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) +
                         shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))
    return all_anchor_points


class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = np.array(image.shape[2:])
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x)
                        for x in self.pyramid_levels]
        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2 ** p, row=self.row, line=self.line)
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx],
                                           anchor_points)
            all_anchor_points = np.append(all_anchor_points,
                                          shifted_anchor_points, axis=0)
        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        return torch.from_numpy(all_anchor_points.astype(np.float32)).to(image.device)


class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(Decoder, self).__init__()
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


class UncertaintyHead(nn.Module):
    """Patch-level uncertainty branch (CU paper Sec. 3.3).

    Input : shared VGG front-13-layer feature map (1/2 resolution, 128 ch).
    Output: sigmoid map with ONE cell per patch_size x patch_size pixels of
            the network input (patch_size=64 -> 128x128 crop gives 2x2 map).

    The stacked stride-2 convs give a receptive field (~129 px) covering the
    whole patch plus context, so each output value can judge its patch.
    """

    def __init__(self, in_channels=128, width=256, patch_size=PATCH_SIZE_DEFAULT):
        super().__init__()
        assert patch_size >= 2 and (patch_size & (patch_size - 1)) == 0, \
            'patch_size must be a power of two (32/64/128 for the ablation)'
        n_stride = int(math.log2(patch_size // 2))  # 1/2 feature -> 1/patch_size
        layers = [nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
                  nn.ReLU(inplace=True)]
        for _ in range(n_stride):
            layers += [nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(width, 1, kernel_size=3, padding=1)]
        self.body = nn.Sequential(*layers)
        self.patch_size = patch_size

    def forward(self, x):
        return torch.sigmoid(self.body(x))


class P2PNet(nn.Module):
    def __init__(self, backbone, row=2, line=2,
                 patch_size=PATCH_SIZE_DEFAULT):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2  # 0: background, 1: person
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=256,
                                          num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(
            num_features_in=256,
            num_classes=self.num_classes,
            num_anchor_points=num_anchor_points)
        # uncertainty branch on the shared front-13-layer feature (1/2 res)
        self.uncertainty = UncertaintyHead(in_channels=128,
                                           patch_size=patch_size)

        self.anchor_points = AnchorPoints(pyramid_levels=[3, ], row=row, line=line)
        self.fpn = Decoder(256, 512, 512)

    def forward(self, samples):
        features = self.backbone(samples)
        features_fpn = self.fpn([features[1], features[2], features[3]])

        batch_size = features[0].shape[0]
        regression = self.regression(features_fpn[1]) * 100  # 8x downsample level
        classification = self.classification(features_fpn[1])
        uncertainty = self.uncertainty(features[0])          # [B, 1, H/p, W/p]

        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        output_coord = regression + anchor_points

        out = {'pred_logits': classification,      # [B, M, 2] raw logits
               'pred_points': output_coord,         # [B, M, 2] absolute coords
               'anchor_points': anchor_points,      # [B, M, 2] reference coords
               'pred_uncer': uncertainty}           # [B, 1, H/p, W/p] patch unc.
        return out


class SetCriterion_Crowd(nn.Module):
    """P2PNet criterion (identical for labeled GT and unlabeled pseudo-points).

    loss_ce   : CE over all proposals (background weighted by eos_coef)
    loss_point: MSE over matched pairs, weight = point_loss_coef

    CU supervises pseudo-points UNWEIGHTED (no instance-wise weighting --
    that is Consistent-Point's IUC, not used here).
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(size=src_logits.shape[:2], fill_value=0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes,
                                  weight=self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_points(self, outputs, targets, indices, num_points):
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat(
            [t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if src_points.numel() == 0:
            return {'loss_point': torch.zeros(
                1, device=outputs['pred_points'].device).squeeze()}

        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')
        losses = {'loss_point': loss_bbox.sum() / max(num_points, 1)}
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(input=src, fill_value=i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """L_pred for BOTH flows: target dicts carry 'point' + 'labels'."""
        output1 = {'pred_logits': outputs['pred_logits'],
                   'pred_points': outputs['pred_points']}
        indices1, num_boxes = self._prepare(output1, targets)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))
        return losses

    def _prepare(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_points = sum(len(t['labels']) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float,
                                     device=outputs['pred_points'].device)
        num_boxes = torch.clamp(num_points, min=1).item()
        return indices, num_boxes


def build(args, training, device=None):
    num_classes = 1

    backbone = build_backbone(args)
    patch_size = getattr(args, 'patch_size', PATCH_SIZE_DEFAULT)
    model = P2PNet(backbone, args.row, args.line, patch_size=patch_size)
    if not training:
        return model

    weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(
        num_classes, matcher=matcher, weight_dict=weight_dict,
        eos_coef=args.eos_coef, losses=losses)

    return model, criterion
