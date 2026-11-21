import torch
import torch.nn.functional as F
from torch import nn

from ..util.misc import (NestedTensor,get_world_size,is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher_crowd

import numpy as np

# the network frmawork of the regression branch
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

        #output points: [(x1,y1),(x2,y2),...] = num_anchor_points * 2，每一个坐标对应两个值
        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)
    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        #[B,num_anchor_points * 2,H,W] => [B,H,W,num_anchor_points * 2]
        out = out.permute(0, 2, 3, 1)

        #num_anchor_points * H * W == 所有anchor_points的坐标点
        return out.contiguous().view(out.shape[0], -1, 2)

# the network frmawork of the classification branch(classification scores)
class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, num_classes=80, prior=0.01, feature_size=256):
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

        # 每一个类别对应num_anchor_points数量的参考点
        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()
    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, _ = out1.shape

        #[B,num_anchor_points * num_classes,H,W] => [B,H,W,num_anchor_points,num_classes]
        out2 = out1.view(batch_size, width, height, self.num_anchor_points, self.num_classes)

        #H * W * num_anchor_points == 图像中所有anchor_points的scores
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

# generate the reference points in grid layout
def generate_anchor_points(stride=16, row=2, line=2):
    #默认stride = 8 表示当前的图像下采样车倍数
    row_step = stride / row
    line_step = stride / line

    #np.arange(1, line + 1) - 0.5表示将点定义在一个网格中心
    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points

# print(np.arange(1, 3 + 1) - 0.5)
# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    """
        shape:特征图的大小
        stride:默认值为8，当前特征图下采样的步长
        anchor_points = generate_anchor_points()
    """
    #根据给定特征图大小生成一个网格，乘以stride还原为相对原图大小
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    #每一个网格对应的anchor数量
    A = anchor_points.shape[0]
    #根据当前特征图大小，划分原图，并得到划分一张图像对应网格的数量
    K = shifts.shape[0]
    #将生成的anchor放置到划分的对应的特征图上(和目标检测中的一样，将定义好的anchor放置在划分的特征图上)
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    #[A]
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points


# this class generate all reference points on all pyramid levels
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3, device = "cpu"):
        super(AnchorPoints, self).__init__()

        #使用金字塔结构，也就是使用了多尺度的方法，对于图像中的不同尺度进行预测，默认pyramid_levels = [3, ]
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        # strides [8,]只进行8倍的下采样
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.row = row
        self.line = line
        self.device = device

    def forward(self, image):
        #得到图像的height 和 width
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        #对图像进行下采样，对应于pyramid_levels各层的特征图大小
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        #创建一个[num_points,2]表示点的坐标
        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            #根据给定的尺度p,划分的网格大小row和line，生成相应的anchors，默认只有p = 3
            anchor_points = generate_anchor_points(2**p, row=self.row, line=self.line)
            #生成一个具有anchor points的特征图
            shifted_anchor_points = shift(image_shapes[idx], self.strides[idx], anchor_points)
            #添加生成的anchor points feature
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

        #增加维度
        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        if self.device == torch.device("cuda") or self.device == torch.device("cuda:0"):
            # print("cuda is avaiable...")
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))

#根据VGG16输出不同层的结果进行融合
class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(Decoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
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

# the defenition of the P2PNet model
class P2PNet(nn.Module):
    def __init__(self, backbone, row=2, line=2, device = "cpu"):
        super().__init__()
        self.backbone = backbone
        self.device = device
        # 背景和人群
        self.num_classes = 2
        # the number of all anchor points，根据划分网格的大小得到anchor points的数量
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=256,
                                          num_anchor_points=num_anchor_points)

        self.classification = ClassificationModel(num_features_in=256, \
                                            num_classes=self.num_classes, \
                                            num_anchor_points=num_anchor_points)

        # 生成anchor points
        self.anchor_points = AnchorPoints(pyramid_levels=[3,],
                                          row=row, line=line,
                                          device=self.device)

        self.fpn = Decoder(256, 512, 512)

    # def forward(self, samples: NestedTensor):
    def forward(self, samples):
        # get the backbone features
        features = self.backbone(samples)
        # forward the feature pyramid
        features_fpn = self.fpn([features[1], features[2], features[3]])

        batch_size = features[0].shape[0]
        # run the regression and classification branch
        regression = self.regression(features_fpn[1]) * 100 # 8x
        classification = self.classification(features_fpn[1])
        #根据anchor_points，将定义好的anchor_points用于预测的偏移
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        # decode the points as prediction，将特征图上生成的anchor points用于回归偏移
        output_coord = regression + anchor_points
        output_class = classification
        out = {'pred_logits': output_class, 'pred_points': output_coord}
       
        return out

class SetCriterion_Crowd(nn.Module):

    """
    weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    matcher表示实例化的匈牙利匹配算法
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        #实例化的匈牙利匹配算法
        self.matcher = matcher
        #定义的损失权重系数
        self.weight_dict = weight_dict
        #应用于无对象类别的相对分类权重
        self.eos_coef = eos_coef
        #losses = ['labels', 'points']
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        #第一个类别为背景，所以使用eos_coef初始化一个值
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        #预测得到的classification scores
        src_logits = outputs['pred_logits']

        #batch_idx, src_idx
        #得到outputs在target中匹配的索引
        idx = self._get_src_permutation_idx(indices)
        #根据匹配的索引，得到labels中在outputs中匹配的类别
        target_classes_o = torch.cat(
            [
                t["labels"][J] for t, (_, J) in zip(targets, indices)
            ]
        )
        #使用0来填充大小为size的矩阵
        target_classes = torch.full(size=src_logits.shape[:2], fill_value=0,
                                    dtype=torch.int64, device=src_logits.device)
        #将outputs中和target匹配的索引设置为target_classes_o类别
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_points(self, outputs, targets, indices, num_points):

        assert 'pred_points' in outputs
        #得到pred_points和target匹配对应的索引坐标
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        #得到target和pred_points中points匹配的索引对应坐标
        target_points = torch.cat(
            [
                t['point'][i] for t, (_, i) in zip(targets, indices)
            ],
            dim=0
        )

        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')

        losses = {}
        losses['loss_point'] = loss_bbox.sum() / num_points

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        #full_like：返回一个形状与input相同且值全为fill_value的张量
        #其中i表示遍历batch的索引
        batch_idx = torch.cat(
            [
                torch.full_like(input=src, fill_value=i) for i, (src, _) in enumerate(indices)
            ]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [
                torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
            ]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points']}

        #根据匈牙利匹配算法得到预测points和真实points之间的最佳匹配索引
        indices1 = self.matcher(output1, targets)

        #遍历batch得到一个batch中所有图像的points数量
        num_points = sum(len(t["labels"]) for t in targets)
        #转换为tensor
        num_points = torch.as_tensor([num_points], dtype=torch.float,
                                     device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        # 根据自己的机器节点数量划分boxes进行训练
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, output1, targets, indices1, num_boxes)
            )

        return losses

# create the P2PNet model
def build(args, training, device):
    # treats persons as a single class
    num_classes = 1

    #加载模型
    backbone = build_backbone(args)
    model = P2PNet(backbone, args.row, args.line, device=device)
    if not training: 
        return model

    #得到给定的损失函数的系数
    weight_dict = {'loss_ce': 1, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    #匹配算法
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses
    )

    return model, criterion

if __name__ == '__main__':
    anchor_points = generate_anchor_points(stride=8, row=2, line=2)
    shift(shape=(32, 32), stride=8, anchor_points=anchor_points)
    pass