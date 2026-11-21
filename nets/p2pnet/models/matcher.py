
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher_Crowd(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """Creates the matcher

        Params:
        为目标的相关权重
            cost_class: This is the relative weight of the foreground object
        这是匹配代价中点坐标L1误差的相对权重
            cost_point: This is the relative weight of the L1 error of the points coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "points": Tensor of dim [batch_size, num_queries, 2] with the predicted point coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_points] (where num_target_points is the number of ground-truth
                           objects in the target) containing the class labels
                 "points": Tensor of dim [num_target_points, 2] containing the target point coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                Index_i是所选预测的指数
                - index_i is the indices of the selected predictions (in order)
                Index_j是所选目标对应的索引
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """
        #得到Batch_size 和num_anchor_points * H * W数量
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        #[B,num_anchor_points * H * W,num_classes] => [B * num_anchor_points * H * W,num_classes] => 对类别进行归一化处理
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        #[B,num_anchor_points * H * W,2] => [B * num_anchor_points * H * W,2]
        out_points = outputs["pred_points"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and points
        # tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_ids = torch.cat([v["labels"] for v in targets]) #遍历batch，对应每一个坐标点point的索引id
        tgt_points = torch.cat([v["point"] for v in targets]) #遍历batch，对应的目标坐标points

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids] #根据真实的label，得到预测结果中的坐标点对应概率

        # Compute the L2 cost between point
        #批量计算两个向量集合的距离。功能上等同于 scipy.spatial.distance.cdist(input,’minkowski’, p=p)
        #p = 2表示使用欧式距离计算
        cost_point = torch.cdist(out_points, tgt_points, p=2)

        # Compute the giou cost between point
        # Final cost matrix
        #乘以相关系数
        """
        1.If they have the same confidence score, the one closer topi should be matched as positive and encouraged to achieve
          higher localization accuracy.
        2.On the contrary, if the two proposalsshare the same distance from pithe one with higher confidence
          should be trained to be closer to pi with a muchhigher confidence.
        """
        #也就是之所以加上class_score，主要看上面给出的1,2
        C = self.cost_point * cost_point + self.cost_class * cost_class
        #[B,num_anchor_points * H * W,-1]
        C = C.view(bs, num_queries, -1).cpu()

        #得到一个batch中所有points的数量
        sizes = [len(v["point"]) for v in targets]
        #根据计算的集合距离，按照线性匈牙利算法linear_sum_assignment进行匹配
        #得到预测结果points和真实值points最佳的匹配索引
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        #返回匹配的索引值
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

#匈牙利匹配算法
def build_matcher_crowd(args):
    return HungarianMatcher_Crowd(cost_class=args.set_cost_class, cost_point=args.set_cost_point)
