# Hungarian matcher, adapted from P2PNet (DETR).
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher_Crowd(nn.Module):
    """1-to-1 matching between GT points and proposals.

    Cost = cost_point * L2(pred, gt) + cost_class * (-prob[target class])
    """

    def __init__(self, cost_class: float = 1, cost_point: float = 0.05):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [bs*M, 2]
        out_points = outputs["pred_points"].flatten(0, 1)            # [bs*M, 2]

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_points = torch.cat([v["point"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_point = torch.cdist(out_points, tgt_points, p=2)

        C = self.cost_point * cost_point + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["point"]) for v in targets]
        # guard: scipy handles empty cost matrices poorly in some versions
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            if c.shape[1] == 0:
                indices.append((torch.empty(0, dtype=torch.int64),
                                torch.empty(0, dtype=torch.int64)))
            else:
                row, col = linear_sum_assignment(c[i].numpy())
                indices.append((torch.as_tensor(row, dtype=torch.int64),
                                torch.as_tensor(col, dtype=torch.int64)))
        return indices


def build_matcher_crowd(args):
    return HungarianMatcher_Crowd(cost_class=args.set_cost_class,
                                  cost_point=args.set_cost_point)
