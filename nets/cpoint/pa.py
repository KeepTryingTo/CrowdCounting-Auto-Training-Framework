# PA (Position Aggregation), paper Sec.3.2. K=4 is the sweet spot (Tab.6).
#
# Under P2PNet's 1-1 matching the neighbours of a positive proposal get
# suppressed as negatives, yet their regression output stays close to the
# positive's own prediction -- averaging the K nearest (by anchor distance)
# shrinks the position variance to 1/K.
import torch


@torch.no_grad()
def position_aggregation(pred_points, scores, anchor_points,
                         threshold=0.5, K=4, img_size=None):
    """Aggregate teacher pseudo-points.

    pred_points/scores/anchor_points: teacher outputs, [B, M, 2] / [B, M] / [B, M, 2].
    Returns one {'points': [P, 2], 'scores': [P]} per image.
    """
    B, M, _ = pred_points.shape
    results = []
    for b in range(B):
        pos_idx = (scores[b] > threshold).nonzero(as_tuple=True)[0]
        P = pos_idx.numel()
        if P == 0:
            results.append({'points': pred_points.new_zeros((0, 2)),
                            'scores': scores.new_zeros((0,))})
            continue

        d = torch.cdist(anchor_points[b, pos_idx], anchor_points[b])  # [P, M]
        d[torch.arange(P, device=d.device), pos_idx] = float('inf')  # skip self

        k = min(K, M - 1)
        if k <= 0:
            pts = pred_points[b, pos_idx]
        else:
            knn = d.topk(k, largest=False).indices
            pts = pred_points[b][knn].mean(dim=1)

        if img_size is not None:
            h, w = int(img_size[0]), int(img_size[1])
            pts = torch.stack([pts[:, 0].clamp(0, w - 1),
                               pts[:, 1].clamp(0, h - 1)], dim=1)

        results.append({'points': pts, 'scores': scores[b, pos_idx]})
    return results
