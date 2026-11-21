# Pairwise ranking loss for the uncertainty branch (CU paper Eq. 2).
#
#   L_uncer = sum_{a_i > a_j} |a_i - a_j| * log2(1 + exp(-(k_i - k_j)))
#
# a_i : batch min-max normalized ASM (surrogate "pseudo-ground-truth" of
#       uncertainty); k_i : uncertainty-branch output (sigmoid).
# Only the ORDER is supervised (ASM absolute values drift during training),
# each pair weighted by the surrogate gap |a_i - a_j|. We normalize the sum
# by the number of valid pairs for a stable gradient scale.
import torch


def ranking_loss(asm, kappa):
    """LambdaLoss-style pairwise ranking loss.

    Args:
        asm:    torch.Tensor [N] surrogate ASM values (un-normalized);
                min-max normalization is applied inside, as in the paper.
        kappa:  torch.Tensor [N] predicted uncertainties in (0, 1)
                (sigmoid outputs of the uncertainty branch).
    Returns:
        scalar loss (0 when fewer than 2 patches / degenerate ASM range).
    """
    if asm.numel() < 2:
        return kappa.new_zeros(())
    a_min, a_max = asm.min(), asm.max()
    a_range = a_max - a_min
    if a_range.item() < 1e-8:
        return kappa.new_zeros(())
    a = (asm - a_min) / a_range  # batch min-max normalization

    ai = a.unsqueeze(1)          # [N, 1]
    aj = a.unsqueeze(0)          # [1, N]
    ki = kappa.unsqueeze(1)
    kj = kappa.unsqueeze(0)

    mask = ai > aj                          # strictly ordered surrogate pairs
    if not mask.any():
        return kappa.new_zeros(())
    w = (ai - aj)[mask]                      # |a_i - a_j| weighting
    term = torch.log2(1 + torch.exp(-(ki - kj)[mask]))  # log2(1 + e^{-(k_i-k_j)})

    loss = (w * term).sum() / mask.sum().clamp(min=1).to(w.dtype)
    return loss
