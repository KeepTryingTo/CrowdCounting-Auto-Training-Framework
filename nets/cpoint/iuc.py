# IUC (Instance-wise Uncertainty Calibration), paper Sec.3.3.
# The teacher score of a pseudo-point doubles as its uncertainty: points near
# the 0.5 boundary barely supervise the student, confident ones fully do.
# Only the positive term of the unlabeled cls loss gets weighted.
import torch


def iuc_weights(scores):
    """w = (c - 0.5) / 0.5, clipped to [0, 1]. scores > 0.5 by construction."""
    return ((scores - 0.5) / 0.5).clamp_(0.0, 1.0)
