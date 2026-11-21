# teacher = EMA of the student
import copy

import torch
import torch.nn as nn


class ModelEMA:
    """Keeps an eval-mode EMA copy of the student (the teacher model)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for name, tparam in self.module.state_dict().items():
            sparam = msd[name].detach()
            # EMA the floats (params + BN running stats), copy the rest
            if tparam.dtype.is_floating_point:
                tparam.mul_(self.decay).add_(sparam, alpha=1.0 - self.decay)
            else:
                tparam.copy_(sparam)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)
