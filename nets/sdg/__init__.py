# -*- coding: utf-8 -*-
"""单域泛化（Single-Domain Generalization）人群计数模型与损失（vendor 移植）

来源：
- DCCUS：AAAI 2023《Domain-general Crowd Counting in Unseen Scenarios》
  （元学习 + 语义/风格 memory，vendored 自 DCCUS/networks/MetaMemNet.py）
- LLB_Count：DCCUS 的扩展（LocalPerceptionLoss + BoundAwareLoss，
  移植自 LLB_Count/losses/）
- MPCount：CVPR 2024《Single Domain Generalization for Crowd Counting》
  （自适应 memory bank + patch 分类 + 双分支一致性，
  移植自 MPCount-main/models/models.py）
"""

from .mpcount import DGModelFinal, build_mpcount          # noqa: F401
from .metamem import MetaMemNet, build_metamemnet        # noqa: F401
from .llb_losses import LocalPerceptionLoss, BoundAwareLoss  # noqa: F401
