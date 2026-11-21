# -*- coding: utf-8 -*-
"""P2PNet vendor（点预测人群计数参考实现）

vendor 自 CrowdCounting-P2PNet-main（P2PNet: Rethinking Spatial Invariance
of Convolutional Networks for Object Counting, CVPR 2022）：
- models/：P2PNet 模型（VGG16_bn 骨干 + FPN + 点回归/分类头）、匈牙利匹配器
- util/misc.py：collate / NestedTensor 等工具

vendored 改动：
- 包内绝对导入改为相对导入
- VGG 预训练权重加载改为「本地 weights/pretrained → torch hub URL」回退
  （原实现硬编码 Linux 绝对路径）
- 内置 SHTechA.pth（ShanghaiTech Part A 训练产物）位于
  weights/pretrained/p2pnet/，可直接推理
"""
