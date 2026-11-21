# -*- coding: utf-8 -*-
"""P2PNet 自定义骨干示例：用 ResNet-18 替换默认的 VGG16_bn。

此文件演示如何为 P2PNet 提供自定义骨干网络。
P2PNet 的训练管线要求骨干输出 4 级特征（[C2, C3, C4, C5]），
分别对应 1/4、1/8、1/16、1/32 分辨率。

注意：
- 此文件名为 example_*.py，扫描器会跳过，需复制并改名才会注册
- P2PNet 的骨干接口是返回 4 级特征的 nn.Module，非标准分类骨干
- 注册后界面点预测路线的模型下拉框中可选用

用法：
  1. 复制此文件为 nets/custom_models/my_p2pnet_backbone.py
  2. 修改模型结构
  3. 重启服务或点击「刷新模型列表」，新模型出现在下拉框
"""
import torch
import torch.nn as nn

from app.core.registry import register_custom_model


class P2PNetResNetBackbone(nn.Module):
    """ResNet-18 骨干，输出 P2PNet 所需的 4 级特征。

    P2PNet 的 Decoder(C3_size, C4_size, C5_size) 接收
    [features[1], features[2], features[3]] 作为 [C3, C4, C5]，
    对应 1/8、1/16、1/32 分辨率。

    此骨干返回 [x1, x2, x3, x4]（兼容 vendored 代码的索引方式）。
    """
    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision.models as tvm
        # 加载 torchvision ResNet-18（可换为 resnet34/50 等）
        try:
            resnet = tvm.resnet18(
                weights="DEFAULT" if pretrained else None)
        except (TypeError, AttributeError):
            resnet = tvm.resnet18(pretrained=pretrained)

        # 提取各层
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)   # 1/4,  C=64
        x2 = self.layer2(x1)   # 1/8,  C=128
        x3 = self.layer3(x2)   # 1/16, C=256
        x4 = self.layer4(x3)   # 1/32, C=512
        return [x1, x2, x3, x4]


@register_custom_model("p2pnet_resnet18",
                       desc="P2PNet + ResNet18 骨干示例（P2PNet 输出契约）",
                       tasks=["crowd_point"])
def build_p2pnet_resnet18(num_classes=1, pretrained=False, **kwargs):
    """构建使用 ResNet-18 骨干的 P2PNet。

    P2PNet 模型本体来自 nets/p2pnet/（vendored），
    此处只替换骨干，Decoder + RegressionModel + ClassificationModel 不变。
    """
    # 此示例返回完整的 P2PNet（自定义骨干 + FPN + 检测头）。
    # 注册 tasks=["crowd_point"] 后，训练会直接使用该模型，并复用
    # 内置匈牙利匹配损失（forward 输出需含 pred_logits / pred_points）。
    backbone = P2PNetResNetBackbone(pretrained=pretrained)

    # 构建 P2PNet 本体（FPN 解码器通道数须匹配骨干输出）
    from nets.p2pnet.models.p2pnet import P2PNet, Decoder
    from types import SimpleNamespace

    # ResNet-18 各级通道数
    C3_size = 128   # 1/8
    C4_size = 256   # 1/16
    C5_size = 512   # 1/32

    # P2PNet 的 Decoder 接收 [C3, C4, C5] 通道数
    decoder = Decoder(C3_size, C4_size, C5_size, feature_size=256)

    # P2PNet 本体（不含骨干）
    model = P2PNet(backbone=backbone, row=2, line=2,
                   device=torch.device("cpu"))
    # P2PNet.__init__ 中会调用 self.backbone(samples) 获取 features，
    # 我们的 ResNet 骨干返回 4 级特征，与 vendored Backbone 的接口一致。

    return model
