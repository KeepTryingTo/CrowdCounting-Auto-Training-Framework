# -*- coding: utf-8 -*-
"""
自定义模型放这里，启动时自动扫描注册（前端按路线分区显示）。

=== 注册方式 ===

    from app.core.registry import register_custom_model

    @register_custom_model("my_model", desc="说明", tasks=["crowd_density"])
    def build(...):
        return MyModel()

也可以使用固定函数名 build_model(...)：文件名即注册名，tasks 默认为
crowd_density。

tasks 可选值（可多选，缺省 = crowd_density）：
    crowd_density   密度图
    crowd_point     点预测
    crowd_box       坐标框（头部框）

=== 三条路线的输出契约 ===

一、密度图（crowd_density）—— 自定义模型 + 损失 + 优化器
    模型: factory(num_classes=1, pretrained=False) -> nn.Module
          forward(x) -> [N,1,H',W'] 密度图（sum≈人数；
          与 GT 尺寸不一致时训练管线自动双线性对齐）
    损失: forward(pred[N,1,h,w], gt[N,1,h,w]) -> 标量
          （tasks=["crowd_density"]）

二、点预测（crowd_point）—— 自定义模型（P2PNet 接口）+ 附加损失
    模型: factory(num_classes=1, pretrained=False) -> nn.Module
          forward(x) -> {"pred_logits": [B,M,2],
                         "pred_points": [B,M,2]}
          内置匈牙利匹配损失直接复用；骨干/FPN/检测头可参考
          example_p2pnet_backbone.py
    损失: forward(outputs, targets) -> 标量
          targets 为 list[{point:[N,2], labels:[N]}]；
          作为【附加项】叠加到内置匹配损失上
          （tasks=["crowd_point"]）

三、头部框（crowd_box）—— 自定义 ultralytics 模型源
    factory() 返回 ultralytics 可加载的 .yaml / .pt 路径或模型名；
    前端选择后由 ultralytics 引擎训练/推理。
    损失由 ultralytics 引擎托管（box/cls/dfl），不支持自定义损失。
    示例见 example_box_model.py

=== 其它 ===

优化器：nets/custom_optimizers.py（三条路线通用）
损失：nets/custom_losses.py 或 nets/custom_losses/*.py
注意：example_*.py 会被扫描器跳过，复制并改名后才会注册。
"""
