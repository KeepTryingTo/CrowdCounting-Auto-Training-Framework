# -*- coding: utf-8 -*-
"""自定义头部框（坐标框）模型示例（ultralytics 源）。

复制为 my_box_model.py 并修改返回的源路径，即可出现在「坐标框」路线的
模型下拉框中（example_*.py 会被扫描器跳过，改名后才注册）。

约定：
- factory() 返回 ultralytics 可加载的源：
    * .yaml 架构文件路径（从头训练）
    * .pt 权重文件路径（微调/继续训练）
    * ultralytics 内置模型名（如 "yolov8n.yaml"）
- 训练时若在界面选择了「初始权重」，会优先使用该权重；
- 损失由 ultralytics 引擎托管（box/cls/dfl），暂不支持自定义损失。
"""
from app.core.registry import register_custom_model


@register_custom_model("my_box_model",
                       desc="自定义头部框模型（返回 YAML/PT 源）",
                       tasks=["crowd_box"])
def build(**kwargs):
    # 替换为你的 .yaml / .pt 绝对路径；也可是 ultralytics 内置模型名
    return "yolov8n.yaml"
