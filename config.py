# -*- coding: utf-8 -*-
"""全局配置：路径、常量"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 数据目录
WEIGHTS_DIR = BASE_DIR / "weights" / "pretrained"       # 内置预训练权重
UPLOAD_MODEL_DIR = BASE_DIR / "uploads" / "models"      # 用户上传的权重
UPLOAD_DATASET_DIR = BASE_DIR / "uploads" / "datasets"  # 用户上传的数据集
RUNS_DIR = BASE_DIR / "runs"                            # 训练产物（checkpoint/日志/曲线）
LOGS_DIR = BASE_DIR / "logs"                            # 运行日志
RESULTS_DIR = BASE_DIR / "results"                      # 推理结果
STATIC_DIR = BASE_DIR / "static"                        # 前端静态文件
CUSTOM_MODELS_DIR = BASE_DIR / "nets" / "custom_models"  # 自定义密度网

for _d in (WEIGHTS_DIR, UPLOAD_MODEL_DIR, UPLOAD_DATASET_DIR, RUNS_DIR,
           LOGS_DIR, RESULTS_DIR, CUSTOM_MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 允许上传的权重后缀（.onnx 供推理面板 ONNX Runtime 验证部署产物 /
# 外部模型推理，训练与导出仍只认 PyTorch 检查点）
WEIGHT_EXTS = {".pth", ".pt", ".pth.tar", ".onnx"}

# 允许上传的图片后缀
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
