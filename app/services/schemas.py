# -*- coding: utf-8 -*-
"""API 请求/响应模型"""
from typing import Optional
from pydantic import BaseModel, Field


class TrainParams(BaseModel):
    """训练请求参数"""
    task: str = Field(..., description="子任务: crowd_density（密度图）/ "
                       "crowd_point（点预测）/ crowd_box（头部框，YOLO） / "
                       "crowd_counting（旧版兼容）")
    model: str = Field(..., description="模型名")
    dataset_path: str = Field(..., description="数据集目录（服务器路径或已上传数据集名）")
    epochs: int = Field(10, ge=1, le=1000)
    batch_size: int = Field(2, ge=1)
    lr: float = Field(0, ge=0, description="学习率；0=按所选路线/算法的论文默认"
                                          "（密度图 1e-5 / P2PNet 1e-4 / "
                                          "P2R 5e-5 / CP、CU、PET、CLTR 1e-4 / "
                                          "YOLO 1e-3）")
    weight_decay: float = Field(1e-4, ge=0)
    momentum: float = Field(0.9, ge=0, le=1)
    img_size: int = Field(0, description="训练输入/裁剪尺寸；0=按路线默认"
                                         "（密度图 384 / P2R 256 / CP、CU 256 / "
                                         "PET、CLTR 256 / P2PNet 1024 / YOLO 640；"
                                         "界面可选范围 256~2048）")
    optimizer: str = Field("", description="优化器；空=按算法论文默认"
                                          "（P2PNet/P2R/CP/CU/CLTR 为 Adam，"
                                          "PET 为 AdamW，密度图为 Adam）")
    scheduler: str = Field("", description="学习率调度器；空=默认 cosine（余弦退火），"
                                          "另可选 step/multistep/poly/warmup 等 9 种")
    loss: str = Field("mse", description="损失函数；点预测默认对应算法的"
                                        "论文损失（p2pnet/p2r/cp/cu/pet/cltr），"
                                        "可选 custom:* 自定义附加损失")
    loss_weights: Optional[dict] = Field(
        None, description="损失分项权重覆盖（键与缺省值随算法而定，缺省=论文设置；"
                          "例：YOLO box_w/cls_w/dfl_w、P2PNet ce_w/points_w、"
                          "DCCUS den_w/mask_w/sim_w/orth_w、自定义附加损失 aux_w）")
    pretrained: bool = Field(True, description="使用预训练权重（骨干 ImageNet）")
    weight_path: Optional[str] = Field(None, description="自定义初始权重路径（上传的权重）")
    resume_weight: Optional[str] = Field(
        None, description="断点续训权重路径（中断的 last.pth：恢复轮次/优化器/"
                          "最优指标继续训练，而非从第 1 轮微调）")
    device: str = Field("cpu", description="cpu / cuda / cuda:0 ...")
    num_workers: int = Field(0, ge=0)
    run_name: Optional[str] = Field(None, description="本次训练名称（用于目录名）")
    # 人群计数专用
    sigma: float = Field(8.0, gt=0)
    adaptive_sigma: bool = Field(False)
    val_ratio: float = Field(0.2, gt=0, lt=1)
    # P2R 半监督专用：有标注样本比例（1.0=全监督；<1 时仅该比例样本带
    # 标注，其余作为无标注数据参与伪标签训练，模拟论文 protocol 划分）
    label_ratio: float = Field(1.0, gt=0, le=1.0,
                               description="P2R 半监督标注比例（1=全监督；"
                                           "如 0.05 表示仅 5% 图像带标注）")
    # 单域泛化路线专用（crowd_sdg：DCCUS / LLB_Count / MPCount）
    # RAZ 路线专用：all / density / localization / attention / raz
    raz_stage: str = Field(
        "all", description="RAZ 训练阶段：all / density / localization / "
                          "attention / raz")
    target_dataset_path: Optional[str] = Field(
        None, description="单域泛化目标数据集（跨域验证集，可选；仅用于"
                          "每轮评估与最优模型选择，不参与梯度更新）")
    sdg_backbone: Optional[str] = Field(
        None, description="单域泛化骨干（convNeXt_tiny / vgg；MPCount 固定 "
                          "vgg16_bn）")
    sdg_num_clusters: int = Field(4, ge=2, le=16,
                                  description="DCCUS/LLB 在线伪域聚类数")
    sdg_iters: int = Field(20, ge=1, le=5000,
                           description="DCCUS/LLB 每 epoch 元学习迭代次数")
    sdg_eval_every: int = Field(1, ge=1, le=100,
                                description="目标域验证间隔（epoch）")
    sdg_test_patch: int = Field(0, ge=0,
                                description="测试/验证切块窗口（0=整图；"
                                            "大图建议 512/640/1024）")
    # 头部框路线专用（crowd_box：标注格式 yolo/voc/csv + YOLO 引擎增强）
    annotation_format: str = Field("yolo")
    yolo_mosaic: float = Field(1.0, ge=0.0, le=1.0,
                               description="YOLO Mosaic 马赛克增强概率")
    yolo_scale: float = Field(0.5, ge=0.0, le=1.0,
                              description="YOLO Scale 随机缩放增强幅度")
    yolo_mixup: float = Field(0.0, ge=0.0, le=1.0,
                              description="YOLO Mixup 混合增强概率")
    yolo_copy_paste: float = Field(0.1, ge=0.0, le=1.0,
                                    description="YOLO Copy-Paste 增强")
    # 头部框路线 · 知识蒸馏（ultralytics 引擎原生 DistillationModel：
    # 教师-学生 neck 特征投影对齐 + score 加权 L2 蒸馏损失，dis 为其权重）
    distill: bool = Field(False,
                          description="启用知识蒸馏（仅 crowd_box 头部框路线："
                                      "以大模型教师的中间特征为软目标训练"
                                      "参数量更少的学生模型）")
    distill_teacher_model: Optional[str] = Field(
        None, description="教师模型名（官方 COCO 预训练，自动下载；"
                          "与教师权重文件同时给出时后者优先）")
    distill_teacher_weight: Optional[str] = Field(
        None, description="教师权重文件路径（训练产物 best.pt / 上传 .pt；"
                          "推荐同数据集训练产物，类别一致蒸馏效果最好）")
    distill_lambda: float = Field(6.0, ge=0.0, le=100.0,
                                  description="蒸馏损失权重（ultralytics dis，"
                                              "引擎默认 6.0）")
    # 通用超参数增强
    early_stopping: int = Field(0, ge=0,
                                description="早停耐心轮数，0=不启用")
    eval_every: int = Field(1, ge=1, le=1000,
                            description="测试集评估间隔：每 N 个 epoch 在"
                                        "验证/测试集上评估一次（1=每轮；"
                                        "训练最后一轮总会评估）")
    # 多 GPU 训练模式：dp（DataParallel 单进程多卡，默认）/ ddp（分布式，每卡一进程）
    multi_gpu_mode: Optional[str] = Field(
        None, description="多 GPU 模式：dp=DataParallel（默认） / "
                          "ddp=DistributedDataParallel（真分布式，mp.spawn "
                          "每卡一进程 + DistributedSampler 数据分片；早停自动"
                          "禁用以保证各 rank 一致）")
    ddp_world_size: Optional[int] = Field(
        None, ge=2, description="DDP 世界大小（进程数），缺省=GPU 数")
    augment: bool = Field(True, description="数据增强开关")
    augment_strength: float = Field(1.0, ge=0.0, le=2.0,
                                    description="数据增强强度（0=关闭等效）")
    # 调度器专用（warmup_cosine 预热轮数 / warm_restarts 周期 / poly 幂）
    warmup_epochs: int = Field(5, ge=1, description="warmup_cosine 预热轮数")
    t0_epochs: int = Field(10, ge=1, description="warm_restarts 初始周期")
    poly_power: float = Field(0.9, gt=0, le=2.0, description="poly 衰减幂")
    seed: int = Field(42, description="随机种子")
    # 批量推理
    save_dir: Optional[str] = Field(None, description="推理结果保存目录")


class StopResponse(BaseModel):
    ok: bool
    message: str


class OnnxExportParams(BaseModel):
    """ONNX 部署导出请求参数（部署模式：权重 → ONNX + 部署片段代码）"""
    weight_path: str = Field(..., description="权重文件路径（训练产物 / "
                              "上传 .pth/.pt；模型类型由格式推断模块自动识别）")
    run_name: Optional[str] = Field(None, description="导出名称（留空自动生成）")
    imgsz: int = Field(0, ge=0, le=4096,
                       description="输入边长（0=按模型类型默认；点预测自动对齐"
                                   "16/32 倍数）")
    opset: int = Field(13, ge=9, le=19, description="ONNX 算子集版本")
    dynamic_batch: bool = Field(False, description="动态 batch 维（YOLO 为引擎 "
                                "dynamic，同时放开空间维）")
    simplify: bool = Field(True, description="onnxslim 图简化")
    half: bool = Field(False, description="FP16（仅 YOLO 且 CUDA 导出有效，"
                                      "CPU 自动回退 fp32）")
    nms: bool = Field(False, description="YOLO 专用：NMS 内嵌进 ONNX 图"
                                  "（端到端输出 [max_det,6]，片段代码免后处理）")
    conf_threshold: float = Field(0.5, ge=0.0, le=1.0,
                                  description="片段代码默认置信度阈值")
