# 人群计数自动训练和推理框架（CrowdCounting Auto Training Framework）

[中文](README.md) | [English](README_EN.md)

> FastAPI + PyTorch 的人群计数领域自动训练与推理平台，界面搬移自「深度学习自动训练和推理框架」，
> 专注人群计数单一领域（移除 MindSpore / PaddlePaddle 与其他领域部分）。
>
> 三条技术路线：
>
> - **基于密度图**（Density Map-based Crowd Counting Algorithm）
> - **基于点预测**（Point-based Crowd Counting Algorithm：P2PNet / P2RLoss / Consistent-Point / CU Uncertainty / PET / CLTR）
> - **基于坐标框**（头部定位，ultralytics YOLO 系列）

## 作者主页

| 平台      | 链接                                                                                                                                    |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| CSDN 博客 | [https://blog.csdn.net/Keep_Trying_Go](https://blog.csdn.net/Keep_Trying_Go)                                                             |
| 哔哩哔哩  | [https://space.bilibili.com/625095571](https://space.bilibili.com/625095571)                                                             |
| GitHub    | [https://github.com/KeepTryingTo](https://github.com/KeepTryingTo)                                                                       |
| 抖音      | [https://www.douyin.com/user/self?from_tab_name=main&amp;showTab=post](https://www.douyin.com/user/self?from_tab_name=main&showTab=post) |

## 功能总览

[人群计数自动训练框架演示](./video/人群计数自动训练框架界面功能演示.mp4)

<video width="320" height="240" controls>
  <source src="./video/人群计数自动训练框架界面功能演示.mp4" type="video/mp4">
  你的浏览器不支持视频标签。
</video>

| 能力                | 训练 | 推理                                | 内置模型                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------- | ---- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 密度图估计          | ✅   | ✅ 人数估计 + 密度热力图/叠加图     | **GhostNet v1/v2** 密度网（本地离线权重）、**torchvision 全部常用分类骨干**（resnet/vgg/densenet/mobilenet/efficientnet/convnext/shufflenet）、**参考项目密度网 crowdvgg / crowdresnet**（vendor 于 `nets/crowd/`）、自定义密度网（`nets/custom_models/`）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 点预测（P2PNet 系） | ✅   | ✅ 人数 + 预测点可视化（红色圆点）  | **P2PNet**（VGG16_bn + FPN + 点回归/分类 + 匈牙利匹配，vendor 于 `nets/p2pnet/`；内置 SHTechA 预训练检查点开箱即用）、**P2RLoss**（CVPR 2025 半监督：教师-学生 EMA + Point-to-Region 损失，`nets/p2r/`）、**Consistent-Point**（arXiv 2503.12441 半监督：PA 位置聚合 + IUC 不确定性校准 + Mean-Teacher，`nets/cpoint/`）、**CU**（ICCV 2023 半监督：patch 级不确定性分支 + PatchBank ASM + 排序损失 + 课程伪标签过滤，`nets/cu/`）、**PET**（ICCV 2023 点查询四叉树：稀疏点查询按需分裂；模型内计算点查询损失 + 四叉树划分损失，`nets/pet/`；内置 SHA 预训练权重可离线推理）、**CLTR**（ECCV 2022 人群定位 Transformer：Conditional DETR + ResNet50 + 500 点查询，256 crop，`nets/cltr/`；无内置权重，需训练/上传）——半监督模型设「有标注样本比例」即可少量标注训练 |
| 头部定位框（YOLO）  | ✅   | ✅ 人头数 + 检测框可视化 + 各类计数 | **ultralytics 官方全系列**——YOLOv3u / v5nu-v5xu / v6（yaml）/ v8 / v9 / v10 / 11 / 12 / 26 / **RT-DETR** / **YOLOv13**（官方 Release 权重自动下载）/ yolo_custom（任意 .pt 微调）——微调时自动替换分类头，训练产物 `.pt` 推理                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

**通用训练能力**

- 训练/验证曲线 PNG + 界面**逐 batch 实时损失曲线**与**逐 epoch 验证指标曲线**、历史任务曲线回看
- 训练参数随算法自适应：损失函数 / 输入分辨率按密度图·点预测·头部框分别给默认值与可选值；权重以模型名前缀命名（如 `mobilenet_v2_best.pth`）
- 早停（early stopping）、随机种子、完整 checkpoint（权重 + optimizer/scheduler state + epoch + 最优指标）
- **断点续训**（已暂停/失败任务一键继续；上传中断的 `<模型名>_last.pth` 或运行目录 zip 恢复）
- 多 GPU：DataParallel（默认）/ **DDP 分布式**（mp.spawn 每卡一进程 + DistributedSampler，GPU→NCCL、无 GPU→gloo）
- 训练产物经 `/runs` 下载；任务历史表可下载日志 / 最优权重 / 最末权重 / params / metrics
- 数据集与权重**上传进度条**、**批量推理**（多图 → zip 打包下载）、**摄像头实时计数**（MJPEG 推流 + 左上角 FPS/人数；需 GUI 版 OpenCV，headless 会提示安装）
- **数据集结构浏览**（上传/下载模块内，逐级展开查看真实目录与文件列表）
- 中英文双语界面（右上角切换，localStorage 记忆）
- **系统状态页**：CPU/内存/GPU/磁盘/网络实时监控 + 后台日志实时查看 + 网络代理设置（proxy.json 持久化）
- **ONNX 部署**：导出一键校验 + 自动生成部署片段（单张图像 / 摄像头实时推理，`python snippet.py camera [编号]`）

**人群计数损失函数**（`app/services/crowd_losses.py`，移植自 CrowdCounting-framework-PyTorch/losses，纯 PyTorch 零依赖）：MSE、MAE、**SSIM**、**MSE+SSIM 组合**、**DM-Count**（NeurIPS 2020：计数损失 + Sinkhorn **OT 分布匹配** + **TV 损失**，权重/正则/迭代数对齐官方实现）、**OT 最优传输**（Sinkhorn + 点标注）、**MMD**（域适应，库函数）；另支持**按路线分区**的自定义模型 / 损失注册（`nets/custom_models/`、`nets/custom_losses.py`，`tasks` 取 `crowd_density` / `crowd_point` / `crowd_box`），详见「自定义模型 / 损失」小节。

## 快速开始

### 环境要求

- Python 3.7+，PyTorch 1.11.0+，torchvision
- 默认 CPU，检测到 CUDA 时可选单卡 / 多卡
- 依赖安装：`pip install -r requirements.txt`
- 可选依赖（缺了只影响对应功能）：ultralytics（头部框路线 YOLO 系列）、psutil（系统监控）、pynvml（GPU 利用率/温度）、pysocks（socks5 代理测试）

### 启动

**Windows**：双击 `run.bat`（自动探测 Python 环境，与项目所在位置无关，可整体迁移），或 `run.bat --check` 仅检测环境。

**Linux / macOS**：`bash run.sh`（或 `chmod +x run.sh && ./run.sh`），`bash run.sh --check` 仅检测环境。

**手动启动**：

```bash
python main.py
# 或
uvicorn main:app --host 127.0.0.1 --port 8060
```

浏览器访问 **http://127.0.0.1:8060**（默认端口 8060，避开常见 8000 端口冲突；`python main.py --port <端口>` 可自定义）

## 数据集格式（训练）

```
dataset_root/
├── Train/
│   ├── Images/       # *.jpg / *.png
│   └── GT/           # *.npy（密度图，sum≈人数）或 *.txt（每行 "x y" 点标注；P2PNet 也支持 .mat）
└── Test/
    ├── Images/
    └── GT/
```

兼容 `imgs/ + gt/npys/` 扁平结构与大小写目录名（imgs/images、GT/gt/npys 等）；无 Test 时自动按比例划分验证集。

> **σ 一致性提示**：`.txt` 点标注在训练时按「高斯 sigma」参数现场生成密度图（默认 8），
> `.npy` 密度图的 σ 以生成时为准（`.mat→npy` 工具默认 2，与参考框架对齐）。
> **混用两种 GT 时请保持 σ 一致**（工具界面与训练卡片的 sigma 均可调），
> 否则监督目标的高斯尺度不同；训练日志会显式提示本次 GT 的构成与 σ。

### 内置数据工具（训练卡片内，后台任务 + 进度显示）

**① .mat → GT 密度图转换工具**

输入 .mat 目录 + 保存目录（可选图像目录），自动探测 **ShanghaiTech(image_info)** 与 **annPoints**（QNRF/NWPU/UCF）两种 .mat 格式：

- `image_info` 支持 MATLAB struct 嵌套结构（含 `location`/`number` 及可选 `height`/`width`，逐字段递归下钻）
- 生成与参考框架语义对齐的密度图（固定 sigma=2，可选 k 近邻自适应），**按点局部贴片叠加归一化高斯小核**（与逐点整幅滤波数值等价），大图密集标注从分钟级降到毫秒级
- 图像定位支持标签文件带任意标记前缀/后缀（GT_IMG_x.mat↔IMG_x.jpg、mycustom_IMG_x.mat↔IMG_x.jpg，token 级对齐防误配）
- 输出 .npy 以对应图像主名命名，训练配对无需手动重命名

**② .npy 密度图可视化工具（.npy → .png）**

三种方式：伪彩色热力图（jet，与推理热力图一致）/ 灰度 / **叠加融合**（移植自参考项目 vis_img_den.py 的 save_density_map：密度归一化 → 反转 → JET 着色 → (1-alpha)·img + alpha·density → 重归一化，与参考实现逐像素一致，融合比例可调）；左上角标注 count≈密度总和；已存在的 PNG 自动跳过（可增量重复执行）。

### 上传 / 下载数据集

- 同名数据集自动追加 `_1` / `_2` 后缀，不覆盖已有数据；
- 上传或下载完成后，可在「数据集结构浏览」中逐级展开查看真实目录与文件列表。

### 训练（密度图估计）

- 模型：GhostNet 系列 / torchvision 分类骨干 / **参考项目 crowdvgg / crowdresnet** / 自定义密度网
- 损失：MSE / MAE / SSIM / MSE+SSIM / **DM-Count** / custom:*（默认 MSE）；优化器 Adam/SGD/AdamW 等，调度器可选（默认 cosine 余弦退火，另支持 step / multistep / warmup 等）
  - **DM-Count（NeurIPS 2020）**：`L = L_count + 0.1·L_OT + 0.01·L_TV`——计数 L1 + Sinkhorn 熵正则最优传输的分布匹配（包络定理梯度写回）+ 归一化密度 TV；Sinkhorn reg=10 / 100 次迭代，OT 目标测度取归一化 GT 密度（.npy / .txt GT 均可用），验证阶段自动跳过 OT 项
- 数据增强：随机裁剪（分辨率可选）/ 随机翻转；点标注自动按 sigma（可配，支持 k 近邻自适应）生成密度图
- 验证指标：**MAE**（|预测计数 - 真实计数| 均值），最优保存 `<模型名>_best.pth` / `<模型名>_last.pth`；训练中逐 batch 刷新损失曲线，验证 MAE 逐 epoch 刷新

### 训练（点预测 — P2PNet / P2RLoss / Consistent-Point / CU / PET / CLTR）

- 需点标注：`Train/Images + GT/*.txt`（每行 "x y"）/ ShanghaiTech `.mat` / `.npy`（(N,2) 坐标，P2RLoss 源格式）
- **P2PNet**：Adam 分组学习率（骨干 lr_backbone=1e-5 / 其余 lr=1e-4）+ StepLR + 梯度裁剪 0.1；随机缩放增强 [0.7, 1.3]（图像与坐标同步）
- **半监督三件套（P2RLoss / Consistent-Point / CU）**：教师-学生 Mean-Teacher 架构，「有标注样本比例」<1 时仅该比例图像带标注，其余作无标注数据参与伪标签训练（种子确定性划分）；默认超参对齐各自论文设定（P2R Adam 5e-5 / CP、CU Adam 1e-4，骨干 1e-5，LR 不衰减），优化器 / 学习率 / 调度器 / 损失均可在训练界面调整
  - P2RLoss：Point-to-Region 匹配损失（伪点置信度区域化传播）
  - Consistent-Point：PA 位置聚合（K=4 邻居均值）+ IUC 不确定性加权
  - CU：patch 级不确定性分支 + PatchBank ASM 代理真值 + 成对排序损失 + 课程伪标签阈值过滤（λ₁=0.3 / patch=64 / u_max=0.5 / EMA 0.999）
- **PET（ICCV 2023）**：点查询四叉树，模型内计算“点查询损失 + 四叉树划分损失”；默认 AdamW 分组学习率 + LR 不衰减，输入 patch=256；Batch Size 建议 ≥2；内置 `weights/pretrained/pet/SHA_model.pth` 可直接推理
- **CLTR（ECCV 2022）**：Conditional DETR + ResNet50 人群定位；256×256 crop、500 queries；默认 Adam + MultiStepLR 衰减；验证/推理按 256×256 分块（score 阈值 0.35）；无内置权重，需训练或上传 .pth
- 损失：点预测下拉默认 = 该算法的论文损失（P2PNet 匈牙利匹配 / P2R / CP / CU / CLTR 点定位），优化器、学习率、调度器均可自由选择；P2PNet / P2R / CP / CU / CLTR 另可追加注册到 `crowd_point` 的自定义附加损失（PET 损失在模型内计算，不可替换；头部框由 ultralytics 托管）
- 分辨率：长边可选（原始 / 384 / 512 / 640 / 768 / 1024）；自定义点模型需实现 P2PNet 输出契约（`pred_logits` / `pred_points`），参考 `nets/custom_models/example_p2pnet_backbone.py`
- VGG16_bn 骨干预训练：`weights/pretrained/vgg16_bn.pth` → torch hub 缓存 → 在线下载
- 每轮验证输出 MAE（softmax 得分 > 0.5 计数 vs 真实点数），最优保存 `<模型名>_best.pth` / `<模型名>_last.pth`（半监督模型同时保存教师 / bank 状态，支持断点续训完整恢复）

### 自定义模型 / 损失（按路线注册）

在 `nets/custom_models/` 实现模型、在 `nets/custom_losses.py` 实现损失，用 `tasks` 标明适用路线，前端会按路线自动分区显示：

| 路线                     | 模型契约                                             | 损失                                              |
| ------------------------ | ---------------------------------------------------- | ------------------------------------------------- |
| 密度图 `crowd_density` | `forward(x) -> [N,1,H',W']`                        | `forward(pred, gt) -> 标量`                     |
| 点预测 `crowd_point`   | `forward(x) -> {"pred_logits","pred_points"}`      | `forward(outputs, targets) -> 标量`（附加损失） |
| 坐标框 `crowd_box`     | 工厂返回 ultralytics 可加载的 `.yaml` / `.pt` 源 | 引擎托管（box/cls/dfl）                           |

```python
@register_custom_model("my_density", desc="我的密度网", tasks=["crowd_density"])
def build(...): return MyNet()

@register_custom_loss("my_loss", desc="我的损失", tasks=["crowd_density"])
def factory(...): return MyLoss()
```

`example_*.py` 默认不注册，复制改名后生效；示例见 `nets/custom_models/`。

### 推理

三种权重来源：内置预训练（**P2PNet SHTechA 检查点离线可推理** / GhostNetV2 本地骨干）、训练产物权重（自动进入下拉框）、上传 .pth/.pt。

- **密度网**：返回估计人数 + 密度热力图叠加图 / 纯热力图（与原图同尺寸）
- **点预测（P2PNet / P2RLoss / Consistent-Point / CU / PET / CLTR / 自定义点模型）**：返回人数 + 预测点可视化（红色实心圆点）；P2PNet / Consistent-Point / CU 点置信度阈值可调（默认 0.5），P2RLoss 点由 logit>0 判定；半监督 / 自定义点模型选择训练产物权重推理，PET 可用内置 SHA，CLTR 需训练/上传权重
- 勾选「保存预测结果」后结果图保存到 `results/` 并提供下载链接；批量推理 zip 打包下载
- **摄像头实时计数**：推理面板「摄像头实时计数」按钮，OpenCV 采集 + 逐帧推理 + MJPEG 推流

## 项目结构

```text
Codes/
├── main.py                         # FastAPI 入口（python main.py → http://127.0.0.1:8060）
├── launcher.py                     # 交互式启动器（列出全部候选 Python 环境供选择）
├── run.bat                         # Windows 一键启动脚本（自动探测 Python 环境；--check 仅检测）
├── run.sh                          # Linux/macOS 一键启动脚本（自动探测 Python；--check 仅检测）
├── run_python.txt                  # 记录 python.exe 完整路径（供 run.bat / run.sh 优先使用）
├── config.py                       # 全局配置（数据目录路径、上传后缀白名单等）
├── requirements.txt                # Python 依赖清单
│
├── app/                            # ★ 后端应用层（FastAPI）
│   ├── core/
│   │   ├── backend_log.py          #   后台日志：stdout/stderr 分流进环形缓冲
│   │   ├── registry.py             #   模型注册表：内置工厂 + 按路线分区的自定义模型/损失
│   │   ├── dataset_naming.py       #   同名数据集唯一目录（_1/_2 后缀）
│   │   ├── model_cache.py          #   预训练权重统一缓存目录（pretrained model/）
│   │   ├── optim_builders.py        #   torch 优化器统一构建
│   │   ├── proxy.py                #   网络代理（proxy.json 持久化 + 连通性测试）
│   │   └── task_manager.py         #   训练任务管理器（后台线程/停止/续训/历史重建/优雅停机）
│   ├── routers/
│   │   ├── crowd.py                #   人群计数路由（/api/crowd：密度图/点预测/头部框
│   │   │                           #   三条路线的训练/推理/工具/摄像头 + 旧 key 兼容）
│   │   └── system_monitor.py        #   系统状态路由（CPU/内存/GPU/磁盘/网络 + 代理）
│   └── services/
│       ├── crowd_counting.py       #   人群计数核心服务（密度网/点预测训练与推理 + 数据工具）
│       ├── crowd_losses.py         #   损失库（MSE/MAE/SSIM/MSE+SSIM/OT/MMD）
│       ├── onnx_export.py          #   ONNX 导出 + 部署片段生成（单图 / 摄像头）
│       ├── onnx_infer.py           #   ONNX 推理（onnxruntime）
│       ├── camera_service.py        #   摄像头实时计数（MJPEG 推流；headless 自动提示）
│       ├── ddp_launcher.py         #   DDP 分布式启动器（mp.spawn 每卡一进程）
│       ├── train_common.py          #   训练通用（种子/早停/原子 checkpoint/续训/DDP 包装）
│       ├── eval_metrics.py          #   指标计算 + 损失曲线 PNG（Figure API 线程安全）
│       └── schemas.py              #   API 请求/响应 Pydantic 模型
│
├── nets/                           # ★ 模型定义层
│   ├── crowd/                      #   Density Map-based 参考模型 vendor
│   │   ├── backbones/vgg.py        #     VGG16 前端 + FPN 风格多尺度解码（896 通道）
│   │   ├── backbones/resnet.py     #     ResNet-IBN（IBN-Net）骨干
│   │   ├── heads/pred_density.py   #     PreDensityHead 密度头
│   │   ├── utils/modules.py        #     IBN / 上采样等公共模块
│   │   ├── crowd_vgg.py            #     CrowdVGG
│   │   └── crowd_resnet.py         #     CrowdResNet
│   ├── p2pnet/                     #   Point-based 参考模型 vendor（P2PNet，CVPR 2022）
│   ├── p2r/                        #   P2RLoss vendor（CVPR 2025 半监督点预测）
│   ├── cpoint/                     #   Consistent-Point vendor（arXiv 2503.12441 半监督）
│   ├── cu/                         #   CU vendor（ICCV 2023 半监督：不确定性分支 + PatchBank）
│   ├── pet/                        #   PET vendor（ICCV 2023 点查询四叉树）
│   ├── cltr/                       #   CLTR vendor（ECCV 2022 人群定位 Transformer）
│   ├── custom_models/              #   自定义模型（tasks 分区：密度图/点预测/头部框）
│   ├── custom_losses.py            #   自定义损失（tasks 分区注册）
│   ├── ghostnet.py / ghostnetv2.py #   GhostNet v1/v2（内置密度网骨干）
│   └── ms_/pp_ 前缀目录已移除      #   （本框架不含 MindSpore / PaddlePaddle）
│
├── static/                         # ★ 前端静态资源
│   ├── index.html                  #   单页应用（人群计数 + 系统状态/代理/指标/学习版图/个人主页）
│   ├── css/style.css               #   全局样式
│   └── js/app.js + i18n.js         #   前端逻辑 + 中英文文案
│
├── weights/pretrained/             # ★ 内置离线权重
│   ├── ck_ghostnetv2_*.pth.tar     #   GhostNetV2 ImageNet 权重
│   ├── p2pnet/SHTechA.pth          #   P2PNet ShanghaiTech A 预训练检查点
│   └── pet/SHA_model.pth           #   PET ShanghaiTech A 预训练检查点
├── uploads/                        # 用户上传（数据集 zip / 权重）
├── runs/                           # 训练产物（<模型名>_best.pth / _last.pth + metrics.csv + 曲线 + 日志）
├── results/                        # 推理结果保存
├── logs/                           # 日志
└── pretrained model/               # 自动下载的预训练权重统一缓存
```

## 界面使用流程

1. **启动**：`run.bat` 自动探测 Python 环境（`run_python.txt` → 环境变量 → PATH/conda），或 `python main.py`
2. **进入主界面**：两级导航 = 一级 PyTorch + 全局模块（**上传数据集** / 系统状态 / 代理设置 / 评价指标 / 学习版图 / 个人主页），二级「人群计数」；右上角切换中/英文
3. **上传和下载数据**（全局模块）：拖拽/点击选择 zip 压缩包上传（带进度条），或从 URL 下载数据集（预设链接 + 自定义 URL，后台下载带进度 + 自动解压）；同名数据集自动追加 `_1`/`_2` 后缀；已上传列表支持路径查看/删除，并可在「数据集结构浏览」中逐级展开查看真实目录与文件
4. **训练模式**（三卡片）：① 模型与设备 → ② 训练超参数（损失函数 / 分辨率按路线自适应）→ ③ 数据集 → ④ 开始训练（实时进度 / 逐 batch 损失曲线 + 逐 epoch 验证指标 / 日志 / 历史任务，支持停止、续训、恢复中断；权重以模型名前缀命名）。训练任务严格串行：已有训练运行时，后续训练按提交顺序进入 FIFO 队列，前一个结束后自动开始下一个，历史表显示队列序号
   - **技术路线选择**：「具体领域」下拉区分三条路线：
     - **人群计数 · 密度图**（GhostNet / torchvision / 参考项目密度网；GT 为 .npy/.txt，提供 .mat→npy 转换与 .npy→png 可视化工具，σ 可调）
     - **人群计数 · 点预测**（P2PNet 全监督；P2RLoss / Consistent-Point / CU 半监督，设「有标注样本比例」即可少量标注训练；另有 RAZ：Main-Net 顺序训练密度/定位/注意力并生成 zoom crops 训练 RAZ-Net，联合计数与定位；GT 为 .txt/.mat/.npy 点标注，无需密度图转换，隐藏 σ 与转换工具，推理时显示点置信度阈值）
     - **人群计数 · 头部框**（YOLO 系列；GT 为 YOLO/VOC/CSV 标注（单类 head），ultralytics 引擎托管训练，显示标注格式选择与 YOLO 专属增强参数，推理输出检测框 + 各类计数）
     - 头部框路线支持**知识蒸馏**（勾选后可选教师）：以大模型（教师）的中间特征为软目标训练参数量更少的小模型（学生）。教师可选同数据集训练产物（推荐，类别一致）或官方 COCO 预训练模型（自动下载，投影层自动对齐通道差异）；蒸馏损失权重 dis 可调（引擎默认 6.0）。师生需同为 3 尺度 Detect 头的 YOLO 系列（RT-DETR / YOLOv6 暂不支持），需 ultralytics ≥ 8.4.0（引擎原生 `DistillationModel`）
   - 训练产物分别保存在 `runs/crowd_density/`、`runs/crowd_point/`、`runs/crowd_box/`；路线与模型交叉混用会被拒绝（点预测 × DDP / 头部框 × DDP 同样被拒——YOLO 多卡由 ultralytics 引擎原生调度）
5. **推理模式**：选权重来源（内置预训练 / 训练产物 / 上传）→ 上传图像（可批量）→ 开始推理 → 人数 + 热力图 / 预测点可视化；可保存结果、开启摄像头实时计数
6. **部署模式（ONNX 导出）**：选择权重（训练产物 / 上传）→ **独立格式推断模块**自动识别模型类型（优先级：同目录 params.json → ultralytics 检查点 → state_dict 架构指纹；密度网具体骨干在导出任务中逐候选严格加载探测）→ 可调导出参数（输入边长按模型自动对齐 16/32 倍数、opset、动态 batch、onnxslim 简化、YOLO 专属 FP16 / NMS 内嵌端到端）→ 后台任务在**隔离子进程**中导出（原生库冲突 / 内存峰值不影响主服务）并经 onnxruntime 校验 → 产出 `model.onnx` + **按模型类型生成的部署片段代码 snippet.py**（密度图求和计数 / P2PNet 置信度筛点 / P2R logit>0 取点 / YOLO letterbox+NMS 或端到端，含 Windows 非 ASCII 路径安全读写）+ README/meta —— 复制片段 + onnx 到部署设备（ONNX Runtime / TensorRT / OpenVINO 等）即可运行，无需本框架与 PyTorch；片段支持单张图像与摄像头实时推理（`python snippet.py camera [编号]`，需 GUI 版 OpenCV）；产物保存于 `runs/onnx/<名称>/`。注意：**PET / CLTR / 自定义点预测模型暂不支持 ONNX 导出**，请使用 PyTorch 权重推理
7. **测试模式**（独立于训练 / 推理 / 部署）：选择算法与路线 → 选择训练产物权重或上传 `.pth/.pt` → 选择测试数据集 → 设置切块 patch（0=整图）→ 开始测试；后台仅推理、不更新权重，实时输出日志，完成后显示 **MAE / RMSE / 样本数**，逐图结果写入 `runs/test/<名称>/test_results.csv`。支持单域泛化（DCCUS / LLB_Count / MPCount）、密度图和点预测路线
8. **网络代理**：预训练权重下载网络受限时，在「代理设置」配置 HTTP(S)/SOCKS5 代理 → 测试连通性 → 保存并应用；配置持久化于 `proxy.json`，重启自动应用

## 评价指标

- 人群计数：**MAE**（训练每轮验证 + 评价指标页可交互演示 MAE/RMSE 计算过程）

## 参考与致谢

- [Density Map-based Crowd Counting Algorithm](<../Density%20Map-based%20Crowd%20Counting%20Algorithm/>)（本仓库算法目录）：损失函数 MSE/MAE/SSIM/MMD/OT、可视化、`nets/crowd/`（CrowdVGG / CrowdResNet）的移植来源
- [Point-based Crowd Counting Algorithm / CrowdCounting-P2PNet](<../Point-based%20Crowd%20Counting%20Algorithm/>)（本仓库算法目录）：`nets/p2pnet/` 的 vendor 来源（P2PNet, CVPR 2022）
- [KeepTryingTo/Uncertainty-Semi-Supervised-Crowd-Counting](https://github.com/KeepTryingTo/Uncertainty-Semi-Supervised-Crowd-Counting)：`nets/cu/` 的 vendor 来源（CU: Calibrating Uncertainty, ICCV 2023 独立复现）
- [cxliu0/PET](https://github.com/cxliu0/PET)：`nets/pet/` 的 vendor 来源（Point-Query Quadtree, ICCV 2023）
- [CLTR Project Page](https://dk-liang.github.io/CLTR/)：`nets/cltr/` 的 vendor 来源（An end-to-end transformer model for crowd localization, ECCV 2022）
- [https://github.com/cvlab-stonybrook/DM-Count](https://github.com/cvlab-stonybrook/DM-Count)（OT 损失参考实现）
- [https://github.com/gjy3035/Awesome-Crowd-Counting](https://github.com/gjy3035/Awesome-Crowd-Counting)（人群计数论文与代码大全 / C-3-Framework 统一评测）
- [KeepTryingTo/Crowd-Counting-Papers-with-Code](https://github.com/KeepTryingTo/Crowd-Counting-Papers-with-Code)（人群计数论文与代码汇总：276 篇论文下载链接 + 31 个代码仓库 + CSDN 博客/B 站视频讲解索引，配套本框架使用）
- [https://github.com/huawei-noah/Efficient-AI-Backbones](https://github.com/huawei-noah/Efficient-AI-Backbones)（GhostNet v1/v2）
- [https://github.com/KeepTryingTo/CrowdCounting-framework-PyTorch](https://github.com/KeepTryingTo/CrowdCounting-framework-PyTorch)（本仓库主项目）
- 界面与框架骨架搬移自「深度学习自动训练和推理框架」（[https://github.com/KeepTryingTo](https://github.com/KeepTryingTo)）
