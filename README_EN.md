# Crowd Counting Auto Training Framework

[中文](README.md) | English

> A FastAPI + PyTorch platform for crowd-counting training and inference. The UI is adapted from the
> "Deep Learning Auto Training and Inference Framework" and focuses on crowd counting only
> (MindSpore / PaddlePaddle and other domains removed).
>
> Three technical routes:
> - **Density map based** (Density Map-based Crowd Counting Algorithm)
> - **Point based** (Point-based Crowd Counting Algorithm: P2PNet / P2RLoss / Consistent-Point / CU Uncertainty / PET / CLTR)
> - **Bounding box based** (head localization, ultralytics YOLO series)

## Author

| Platform | Link |
|---|---|
| CSDN blog | <https://blog.csdn.net/Keep_Trying_Go> |
| Bilibili | <https://space.bilibili.com/625095571> |
| GitHub | <https://github.com/KeepTryingTo> |
| Douyin | <https://www.douyin.com/user/self?from_tab_name=main&showTab=post> |

## Features

| Capability | Training | Inference | Built-in models |
|---|---|---|---|
| Density map estimation | ✅ | ✅ Count + density heatmap / overlay | **GhostNet v1/v2** density nets (local offline weights), **torchvision backbones** (resnet/vgg/densenet/mobilenet/efficientnet/convnext/shufflenet), **reference density nets crowdvgg / crowdresnet** (vendored in `nets/crowd/`), custom density nets (`nets/custom_models/`) |
| Point prediction (P2PNet family) | ✅ | ✅ Count + predicted points (red dots) | **P2PNet** (VGG16_bn + FPN + point regression/classification + Hungarian matching, vendored in `nets/p2pnet/`; bundled SHTechA checkpoint works out of the box), **P2RLoss** (CVPR 2025 semi-supervised: teacher-student EMA + Point-to-Region loss, `nets/p2r/`), **Consistent-Point** (arXiv 2503.12441 semi-supervised: PA position aggregation + IUC uncertainty calibration + Mean-Teacher, `nets/cpoint/`), **CU** (ICCV 2023 semi-supervised: patch-level uncertainty branch + PatchBank ASM + ranking loss + curriculum pseudo-label filtering, `nets/cu/`), **PET** (ICCV 2023 point-query quadtree: sparse point queries split on demand; point-query and quadtree-split losses are computed inside the model, `nets/pet/`; bundled SHA checkpoint works offline for inference), **CLTR** (ECCV 2022 crowd-localization Transformer: Conditional DETR + ResNet50 + 500 point queries, 256 crop, `nets/cltr/`; no bundled weights — train or upload) — semi-supervised models train with few labels via the "labeled ratio" option |
| Head bounding box (YOLO) | ✅ | ✅ Head count + box visualization + per-class counts | **Full ultralytics lineup** — YOLOv3u / v5nu-v5xu / v6 (yaml) / v8 / v9 / v10 / 11 / 12 / 26 / **RT-DETR** / **YOLOv13** (official Release weights auto-downloaded) / yolo_custom (any `.pt` fine-tuning) — the classification head is replaced automatically during fine-tuning; `.pt` artifacts are used for inference |

**General training capabilities**

- Training/validation curve PNGs + **per-batch live loss curve** and **per-epoch validation metric curve** in the UI, plus history chart review
- Parameters adapt to the algorithm: loss and input resolution get route-specific defaults/options; weights are prefixed with the model name (e.g. `mobilenet_v2_best.pth`)
- Early stopping, random seed, full checkpoint (weights + optimizer/scheduler state + epoch + best metric)
- **Resume training** (one click for paused/failed tasks; upload an interrupted `<model>_last.pth` or the whole run-directory zip)
- Multi-GPU: DataParallel (default) / **DDP** (mp.spawn one process per GPU + DistributedSampler; NCCL on GPU, gloo on CPU)
- Training artifacts downloadable via `/runs`; the history table downloads logs / best weight / last weight / params / metrics
- **Upload progress bars** for datasets and weights, **batch inference** (many images → zip), **real-time camera counting** (MJPEG stream + FPS/count overlay; requires a GUI build of OpenCV — headless builds show an install hint)
- **Dataset structure browser** (inside Upload/Download; expand directories and list files level by level)
- Bilingual UI (Chinese/English, top-right toggle, remembered via localStorage)
- **System status page**: live CPU/memory/GPU/disk/network monitoring + live backend logs + network proxy settings (persisted in `proxy.json`)
- **ONNX deployment**: one-click export validation + generated deployment snippet (single image / real-time camera, `python snippet.py camera [index]`)

**Crowd-counting losses** (`app/services/crowd_losses.py`, ported from CrowdCounting-framework-PyTorch/losses, pure PyTorch with zero extra deps): MSE, MAE, **SSIM**, **MSE+SSIM**, **DM-Count** (NeurIPS 2020: counting loss + Sinkhorn **OT distribution matching** + **TV loss**, weights/reg/iterations aligned with the official implementation), **OT** (Sinkhorn + point annotations), **MMD** (domain adaptation, library function). Custom models/losses are registered **per route** (`nets/custom_models/`, `nets/custom_losses.py`; `tasks` = `crowd_density` / `crowd_point` / `crowd_box`), see "Custom models / losses" below.

## Quick start

### Requirements

- Python 3.7+, PyTorch 1.11.0+, torchvision
- CPU by default; single-GPU / multi-GPU when CUDA is available
- Install dependencies: `pip install -r requirements.txt`
- Optional (only affects the corresponding feature): ultralytics (head-box route), psutil (system monitor), pynvml (GPU utilization/temperature), pysocks (SOCKS5 proxy test)

### Launch

**Windows**: double-click `run.bat` (auto-detects a Python environment; works after moving the project), or `run.bat --check` to only check the environment.

**Linux / macOS**: `bash run.sh` (or `chmod +x run.sh && ./run.sh`); `bash run.sh --check` only checks the environment.

**Manual launch**:

```bash
python main.py
# or
uvicorn main:app --host 127.0.0.1 --port 8060
```

Open **http://127.0.0.1:8060** (default port 8060 to avoid the common 8000 conflict; `python main.py --port <port>` overrides).

## Dataset layout (training)

```
dataset_root/
├── Train/
│   ├── Images/       # *.jpg / *.png
│   └── GT/           # *.npy (density map, sum≈count) or *.txt (one "x y" per line; P2PNet also accepts .mat)
└── Test/
    ├── Images/
    └── GT/
```

Also supports a flat `imgs/ + gt/npys/` layout and case variants (imgs/images, GT/gt/npys, ...). If `Test/` is absent, a validation split is created automatically.

> **σ consistency note**: `.txt` point annotations are rasterized with the Gaussian `sigma` parameter at training time (default 8), while `.npy` density maps keep the σ used when they were generated (the `.mat→npy` tool defaults to 2, aligned with the reference framework).
> **Keep σ consistent when mixing the two GT types** (both the tool and the training card expose sigma); otherwise the supervision scale differs. The training log reports the GT composition and σ.

### Built-in data tools (in the training card, background tasks with progress)

**① .mat → GT density map**

Input a `.mat` directory + output directory (optional image directory). Detects **ShanghaiTech (image_info)** and **annPoints** (QNRF/NWPU/UCF) formats:

- `image_info` supports nested MATLAB structs (`location`/`number` and optional `height`/`width`, recursively)
- Produces reference-aligned density maps (fixed sigma=2, optional k-NN adaptive) by **summing normalized Gaussian patches per point** (numerically equivalent to full-image filtering, but milliseconds instead of minutes on large dense images)
- Matches labels with arbitrary prefixes/suffixes (GT_IMG_x.mat ↔ IMG_x.jpg, token-level matching avoids false pairs)
- `.npy` files are named after the image stem, so training pairs need no renaming

**② .npy density map visualization (.npy → .png)**

Three modes: pseudo-color heatmap (jet, same as inference) / grayscale / **overlay** (ported from `vis_img_den.py`'s `save_density_map`: normalize → invert → JET → `(1-alpha)·img + alpha·density` → renormalize; alpha adjustable); `count≈density sum` is annotated at the top-left; existing PNGs are skipped for incremental runs.

### Upload / download datasets

- Same-name datasets get an automatic `_1` / `_2` suffix instead of overwriting existing data;
- After upload/download, expand the real directory and file list level by level in **Dataset structure**.

### Training (density map estimation)

- Models: GhostNet series / torchvision backbones / reference **crowdvgg / crowdresnet** / custom density nets
- Loss: MSE / MAE / SSIM / MSE+SSIM / **DM-Count** / custom:* (default MSE); optimizers Adam/SGD/AdamW etc.; selectable schedulers (default cosine annealing; step / multistep / warmup etc.)
  - **DM-Count (NeurIPS 2020)**: `L = L_count + 0.1·L_OT + 0.01·L_TV` — counting L1 + Sinkhorn entropic-regularized optimal transport distribution matching (envelope-theorem gradient write-back) + normalized-density TV; Sinkhorn reg=10 / 100 iterations, OT target measure = normalized GT density (works with both .npy and .txt GT), OT is skipped during validation
- Augmentation: random crop (resolution selectable) / random flip; point annotations are rasterized with the adjustable sigma (optional k-NN adaptive)
- Metric: **MAE** (mean |predicted count − GT count|); saves `<model>_best.pth` / `<model>_last.pth`; the loss curve updates per batch, validation MAE per epoch

### Training (point prediction — P2PNet / P2RLoss / Consistent-Point / CU / PET / CLTR)

- Requires point annotations: `Train/Images + GT/*.txt` (one "x y" per line) / ShanghaiTech `.mat` / `.npy` ((N,2) coordinates, P2RLoss format)
- **P2PNet**: Adam with grouped LR (backbone `lr_backbone=1e-5`, rest `lr=1e-4`) + StepLR + grad clip 0.1; random scale [0.7, 1.3] (image and coordinates together)
- **Semi-supervised trio (P2RLoss / Consistent-Point / CU)**: teacher-student Mean-Teacher; with "labeled ratio" < 1 only that fraction is labeled and the rest are used as unlabeled data for pseudo-label training (deterministic seed split); defaults follow each paper (P2R Adam 5e-5 / CP, CU Adam 1e-4, backbone 1e-5, constant LR) while optimizer / LR / scheduler / loss are adjustable in the training UI
  - P2RLoss: Point-to-Region matching loss (regional confidence propagation)
  - Consistent-Point: PA position aggregation (K=4 neighbor mean) + IUC uncertainty weighting
  - CU: patch-level uncertainty branch + PatchBank ASM proxy labels + pairwise ranking loss + curriculum pseudo-label filtering (λ₁=0.3 / patch=64 / u_max=0.5 / EMA 0.999)
- **PET (ICCV 2023)**: point-query quadtree; computes the point-query loss and quadtree-split loss internally; default AdamW grouped LR with constant LR, patch=256; Batch Size ≥ 2 recommended; bundled `weights/pretrained/pet/SHA_model.pth` works for inference out of the box
- **CLTR (ECCV 2022)**: Conditional DETR + ResNet50 crowd localization; 256×256 crop, 500 queries; default Adam + MultiStepLR; validation/inference tile the image into 256×256 patches (score threshold 0.35); no bundled weights — train or upload a `.pth`
- Loss: the point-prediction dropdown defaults to each algorithm's paper loss (P2PNet Hungarian matching / P2R / CP / CU / CLTR point-localization); optimizer, LR and scheduler are freely selectable, and P2PNet / P2R / CP / CU / CLTR may add custom auxiliary losses registered for `crowd_point` (PET computes its loss inside the model and cannot be replaced; the head-box route is engine-managed)
- Resolution: long side selectable (Original / 384 / 512 / 640 / 768 / 1024); custom point models must implement the P2PNet output contract (`pred_logits` / `pred_points`), see `nets/custom_models/example_p2pnet_backbone.py`
- VGG16_bn backbone pretraining: `weights/pretrained/vgg16_bn.pth` → torch hub cache → online download
- Per-epoch validation MAE (points with softmax score > 0.5 vs GT count); saves `<model>_best.pth` / `<model>_last.pth` (semi-supervised models also save teacher/bank state for full resume)

### Custom models / losses (registered per route)

Implement models in `nets/custom_models/` and losses in `nets/custom_losses.py`, mark the applicable route with `tasks`, and the UI loads/displays them under the matching route:

| Route | Model contract | Loss |
|---|---|---|
| Density `crowd_density` | `forward(x) -> [N,1,H',W']` | `forward(pred, gt) -> scalar` |
| Point `crowd_point` | `forward(x) -> {"pred_logits","pred_points"}` | `forward(outputs, targets) -> scalar` (auxiliary) |
| Box `crowd_box` | Factory returns an ultralytics-loadable `.yaml` / `.pt` source | Engine-managed (box/cls/dfl) |

```python
@register_custom_model("my_density", desc="My density net", tasks=["crowd_density"])
def build(...): return MyNet()

@register_custom_loss("my_loss", desc="My loss", tasks=["crowd_density"])
def factory(...): return MyLoss()
```

`example_*.py` files are not registered by default — copy and rename them. See `nets/custom_models/`.

### Inference

Three weight sources: built-in pretrained (**P2PNet SHTechA** works offline / GhostNetV2 local backbones), training artifacts (automatically listed), and uploaded `.pth/.pt`.

- **Density nets**: estimated count + density heatmap overlay / pure heatmap (same size as the original image)
- **Point prediction (P2PNet / P2RLoss / Consistent-Point / CU / PET / CLTR / custom point models)**: count + predicted points (red dots); confidence threshold adjustable for P2PNet / Consistent-Point / CU / PET / CLTR (CLTR default 0.35, others 0.5), P2RLoss uses `logit>0`; semi-supervised / custom point models use training-artifact weights, PET can use the bundled SHA checkpoint, and CLTR requires a trained/uploaded weight
- "Save results" writes images to `results/` with download links; batch inference is packaged as a zip
- **Real-time camera counting**: the "Camera live counting" button in the inference panel; OpenCV capture + per-frame inference + MJPEG stream

## Project structure

```text
Codes/
├── main.py                         # FastAPI entry (python main.py → http://127.0.0.1:8060)
├── launcher.py                     # Interactive launcher (lists candidate Python envs)
├── run.bat                         # Windows one-click launcher (auto-detect Python; --check only)
├── run.sh                          # Linux/macOS one-click launcher (auto-detect Python; --check only)
├── run_python.txt                  # Full python.exe path (preferred by run.bat / run.sh)
├── config.py                       # Global config (data paths, allowed upload extensions, ...)
├── requirements.txt                # Python dependencies
│
├── app/                            # ★ Backend (FastAPI)
│   ├── core/
│   │   ├── backend_log.py          #   Backend logs: stdout/stderr ring buffer
│   │   ├── registry.py             #   Model registry: built-in factories + route-partitioned custom models/losses
│   │   ├── dataset_naming.py       #   Unique dataset dirs (_1/_2 suffix)
│   │   ├── model_cache.py          #   Unified pretrained weight cache (pretrained model/)
│   │   ├── optim_builders.py       #   Unified torch optimizer builder
│   │   ├── proxy.py                #   Network proxy (proxy.json + connectivity test)
│   │   └── task_manager.py         #   Training task manager (threads/stop/resume/history/graceful shutdown)
│   ├── routers/
│   │   ├── crowd.py                #   Crowd routes (/api/crowd: density/point/box
│   │   │                           #   training/inference/tools/camera + legacy keys)
│   │   └── system_monitor.py       #   System routes (CPU/memory/GPU/disk/network + proxy)
│   └── services/
│       ├── crowd_counting.py       #   Core service (density/point training & inference + data tools)
│       ├── crowd_losses.py         #   Loss library (MSE/MAE/SSIM/MSE+SSIM/OT/MMD)
│       ├── onnx_export.py          #   ONNX export + deployment snippet (image / camera)
│       ├── onnx_infer.py           #   ONNX inference (onnxruntime)
│       ├── camera_service.py       #   Real-time camera counting (MJPEG; headless hint)
│       ├── ddp_launcher.py         #   DDP launcher (mp.spawn, one process per GPU)
│       ├── train_common.py         #   Training utilities (seed/early stop/atomic checkpoint/resume/DDP)
│       ├── eval_metrics.py         #   Metrics + loss curve PNG (thread-safe Figure API)
│       └── schemas.py              #   Pydantic request/response models
│
├── nets/                           # ★ Model definitions
│   ├── crowd/                      #   Density-map reference models (vendored)
│   │   ├── backbones/vgg.py        #     VGG16 front-end + FPN-style multi-scale decoder (896 ch)
│   │   ├── backbones/resnet.py     #     ResNet-IBN (IBN-Net) backbone
│   │   ├── heads/pred_density.py   #     PreDensityHead
│   │   ├── utils/modules.py        #     IBN / upsampling modules
│   │   ├── crowd_vgg.py            #     CrowdVGG
│   │   └── crowd_resnet.py         #     CrowdResNet
│   ├── p2pnet/                     #   Point-based reference (P2PNet, CVPR 2022)
│   ├── p2r/                        #   P2RLoss (CVPR 2025 semi-supervised point prediction)
│   ├── cpoint/                     #   Consistent-Point (arXiv 2503.12441 semi-supervised)
│   ├── cu/                         #   CU (ICCV 2023 semi-supervised: uncertainty + PatchBank)
│   ├── pet/                        #   PET (ICCV 2023 point-query quadtree)
│   ├── cltr/                       #   CLTR (ECCV 2022 crowd-localization Transformer)
│   ├── custom_models/              #   Custom models (tasks: density/point/box)
│   ├── custom_losses.py            #   Custom losses (task-partitioned)
│   ├── ghostnet.py / ghostnetv2.py #   GhostNet v1/v2 backbones
│   └── ms_/pp_ prefixed dirs removed  #   (no MindSpore / PaddlePaddle)
│
├── static/                         # ★ Frontend assets
│   ├── index.html                  #   SPA (crowd counting + system/proxy/metrics/learning map/author)
│   ├── css/style.css               #   Global styles
│   └── js/app.js + i18n.js         #   Frontend logic + bilingual texts
│
├── weights/pretrained/             # ★ Built-in offline weights
│   ├── ck_ghostnetv2_*.pth.tar     #   GhostNetV2 ImageNet weights
│   ├── p2pnet/SHTechA.pth          #   P2PNet ShanghaiTech A checkpoint
│   └── pet/SHA_model.pth           #   PET ShanghaiTech A checkpoint
├── uploads/                        # User uploads (dataset zip / weights)
├── runs/                           # Training artifacts (<model>_best.pth / _last.pth + metrics.csv + curves + logs)
├── results/                        # Inference outputs
├── logs/                           # Logs
└── pretrained model/               # Auto-downloaded weight cache
```

## UI workflow

1. **Launch**: `run.bat` auto-detects Python (`run_python.txt` → env var → PATH/conda), or run `python main.py`
2. **Main UI**: two-level navigation = PyTorch + global modules (**Upload datasets** / System status / Proxy / Metrics / Learning map / Author), with a secondary "Crowd counting" entry; toggle Chinese/English at the top-right
3. **Upload and download data**: drag/click a zip to upload (progress bar), or download a dataset from a URL (presets + custom URL, background download with progress + auto-extract); same-name datasets get `_1`/`_2` suffixes; the uploaded list supports path view/delete and the **Dataset structure** browser
4. **Training mode** (three cards): ① model & device → ② hyperparameters (loss / resolution adapt per route) → ③ dataset → ④ start training (live progress / per-batch loss + per-epoch validation metrics / log / history; stop, resume, recover interrupted runs; weights are named with the model prefix). Training is strictly serial: while one run is active, subsequent runs enter a FIFO queue in submission order and start automatically when the previous run finishes; the history table shows the queue position
   - **Route selection** — three routes in the "domain" dropdown:
     - **Crowd counting · Density** (GhostNet / torchvision / reference density nets; GT `.npy`/`.txt`; `.mat→npy` and `.npy→png` tools; sigma adjustable)
     - **Crowd counting · Point** (P2PNet supervised; P2RLoss / Consistent-Point / CU semi-supervised via "labeled ratio"; GT `.txt`/`.mat`/`.npy`; no density conversion, sigma/tools hidden, point confidence threshold at inference)
     - **Crowd counting · Head box** (YOLO series; YOLO/VOC/CSV single-class "head" annotations; ultralytics-managed training; annotation format and YOLO augmentation options; boxes + per-class counts at inference)
       - Head-box knowledge distillation (optional teacher): trains a smaller student using the teacher's intermediate features as soft targets. Teacher = a same-dataset training artifact (recommended) or official COCO pretrained weights (auto-downloaded, projection layers align channels); distillation weight `dis` adjustable (engine default 6.0). Both must be 3-scale Detect YOLO models (RT-DETR / YOLOv6 not supported); requires ultralytics ≥ 8.4.0 (native `DistillationModel`)
     - Artifacts are stored under `runs/crowd_density/`, `runs/crowd_point/`, `runs/crowd_box/`; route/model mismatches are rejected (point × DDP / box × DDP are also rejected — YOLO multi-GPU is handled natively by ultralytics)
5. **Inference**: choose a weight source (built-in / trained / uploaded) → upload images (batch okay) → run → count + heatmap / predicted points; save results and start real-time camera counting
6. **Deployment (ONNX export)**: choose a weight → the **independent format-inference module** detects the model type (priority: `params.json` in the same directory → ultralytics checkpoint → `state_dict` architecture fingerprint; density backbones are probed strictly during export) → adjustable export options (input size aligned to 16/32, opset, dynamic batch, onnxslim simplification, YOLO-only FP16 / end-to-end NMS) → a background task exports in an **isolated subprocess** (native-library conflicts / peak memory do not affect the server) and validates with onnxruntime → outputs `model.onnx` + a **route-specific `snippet.py`** (density sum / P2PNet thresholding / P2R `logit>0` / YOLO letterbox+NMS or end-to-end; Windows non-ASCII path safe) + README/meta. Copy the snippet and `model.onnx` to the deployment device (ONNX Runtime / TensorRT / OpenVINO, etc.); the framework and PyTorch are not required. The snippet supports single images and real-time camera inference (`python snippet.py camera [index]`, GUI OpenCV required). Artifacts go to `runs/onnx/<name>/`. Note: **PET / CLTR / custom point models do not support ONNX export**; use PyTorch weights for inference
7. **Test mode** (independent of train / inference / deploy): choose a route and algorithm → choose a training artifact or upload a `.pth/.pt` → choose a test dataset → set the patch window (0 = whole image) → run; inference only, no weight updates, with live logs and final **MAE / RMSE / sample count**. Per-image rows are written to `runs/test/<name>/test_results.csv`. Supports SDG (DCCUS / LLB_Count / MPCount), density and point routes
8. **Network proxy**: if pretrained-weight downloads are blocked, configure an HTTP(S)/SOCKS5 proxy under "Proxy settings" → test connectivity → save and apply; persisted in `proxy.json` and applied on restart

## Evaluation metrics

- Crowd counting: **MAE** (per-epoch validation + an interactive MAE/RMSE demonstration on the metrics page)

## References & acknowledgements

- [Density Map-based Crowd Counting Algorithm](../Density%20Map-based%20Crowd%20Counting%20Algorithm/) (algorithm directory in this repo): source of MSE/MAE/SSIM/MMD/OT losses, visualization, and `nets/crowd/` (CrowdVGG / CrowdResNet)
- [Point-based Crowd Counting Algorithm / CrowdCounting-P2PNet](../Point-based%20Crowd%20Counting%20Algorithm/) (algorithm directory in this repo): vendor source of `nets/p2pnet/` (P2PNet, CVPR 2022)
- [KeepTryingTo/Uncertainty-Semi-Supervised-Crowd-Counting](https://github.com/KeepTryingTo/Uncertainty-Semi-Supervised-Crowd-Counting): vendor source of `nets/cu/` (CU: Calibrating Uncertainty, ICCV 2023 independent reproduction)
- [cxliu0/PET](https://github.com/cxliu0/PET): vendor source of `nets/pet/` (Point-Query Quadtree, ICCV 2023)
- [CLTR Project Page](https://dk-liang.github.io/CLTR/): vendor source of `nets/cltr/` (An end-to-end transformer model for crowd localization, ECCV 2022)
- <https://github.com/cvlab-stonybrook/DM-Count> (OT loss reference)
- <https://github.com/gjy3035/Awesome-Crowd-Counting> (crowd-counting papers/code collection, C-3-Framework evaluation)
- [KeepTryingTo/Crowd-Counting-Papers-with-Code](https://github.com/KeepTryingTo/Crowd-Counting-Papers-with-Code) (276 papers with download links + 31 code repos + blog/video walkthroughs)
- <https://github.com/huawei-noah/Efficient-AI-Backbones> (GhostNet v1/v2)
- <https://github.com/KeepTryingTo/CrowdCounting-framework-PyTorch> (the main project)
- UI and framework skeleton adapted from the "Deep Learning Auto Training and Inference Framework" (<https://github.com/KeepTryingTo>)
