# -*- coding: utf-8 -*-
"""人群计数领域路由：密度图估计 + 点预测（P2PNet）

子任务与工具：
- 训练 / 推理（密度网 + P2PNet 点预测）
- .mat 点标注 → GT 密度图转换工具
- .npy 密度图 → PNG 可视化工具
- 摄像头实时计数
"""
import base64
import io
import json
import re
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from config import (UPLOAD_DATASET_DIR, UPLOAD_MODEL_DIR, RUNS_DIR,
                    RESULTS_DIR)
from app.core.task_manager import (task_manager, find_best_weight,
                                   find_last_weight)
from app.core.registry import (scan_custom_models, custom_losses_for,
                               ensure_scanned)
from app.services import crowd_counting
from app.services.schemas import TrainParams, OnnxExportParams

router = APIRouter(prefix="/api/crowd", tags=["crowd_counting"])

# 子任务定义：人群计数按技术路线拆分为「密度图」「点预测」「头部框」三条，
# 界面「具体领域」下拉即区分（点预测/头部框无需密度图转换工具）。
# crowd_counting 为旧版合并 key，仅作历史任务续训/旧客户端兼容，不再列出。
CROWD_TASKS = {
    "crowd_density": {
        "label": "人群计数 · 密度图", "trainable": True,
        "desc": "密度图回归：GhostNet / torchvision / 参考项目密度网；"
                "GT 为 .npy 密度图或 .txt 点标注（按 σ 生成），"
                ".mat 可用内置工具转换",
    },
    "crowd_point": {
        "label": "人群计数 · 点预测", "trainable": True,
        "desc": "点预测：P2PNet 直接回归人头坐标（.txt / ShanghaiTech .mat "
                "点标注），无需密度图转换；P2RLoss（CVPR 2025）与 "
                "Consistent-Point（arXiv 2503.12441）半监督点预测（教师-学生，"
                "label_ratio 设定标注比例，少量标注即可训练；另支持 "
                "(N,2) 坐标 .npy 标注）；RAZ（Main-Net 密度/定位/注意力 + "
                "RAZ-Net 循环局部放大，联合计数与定位）",
    },
    "crowd_box": {
        "label": "人群计数 · 头部框", "trainable": True,
        "desc": "头部定位框预测：YOLO 系列检测人头 bounding box"
                "（YOLO/VOC/CSV 标注，ultralytics 引擎），"
                "人头数量 = 检测框数量；需安装 ultralytics",
    },
    "crowd_sdg": {
        "label": "人群计数 · 单域泛化", "trainable": True,
        "desc": "单域泛化（SDG，基于密度图）：DCCUS（AAAI 2023 元学习 + "
                "语义/风格记忆）/ LLB_Count（局部-边界感知损失）/ MPCount"
                "（CVPR 2024 自适应记忆库 + patch 分类一致性）。"
                "源数据集训练、目标数据集跨域验证；独立「测试」模式在"
                "其它域测试集上评估 MAE / MSE",
    },
    "crowd_counting": {   # 兼容旧任务 kind（历史续训 / 旧前端）
        "label": "人群计数", "trainable": True,
        "desc": "密度图 + 点预测（旧版合并入口）",
    },
}

# 界面下拉展示的子任务（不含兼容 key）
_TASKS_VISIBLE = ("crowd_density", "crowd_point", "crowd_box", "crowd_sdg")


def _resolve_dataset(path_or_name: str) -> str:
    """数据集路径解析：支持绝对路径或已上传数据集名"""
    p = Path(path_or_name)
    if p.exists():
        return str(p.resolve())
    up = UPLOAD_DATASET_DIR / path_or_name
    if up.exists():
        return str(up.resolve())
    raise HTTPException(400, f"数据集不存在: {path_or_name}（可先上传 zip 数据集）")


def _persist_task_params(save_dir, params):
    """把后台任务参数写入 save_dir/params.json，供数据集占用检查使用。"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    (Path(save_dir) / "params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")


def _resolve_weight(path_or_name: str, task: str = None) -> str:
    """权重路径解析：绝对路径 / 上传的权重名 / 训练产物（限定同任务目录）"""
    p = Path(path_or_name)
    if p.exists():
        return str(p.resolve())
    up = UPLOAD_MODEL_DIR / path_or_name
    if up.exists():
        return str(up.resolve())
    # 训练产物：先在当前任务目录下找，再全 runs/ 搜
    # （rglob 仅支持纯相对文件名作模式：带盘符/根锚点的路径搜索无意义，
    #   直接判不存在给出明确报错，而非 rglob 抛 NotImplementedError → 500）
    if p.drive or p.root:
        raise HTTPException(400, f"权重文件不存在: {path_or_name}")
    # runs/ 下的搜索只接受纯文件名，禁止通配符与 .. 反查
    if any(ch in path_or_name for ch in "*?[]") or \
            any(part == ".." for part in p.parts):
        raise HTTPException(
            400, f"权重名不合法（不支持通配符/..）: {path_or_name}")
    if task:
        for r in sorted((RUNS_DIR / task).rglob(path_or_name),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            return str(r.resolve())
    for r in sorted(RUNS_DIR.rglob(path_or_name),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        return str(r.resolve())
    raise HTTPException(400, f"权重文件不存在: {path_or_name}")


# ------------------------- 元信息 -------------------------

def _custom_models_for_task(task_key):
    """按路线取自定义模型并转成前端模型条目（类型区分三种算法）"""
    from app.core.registry import custom_models_for
    type_map = {"crowd_density": "custom",
                "crowd_point": "custom_point",
                "crowd_box": "custom_box"}
    default_desc = {"crowd_density": "自定义密度网（密度图输出）",
                    "crowd_point": "自定义点预测模型（P2PNet 输出契约）",
                    "crowd_box": "自定义头部框模型（ultralytics 源：.yaml/.pt）"}
    out = []
    for meta in custom_models_for(task_key):
        out.append({
            "name": meta["name"],
            "desc": meta.get("desc") or default_desc[task_key],
            "type": type_map[task_key],
            "pretrained": False, "pretrained_local": False,
            "need_weight": task_key != "crowd_box",
        })
    return out


def _merge_models(base, extra):
    """按模型名去重合并（自定义模型与内置同名时以内置为准）"""
    names = {m.get("name") for m in base}
    return list(base) + [m for m in extra if m.get("name") not in names]


@router.get("/tasks")
def list_tasks():
    scan_custom_models(Path(__file__).resolve().parents[2] / "nets" / "custom_models")
    all_models = crowd_counting.list_models()
    tasks = []
    for key in _TASKS_VISIBLE:
        meta = CROWD_TASKS[key]
        custom = _custom_models_for_task(key)
        if key == "crowd_box":
            # 头部框路线：YOLO 系列（ultralytics 引擎，未安装时清单带提示）
            # 推理需训练产物权重（COCO 无 head 类）；训练可微调/从头训练
            from app.services import yolo_service
            models = _merge_models(yolo_service.list_models(), custom)
            train_models = _merge_models(
                yolo_service.list_train_models(),
                [{**m, "need_weight": False} for m in custom])
        elif key == "crowd_point":
            # 点预测：P2PNet/P2R/CP/CU + 注册到 crowd_point 的自定义模型
            base = [m for m in all_models if m.get("type") == "p2pnet"]
            models = _merge_models(base, custom)
            train_models = models
        elif key == "crowd_sdg":
            # 单域泛化：DCCUS / LLB_Count / MPCount（独立分支）
            from app.services import sdg_service
            models = sdg_service.list_models()
            train_models = models
        else:
            # 密度图：内置/参考/torchvision 密度网 + 注册到 crowd_density 的自定义模型
            base = [m for m in all_models if m.get("type") != "p2pnet"]
            models = _merge_models(base, custom)
            train_models = models
        tasks.append({"key": key, **meta, "models": models,
                      "train_models": train_models})
    return tasks


@router.get("/options")
def train_options():
    ensure_scanned()   # 确保自定义损失已按 tasks 注册（mtime 缓存，开销很小）
    from app.services.crowd_counting import CROWD_SCHEDULERS
    from app.core.optim_builders import torch_optimizers
    density_custom = [f"custom:{m['name']}"
                      for m in custom_losses_for("crowd_density")]
    point_custom = [f"custom:{m['name']}"
                    for m in custom_losses_for("crowd_point")]
    sdg_custom = [f"custom:{m['name']}"
                  for m in custom_losses_for("crowd_sdg")]
    density_losses = ["mse", "mae", "ssim", "mse_ssim", "dmcount"] \
        + density_custom
    return {
        "optimizers": torch_optimizers(),
        "schedulers": {
            "crowd_density": CROWD_SCHEDULERS,
            # 点预测与密度图共用 9 种调度器（各算法论文默认在界面预置：
            # 半监督三件套 / PET 恒定、P2PNet step、CLTR multistep）
            "crowd_point": CROWD_SCHEDULERS,
            # 单域泛化：DCCUS/LLB 默认恒定、MPCount 默认 cosine
            "crowd_sdg": CROWD_SCHEDULERS,
            "crowd_box": ["linear", "cosine"],
        },
        "losses": {
            "crowd_density": density_losses,
            # 点预测：内置默认损失（p2pnet）+ 注册到 crowd_point 的自定义
            # 附加损失；前端按模型把默认项设为该算法的论文损失
            # （p2r/cp/cu/pet/cltr），并可追加自定义附加损失
            "crowd_point": ["p2pnet", "raz", "d2c"] + point_custom,
            # 单域泛化：损失由算法内置（DCCUS/LLB/MPCount 默认项），可追加
            # 注册到 crowd_sdg 的自定义附加损失
            "crowd_sdg": ["dccus", "llb", "mpcount"] + sdg_custom,
            # 头部框：YOLO 引擎内置 box/cls/dfl 损失（引擎托管，不使用自定义损失）
            "crowd_box": ["yolo"],
        },
        "devices": _devices(),
        # 多 GPU 训练支持（gpu_count>1 时前端显示模式选择）
        "gpu_count": (torch.cuda.device_count()
                      if torch.cuda.is_available() else 0),
        "multi_gpu_modes": [
            {"id": "dp", "label": "DataParallel（单进程多卡，默认）"},
            {"id": "ddp", "label": "DDP 分布式（每卡一进程 + 数据分片）"},
        ],
    }


def _devices():
    devs = [{"id": "cpu", "label": "CPU"}]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devs.append({"id": f"cuda:{i}", "label": f"GPU{i}: {name}"})
        devs.append({"id": "cuda", "label": "全部 GPU (DataParallel)"})
    return devs


# ------------------------- 训练 -------------------------

TRAIN_RUNNERS = {
    "crowd_density": crowd_counting.run_crowd_train,
    "crowd_point": crowd_counting.run_crowd_train,
    "crowd_box": None,   # 延迟导入（ultralytics 可选依赖，下面分支处理）
    "crowd_sdg": None,   # 延迟导入（单域泛化独立服务）
    "crowd_counting": crowd_counting.run_crowd_train,   # 兼容旧任务 kind
}


def _get_sdg_runner():
    """单域泛化训练 runner（延迟导入，避免拖慢其它路线启动）"""
    from app.services import sdg_service
    return sdg_service.run_sdg_train


def _get_box_runner():
    """头部框路线 runner（延迟导入 ultralytics，未安装时报错友好）"""
    from app.services import yolo_service
    return yolo_service.run_yolo_train


def _check_route_model(task: str, model: str):
    """路线 × 模型匹配校验（防止手工请求混用，产物目录按路线分列）"""
    from app.services import yolo_service
    from app.core.registry import custom_model_tasks
    # 自定义模型：按其注册的 tasks 分区校验
    _tasks = custom_model_tasks(model)
    if _tasks:
        if task in _tasks:
            return
        raise HTTPException(
            400, f"自定义模型 {model} 仅可用于路线 {_tasks}，"
                 f"当前路线为 {task}（请在注册时设置正确的 tasks）")
    if task == "crowd_density" and model in ("p2pnet", "p2r", "cp", "cu",
                                              "pet", "cltr", "raz", "d2c"):
        raise HTTPException(
            400, "密度图路线不支持点预测模型（P2PNet / P2R / "
                 "Consistent-Point / CU / PET / CLTR 请在「人群计数 · 点预测」"
                 "路线下使用）")
    if task == "crowd_point" and model not in ("p2pnet", "p2r", "cp", "cu",
                                                "pet", "cltr", "raz", "d2c"):
        raise HTTPException(
            400, "点预测路线仅支持 P2PNet / P2R / Consistent-Point / CU / "
                 "PET / CLTR 模型（密度网请在「人群计数 · 密度图」路线使用）")
    from app.services.sdg_service import SDG_MODELS as _SDG_MODELS
    if task == "crowd_sdg" and model not in _SDG_MODELS:
        raise HTTPException(
            400, f"单域泛化路线仅支持 DCCUS / LLB_Count / MPCount 模型"
                 f"（可选: {list(_SDG_MODELS)}）；密度网/点预测模型请在"
                 f"各自的路线中使用")
    if task != "crowd_sdg" and model in _SDG_MODELS:
        raise HTTPException(
            400, "单域泛化模型（DCCUS / LLB_Count / MPCount）仅用于"
                 "「人群计数 · 单域泛化」路线")
    if task == "crowd_box" and not yolo_service.is_yolo_model(model):
        raise HTTPException(
            400, "头部框路线仅支持 YOLO 系列模型（密度网/点预测模型请在"
                 "「人群计数 · 密度图 / 点预测」路线下使用）")
    if task != "crowd_box" and yolo_service.is_yolo_model(model):
        raise HTTPException(
            400, "YOLO 系列模型属于头部框路线（请在"
                 "「人群计数 · 头部框」路线下使用）")


# ------------------------- ONNX 部署导出（部署模式） -------------------------

@router.post("/onnx/format")
def onnx_infer_format(body: dict):
    """权重格式推断（独立模块 weight_format）：部署面板选择权重后调用，
    自动识别模型路线/架构/建议输入尺寸，前端据此渲染导出参数与提示"""
    from app.services import weight_format
    wp = (body or {}).get("weight_path", "")
    if not wp:
        raise HTTPException(400, "请先选择权重文件")
    wp = _resolve_weight(wp, None)
    try:
        return weight_format.infer_weight_format(wp)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/onnx/export")
def onnx_export(params: OnnxExportParams):
    """ONNX 部署导出（后台任务）：格式推断 → 导出（密度网/点预测
    torch.onnx.export，YOLO 走 ultralytics 原生）→ onnxslim 简化 +
    onnxruntime 校验 → 生成部署片段代码（按模型类型）"""
    from app.services import onnx_export as _oe
    params.weight_path = _resolve_weight(params.weight_path, None)
    run_name = params.run_name or (
        f"onnx_{Path(params.weight_path).stem}_{params.imgsz or 'auto'}")
    run_name = re.sub(r"[^\w\-]", "_", run_name)
    save_dir = RUNS_DIR / "onnx" / run_name
    if save_dir.exists():
        save_dir = RUNS_DIR / "onnx" / f"{run_name}_{uuid.uuid4().hex[:4]}"
    save_dir.mkdir(parents=True, exist_ok=True)
    # params.json 持久化（与训练一致）：服务重启后磁盘扫描重建该任务，
    # 部署产物（onnx / 片段代码下载）在历史列表中仍可达
    (save_dir / "params.json").write_text(
        json.dumps(params.dict(), ensure_ascii=False, indent=2),
        encoding="utf-8")
    task = task_manager.create(run_name, "onnx", save_dir)
    task_manager.submit(task, _oe.run_onnx_export, params.dict())
    return {"task_id": task.id, "save_dir": str(save_dir),
            "run_name": run_name}


@router.post("/onnx/infer_format")
def onnx_infer_format(body: dict):
    """ONNX 模型类型识别（推理面板选择 onnx 后调用，子进程 ort 探测）：
    本框架导出产物按同目录 meta.json 精确识别；上传的外部 onnx 按图结构
    启发式识别（双输出 P2PNet / [max_det,6] 端到端 YOLO / [4+nc,N] 标准
    YOLO / 1/4 logit 图 P2R / 单通道密度图）"""
    from app.services import onnx_infer
    op = (body or {}).get("onnx_path", "")
    if not op:
        raise HTTPException(400, "请先选择 ONNX 模型文件")
    op = _resolve_weight(op, None)
    if not str(op).lower().endswith(".onnx"):
        raise HTTPException(400, "该接口仅识别 .onnx 模型文件")
    try:
        return onnx_infer.probe_format(op)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/train/resume")
def resume_train(body: dict):
    """断点续训：复用原运行目录与原参数，恢复模型/optimizer/scheduler/epoch，
    metrics.csv 截断到恢复点后续写（界面图表连续显示）"""
    from app.core.task_manager import find_resume_weight
    task_id = (body or {}).get("task_id", "")
    t = task_manager.get(task_id)
    if t is None:
        raise HTTPException(404, "任务不存在")
    if t.status in ("running", "pending"):
        raise HTTPException(400, "任务正在运行中，无法续训")
    if t.status == "completed":
        raise HTTPException(400, "任务已完成，无需续训（可发起新训练）")
    save_dir = Path(t.save_dir)
    params_f = save_dir / "params.json"
    if not params_f.exists():
        raise HTTPException(400, f"运行目录缺少 params.json，无法续训: {save_dir}")
    try:
        params = json.loads(params_f.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(400, "params.json 解析失败，无法续训")
    kind = params.get("task") or t.kind
    if kind not in TRAIN_RUNNERS:
        raise HTTPException(400, f"该任务类型不支持续训: {kind}"
                                 f"（支持: {list(TRAIN_RUNNERS)}）")
    weight = find_resume_weight(save_dir)
    if weight is None:
        raise HTTPException(400, "未找到可恢复的权重（*_last.pth / last.pt），"
                                 "无法续训")
    # 数据集存在性预检查（给出更友好的错误）
    ds = params.get("dataset_path")
    if ds and not Path(ds).exists():
        raise HTTPException(400, f"原数据集路径已不存在: {ds}（续训需要原始数据集）")
    if params.get("device", "cpu").startswith("cuda") and \
            not torch.cuda.is_available():
        params["device"] = "cpu"
    params["resume_weight"] = str(weight)
    # 运行器解析：crowd_box 走 ultralytics 引擎（TRAIN_RUNNERS 中为 None）
    runner = TRAIN_RUNNERS[kind]
    if kind == "crowd_box":
        runner = _get_box_runner()
    if runner is None:
        raise HTTPException(400, f"该任务类型未注册训练运行器: {kind}")
    # 清理删除标记，创建新任务接管同一运行目录
    (save_dir / ".deleted").unlink(missing_ok=True)
    task = task_manager.create(t.name, kind, save_dir)
    task_manager.submit(task, runner, params)
    return {"task_id": task.id, "save_dir": str(save_dir),
            "run_name": t.name, "resume_from": str(weight),
            "kind": kind, **_queue_fields(task)}


@router.delete("/train/tasks/{task_id}")
def delete_train_task(task_id: str, artifacts: bool = False):
    """删除任务记录与日志（默认保留训练产物；artifacts=true 连同运行目录一并删除）"""
    t = task_manager.get(task_id)
    if t is None:
        raise HTTPException(404, "任务不存在")
    if t.status in ("running", "pending"):
        raise HTTPException(400, "任务运行中，请先暂停再删除")
    ok = task_manager.delete(task_id, artifacts=artifacts)
    if not ok:
        raise HTTPException(400, "删除失败")
    return {"ok": True, "artifacts_deleted": artifacts}


@router.post("/mat2density")
def mat_to_density(mat_dir: str, save_dir: str, img_dir: str = "",
                   sigma: float = 2.0, adaptive: bool = False):
    """.mat 点标注 → GT 密度图 .npy（后台任务，进度在任务列表查看）

    自动探测 ShanghaiTech(image_info) 与 annPoints 两种 .mat 格式；
    图像按同名或剥离常见标记前后缀匹配（大小写不敏感），输出 .npy
    以图像主名命名，训练配对无需重命名
    """
    md = Path(mat_dir)
    if not md.exists():
        up = UPLOAD_DATASET_DIR / mat_dir
        if up.exists():
            md = up
        else:
            raise HTTPException(400, f".mat 目录不存在: {mat_dir}")
    sd = Path(save_dir)
    if not sd.is_absolute():
        sd = RUNS_DIR / "mat2density" / save_dir
    sd.mkdir(parents=True, exist_ok=True)
    params = {"mat_dir": str(md), "save_dir": str(sd), "img_dir": img_dir,
              "sigma": sigma, "adaptive": adaptive}
    _persist_task_params(sd, params)
    task = task_manager.create(f"mat2density_{md.name}", "mat2density", sd)
    task_manager.submit(task, crowd_counting.run_mat2density, params)
    return {"task_id": task.id, "save_dir": str(sd)}


@router.post("/train/upload_run")
async def upload_train_run(file: UploadFile = File(...)):
    """上传中断训练的运行目录 zip，恢复为历史任务（可一键续训）

    zip 需含 params.json 及按原结构的训练产物（last.pth / metrics.csv 等）；
    原数据集路径仍需在本机存在，否则续训时会明确提示
    """
    import shutil
    import zipfile as _zf
    from app.core.task_manager import find_resume_weight
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "请上传 zip 格式的运行目录压缩包")
    tmp = RUNS_DIR / f"_tmp_upload_run_{uuid.uuid4().hex[:6]}.zip"
    try:
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            zf = _zf.ZipFile(tmp)
        except _zf.BadZipFile:
            raise HTTPException(400, "无效的 zip 文件")
        with zf:
            names = [n.replace("\\", "/") for n in zf.namelist()
                     if not n.startswith("__MACOSX") and not n.endswith("/")]

            def _find_params(prefix):
                cands = [n for n in names if n == f"{prefix}params.json"]
                return cands[0] if len(cands) == 1 else None

            params_name = _find_params("")
            sub = ""
            if params_name is None:
                # 单一顶层目录则下探一层（用户连文件夹一起压缩）
                tops = {n.split("/", 1)[0] for n in names}
                if len(tops) == 1:
                    sub = next(iter(tops)) + "/"
                    params_name = _find_params(sub)
            if params_name is None:
                raise HTTPException(
                    400, "zip 中未找到 params.json（请压缩运行目录本身，"
                         "需包含 params.json / last.pth 等训练产物）")
            try:
                params = json.loads(zf.read(params_name).decode("utf-8"))
            except Exception:
                raise HTTPException(400, "params.json 解析失败，无法恢复")
            kind = params.get("task")
            if kind not in TRAIN_RUNNERS:
                raise HTTPException(
                    400, f"不支持恢复的任务类型: {kind}"
                         f"（支持: {sorted(TRAIN_RUNNERS)}）")
            run_name = re.sub(r"[^\w\-]", "_", Path(file.filename).stem) \
                or "restored"
            save_dir = RUNS_DIR / kind / run_name
            if save_dir.exists():
                save_dir = RUNS_DIR / kind / f"{run_name}_{uuid.uuid4().hex[:4]}"
            save_dir.mkdir(parents=True)
            # 解压（防 zip-slip：条目必须落在 save_dir 内）
            root = save_dir.resolve()
            n_files = 0
            for info in zf.infolist():
                if info.is_dir() or info.filename.startswith("__MACOSX"):
                    continue
                rel = info.filename.replace("\\", "/")
                if sub and rel.startswith(sub):
                    rel = rel[len(sub):]
                if not rel:
                    continue
                target = (save_dir / rel).resolve()
                if root != target and root not in target.parents:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n_files += 1
    finally:
        tmp.unlink(missing_ok=True)
    # 强制重扫磁盘任务 → 出现在历史列表
    task_manager.scan_disk_tasks(force=True)
    task = task_manager.get(f"disk-{kind}-{save_dir.name}")
    return {"ok": True, "task_id": task.id if task else None,
            "run_name": save_dir.name, "kind": kind,
            "save_dir": str(save_dir), "num_files": n_files,
            "resumable": find_resume_weight(save_dir) is not None}


@router.post("/npy2png")
def npy_to_png(npy_dir: str, save_dir: str = "", mode: str = "heat",
               annotate: bool = True, img_dir: str = "",
               alpha: float = 0.67):
    """.npy 密度图 → PNG 可视化（后台任务，进度在任务列表查看）

    mode: heat 热力图（默认）/ gray 灰度 / overlay 与原图融合（需 img_dir）；
    save_dir 留空 → runs/npy2png/<npy目录名>；alpha 为融合的密度权重（0~1）
    """
    nd = Path(npy_dir)
    if not nd.exists():
        up = UPLOAD_DATASET_DIR / npy_dir
        if up.exists():
            nd = up
        else:
            raise HTTPException(400, f".npy 目录不存在: {npy_dir}")
    idir = None
    if (mode or "").lower() == "overlay":
        if not img_dir:
            raise HTTPException(400, "叠加融合模式（overlay）需要提供原图目录 img_dir")
        idir = Path(img_dir)
        if not idir.exists():
            up = UPLOAD_DATASET_DIR / img_dir
            if up.exists():
                idir = up
            else:
                raise HTTPException(400, f"原图目录不存在: {img_dir}")
    if save_dir:
        sd = Path(save_dir)
        if not sd.is_absolute():
            sd = RUNS_DIR / "npy2png" / save_dir
    else:
        sd = RUNS_DIR / "npy2png" / nd.name
    sd.mkdir(parents=True, exist_ok=True)
    params = {"npy_dir": str(nd), "save_dir": str(sd),
              "mode": mode, "annotate": annotate,
              "img_dir": str(idir) if idir else "",
              "alpha": alpha}
    _persist_task_params(sd, params)
    task = task_manager.create(f"npy2png_{nd.name}", "npy2png", sd)
    task_manager.submit(task, crowd_counting.run_npy2png, params)
    return {"task_id": task.id, "save_dir": str(sd)}


def _unique_run_dir(base_dir: Path, run_name: str) -> Tuple[str, Path]:
    """Return a unique training run name/directory.

    If the requested run name already exists (same dataset/algorithm/model
    name), append a readable timestamp. A numeric suffix handles the rare
    case where two tasks are created in the same second.
    """
    candidate = run_name
    path = Path(base_dir) / candidate
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = f"{run_name}_{stamp}"
        path = Path(base_dir) / candidate
        n = 2
        while path.exists():
            candidate = f"{run_name}_{stamp}_{n}"
            path = Path(base_dir) / candidate
            n += 1
    return candidate, path


def _queue_fields(task):
    """训练任务提交后的排队信息（queue_position>0 表示还在等待）。"""
    pos = int(getattr(task, "queue_position", 0) or 0)
    return {"queued": pos > 0, "queue_position": pos,
            "status": getattr(task, "status", "")}


@router.post("/train")
def start_train(params: TrainParams):
    if params.task not in TRAIN_RUNNERS:
        raise HTTPException(400, f"该任务暂不支持训练: {params.task}"
                                 f"（可训练: {list(TRAIN_RUNNERS)}）")
    # 路线与模型匹配校验：防止手工请求混用（产物目录按路线分列，混用会造成
    # kind 与产物不符）
    _check_route_model(params.task, params.model)
    params.dataset_path = _resolve_dataset(params.dataset_path)
    if params.task == "crowd_sdg" and (params.target_dataset_path or "").strip():
        # 单域泛化：目标数据集（跨域验证集）同样支持已上传数据集名
        params.target_dataset_path = _resolve_dataset(
            params.target_dataset_path.strip())
    if params.weight_path:
        params.weight_path = _resolve_weight(params.weight_path, params.task)
    if params.distill and params.distill_teacher_weight:
        # 蒸馏教师权重（训练产物 best.pt / 上传 .pt：训练产物名限定同任务目录）
        params.distill_teacher_weight = _resolve_weight(
            params.distill_teacher_weight, params.task)
    if params.resume_weight:
        # 断点续训权重（上传的中断 last.pth / 运行目录中的权重）
        params.resume_weight = _resolve_weight(params.resume_weight,
                                               params.task)
    if params.device.startswith("cuda") and not torch.cuda.is_available():
        params.device = "cpu"

    # DDP + 点预测模型拒绝：P2PNet / P2R / Consistent-Point / CU 训练管线均
    # 无 DP/DDP 支持，DDP 会退化为各 rank 全量重复训练（须在创建任务之前
    # 校验，否则留下 pending 孤儿任务）
    if params.multi_gpu_mode == "ddp" and params.model in ("p2pnet", "p2r",
                                                           "cp", "cu", "pet",
                                                           "cltr", "raz", "d2c"):
        raise HTTPException(
            400, "点预测模型（P2PNet / P2R / Consistent-Point / CU / PET / "
                 "CLTR）训练暂不支持 DDP 分布式模式（请选择单卡 / "
                 "DataParallel，或改用密度网模型）")
    if params.multi_gpu_mode == "ddp" and params.task == "crowd_sdg":
        raise HTTPException(
            400, "单域泛化（DCCUS / LLB_Count / MPCount）训练暂不支持 DDP"
                 "（元学习内外循环按单卡实现，请选择单卡训练）")
    # 头部框路线的多 GPU 由 ultralytics 引擎原生调度
    # （设备选「全部 GPU」即可），不走 mp.spawn 框架 DDP
    if params.multi_gpu_mode == "ddp" and params.task == "crowd_box":
        raise HTTPException(
            400, "头部框路线的多 GPU 训练由 ultralytics 引擎原生调度"
                 "（计算设备选「全部 GPU (DataParallel)」即可），"
                 "无需选择 DDP 模式")
    # DDP 需要至少 2 张 GPU；在创建任务之前拒绝，避免留下失败任务记录
    if params.multi_gpu_mode == "ddp":
        from app.services import ddp_launcher
        if ddp_launcher.gpu_count() < 2:
            raise HTTPException(
                400, "DDP 分布式至少需要 2 张 GPU；单卡请选择单卡 / "
                     "DataParallel 模式")

    run_name = params.run_name or f"{params.task}_{params.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = re.sub(r"[^\w\-]", "_", run_name)
    task_dir = RUNS_DIR / f"{params.task}"
    run_name, save_dir = _unique_run_dir(task_dir, run_name)
    params.run_name = run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "params.json").write_text(
        json.dumps(params.dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    task = task_manager.create(run_name, params.task, save_dir)
    # 头部框（ultralytics 引擎）：不区分 DP/DDP（引擎原生多卡），直接提交
    if params.task == "crowd_box":
        p = params.dict()
        task_manager.submit(task, _get_box_runner(), p)
        return {"task_id": task.id, "save_dir": str(save_dir),
                "run_name": run_name, **_queue_fields(task)}
    # 单域泛化：独立 runner（元学习 / 双分支一致性）
    if params.task == "crowd_sdg":
        p = params.dict()
        task_manager.submit(task, _get_sdg_runner(), p)
        return {"task_id": task.id, "save_dir": str(save_dir),
                "run_name": run_name, **_queue_fields(task)}
    # DDP 分布式模式（multi_gpu_mode="ddp"）：经 mp.spawn 每卡一进程
    from app.services import ddp_launcher
    p = params.dict()
    if p.get("multi_gpu_mode") == "ddp":
        n_gpu = ddp_launcher.gpu_count()
        world = int(p.get("ddp_world_size") or 0) or max(2, n_gpu)
        if n_gpu > 1:
            world = min(world, n_gpu)
        p["ddp_world_size"] = world
        task_manager.submit(task, ddp_launcher.run_ddp_train, p, params.task)
        return {"task_id": task.id, "save_dir": str(save_dir),
                "run_name": run_name, "distributed": True,
                "world_size": world, **_queue_fields(task)}
    task_manager.submit(task, TRAIN_RUNNERS[params.task], p)
    return {"task_id": task.id, "save_dir": str(save_dir),
            "run_name": run_name, **_queue_fields(task)}


@router.post("/sdg/test")
def sdg_test(body: dict):
    """单域泛化：仅测试（不训练）——加载训练权重在目标域测试集上评估泛化性

    body: weight_path（训练产物 best/last.pth）、dataset_path（测试数据集，
    路径或已上传名）、model（sdg_*，可省略，优先读权重内记录的 sdg_meta）、
    sdg_backbone / sdg_test_patch（窗口大小，0=整图）/ device
    结果：MAE / MSE + 逐图 CSV（save_dir/test_results.csv）
    """
    from app.services import sdg_service
    wp = (body or {}).get("weight_path") or ""
    ds = (body or {}).get("dataset_path") or ""
    if not wp:
        raise HTTPException(400, "请选择训练产物权重（best/last.pth）")
    wp = _resolve_weight(wp, "crowd_sdg")
    if not ds:
        raise HTTPException(400, "请选择测试数据集（其它领域的测试集）")
    ds = _resolve_dataset(ds)
    params = {"weight_path": str(wp), "dataset_path": str(ds),
              "model": (body or {}).get("model") or "sdg_dccus",
              "sdg_backbone": (body or {}).get("sdg_backbone") or None,
              "sdg_num_clusters": int((body or {}).get("sdg_num_clusters")
                                      or 4),
              "sdg_test_patch": int((body or {}).get("sdg_test_patch") or 0),
              "device": (body or {}).get("device") or "cpu"}
    if params["device"].startswith("cuda") and not torch.cuda.is_available():
        params["device"] = "cpu"
    run_name = ("sdg_test_" + Path(wp).parent.name + "_" +
                datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_name = re.sub(r"[^\w\-]", "_", run_name)
    save_dir = RUNS_DIR / "sdg_test" / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    task = task_manager.create(run_name, "sdg_test", save_dir)
    task_manager.submit(task, sdg_service.run_sdg_test, params)
    return {"task_id": task.id, "save_dir": str(save_dir),
            "run_name": run_name}


@router.post("/test")
def start_test(body: dict):
    """独立测试模式：算法 + 权重 + 数据集 + 可选切块窗口。

    与训练解耦，只做推理评估；结果写入 runs/test/<name>/test_results.csv，
    任务日志通过通用 /train/tasks/{id}/log 读取。
    """
    body = body or {}
    task_key = (body.get("task") or "crowd_sdg").strip()
    model = (body.get("model") or "").strip()
    if task_key not in ("crowd_sdg", "crowd_density", "crowd_point",
                        "crowd_box"):
        raise HTTPException(400, "测试模式支持 crowd_sdg / crowd_density / "
                                 "crowd_point / crowd_box")
    if not model:
        raise HTTPException(400, "请选择测试算法 / 模型")
    _check_route_model(task_key, model)
    wp = (body.get("weight_path") or "").strip()
    if not wp:
        raise HTTPException(400, "请选择训练产物权重或上传权重")
    wp = _resolve_weight(wp, task_key)
    ds = (body.get("dataset_path") or "").strip()
    if not ds:
        raise HTTPException(400, "请选择测试数据集")
    ds = _resolve_dataset(ds)
    device = (body.get("device") or "cpu").strip() or "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    split = (body.get("split") or "test").strip().lower()
    if split not in ("test", "train"):
        split = "test"
    patch = max(0, int(body.get("patch_size") or 0))
    save_outputs = bool(body.get("save_outputs"))
    point_format = (body.get("point_format") or "txt").strip().lower()
    if point_format not in ("txt", "mat"):
        point_format = "txt"
    output_dir = ""
    if save_outputs:
        raw_out = (body.get("output_dir") or "").strip()
        if raw_out:
            out_p = Path(raw_out).expanduser()
            if not out_p.is_absolute():
                out_p = RUNS_DIR / "test" / out_p
        else:
            out_p = None
        if out_p is not None:
            try:
                out_p.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise HTTPException(400, f"无法创建保存目录: {out_p} ({exc})")
            output_dir = str(out_p)
    params = {"task": task_key, "model": model, "weight_path": str(wp),
              "dataset_path": str(ds), "split": split,
              "patch_size": patch, "sdg_test_patch": patch,
              "device": device,
              "sdg_backbone": body.get("sdg_backbone") or None,
              "sdg_num_clusters": int(body.get("sdg_num_clusters") or 4),
              "conf_threshold": float(body.get("conf_threshold") or 0.5),
              "save_outputs": save_outputs,
              "output_dir": output_dir,
              "point_format": point_format}
    run_name = (f"test_{task_key.replace('crowd_', '')}_{model}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_name = re.sub(r"[^\w\-]", "_", run_name)
    save_dir = RUNS_DIR / "test" / run_name
    if save_dir.exists():
        save_dir = RUNS_DIR / "test" / f"{run_name}_{uuid.uuid4().hex[:4]}"
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    task = task_manager.create(run_name, "test", save_dir)
    if task_key == "crowd_sdg":
        from app.services import sdg_service
        runner = sdg_service.run_sdg_test
    elif task_key == "crowd_density":
        runner = crowd_counting.run_density_test
    elif task_key == "crowd_box":
        from app.services import yolo_service
        runner = yolo_service.run_box_test
    else:
        runner = crowd_counting.run_point_test
    task_manager.submit(task, runner, params)
    return {"task_id": task.id, "save_dir": str(save_dir),
            "run_name": run_name}


@router.post("/pick-dir")
def pick_dir(body: dict = None):
    """Open a native folder picker on the server host and return its path.

    The web UI cannot read an absolute local path from a browser folder
    input, so the local deployment opens a dialog in a child process and
    returns the selected directory ("" when the user cancels).
    """
    import os
    import subprocess
    import sys
    initial = ((body or {}).get("initial_dir") or "").strip()
    script = (
        "import sys\n"
        "try:\n"
        "    import tkinter as tk\n"
        "    from tkinter import filedialog\n"
        "    root = tk.Tk()\n"
        "    root.withdraw()\n"
        "    root.attributes('-topmost', True)\n"
        "    start = sys.argv[1] if len(sys.argv) > 1 else None\n"
        "    p = filedialog.askdirectory(initialdir=start or None,\n"
        "                                title='选择保存目录', mustexist=False)\n"
        "    root.destroy()\n"
        "    print(p or '')\n"
        "except Exception as exc:\n"
        "    print('__ERROR__' + str(exc))\n"
    )
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run([sys.executable, "-c", script, initial],
                              capture_output=True, text=True, timeout=600,
                              encoding="utf-8", errors="replace", env=env)
    except Exception as exc:
        raise HTTPException(500, f"无法打开目录选择窗口: {exc}")
    out = (proc.stdout or "").strip().splitlines()
    path = out[-1].strip() if out else ""
    if path.startswith("__ERROR__"):
        raise HTTPException(
            500, "无法打开目录选择窗口（当前环境可能没有图形界面），"
                 "请手动输入保存路径: " + path[len("__ERROR__"):])
    return {"path": path}


@router.get("/train/tasks")
def list_train_tasks():
    return task_manager.list()


@router.get("/train/tasks/{task_id}")
def get_train_task(task_id: str):
    t = task_manager.get(task_id)
    if t is None:
        raise HTTPException(404, "任务不存在")
    return t.to_dict()


@router.post("/train/tasks/{task_id}/stop")
def stop_train_task(task_id: str):
    t = task_manager.get(task_id)
    if t is None:
        raise HTTPException(404, "任务不存在")
    ok = task_manager.stop(task_id)
    return {"ok": ok, "message": "已发送停止信号" if ok else f"任务状态: {t.status}，无法停止"}


@router.get("/train/tasks/{task_id}/log")
def get_train_log(task_id: str, tail: int = 100):
    t = task_manager.get(task_id)
    if t is None:
        raise HTTPException(404, "任务不存在")
    return {"log": task_manager.read_log(task_id, tail)}


@router.get("/train/tasks/{task_id}/download")
def download_train_artifact(task_id: str, kind: str = "log"):
    """下载训练产物（附件方式，任务管理器全局共享）

    kind: log（训练日志）/ best（最优权重）/ last（最末权重）/
          params（训练参数 params.json）/ metrics（指标记录 metrics.csv）
    """
    from fastapi.responses import FileResponse
    t = task_manager.get(task_id)
    if t is None:
        raise HTTPException(404, "任务不存在")
    save_dir = Path(t.save_dir)

    def _pick(cands):
        for c in cands:
            if not c:
                continue
            p = Path(c)
            if not p.is_absolute():
                p = save_dir / p
            if p.is_file():
                return p
        return None

    result = t.result or {}
    if kind == "log":
        p = _pick(["train_log.txt"])
    elif kind == "best":
        # 结果路径优先；缺失时按新/旧命名与 YOLO 产物兜底查找
        p = _pick([result.get("best_weight")]) or find_best_weight(save_dir)
    elif kind == "last":
        p = _pick([result.get("last_weight")]) or find_last_weight(save_dir)
    elif kind == "params":
        p = _pick(["params.json"])
    elif kind == "metrics":
        p = _pick(["metrics.csv"])
    elif kind in ("test_csv", "results"):
        p = _pick([result.get("per_image_csv"), "test_results.csv"])
    else:
        raise HTTPException(400, f"未知产物类型: {kind}"
                                 f"（可选: log / best / last / params / metrics）")
    if p is None:
        raise HTTPException(404, f"产物不存在（{kind}）：该任务尚未产出此文件")
    return FileResponse(str(p), filename=p.name,
                        media_type="application/octet-stream")


# ------------------------- 推理 -------------------------

async def _read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    try:
        return Image.open(__import__("io").BytesIO(data))
    except Exception:
        raise HTTPException(400, "无法解析上传的图片文件")


def _save_result(vis_b64: str, prefix: str) -> str:
    import base64
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    path = RESULTS_DIR / fname
    path.write_bytes(base64.b64decode(vis_b64))
    return f"/results/{fname}"


@router.post("/infer")
async def infer(
    images: List[UploadFile] = File(...),
    task: str = Form(...),
    model: str = Form(...),
    weight_path: str = Form(""),
    pretrained: bool = Form(False),
    device: str = Form("cpu"),
    conf_threshold: float = Form(None),
    save_result: bool = Form(False),
    save_dir: str = Form(""),
):
    """单张/批量图像推理；多张时逐张推理并打包 zip 供下载。

    密度网：返回人数 + 密度热力图（叠加图 / 纯热力图）；
    P2PNet：返回人数 + 预测点可视化（红色圆点），conf_threshold 为点置信度阈值。
    """
    if task not in CROWD_TASKS:
        raise HTTPException(400, f"未知任务: {task}")
    # 未显式传阈值时按模型取官方默认：CLTR 验证/推理使用 0.35，
    # 其余点模型/密度网沿用 0.5
    if conf_threshold is None:
        conf_threshold = 0.35 if model == "cltr" else 0.5
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    wp = _resolve_weight(weight_path, task) if weight_path else None
    out_dir = Path(save_dir) if save_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    previews = []
    batch_vis = []   # 批量模式：可视化图直接写入 zip，不再依赖「保存结果」

    def _finalize(f, result):
        """逐图结果收尾：落盘保存 / 批量留档与裁剪 / 汇总（两路线共用）"""
        result["filename"] = f.filename
        if save_result or out_dir is not None:
            b64 = result.get("visualization") or result.get("density_map")
            if b64:
                fname = f"{task}_{Path(f.filename).stem}_{uuid.uuid4().hex[:6]}.png"
                target = out_dir if out_dir is not None else RESULTS_DIR
                saved = target / fname
                saved.write_bytes(base64.b64decode(b64))
                result["saved_path"] = str(saved)
                result["saved_url"] = (_served_url(saved) if out_dir is not None
                                       else f"/results/{fname}")
        # 批量时先留档可视化字节（供 zip），再裁掉 base64 大字段
        if len(images) > 1:
            b64 = result.get("visualization") or result.get("density_map")
            if b64:
                if len(previews) < 5:
                    from app.services.visualize_service import _make_preview
                    previews.append(_make_preview(
                        b64, f.filename or Path(f.filename).stem,
                        result.get("count", 0)))
                batch_vis.append((f"{Path(f.filename).stem}.png",
                                  base64.b64decode(b64)))
            for k in ("visualization", "density_map"):
                result.pop(k, None)
        results.append(result)

    # ONNX 部署产物 / 上传的 onnx：onnxruntime 子进程批量推理（模型与设备
    # 选项不参与 —— 输入契约在导出时已固化；单次会话处理全部图像）
    if wp and str(wp).lower().endswith(".onnx"):
        from app.services import onnx_infer
        try:
            onnx_results = onnx_infer.run_infer_batch(
                [await _read_image(f) for f in images], wp,
                conf_threshold=conf_threshold)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if len(onnx_results) != len(images):
            raise HTTPException(400, "ONNX 推理结果数量与图像数不一致")
        for f, result in zip(images, onnx_results):
            _finalize(f, result)
    else:
        for f in images:
            img = await _read_image(f)
            try:
                if task == "crowd_box":
                    # 头部框路线：ultralytics 引擎（.pt 权重路径直接给引擎）
                    from app.services import yolo_service
                    result = yolo_service.headbox_infer(
                        img, model, weight_path=wp, conf_threshold=conf_threshold,
                        device=device, pretrained=pretrained)
                else:
                    result = crowd_counting.crowd_infer(
                        img, model_name=model, weight_path=wp, device=device,
                        pretrained=pretrained, conf_threshold=conf_threshold)
            except (ValueError, KeyError) as e:
                raise HTTPException(400, str(e))
            except HTTPException:
                raise
            except OSError as e:
                try:
                    _cache = Path(torch.hub.get_dir()) / "checkpoints"
                except Exception:
                    _cache = "~/.cache/torch/hub/checkpoints"
                raise HTTPException(
                    400, f"预训练权重下载失败（网络不可用）：{e}。"
                         f"若多次失败可手动下载权重（地址见上方错误信息）放到 "
                         f"{_cache} 后重试，"
                         f"或改用本地预训练模型（ghostnetv2 系列 / p2pnet），"
                         f"或上传权重文件")
            _finalize(f, result)

    if len(results) == 1:
        return results[0]

    import zipfile as _zf
    zip_name = f"batch_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.zip"
    zip_path = RESULTS_DIR / zip_name
    with _zf.ZipFile(zip_path, "w") as zf:
        # 可视化图直接入 zip（不依赖「保存预测结果」勾选）
        for fname, data in batch_vis:
            zf.writestr(fname, data)
        summary = [{k: v for k, v in r.items()} for r in results]
        zf.writestr("results.json", json.dumps(summary, ensure_ascii=False, indent=2))
    return {"batch": True, "num_images": len(results), "results": results,
            "preview_images": previews,
            "save_dir": str(out_dir) if out_dir is not None else "",
            "zip_url": f"/results/{zip_name}"}


# ------------------------- 摄像头实时计数（OpenCV） -------------------------

@router.post("/camera/start")
def camera_start(params: dict):
    """启动摄像头实时计数会话（人群计数）"""
    from app.services import camera_service
    task = params.get("task")
    if task not in camera_service.SUPPORTED_TASKS["pytorch"]:
        raise HTTPException(
            400, f"摄像头实时计数不支持任务: {task}（支持: "
                 f"{sorted(camera_service.SUPPORTED_TASKS['pytorch'])}）")
    if not params.get("model"):
        raise HTTPException(400, "请先选择网络模型")
    _check_route_model(task, params["model"])
    wp = params.get("weight_path") or ""
    if wp.lower().endswith(".onnx"):
        raise HTTPException(
            400, "摄像头实时检测暂不支持 ONNX 权重，请选择 .pth/.pt 权重")
    if wp:
        try:
            params = {**params, "weight_path": _resolve_weight(wp, task)}
        except HTTPException:
            raise
    # CLTR 与自定义点模型没有内置权重，必须在启动摄像头前给出明确提示，
    # 否则会在推理线程里逐帧失败
    model = params.get("model") or ""
    if model == "cltr" and not wp:
        raise HTTPException(400, "CLTR 摄像头推理需要权重：请选择训练产物 / 上传 .pth")
    from app.core.registry import is_custom_model_for
    if is_custom_model_for(model, "crowd_point") and not wp:
        raise HTTPException(
            400, f"自定义点模型 {model} 摄像头推理需要权重：请选择训练产物 / 上传 .pth")
    device = params.get("device") or "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        params = {**params, "device": "cpu"}
    try:
        session = camera_service.manager.start("pytorch", params)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return session.status()


@router.get("/camera/stream")
def camera_stream():
    """MJPEG 实时画面流（<img> 标签直接渲染）"""
    from app.services import camera_service
    return camera_service.stream_response()


@router.post("/camera/stop")
def camera_stop():
    from app.services import camera_service
    camera_service.manager.stop()
    return {"ok": True}


@router.get("/camera/status")
def camera_status():
    from app.services import camera_service
    return camera_service.manager.status()


# ------------------------- 数据读取预处理（全局模块） -------------------------

@router.get("/preprocess/datasets")
def preprocess_datasets():
    """返回支持的常用数据集预设清单（含目录结构 tree 与标注说明）"""
    from app.services.preprocess_service import DATASET_PRESETS
    out = []
    for key, meta in DATASET_PRESETS.items():
        out.append({"key": key, "label": meta["label"],
                    "annotation": meta["annotation"],
                    "tree": meta["tree"]})
    return out


@router.post("/preprocess")
async def run_preprocess(
    dataset_path: str,
    preset: str = "custom_txt",
    box_source: str = "adaptive",
    yolo_weight: str = "",
    yolo_conf: float = 0.25,
    adaptive_k: int = 4,
    adaptive_scale: float = 0.4,
    adaptive_default: float = 20,
    save_dir: str = "",
):
    """启动数据读取预处理后台任务（常用数据集 → YOLO 头部框格式）

    参数通过查询参数（query string）传入，与前端 `/api/crowd/preprocess?…` 对应。
    box_source: 无框标注时的策略
      - adaptive: k 近邻自适应推框（从头部坐标点大致推出头部坐标框）
      - yolo_detect: 用已训练 YOLO 头部框模型对图像检测取框（需 yolo_weight）
      - skip: 跳过无框标注样本
    """
    from app.services.preprocess_service import run_preprocess as _run

    ds = Path(dataset_path)
    if not ds.exists():
        up = UPLOAD_DATASET_DIR / dataset_path
        if up.exists():
            ds = up
        else:
            raise HTTPException(400, f"数据集不存在: {dataset_path}")
    ds = ds.resolve()   # 统一为绝对路径，确保默认输出与源数据集同目录

    if save_dir:
        sd = Path(save_dir)
        if not sd.is_absolute():
            sd = RUNS_DIR / "preprocess" / save_dir
    else:
        # 默认与源数据集同级目录：<数据集目录名>_box
        # （结构：train/test × images/labels + data.yaml，可直接用于坐标框训练）
        sd = ds.parent / f"{ds.name}_box"
    if sd.resolve() == ds.resolve():
        raise HTTPException(400, "输出目录不能与源数据集相同（请留空使用 "
                                 "<数据集名>_box，或指定其它目录）")

    # yolo_weight 解析（可选：训练产物路径）
    yw = ""
    if yolo_weight:
        try:
            yw = _resolve_weight(yolo_weight, "crowd_box")
        except HTTPException:
            yw = yolo_weight  # 直接路径

    params = {
        "dataset_path": str(ds),
        "preset": preset,
        "box_source": box_source,
        "yolo_weight": yw,
        "yolo_conf": yolo_conf,
        "adaptive_k": adaptive_k,
        "adaptive_scale": adaptive_scale,
        "adaptive_default": adaptive_default,
        "save_dir": str(sd),
    }
    _persist_task_params(sd, params)
    task = task_manager.create(f"preprocess_{ds.name}", "preprocess", sd)
    task_manager.submit(task, _run, params)
    return {"task_id": task.id, "save_dir": str(sd)}


# ------------------------- 可视化模块（全局模块） -------------------------

def _visualize_output_dir(save_dir: str = "") -> Path:
    """Resolve a visualization output directory.

    Relative values stay under runs/visualize so the existing /runs static
    mount can serve them; absolute paths are honoured for explicit server
    directories.
    """
    if save_dir:
        p = Path(save_dir)
        if not p.is_absolute():
            p = RUNS_DIR / "visualize" / p
    else:
        p = RUNS_DIR / "visualize"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _served_url(path: Path) -> str:
    try:
        rel = Path(path).resolve().relative_to(RUNS_DIR.resolve())
        return f"/runs/{rel.as_posix()}"
    except (ValueError, OSError):
        return ""


@router.post("/visualize/single")
async def visualize_single(
    mode: str = Form(...),
    route: str = Form("crowd_density"),
    image: UploadFile = File(...),
    label_file: UploadFile = File(None),
    label_format: str = Form("npy"),
    model: str = Form(""),
    weight_path: str = Form(""),
    pretrained: bool = Form(False),
    device: str = Form("cpu"),
    conf_threshold: float = Form(0.5),
    save_result: bool = Form(False),
    save_dir: str = Form(""),
):
    """单张图像可视化（同步请求，返回 base64 可视化图）

    mode: "label"（标签可视化） | "model"（模型推理可视化）
    route: crowd_density / crowd_point / crowd_box
    label_file: 标签文件（label 模式必填：.npy / .mat / .txt / .xml）
    """
    from app.services import visualize_service as vs

    if mode not in ("label", "model"):
        raise HTTPException(400, f"无效模式: {mode}（支持 label / model）")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # 读取图像
    data = await image.read()
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "无法解析上传的图片文件")

    try:
        if mode == "label":
            if not label_file:
                raise HTTPException(400, "标签可视化模式需要上传标签文件")
            # 保存标签到临时文件
            lbl_data = await label_file.read()
            lbl_suffix = Path(label_file.filename or "").suffix.lower()
            tmp_lbl = RESULTS_DIR / f"_tmp_vis_{uuid.uuid4().hex[:8]}{lbl_suffix}"
            tmp_lbl.write_bytes(lbl_data)
            try:
                # 标签格式解析（与前端约定一致：路线优先，头部框路线按显式格式）
                if route == "crowd_density" or label_format == "npy":
                    if lbl_suffix != ".npy":
                        raise HTTPException(
                            400, f"密度图路线的标签文件应为 .npy 密度图"
                                 f"（当前: {lbl_suffix or '无后缀'}）")
                    result = vs.vis_label_density(img, tmp_lbl)
                elif route == "crowd_point" or label_format == "mat":
                    if lbl_suffix != ".mat":
                        raise HTTPException(
                            400, f"点预测路线的标签文件应为 .mat 点标注"
                                 f"（当前: {lbl_suffix or '无后缀'}）")
                    result = vs.vis_label_points(img, tmp_lbl)
                elif label_format == "voc_xml":
                    if lbl_suffix != ".xml":
                        raise HTTPException(
                            400, f"VOC 标签文件应为 .xml"
                                 f"（当前: {lbl_suffix or '无后缀'}）")
                    result = vs.vis_label_boxes_voc(img, tmp_lbl)
                else:
                    if lbl_suffix != ".txt":
                        raise HTTPException(
                            400, f"YOLO 标签文件应为 .txt"
                                 f"（当前: {lbl_suffix or '无后缀'}）")
                    result = vs.vis_label_boxes_yolo(img, tmp_lbl)
            finally:
                tmp_lbl.unlink(missing_ok=True)
        else:
            if not model:
                raise HTTPException(400, "模型推理可视化模式需要选择模型")
            wp = _resolve_weight(weight_path, route) if weight_path else None
            result = vs.vis_model(img, route, model, weight_path=wp,
                                  device=device, pretrained=pretrained,
                                  conf_threshold=conf_threshold)
        result["filename"] = image.filename
        if save_result or save_dir:
            target = _visualize_output_dir(save_dir)
            stem = Path(image.filename or "image").stem
            b64 = result.get("visualization") or result.get("density_map")
            if b64:
                out = target / f"{stem}_vis.png"
                out.write_bytes(base64.b64decode(b64))
                result["saved_path"] = str(out)
                result["saved_url"] = _served_url(out)
        return result
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except OSError as e:
        # 预训练权重下载失败（网络不可用）：与推理接口一致的友好提示
        try:
            _cache = Path(torch.hub.get_dir()) / "checkpoints"
        except Exception:
            _cache = "~/.cache/torch/hub/checkpoints"
        raise HTTPException(
            400, f"预训练权重下载失败（网络不可用）：{e}。"
                 f"若多次失败可手动下载权重（地址见上方错误信息）放到 "
                 f"{_cache} 后重试，或上传权重文件")
    except Exception as e:
        raise HTTPException(500, f"可视化失败: {e}")


@router.post("/visualize/batch")
async def visualize_batch(
    mode: str = Form(...),
    route: str = Form("crowd_density"),
    image_dir: str = Form(...),
    label_dir: str = Form(""),
    label_format: str = Form("npy"),
    model: str = Form(""),
    weight_path: str = Form(""),
    device: str = Form("cpu"),
    conf_threshold: float = Form(0.5),
    save_dir: str = Form(""),
    save_result: bool = Form(True),
):
    """批量可视化（后台任务，处理目录下所有图像）

    mode: "label"（标签可视化） | "model"（模型推理可视化）
    """
    from app.services.visualize_service import run_visualize_batch

    if mode not in ("label", "model"):
        raise HTTPException(400, f"无效模式: {mode}")
    # 标签格式按路线推导（与单张接口一致，防手工调用错配）
    if mode == "label":
        if route == "crowd_density":
            label_format = "npy"
        elif route == "crowd_point":
            label_format = "mat"
        if not label_dir:
            raise HTTPException(400, "标签可视化模式需要提供标签目录 label_dir")

    # 解析图像目录
    idir = Path(image_dir)
    if not idir.exists():
        up = UPLOAD_DATASET_DIR / image_dir
        if up.exists():
            idir = up
        else:
            raise HTTPException(400, f"图像目录不存在: {image_dir}")

    # 解析标签目录
    ldir = None
    if mode == "label" and label_dir:
        ldir = Path(label_dir)
        if not ldir.exists():
            up = UPLOAD_DATASET_DIR / label_dir
            if up.exists():
                ldir = up
            else:
                raise HTTPException(400, f"标签目录不存在: {label_dir}")

    # 解析模型权重
    wp = ""
    if mode == "model" and weight_path:
        try:
            wp = _resolve_weight(weight_path, route)
        except HTTPException:
            wp = weight_path

    # 输出目录
    if save_dir:
        sd = Path(save_dir)
        if not sd.is_absolute():
            sd = RUNS_DIR / "visualize" / save_dir
    else:
        sd = RUNS_DIR / "visualize" / f"{idir.name}_{mode}"

    params = {
        "mode": mode, "route": route,
        "image_dir": str(idir),
        "label_dir": str(ldir) if ldir else "",
        "label_format": label_format,
        "model": model, "weight_path": wp,
        "device": device, "conf_threshold": conf_threshold,
        "save_dir": str(sd),
        "output_dir": str(sd),
        "save_result": bool(save_result),
    }
    _persist_task_params(sd, params)
    task = task_manager.create(f"vis_{idir.name}", "visualize", sd)
    task_manager.submit(task, run_visualize_batch, params)
    return {"task_id": task.id, "save_dir": str(sd)}


@router.post("/visualize/upload_batch")
async def visualize_upload_batch(
    mode: str = Form("model"),
    route: str = Form("crowd_density"),
    images: List[UploadFile] = File(...),
    model: str = Form(""),
    weight_path: str = Form(""),
    device: str = Form("cpu"),
    conf_threshold: float = Form(0.5),
    save_result: bool = Form(True),
    save_dir: str = Form(""),
):
    """Visualize a browser-selected image folder (model mode).

    The browser cannot expose a real directory path, so uploaded images are
    processed directly. The first five previews and a ZIP are returned.
    """
    from app.services import visualize_service as vs

    if mode != "model":
        raise HTTPException(400, "本地文件夹上传目前仅支持模型可视化；"
                                 "标签可视化请使用服务器目录批量方式")
    if not model:
        raise HTTPException(400, "模型可视化需要选择模型")
    if not images:
        raise HTTPException(400, "未选择图像")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    wp = _resolve_weight(weight_path, route) if weight_path else None
    if not save_result:
        out_dir = RESULTS_DIR / f"_tmp_vis_{uuid.uuid4().hex[:8]}"
    else:
        out_dir = _visualize_output_dir(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, previews = [], []
    failed = 0
    for f in images:
        try:
            data = await f.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            r = vs.vis_model(img, route, model, weight_path=wp,
                             device=device, conf_threshold=conf_threshold)
            b64 = r.get("visualization")
            vis_file = ""
            if b64:
                out = out_dir / f"{Path(f.filename or 'image').stem}_vis.png"
                out.write_bytes(base64.b64decode(b64))
                vis_file = str(out)
                if len(previews) < vs.MAX_PREVIEW_IMAGES:
                    previews.append(vs._make_preview(
                        b64, f.filename or out.name, r.get("count", 0)))
            results.append({"filename": f.filename, "count": r.get("count", 0),
                            "vis_file": vis_file})
        except Exception as e:
            failed += 1
            results.append({"filename": f.filename, "error": str(e)})

    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_name = (f"vis_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                f"{uuid.uuid4().hex[:6]}.zip")
    zip_path = RESULTS_DIR / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out_dir.glob("*_vis.png"):
            zf.write(f, f.name)
        zf.write(out_dir / "results.json", "results.json")
    result = {"batch": True, "num_images": len(images),
              "processed": len(results) - failed, "failed": failed,
              "preview_images": previews,
              "zip_url": f"/results/{zip_name}",
              "save_dir": str(out_dir) if save_result else ""}
    if not save_result:
        shutil.rmtree(out_dir, ignore_errors=True)
    return result
