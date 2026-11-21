# -*- coding: utf-8 -*-
"""人群计数自动训练和推理框架 — FastAPI 入口

启动: python main.py  →  http://127.0.0.1:8060
"""
import json
import shutil
import sys
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from config import (STATIC_DIR, RESULTS_DIR, UPLOAD_DATASET_DIR,
                    UPLOAD_MODEL_DIR, WEIGHT_EXTS, IMAGE_EXTS, CUSTOM_MODELS_DIR)
from app.core.backend_log import install as install_backend_log
from app.core.registry import scan_custom_models

# proxy.json 须在路由 import 前应用（import 可能触发权重下载）
from app.core.proxy import init as _init_proxy
_init_proxy()

# 缓存目录环境变量须在 torch import 前设置（import 时固化）
from app.core.model_cache import init as _init_model_cache
_init_model_cache()

_available_routers = []

try:
    from app.routers import crowd
    _available_routers.append(crowd)
except ImportError as e:
    print(f"[WARN] Crowd counting router unavailable: {e}")

try:
    from app.routers import system_monitor
    _available_routers.append(system_monitor)
except ImportError as e:
    print(f"[WARN] System monitor router unavailable: {e}")

# stdout/stderr 分流进后台日志缓冲，须在 uvicorn 日志配置前安装
install_backend_log()

# lifespan（on_event 已废弃）
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app):
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    names = scan_custom_models(CUSTOM_MODELS_DIR)
    print(f"[startup] 自定义模型: {names or '无'}")
    yield
    # 优雅停机（H6）：通知运行中的训练任务停止并限时等待，
    # 降低守护线程被硬杀导致 checkpoint 写坏的概率（保存端另有原子写兜底）
    from app.core.task_manager import task_manager
    task_manager.shutdown(timeout=10)


app = FastAPI(
    title="人群计数自动训练和推理框架",
    description="人群计数领域自动训练与推理平台（密度图估计 + 点预测）",
    version="1.0.0",
    lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"])


# 静态资源用 no-cache（非 no-store）：ETag 协商校验，前端更新后立即生效
@app.middleware("http")
async def _static_no_cache(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


for _router_mod in _available_routers:
    app.include_router(_router_mod.router)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
# 训练产物（checkpoint/日志/曲线图），供前端下载
from config import RUNS_DIR as _RUNS_DIR
app.mount("/runs", StaticFiles(directory=str(_RUNS_DIR)), name="runs")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ------------------------- 上传 -------------------------

@app.post("/api/upload/model")
async def upload_model(file: UploadFile = File(...)):
    """上传模型权重文件（.pth/.pt 等）"""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in WEIGHT_EXTS:
        raise HTTPException(400, f"不支持的权重格式: {suffix}，支持: {sorted(WEIGHT_EXTS)}")
    # 只留文件名，防 multipart 文件名里带路径
    safe_name = Path(file.filename).name
    dest = UPLOAD_MODEL_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    await run_in_threadpool(_save_uploaded_file, file, dest)
    return {"ok": True, "name": dest.name, "path": str(dest),
            "size_mb": round(dest.stat().st_size / 1048576, 2)}


# 解压总量/条目数上限（zip bomb 防护；人群计数数据集可到十几 GB）
_MAX_ZIP_UNCOMPRESSED = 50 * 1024 ** 3
_MAX_ZIP_ENTRIES = 200000


def _save_uploaded_file(file: UploadFile, dest: Path):
    """在线程池中把上传流写入临时 zip。"""
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)


def _extract_dataset_zip(tmp_zip: Path, dest_dir: Path):
    """校验并解压数据集 zip，展开单一顶层目录（在线程池执行）。"""
    with zipfile.ZipFile(tmp_zip) as zf:
        infos = zf.infolist()
        if len(infos) > _MAX_ZIP_ENTRIES:
            raise ValueError(
                f"zip 条目数 {len(infos)} 超过上限 {_MAX_ZIP_ENTRIES}，已拒绝")
        total = sum(i.file_size for i in infos)
        if total > _MAX_ZIP_UNCOMPRESSED:
            raise ValueError(
                f"数据集解压后总量 {total / 1024 ** 3:.1f} GB 超过上限 "
                f"{_MAX_ZIP_UNCOMPRESSED // 1024 ** 3} GB，已拒绝")
        zf.extractall(dest_dir)

    # 单一顶层目录 → 下探一层；跳过 __MACOSX 等辅助项，冲突名加后缀
    entries = [p for p in dest_dir.iterdir() if not p.name.startswith("__")]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for p in list(inner.iterdir()):
            if p.name.startswith("__"):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
                continue
            dst = dest_dir / p.name
            if dst.exists():
                i = 1
                while (dest_dir / f"{p.name}_{i}").exists():
                    i += 1
                dst = dest_dir / f"{p.name}_{i}"
            shutil.move(str(p), str(dst))
        try:
            inner.rmdir()
        except OSError:
            pass


def _ensure_dataset_not_in_use(target: Path):
    """数据集被运行中/排队任务占用时抛 400（删除与覆盖上传共用）。

    任务可能使用 dataset_path（训练）、dataset_dir（上传后处理）、
    npy_dir/mat_dir/img_dir（密度图工具）等不同字段；只要目标数据集
    与这些路径存在包含关系，就拒绝删除。
    """
    from app.core.task_manager import task_manager
    target = target.resolve()
    path_keys = ("dataset_path", "dataset_dir", "npy_dir", "mat_dir",
                 "img_dir", "image_dir", "label_dir", "vis_dir")
    for t in task_manager.list():
        # queued 是任务等待训练槽位的真实状态，必须一并保护
        if t["status"] not in ("running", "queued", "pending"):
            continue
        try:
            params = json.loads(
                (Path(t["save_dir"]) / "params.json")
                .read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in path_keys:
            raw = params.get(key) or ""
            if not raw:
                continue
            try:
                p = Path(raw).resolve()
            except (OSError, ValueError):
                continue
            if p == target or target in p.parents or p in target.parents:
                raise HTTPException(
                    400, f"数据集正在被任务使用: {t['name']}（{key}={raw}），"
                         f"请先停止该任务")


@app.post("/api/upload/dataset")
async def upload_dataset(file: UploadFile = File(...),
                         mat_to_npy: bool = Form(False),
                         sigma: float = Form(2.0),
                         adaptive: bool = Form(False),
                         vis_png: bool = Form(False),
                         vis_dir: str = Form(""),
                         convert_txt: bool = Form(True)):
    """上传数据集 zip 包（人群计数格式）

    mat_to_npy: 上传后 GT 点标注（.txt/.mat）→ .npy 密度图后台任务
                （仅删除已成功生成对应 .npy 的源标注；转换失败/无配对
                图像的 .mat 会保留，避免数据丢失）
    vis_png:    转换同时将 .npy 密度图可视化为 .png（vis_dir 可选保存目录，
                留空 → <数据集>/vis_png/<Train|Test>）
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "请上传 zip 格式的数据集压缩包")
    name = Path(file.filename).stem
    tmp_zip = UPLOAD_DATASET_DIR / f"_tmp_{uuid.uuid4().hex[:8]}_{name}.zip"
    await run_in_threadpool(_save_uploaded_file, file, tmp_zip)
    dest_dir = None
    try:
        # 数据集名按最终标注格式加后缀（便于区分同名不同格式的数据集）：
        #   mat_to_npy → _npy；convert_txt → _txt；保留原格式 → 按包内 GT
        #   扩展名取 _mat / _npy / _txt（含 .mat 优先，代表混合/原始标注）
        if mat_to_npy:
            _fmt = "npy"
        elif convert_txt:
            _fmt = "txt"
        else:
            try:
                with zipfile.ZipFile(tmp_zip) as _zf:
                    _gt_exts = {Path(n).suffix.lower()
                                for n in _zf.namelist()
                                if n.lower().endswith((".mat", ".txt", ".npy"))}
            except Exception:
                _gt_exts = set()
            _fmt = ("mat" if ".mat" in _gt_exts
                    else "npy" if ".npy" in _gt_exts
                    else "txt" if ".txt" in _gt_exts else "")
        if _fmt and not name.lower().endswith(f"_{_fmt}"):
            name = f"{name}_{_fmt}"
        # 同名数据集不覆盖：自动追加 _1 / _2 … 后缀区分（原子占位）
        from app.core.dataset_naming import unique_dataset_name
        dest_dir = await run_in_threadpool(
            unique_dataset_name, UPLOAD_DATASET_DIR, name)
        try:
            await run_in_threadpool(_extract_dataset_zip, tmp_zip, dest_dir)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except zipfile.BadZipFile:
            raise HTTPException(400, "无效的 zip 文件")
        except Exception:
            shutil.rmtree(dest_dir, ignore_errors=True)
            dest_dir = None
            raise
    except HTTPException:
        if dest_dir is not None:
            shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    finally:
        tmp_zip.unlink(missing_ok=True)

    # 实际数据集名（同名上传时带 _1 / _2 后缀）
    actual_name = dest_dir.name

    # 常用数据集自动统一为 Train/Images + GT(.txt) 格式（与下载数据集
    # 一致；命中 ShanghaiTech/QNRF/NWPU/UCF_CC_50 布局时原地转换，
    # 转换成功后自动清理原始目录与标注、仅保留统一结构，
    # JHU 原生可读 / 已统一则跳过；线程池执行避免 .mat 解析阻塞事件循环）
    unified = None
    unified_log = []
    try:
        from functools import partial
        from app.services.dataset_unify import unify_dataset_if_common
        # convert_txt=True：GT 统一转写为 .txt；False：保留原始 .mat/.txt
        # 格式（目录仍统一为 Train/Images + GT/）
        unified = await run_in_threadpool(
            partial(unify_dataset_if_common, dest_dir, unified_log.append,
                    convert_txt=convert_txt))
    except Exception as e:
        unified = {"error": str(e)}

    # 上传后处理（GT → .npy 密度图 + 可选 .png 可视化）：后台任务执行
    # （常用/自定义统一格式均可：GT 目录的 .txt/.mat 成功转换后才删除；
    # 失败/无配对图像的标注保留，便于用户修复后重试）
    post_task_id = None
    if mat_to_npy or vis_png:
        from app.core.task_manager import task_manager
        from app.services.dataset_unify import run_upload_postprocess
        from config import RUNS_DIR
        sd = RUNS_DIR / "upload_post" / actual_name
        if sd.exists():
            sd = RUNS_DIR / "upload_post" / (
                f"{actual_name}_{uuid.uuid4().hex[:4]}")
        sd.mkdir(parents=True, exist_ok=True)
        params = {"dataset_dir": str(dest_dir), "mat_to_npy": bool(mat_to_npy),
                  "sigma": float(sigma), "adaptive": bool(adaptive),
                  "vis_png": bool(vis_png), "vis_dir": vis_dir or ""}
        # 持久化参数：_ensure_dataset_not_in_use 依赖 params.json 识别
        # 后处理任务正在使用的 dataset_dir，避免删除时误删
        (sd / "params.json").write_text(
            json.dumps(params, ensure_ascii=False, indent=2),
            encoding="utf-8")
        task = task_manager.create(f"upload_post_{actual_name}",
                                   "postprocess", sd)
        task_manager.submit(task, run_upload_postprocess, params)
        post_task_id = task.id

    stats = await run_in_threadpool(_dataset_stats, dest_dir)
    return {"ok": True, "name": actual_name, "path": str(dest_dir), **stats,
            "unified": unified, "unified_log": unified_log,
            "convert_txt": bool(convert_txt),
            "post_task_id": post_task_id}


@app.delete("/api/upload/dataset/{name}")
def delete_upload_dataset(name: str):
    """删除数据集目录，不可恢复。名称做越界检查，被运行中任务占用的拒绝删除"""
    if not name or "/" in name or "\\" in name or ".." in name or \
            name.startswith("_tmp"):
        raise HTTPException(400, f"无效的数据集名称: {name}")
    target = (UPLOAD_DATASET_DIR / name).resolve()
    if target.parent != UPLOAD_DATASET_DIR.resolve():
        raise HTTPException(400, f"无效的数据集名称（越界）: {name}")
    if not target.is_dir():
        raise HTTPException(404, f"数据集不存在: {name}")
    # 运行中/排队中任务占用检查
    _ensure_dataset_not_in_use(target)
    try:
        shutil.rmtree(target)
    except OSError as e:
        raise HTTPException(500, f"删除失败: {e}")
    # 之前上传失败可能残留临时 zip
    (UPLOAD_DATASET_DIR / f"_tmp_{name}.zip").unlink(missing_ok=True)
    return {"ok": True, "deleted": name, "path": str(target)}


@app.get("/api/upload/dataset/{name}/download")
def download_upload_dataset(name: str):
    """把已上传数据集打包为 zip 下载（含统一后的 Images/GT 结构）"""
    if not name or "/" in name or "\\" in name or ".." in name or \
            name.startswith(("_tmp", "_dl")):
        raise HTTPException(400, f"无效的数据集名称: {name}")
    target = (UPLOAD_DATASET_DIR / name).resolve()
    if target.parent != UPLOAD_DATASET_DIR.resolve():
        raise HTTPException(400, f"无效的数据集名称（越界）: {name}")
    if not target.is_dir():
        raise HTTPException(404, f"数据集不存在: {name}")
    tmp_zip = UPLOAD_DATASET_DIR / f"_dl_{uuid.uuid4().hex[:8]}_{name}.zip"

    def _zip() -> None:
        # 压缩包内保留数据集目录名：<name>/Train/Images/...
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(target.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(target.parent))

    try:
        _zip()
    except Exception as e:
        tmp_zip.unlink(missing_ok=True)
        raise HTTPException(500, f"打包失败: {e}")
    return FileResponse(str(tmp_zip), filename=f"{name}.zip",
                        media_type="application/zip",
                        background=BackgroundTask(lambda: tmp_zip.unlink(
                            missing_ok=True)))


_DATASET_TREE_MAX = 5000


@app.get("/api/upload/dataset/tree")
def dataset_tree(name: str, path: str = ""):
    """浏览已上传数据集的目录结构（逐级懒加载，一次只列一层）。

    name: uploads/datasets 下的数据集目录名
    path: 相对数据集根的路径（"" = 根目录）
    返回 {name, path, type, total, truncated, entries:[{name,path,type,size,ext}]}
    """
    if not name or "/" in name or "\\" in name or ".." in name or \
            name.startswith(("_tmp", "_dl")):
        raise HTTPException(400, f"无效的数据集名称: {name}")
    datasets_root = UPLOAD_DATASET_DIR.resolve()
    base = (UPLOAD_DATASET_DIR / name).resolve()
    if base.parent != datasets_root or not base.is_dir():
        raise HTTPException(404, f"数据集不存在: {name}")

    rel = (path or "").replace("\\", "/").strip("/")
    target = (base / rel).resolve() if rel else base
    # 防目录穿越（含指向数据集外部的符号链接）
    if target != base and base not in target.parents:
        raise HTTPException(400, "非法的目录路径")
    if not target.exists():
        raise HTTPException(404, f"路径不存在: {path}")
    if target.is_file():
        return {"name": name, "path": rel, "type": "file",
                "size": target.stat().st_size, "total": 0,
                "truncated": False, "entries": []}

    try:
        children = sorted(target.iterdir(),
                          key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as e:
        raise HTTPException(500, f"读取目录失败: {e}")
    truncated = len(children) > _DATASET_TREE_MAX
    entries = []
    for p in children[:_DATASET_TREE_MAX]:
        try:
            is_dir = p.is_dir()
            size = 0 if is_dir else p.stat().st_size
        except OSError:
            is_dir, size = False, 0
        entries.append({
            "name": p.name,
            "path": str(p.relative_to(base)).replace("\\", "/"),
            "type": "dir" if is_dir else "file",
            "size": size,
            "ext": "" if is_dir else p.suffix.lower(),
        })
    return {"name": name, "path": rel, "type": "dir",
            "total": len(children), "truncated": truncated,
            "entries": entries}


def _dataset_stats(root: Path) -> dict:
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    gts, npys, mats = [], [], []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".txt":
            gts.append(p)
        elif p.suffix.lower() == ".npy":
            npys.append(p)
        elif p.suffix.lower() == ".mat":
            mats.append(p)
    return {"num_images": len(imgs),
            "num_gt_txt": len(gts), "num_gt_npy": len(npys),
            "num_gt_mat": len(mats)}


@app.get("/api/uploads")
def list_uploads():
    models = [{"name": p.name, "path": str(p),
               "size_mb": round(p.stat().st_size / 1048576, 2)}
              for p in sorted(UPLOAD_MODEL_DIR.iterdir()) if p.is_file()]
    datasets = [{"name": p.name, "path": str(p), **_dataset_stats(p)}
                for p in sorted(UPLOAD_DATASET_DIR.iterdir())
                if p.is_dir() and not p.name.startswith(("_tmp", "_dl"))]
    # 训练产物权重（.pth），显示相对 runs/ 的路径；
    # 过滤 DDP 异常残留的 _rankN 影子目录（孤儿 rank 权重不可用）
    runs = []
    runs_root = RESULTS_DIR.parent / "runs"
    weight_files = [p for pattern in ("*.pth", "*.pt")
                    for p in runs_root.rglob(pattern)
                    if not any(part.startswith("_rank") for part in p.parts)]
    # 同一目录已生成带模型名前缀的副本时，隐藏 ultralytics 原始 best.pt/last.pt，
    # 避免推理权重下拉出现重复项（原文件保留供其原生 resume 使用）
    def _has_prefixed_copy(p):
        return any(p.parent.glob(f"*_{p.name}"))
    weight_files = [
        p for p in weight_files
        if not (p.name in ("best.pt", "last.pt") and _has_prefixed_copy(p))]
    for p in sorted(weight_files, key=lambda x: x.stat().st_mtime, reverse=True):
        runs.append({"name": str(p.relative_to(runs_root)), "path": str(p)})
    # ONNX 部署产物（runs/onnx/<run>/model.onnx）：推理面板可选（onnxruntime
    # 验证导出产物）；仅列导出目录一层，避免误收临时目录内的模型
    onnx_models = []
    onnx_root = runs_root / "onnx"
    if onnx_root.is_dir():
        for p in sorted(onnx_root.glob("*/model.onnx"),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            onnx_models.append({
                "name": f"onnx/{p.parent.name}/model.onnx",
                "run": p.parent.name, "path": str(p),
                "size_mb": round(p.stat().st_size / 1048576, 2)})
    return {"models": models, "datasets": datasets, "trained_weights": runs,
            "onnx_models": onnx_models}


@app.get("/api/devices")
def devices():
    """设备清单（torch 口径，与训练设备下拉/系统状态页一致）"""
    devices_list = [{"id": "cpu", "label": "CPU"}]
    has_gpu = False
    try:
        import torch
        tv = torch.__version__
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            devices_list.append({"id": "cuda", "label": "GPU 0"})
    except Exception:
        tv = "unknown"
    return {"devices": devices_list, "torch_version": tv,
            "cuda_available": has_gpu}


# ------------------------- 数据集下载 -------------------------

@app.get("/api/download/dataset/links")
def dataset_download_links():
    """常用人群计数数据集下载链接预设清单"""
    from app.services.download_service import DATASET_DOWNLOAD_LINKS
    return DATASET_DOWNLOAD_LINKS


@app.post("/api/download/dataset")
async def download_dataset(url: str, name: str = ""):
    """启动数据集下载后台任务（URL → zip → 解压到 uploads/datasets/）"""
    from app.core.task_manager import task_manager
    from app.services.download_service import run_download

    if not url or not url.startswith(("http://", "https://")):
        raise HTTPException(400, f"无效的下载链接: {url}")

    # 名称做越界检查
    if name and ("/" in name or "\\" in name or ".." in name):
        raise HTTPException(400, f"无效的数据集名称: {name}")

    import re
    if name:
        name = re.sub(r"[^\w\-]", "_", name)[:60]
    run_name = f"download_{name or 'dataset'}"
    # 任务临时目录加短随机后缀：并发下载同名数据集时日志/状态互不覆盖
    save_dir = UPLOAD_DATASET_DIR / (
        f"_dl_{name or 'temp'}_{uuid.uuid4().hex[:6]}")
    save_dir.mkdir(parents=True, exist_ok=True)

    task = task_manager.create(run_name, "download", save_dir)
    params = {"url": url, "name": name or "",
              "upload_dir": str(UPLOAD_DATASET_DIR)}
    task_manager.submit(task, run_download, params)
    return {"task_id": task.id, "name": name}


@app.get("/api/frameworks")
def frameworks():
    """后端框架可用性（本平台专注 PyTorch）"""
    try:
        import torch
        pt_ok = True
        tv = torch.__version__
    except Exception:
        pt_ok = False
        tv = "not installed"
    return {
        "current": "pytorch" if pt_ok else "none",
        "frameworks": [
            {"key": "pytorch", "label": "PyTorch", "available": pt_ok,
             "version": tv,
             "desc": "PyTorch 引擎（人群计数：密度图估计 + 点预测）"
                     if pt_ok else "PyTorch 未安装"},
        ],
    }


@app.get("/api/domains")
def domains():
    return {
        "title": "人群计数自动训练和推理框架",
        "domains": [
            {"key": "crowd_counting", "label": "人群计数", "available": True,
             "api": "/api/crowd/tasks"},
        ],
    }


if __name__ == "__main__":
    # 端口默认 8060（避开常见的 8000 端口冲突）；launcher.py 会显式传 --port，
    # 也可手动指定：python main.py --port 9000
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--port", type=int, default=8060)
    _args, _ = _parser.parse_known_args()
    uvicorn.run("main:app", host="127.0.0.1", port=_args.port, reload=False)
