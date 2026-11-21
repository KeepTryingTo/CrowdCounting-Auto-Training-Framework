# -*- coding: utf-8 -*-
"""DDP 分布式训练启动器（mp.spawn 多进程，每卡一进程，Windows 兼容）

- 主进程轮询运行目录 metrics.csv / train_log.txt 刷新任务进度
- worker 经 init_process_group（GPU→NCCL / CPU→gloo）注入
  ddp_rank / ddp_world_size 后复用现有训练函数
- rank 0 写真实目录，rank>0 写 _rankN/ 影子目录（结束后清理）
- DDP 禁用早停（各 rank 判定需严格一致，否则可能死锁）
- 停止按钮 terminate 子进程，产物保留可续训
"""
import json
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# 训练函数注册表：子进程按「模块名.函数名」导入（spawn 不 pickle 闭包）
# crowd_density / crowd_point 为界面新路线 key，crowd_counting 兼容旧任务 kind
TRAIN_FNS = {
    "crowd_density": ("app.services.crowd_counting", "run_crowd_train"),
    "crowd_point": ("app.services.crowd_counting", "run_crowd_train"),
    "crowd_counting": ("app.services.crowd_counting", "run_crowd_train"),
}


def _worker(rank, fn_key, params_json, save_dir):
    """DDP worker：初始化进程组 → 构造 rank 感知的 task → 调用训练函数"""
    import importlib
    import os
    import zlib
    params = json.loads(params_json)
    world = int(params.get("ddp_world_size") or 1)

    # env:// rendezvous 环境变量（mp.spawn 不像 torchrun 自动注入；
    # 端口须各 rank 一致 → 用确定性 crc32 派生，避免 hash() 进程级随机化）
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT",
                          str(29500 + (zlib.crc32(save_dir.encode()) % 500)))

    backend = "nccl" if (torch.cuda.is_available() and
                         torch.cuda.device_count() > 0) else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world)
    if backend == "nccl":
        torch.cuda.set_device(rank)
        params["device"] = f"cuda:{rank}"
    else:
        params["device"] = "cpu"  # gloo（无 GPU 环境：CPU 分布式/测试）

    params["ddp_rank"] = rank
    # DDP 模式禁用早停（各 rank 判定一致性无法保证，避免死锁）
    if params.get("early_stopping"):
        params["early_stopping"] = 0

    mod_name, fn_name = TRAIN_FNS[fn_key]
    fn = getattr(importlib.import_module(mod_name), fn_name)

    from app.core.task_manager import Task
    if rank == 0:
        task = Task(f"ddp-r0", fn_key, "train", Path(save_dir))
    else:
        # rank>0 影子目录：训练函数内所有文件产物自然隔离，结束后整体删除
        shadow = Path(save_dir) / f"_rank{rank}"
        shadow.mkdir(parents=True, exist_ok=True)
        task = Task(f"ddp-r{rank}", fn_key, "train", shadow)

    try:
        fn(task, params)
    finally:
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _read_csv_progress(save_dir):
    """轮询运行目录：从 metrics.csv 回读 (history, last_epoch, last_metrics)"""
    from app.core.task_manager import _read_metrics_csv
    return _read_metrics_csv(Path(save_dir) / "metrics.csv")


def run_ddp_train(task, params: dict, fn_key: str):
    """DDP 启动入口（task_manager 线程中执行）"""
    from app.services.train_common import read_live_progress
    from app.core.task_manager import find_best_weight, find_last_weight
    save_dir = Path(task.save_dir)
    world = int(params.get("ddp_world_size") or 1)
    if fn_key not in TRAIN_FNS:
        raise ValueError(f"该任务不支持 DDP: {fn_key}")
    if world < 2:
        raise ValueError("DDP 分布式要求 ddp_world_size ≥ 2")

    gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    n_gpu = torch.cuda.device_count() if gpu else 0
    if gpu:
        # NCCL 每进程绑定一张卡：单卡无法启动 ≥2 个 rank（旧逻辑 max(2,n_gpu)
        # 在 1 卡时仍为 2，rank1 set_device 会直接崩）
        if n_gpu < 2:
            raise ValueError(
                "DDP 分布式至少需要 2 张 GPU；单卡请选择单卡/DataParallel 模式")
        if world > n_gpu:
            world = n_gpu
            task.log(f"DDP 进程数调整为 GPU 数: {world}")
    params = dict(params)
    params["ddp_world_size"] = world
    params.pop("resume_weight", None)  # DDP 首版不支持断点续训
    params.pop("early_stopping", None)

    task.total_epochs = int(params.get("epochs", 0) or 0)
    task.log(f"启动 DDP 分布式训练: {world} 进程 × "
             f"{'NCCL/GPU' if gpu else 'gloo/CPU'}（{'cuda' if gpu else 'cpu'}）")

    ctx = mp.spawn(_worker,
                   args=(fn_key, json.dumps(params, ensure_ascii=False, default=str),
                         str(save_dir)),
                   nprocs=world, join=False)

    # 轮询刷新 task 进度。train_log.txt 是 rank0 写进运行目录的，
    # /log 读文件，天然实时
    t0 = time.time()
    while True:
        done = ctx.join(timeout=2.0)
        if task.stop_event.is_set():
            task.log("收到停止请求，终止 DDP 子进程...")
            for p in getattr(ctx, "processes", []):
                try:
                    p.terminate()
                except Exception:
                    pass
            time.sleep(2.0)
            # 停止路径也清一遍影子目录
            for r in range(1, world):
                shutil.rmtree(save_dir / f"_rank{r}", ignore_errors=True)
            raise StopIteration()
        history, last_epoch, last_metrics = _read_csv_progress(save_dir)
        if history:
            task.epoch = last_epoch
            if task.total_epochs:
                task.progress = min(100.0, last_epoch / task.total_epochs * 100.0)
            task.metrics = last_metrics
            task.history = history
        # 逐 batch 实时进度（rank0 训练循环写入 live_progress.json）：
        # 仅当实时 epoch 领先于 csv 已完成 epoch 时采用（epoch 内平滑刷新）；
        # csv 已写入本轮（含 val_mae 等验证指标）时以 csv 为准，避免实时
        # 文件里残留的上一轮验证指标覆盖本轮结果
        live = read_live_progress(save_dir)
        if live:
            try:
                live_epoch = int(live.get("epoch"))
            except (TypeError, ValueError):
                live_epoch = None
            if live_epoch is not None and live_epoch > last_epoch:
                task.epoch = live_epoch
                try:
                    task.progress = min(100.0, float(live.get("progress")))
                except (TypeError, ValueError):
                    pass
                lv_metrics = live.get("metrics")
                if isinstance(lv_metrics, dict) and lv_metrics:
                    task.metrics = lv_metrics
                lv_hist = live.get("live_history")
                if isinstance(lv_hist, list):
                    task.live_history = lv_hist
        if done:
            break
        if time.time() - t0 > 7200:
            task.log("警告：训练超过 2 小时仍在运行")
            t0 = time.time()

    # 清理影子目录（rank>0 产物）
    for r in range(1, world):
        shadow = save_dir / f"_rank{r}"
        shutil.rmtree(shadow, ignore_errors=True)

    # 终态回读
    history, last_epoch, last_metrics = _read_csv_progress(save_dir)
    if history:
        task.epoch = last_epoch
        task.metrics = last_metrics
        task.history = history
        if task.total_epochs:
            task.progress = min(100.0, last_epoch / task.total_epochs * 100.0)
    best_w = find_best_weight(save_dir)
    last_w = find_last_weight(save_dir)
    task.result = {"best_weight": str(best_w) if best_w else "",
                   "last_weight": str(last_w) if last_w else "",
                   "save_dir": str(save_dir),
                   "distributed": True, "world_size": world}
    task.log(f"DDP 分布式训练完成（{world} 进程）")


def gpu_count() -> int:
    try:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0
