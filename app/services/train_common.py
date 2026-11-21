# -*- coding: utf-8 -*-
"""训练通用增强：随机种子、早停、checkpoint 完整状态、损失曲线保存、断点续训、DDP 分布式、训练内存及时释放"""
import random
from pathlib import Path

import numpy as np
import torch


# ------------------------- 训练内存及时释放 -------------------------

def release_training_memory(log=None, reason="内存释放"):
    """训练内存及时释放（每轮结束 / 任务收尾统一调用，幂等、毫秒级）：

    1. gc.collect()：回收循环引用垃圾 —— 每轮的 matplotlib 图窗
       （Figure↔Axes↔Artist 互相引用）、数据集/增强临时对象等，
       引用计数无法释放，惰性 GC 阈值触发前会逐轮堆积（长训练可
       累积 GB 级）；
    2. torch.cuda.empty_cache()：caching allocator 把已释放张量的
       显存块留在池中复用、不主动归还驱动 —— 同卡并发训练（任务
       管理器默认并发 2）、训练后同卡推理会被这些「账面已释放」的
       块拖到 OOM，须显式归还。仅归还空闲块，绝不影响在用张量。

    未初始化 CUDA（纯 CPU 训练）时退化为纯 gc。log 传入时仅在实际
    归还了 ≥1MB 显存的情况下记录一行，避免逐轮刷屏。
    """
    import gc
    gc.collect()
    freed_mb = 0.0
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            before = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            freed_mb = (before - torch.cuda.memory_reserved()) / 1048576
    except Exception:
        pass
    if log is not None and freed_mb >= 1.0:
        try:
            log(f"{reason}：已归还 CUDA 缓存显存 {freed_mb:.0f} MB")
        except Exception:
            pass


# ------------------------- DDP 分布式训练支持 -------------------------

def ddp_world_size(params) -> int:
    """当前训练的分布式世界大小（1 = 非分布式）"""
    try:
        return int((params or {}).get("ddp_world_size") or 1)
    except Exception:
        return 1


def ddp_rank(params) -> int:
    """当前进程的分布式 rank（0 = 主进程）"""
    try:
        return int((params or {}).get("ddp_rank") or 0)
    except Exception:
        return 0


def is_main_process(params) -> bool:
    """是否主进程（rank 0：日志 / checkpoint / metrics / 推理产物唯一写入者）"""
    return ddp_rank(params) == 0


def maybe_distributed_sampler(dataset, params):
    """DDP 模式返回 DistributedSampler（数据按 rank 分片），否则 None"""
    if ddp_world_size(params) > 1:
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(
            dataset, num_replicas=ddp_world_size(params),
            rank=ddp_rank(params), shuffle=True)
    return None


def wrap_model_distributed(model, device_str, params, task=None):
    """模型分布式包装：DDP（真分布式，多进程每卡一个）> DataParallel（单进程多卡）

    - DDP：由 ddp_launcher 经 mp.spawn 拉起 N 进程调用，params 含 ddp_rank/
      ddp_world_size；GPU 用 device_ids=[rank]（NCCL），CPU 用 gloo（测试）
    - DataParallel：单进程多卡（默认模式，Detect 检测等亦可回退）
    - 单卡/单进程：原样返回
    """
    import torch.nn as nn
    if ddp_world_size(params) > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        rank = ddp_rank(params)
        if str(device_str).startswith("cuda") and torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")  # gloo CPU（无 GPU 环境测试/回退）
        model.to(device)
        ddp = DDP(model, device_ids=[rank] if device.type == "cuda" else None)
        if task is not None:
            task.log(f"DDP 分布式训练: rank {rank}/{ddp_world_size(params)} "
                     f"({device})")
        return ddp
    n_gpu = torch.cuda.device_count() if str(device_str).startswith("cuda") else 0
    if n_gpu > 1:
        gpu_ids = list(range(n_gpu))
        model = nn.DataParallel(model, device_ids=gpu_ids)
        if task is not None:
            task.log(f"使用 DataParallel 多 GPU: {gpu_ids}")
    return model


def ddp_set_epoch(loader, epoch, params):
    """DDP 模式下每轮刷新 DistributedSampler 的洗牌种子（必须，否则每轮同序）"""
    if ddp_world_size(params) > 1:
        s = getattr(loader, "sampler", None)
        if hasattr(s, "set_epoch"):
            s.set_epoch(epoch)


def ddp_cleanup(params):
    """销毁分布式进程组（worker 退出前调用）"""
    if ddp_world_size(params) > 1:
        import torch.distributed as dist
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def unwrap_ddp(model):
    """解包 DDP / DataParallel → 原始模型（保存/推理用）"""
    import torch.nn as nn
    if isinstance(model, (nn.DataParallel,)) or \
            type(model).__name__ == "DistributedDataParallel":
        return model.module
    return model


def set_seed(seed: int):
    """全局随机种子（python / numpy / torch，含 cudnn 确定性）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopper:
    """早停跟踪器：metric 无提升达 patience 轮后触发停止"""

    def __init__(self, patience: int, mode: str = "min"):
        self.patience = max(0, int(patience))
        self.mode = mode
        self.best = None
        self.counter = 0
        self.stopped = False

    def step(self, metric: float) -> bool:
        """返回 True 表示应停止训练"""
        if self.patience <= 0:
            return False
        if self.best is None or \
                (metric < self.best if self.mode == "min" else metric > self.best):
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                return True
        return False


def save_checkpoint(path, *, net, optimizer=None, scheduler=None,
                    epoch=0, best_metric=None, **extra):
    """保存完整 checkpoint：模型权重 + optimizer/scheduler state + epoch + best metric

    原子写入（先写 .tmp 再 os.replace）：服务退出硬杀线程也不会留下截断的 .pth；
    只写一份 state_dict（加载端兼容 state_dict / model_state_dict 两种键名，
    双写会让每个 .pth 体积翻倍）。
    """
    raw = net.module if isinstance(net, torch.nn.DataParallel) else net
    raw = unwrap_ddp(raw) if type(raw).__name__ == "DistributedDataParallel" else raw
    ckpt = {"state_dict": raw.state_dict(),
            "epoch": epoch, "best_metric": best_metric}
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    ckpt.update(extra)
    import os
    tmp = str(path) + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, str(path))


def save_loss_curves(save_dir, history, extra_keys=()):
    """训练/验证损失曲线 PNG（每 epoch 更新，训练结束与中途均保存）"""
    from pathlib import Path
    from app.services.eval_metrics import plot_loss_curves
    keys = [k for k in ("train_loss", "val_loss", *extra_keys)
            if history and k in history[0]]
    if keys:
        plot_loss_curves(history, Path(save_dir) / "loss_curves.png", keys)


# ------------------------- 训练实时指标（逐 batch） -------------------------

def report_batch_progress(task, *, epoch, epochs, batch_idx, total_batches,
                          running_loss, n_batches, live_metrics=None,
                          log_every=0, is_main=True):
    """训练循环内逐 batch 刷新进度与实时损失，供前端轮询实时展示。

    前端 `/train/tasks/{id}`（约 1.5~2s 轮询）据此可在 epoch 进行中就看到
    train_loss 与进度条连续变化，而不必等整轮结束：

    - task.progress = (已完成 epoch 数 + 当前 epoch 内 batch 完成比例) / 总轮数
      × 100，进度条平滑推进；
    - task.metrics["train_loss"] 为当前 epoch 到目前的平均损失，并保留上一轮
      的验证指标（val_mae / val_loss 等），便于训练中对照；
    - 每 log_every 个 batch 追加一行训练日志（0 表示不写日志，仅刷新状态）；
    - 实时状态原子写入 <save_dir>/live_progress.json，供 DDP 主进程轮询
      （DDP 子进程与父进程的 task 对象不共享，父进程读文件转发给前端）。

    非主进程（DDP rank>0）直接返回，避免影子目录无意义写盘。
    """
    if not is_main:
        return
    try:
        epochs_i = max(1, int(epochs))
        epoch_i = int(epoch)
        total_b = max(1, int(total_batches))
        batch_i = int(batch_idx)
        progress = ((epoch_i - 1) + (batch_i + 1) / total_b) / epochs_i * 100.0
        avg_loss = float(running_loss) / max(1, int(n_batches))
        metrics = dict(getattr(task, "metrics", None) or {})
        metrics["train_loss"] = round(avg_loss, 5)
        if live_metrics:
            for k, v in live_metrics.items():
                if v is None:
                    continue
                try:
                    metrics[k] = round(float(v), 5)
                except (TypeError, ValueError):
                    metrics[k] = v
        task.epoch = epoch_i
        task.progress = progress
        task.metrics = metrics
        # 逐 step 实时曲线点（前端实时绘图用）；限长避免无限增长与轮询膨胀
        step = (epoch_i - 1) * total_b + (batch_i + 1)
        lh = getattr(task, "live_history", None)
        if not isinstance(lh, list):
            lh = []
            task.live_history = lh
        lh.append({"step": step, "epoch": epoch_i,
                   "train_loss": round(avg_loss, 5)})
        if len(lh) > 2000:
            del lh[:-1000]
        # 落盘限频：每 epoch 约 20 次 + 最后一个 batch（DDP 父进程 2s 轮询足够）
        write_every = max(1, total_b // 20)
        if batch_i + 1 >= total_b or (batch_i + 1) % write_every == 0:
            _write_live_progress(task, epoch_i, progress, metrics)
        if int(log_every or 0) > 0 and (
                batch_i + 1 >= total_b or
                (batch_i + 1) % int(log_every) == 0):
            task.log(f"Epoch {epoch_i}/{epochs_i} | "
                     f"batch {batch_i + 1}/{total_b} | loss {avg_loss:.5f}")
    except Exception:
        pass


def _write_live_progress(task, epoch, progress, metrics):
    """原子写实时进度文件（先写 .tmp 再 os.replace，避免 DDP 父进程读到半个文件）"""
    import json as _json
    import os as _os
    try:
        p = Path(task.save_dir) / "live_progress.json"
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(_json.dumps(
            {"epoch": epoch, "progress": round(progress, 2),
             "metrics": metrics,
             "live_history": getattr(task, "live_history", [])[-400:]},
            ensure_ascii=False), encoding="utf-8")
        _os.replace(str(tmp), str(p))
    except OSError:
        pass


def read_live_progress(save_dir):
    """读取 report_batch_progress 落盘的实时进度；无或损坏返回 None"""
    import json as _json
    try:
        p = Path(save_dir) / "live_progress.json"
        data = _json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


# ------------------------- 断点续训 -------------------------

def resume_training_state(resume_weight, model, optimizer=None,
                          scheduler=None, best_key=None, best_mode="max"):
    """断点续训：从 checkpoint 恢复 模型 / optimizer / scheduler / epoch / 最优指标。

    返回 (start_epoch, best_value, best_metric_dict)：
    - start_epoch = ckpt["epoch"] + 1（无 epoch 信息则 1）
    - best_value 取 ckpt["best_metric"][best_key]；缺省按 best_mode 给 -inf/inf
    - best_metric_dict 原样返回（供 task.best 初始化）
    model 可传入 DataParallel 包装后的模型（内部自动取 module）。
    """
    ckpt = torch.load(resume_weight, map_location="cpu")
    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
    if state is None and isinstance(ckpt, dict):
        state = ckpt.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"无效的续训权重（无模型参数）: {resume_weight}")
    if isinstance(ckpt, dict) and not ckpt.get("epoch") and \
            not ckpt.get("optimizer_state_dict") and \
            not ckpt.get("scheduler_state_dict"):
        # 缺少全部恢复信息：疑似纯预训练/推理导出权重，而非中断训练的检查点
        raise ValueError(
            f"所选权重不是断点续训检查点（缺少 epoch / optimizer / scheduler "
            f"恢复信息，疑似纯预训练权重）: {resume_weight}。"
            f"如需以其为初始权重微调（从第 1 轮开始），请取消勾选「断点续训」")
    state = {k[7:] if k.startswith("module.") else k: v
             for k, v in state.items()}
    raw = model.module if hasattr(model, "module") and \
        isinstance(getattr(model, "module", None), torch.nn.Module) else model
    raw.load_state_dict(state, strict=True)
    if optimizer is not None and isinstance(ckpt, dict) and \
            ckpt.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            pass
    if scheduler is not None and isinstance(ckpt, dict) and \
            ckpt.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception:
            pass
    start_epoch = 1
    if isinstance(ckpt, dict) and ckpt.get("epoch"):
        start_epoch = int(ckpt["epoch"]) + 1
    best_metric = ckpt.get("best_metric") if isinstance(ckpt, dict) else None
    best_metric = best_metric if isinstance(best_metric, dict) else {}
    best_value = float("-inf") if best_mode == "max" else float("inf")
    if best_key and best_metric.get(best_key) is not None:
        try:
            best_value = float(best_metric[best_key])
        except (TypeError, ValueError):
            pass
    return start_epoch, best_value, best_metric


def reload_task_history(task, csv_path, start_epoch, header):
    """续训时恢复历史指标：读取 metrics.csv 中 epoch < start_epoch 的行到
    task.history，并重写 csv（表头 + 保留行），后续 epoch 继续追加写入。

    同时将 task.epoch / task.metrics 置为恢复点状态（界面立即显示历史进度）。
    """
    import csv as _csv
    import io as _io
    csv_path = Path(csv_path)
    rows = []
    if csv_path.exists():
        try:
            with open(csv_path, newline="", encoding="utf-8",
                      errors="ignore") as f:
                raw = f.read()
        except OSError:
            raw = ""
        # 并发写入/进程被杀时文件可能含 NUL（csv 遇 NUL 抛 Error），
        # 截断到首个 NUL，末尾无换行的半行丢弃；后续重写 csv 自动清理损坏尾部
        nul = raw.find("\x00")
        if nul != -1:
            raw = raw[:nul]
        if raw and not raw.endswith(("\n", "\r")):
            raw = raw[:raw.rfind("\n") + 1]
        try:
            rows = [r for r in _csv.reader(
                _io.StringIO(raw, newline="")) if r]
        except _csv.Error:
            rows = []
    hdr = rows[0] if rows else header
    kept_raw, history = [], []
    for r in rows[1:]:
        try:
            ep = int(float(r[0]))
        except (ValueError, IndexError):
            continue
        if ep >= start_epoch:
            continue   # 崩溃时可能已写入但未保存对应 checkpoint 的行，丢弃
        kept_raw.append(r)
        item = {"epoch": ep}
        for k, v in zip(hdr[1:], r[1:]):
            try:
                item[k] = float(v)
            except (TypeError, ValueError):
                item[k] = v
        history.append(item)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(hdr)
        for r in kept_raw:
            w.writerow(r)
    task.history = history
    task.epoch = start_epoch - 1
    if history:
        task.metrics = {k: v for k, v in history[-1].items() if k != "epoch"}
    return history
