# -*- coding: utf-8 -*-
"""训练任务管理器：后台线程跑训练、状态跟踪、停止、断点续训

服务重启后从 runs/*/*/params.json 重建历史任务。
训练任务严格串行：同一时间只跑一个，其余按提交顺序 FIFO 排队；
辅助任务（预处理 / 测试 / 导出等）使用独立的小并发槽
（环境变量 DLATF_MAX_CONCURRENT_AUX，默认 2）。
"""
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path


def _find_weight_file(save_dir, kind):
    """定位 best/last 权重：优先新命名 <model>_best.pth，兼容旧命名与 YOLO 产物。

    优先级（同 kind 内）：
      新命名 *.pth → 旧命名 best.pth/last.pth
      → YOLO 标准 yolo/weights/{kind}.pt → YOLO 新命名 yolo/weights/*_{kind}.pt
    """
    save_dir = Path(save_dir)
    patterns = (f"*_{kind}.pth", f"{kind}.pth",
                f"yolo/weights/*_{kind}.pt", f"yolo/weights/{kind}.pt")
    for pat in patterns:
        try:
            hits = [p for p in save_dir.glob(pat) if p.is_file()]
        except OSError:
            hits = []
        if hits:
            return max(hits, key=lambda p: p.stat().st_mtime)
    return None


def find_best_weight(save_dir):
    """最优权重（<model>_best.pth / best.pth / YOLO best.pt）"""
    return _find_weight_file(save_dir, "best")


def find_last_weight(save_dir):
    """最新权重（<model>_last.pth / last.pth / YOLO last.pt）"""
    return _find_weight_file(save_dir, "last")


def find_resume_weight(save_dir):
    """定位可续训权重（last 优先，其次 best；兼容新/旧命名与 YOLO 产物）。

    YOLO 优先使用其原始 yolo/weights/last.pt（原生 resume 语义最稳），
    其余场景优先新命名 <model>_last.pth。
    """
    save_dir = Path(save_dir)
    for pat in ("*_last.pth", "last.pth", "yolo/weights/last.pt",
                "yolo/weights/*_last.pt", "*_best.pth", "best.pth",
                "yolo/weights/best.pt", "yolo/weights/*_best.pt"):
        try:
            hits = [p for p in save_dir.glob(pat) if p.is_file()]
        except OSError:
            hits = []
        if hits:
            return max(hits, key=lambda p: p.stat().st_mtime)
    return None


def _release_task_memory(task=None):
    """任务线程收尾统一释放内存（finally 调用：完成 / 失败 / 停止路径都过这里）。

    训练的模型 / 优化器 / 数据加载器随线程栈帧 unwind 被引用计数即时回收，
    但仍有两类「账面已释放、实际仍占用」须显式处理：
    1. 循环引用垃圾（图窗 / 数据集对象）要等 gc 阈值，长驻服务进程会
       随任务一个个累积；
    2. CUDA caching allocator 的空闲显存块不归还驱动，同卡后续任务 /
       推理会因此 OOM。
    长驻服务进程 + 反复训练的场景，收尾释放是防止内存水位逐任务爬升的关键。
    """
    try:
        from app.services.train_common import release_training_memory
        release_training_memory(task.log if task is not None else None,
                                reason="任务收尾")
        return
    except Exception:
        pass
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            before = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            freed = (before - torch.cuda.memory_reserved()) / 1048576
            if task is not None and freed >= 1.0:
                task.log(f"任务收尾：已归还 CUDA 缓存显存 {freed:.0f} MB")
    except Exception:
        pass


class Task:
    def __init__(self, task_id, name, kind, save_dir, disk=False):
        self.id = task_id
        self.name = name
        self.kind = kind                  # 任务类型（crowd_counting / mat2density / npy2png）
        self.status = "pending"           # pending/queued/running/completed/failed/stopped
        self.progress = 0.0               # 0-100
        self.epoch = 0
        self.total_epochs = 0
        self.queue_position = 0            # 1-based FIFO position for queued training tasks
        self.metrics = {}                 # 当前 epoch 指标
        self.history = []                 # 逐 epoch 指标
        self.live_history = []            # 逐 step 实时损失（前端实时曲线）
        self.best = {}
        self.error = None
        self.stop_event = threading.Event()
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.started_at = None
        self.started_ts = None
        self.finished_at = None
        self.finished_ts = None
        self.save_dir = str(save_dir)
        self.log_file = Path(save_dir) / "train_log.txt"
        self.result = {}                  # 终态产物（best/last 权重路径等）
        self.disk = disk                  # True=磁盘重建的历史任务
        self._thread = None

    def log(self, msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _resumable(self):
        """是否可断点续训：非排队/运行中 / 未完成，且存在可恢复权重"""
        if self.status in ("pending", "queued", "running", "completed"):
            return False
        return find_resume_weight(self.save_dir) is not None

    def to_dict(self):
        now = time.time()
        started_ts = getattr(self, "started_ts", None)
        finished_ts = getattr(self, "finished_ts", None)
        elapsed_seconds = None
        if started_ts is not None:
            end = finished_ts if finished_ts is not None else now
            elapsed_seconds = max(0.0, end - started_ts)
        elif self.status == "queued":
            elapsed_seconds = 0.0
        eta_seconds = None
        if elapsed_seconds is not None and self.status == "running":
            frac = float(self.progress or 0.0) / 100.0
            if frac <= 0 and self.total_epochs and self.epoch:
                frac = float(self.epoch) / float(self.total_epochs)
            if frac > 0:
                eta_seconds = max(0.0, elapsed_seconds * (1.0 - frac) / frac)
        total_seconds = (elapsed_seconds + eta_seconds
                         if elapsed_seconds is not None
                         and eta_seconds is not None else None)
        return {
            "id": self.id, "name": self.name, "kind": self.kind,
            "status": self.status, "progress": round(self.progress, 1),
            "epoch": self.epoch, "total_epochs": self.total_epochs,
            "queue_position": int(getattr(self, "queue_position", 0) or 0),
            "started_at": self.started_at,
            "elapsed_seconds": (round(elapsed_seconds, 1)
                                if elapsed_seconds is not None else None),
            "eta_seconds": (round(eta_seconds, 1)
                            if eta_seconds is not None else None),
            "estimated_total_seconds": (round(total_seconds, 1)
                                        if total_seconds is not None else None),
            "metrics": self.metrics, "history": self.history,
            "live_history": getattr(self, "live_history", [])[-600:],
            "best": self.best, "error": self.error,
            "created_at": self.created_at, "finished_at": self.finished_at,
            "save_dir": self.save_dir, "result": self.result,
            "historical": self.disk, "resumable": self._resumable(),
        }


def _read_metrics_csv(csv_path):
    """读取 metrics.csv → (history, last_epoch, last_metrics)"""
    import csv as _csv
    import io as _io
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return [], 0, {}
    try:
        with open(csv_path, newline="", encoding="utf-8",
                  errors="ignore") as f:
            raw = f.read()
    except OSError:
        return [], 0, {}
    # 训练进程被杀时文件可能残留 NUL（csv 解析会抛 Error）：
    # 截断到首个 NUL，末尾无换行的半行视为撕裂写丢弃
    nul = raw.find("\x00")
    if nul != -1:
        raw = raw[:nul]
    if raw and not raw.endswith(("\n", "\r")):
        raw = raw[:raw.rfind("\n") + 1]
    try:
        rows = [r for r in _csv.reader(_io.StringIO(raw, newline="")) if r]
    except _csv.Error:
        return [], 0, {}
    if len(rows) < 2:
        return [], 0, {}
    hdr = rows[0]
    history, last_epoch, last_metrics = [], 0, {}
    for r in rows[1:]:
        try:
            ep = int(float(r[0]))
        except (ValueError, IndexError):
            continue
        item = {"epoch": ep}
        for k, v in zip(hdr[1:], r[1:]):
            try:
                item[k] = float(v)
            except (TypeError, ValueError):
                item[k] = v
        history.append(item)
        last_epoch = max(last_epoch, ep)
        last_metrics = {k: v for k, v in item.items() if k != "epoch"}
    return history, last_epoch, last_metrics


def _best_from_history(history):
    """从历史指标推断最优指标（磁盘任务的历史表格展示用）"""
    if not history:
        return {}
    for key, mode in (("val_mae", "min"), ("val_loss", "min")):
        vals = [h for h in history if isinstance(h.get(key), (int, float))]
        if not vals:
            continue
        best = (max if mode == "max" else min)(vals, key=lambda h: h[key])
        return {"epoch": best["epoch"], key: best[key]}
    return {}


def _prepare_for_training(task):
    """训练开始前让出资源：清推理模型缓存 + 停摄像头实时检测。

    训练是长驻服务的内存/CPU 峰值场景：LRU 推理缓存（P2PNet 级可达 GB）
    与逐帧摄像头推理会与训练争内存/CPU，轻则拖慢、重则让训练在原生层
    OOM 崩溃。摄像头会话可由用户训练结束后重新开启。
    """
    try:
        from app.services.crowd_counting import clear_inference_cache
        n = clear_inference_cache()
        if n:
            task.log(f"训练前已清空推理模型缓存（{n} 个），为训练让出内存")
    except Exception:
        pass
    try:
        from app.services import camera_service
        if camera_service.manager.get() is not None:
            camera_service.manager.stop()
            task.log("训练前已停止摄像头实时检测，为训练让出 CPU/内存")
    except Exception:
        pass


class TaskManager:
    """全局训练任务管理（单例）：内存任务 + 磁盘历史任务"""

    def __init__(self):
        self._tasks = {}        # 内存任务（本次会话启动）
        self._disk_tasks = {}   # 磁盘重建的历史任务
        self._scanned = False
        self._lock = threading.Lock()          # 保护 _tasks
        self._disk_lock = threading.Lock()     # 保护 _disk_tasks / _scanned
        # 非训练辅助任务仍可使用少量并发槽；训练任务改为显式 FIFO，
        # 同一时间只运行一个，后续任务按提交顺序排队。
        try:
            _max = max(1, int(os.environ.get(
                "DLATF_MAX_CONCURRENT_AUX") or os.environ.get(
                    "DLATF_MAX_CONCURRENT_TRAIN", "2")))
        except (TypeError, ValueError):
            _max = 2
        self.max_concurrent = _max
        self._slots = threading.Semaphore(_max)
        self._training_queue = deque()       # (task, target_fn, args)
        self._queue_lock = threading.Lock()
        self._active_training = None
        self._shutting_down = False

    def create(self, name, kind, save_dir):
        task_id = uuid.uuid4().hex[:12]
        task = Task(task_id, name, kind, save_dir)
        with self._lock:
            self._tasks[task_id] = task
            # 新任务接管同目录的磁盘历史任务（续训场景）
            for k in [k for k, t in self._disk_tasks.items()
                      if t.save_dir == str(save_dir)]:
                del self._disk_tasks[k]
        return task

    @staticmethod
    def _is_training_kind(kind):
        return str(kind or "").startswith("crowd_")

    def _finish_task(self, task, status):
        task.status = status
        task.finished_ts = time.time()
        task.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task.queue_position = 0
        self._write_task_state(task, status)

    def _refresh_queue_positions_locked(self):
        for i, (queued_task, _fn, _args) in enumerate(
                self._training_queue, 1):
            queued_task.queue_position = i

    def _pump_training_queue(self):
        """Start the next queued training task in FIFO order."""
        with self._queue_lock:
            if self._shutting_down or self._active_training is not None:
                return
            while self._training_queue:
                entry = self._training_queue.popleft()
                task = entry[0]
                if task.stop_event.is_set():
                    task.log("排队期间收到停止请求，已取消")
                    self._finish_task(task, "stopped")
                    self._refresh_queue_positions_locked()
                    continue
                self._active_training = task
                task.queue_position = 0
                self._refresh_queue_positions_locked()
                t = threading.Thread(
                    target=self._run_training, args=(entry,), daemon=True,
                    name=f"train-{task.id}")
                task._thread = t
                t.start()
                return

    def _run_training(self, entry):
        task, target_fn, args = entry
        task.log("训练队列：轮到此任务，开始执行")
        try:
            if task.stop_event.is_set():
                self._finish_task(task, "stopped")
                return
            task.started_ts = time.time()
            task.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            task.status = "running"
            _prepare_for_training(task)
            target_fn(task, *args)
            if task.stop_event.is_set():
                self._finish_task(task, "stopped")
            elif task.status == "running":
                task.progress = 100.0
                task.log("训练完成")
                self._finish_task(task, "completed")
        except StopIteration:
            task.log("训练已被用户暂停（可从断点继续训练）")
            self._finish_task(task, "stopped")
        except Exception as e:
            task.error = f"{type(e).__name__}: {e}"
            task.log(f"训练失败: {task.error}")
            task.log(traceback.format_exc())
            self._finish_task(task, "failed")
        finally:
            _release_task_memory(task)
            with self._queue_lock:
                if self._active_training is task:
                    self._active_training = None
                self._refresh_queue_positions_locked()
            self._pump_training_queue()

    def _submit_aux(self, task, target_fn, *args):
        """Auxiliary tasks (preprocess/export/test/...) use the small slot pool."""
        def _finish(status):
            self._finish_task(task, status)

        def runner():
            while not self._slots.acquire(timeout=1.0):
                if task.stop_event.is_set():
                    task.log("排队期间收到停止请求，已取消")
                    _finish("stopped")
                    return
            try:
                if task.stop_event.is_set():
                    _finish("stopped")
                    return
                task.started_ts = time.time()
                task.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                task.status = "running"
                target_fn(task, *args)
                if task.status == "running":
                    task.progress = 100.0
                    task.log("测试完成" if task.kind == "test" else "任务完成")
                    _finish("completed")
            except StopIteration:
                task.log("任务已被用户暂停")
                _finish("stopped")
            except Exception as e:
                task.error = f"{type(e).__name__}: {e}"
                task.log(f"任务失败: {task.error}")
                task.log(traceback.format_exc())
                _finish("failed")
            finally:
                self._slots.release()
                _release_task_memory(task)

        t = threading.Thread(target=runner, daemon=True,
                             name=f"task-{task.id}")
        task._thread = t
        t.start()

    def submit(self, task, target_fn, *args):
        """Submit a task.

        Training tasks (crowd_*) are serialized through an explicit FIFO:
        only one runs at a time and the rest wait in submission order.
        Auxiliary tasks keep the existing concurrent slot pool.
        """
        task.status = "queued"
        if not self._is_training_kind(task.kind):
            self._submit_aux(task, target_fn, *args)
            return task.id

        with self._queue_lock:
            self._training_queue.append((task, target_fn, args))
            self._refresh_queue_positions_locked()
            position = task.queue_position
        task.log(f"已加入训练队列，当前排队第 {position} 位")
        self._pump_training_queue()
        return task.id

    @staticmethod
    def _write_task_state(task, status):
        """任务终态持久化（磁盘扫描据此还原状态）"""
        try:
            Path(task.save_dir).mkdir(parents=True, exist_ok=True)
            (Path(task.save_dir) / "task_state.json").write_text(
                json.dumps({"status": status,
                            "finished_at": task.finished_at},
                           ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    # ------------------------- 磁盘历史任务 -------------------------

    def scan_disk_tasks(self, force=False):
        """扫描 runs/*/*/params.json 重建历史任务（服务重启后可见）"""
        with self._disk_lock:
            if self._scanned and not force:
                return
        from config import RUNS_DIR
        disk = {}
        try:
            params_files = list(RUNS_DIR.glob("*/*/params.json"))
        except OSError:
            params_files = []
        with self._lock:
            mem_tasks = list(self._tasks.values())
        for params_f in params_files:
            run_dir = params_f.parent
            if (run_dir / ".deleted").exists():
                continue   # 已删除记录的任务不再显示
            try:
                params = json.loads(
                    params_f.read_text(encoding="utf-8"))
            except Exception:
                continue
            kind = run_dir.parent.name
            name = run_dir.name
            # 已有内存任务接管同一目录（如续训中）
            if any(t.save_dir == str(run_dir) for t in mem_tasks):
                continue
            task = Task(f"disk-{kind}-{name}", name, kind, run_dir,
                        disk=True)
            history, last_epoch, last_metrics = _read_metrics_csv(
                run_dir / "metrics.csv")
            total = int(params.get("epochs", 0) or 0)
            task.total_epochs = total
            task.epoch = last_epoch
            task.progress = (last_epoch / total * 100.0
                             if total else 0.0)
            task.metrics = last_metrics
            task.history = history
            task.best = _best_from_history(history)
            # 状态：task_state.json 优先 → epoch 达标=完成 → 有权重=已暂停 → 失败
            state_f = run_dir / "task_state.json"
            if state_f.exists():
                try:
                    task.status = json.loads(
                        state_f.read_text(encoding="utf-8")).get(
                            "status", "stopped")
                except Exception:
                    task.status = "stopped"
                _lw = find_last_weight(run_dir)
                task.finished_at = _lw.stat().st_mtime if _lw else None
                if task.finished_at:
                    task.finished_at = datetime.fromtimestamp(
                        task.finished_at).strftime("%Y-%m-%d %H:%M:%S")
            elif total and last_epoch >= total:
                task.status = "completed"
            elif find_resume_weight(run_dir):
                task.status = "stopped"   # 中断（可续训），界面显示「已暂停」
            else:
                task.status = "failed"
            best_w = find_best_weight(run_dir)
            last_w = find_last_weight(run_dir)
            task.result = {
                "best_weight": str(best_w) if best_w else "",
                "last_weight": str(last_w) if last_w else "",
                "save_dir": str(run_dir)}
            try:
                task.created_at = datetime.fromtimestamp(
                    params_f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            except OSError:
                pass
            disk[task.id] = task
        with self._disk_lock:
            self._disk_tasks = disk
            self._scanned = True

    def list(self):
        self.scan_disk_tasks()
        with self._lock:
            mem = list(self._tasks.values())
        mem_dirs = {t.save_dir for t in mem}
        with self._disk_lock:
            disk_all = list(self._disk_tasks.values())
        disk = [t for t in disk_all if t.save_dir not in mem_dirs]
        out = []
        for t in sorted(mem + disk, key=lambda x: x.created_at, reverse=True):
            d = t.to_dict()
            # 列表只回传尾部少量点，避免任务多时轮询响应膨胀
            # （曲线回看/实时图表走详情接口，仍是全量 history）
            d["history"] = d["history"][-2:]
            # 逐 step 实时曲线仅详情接口需要，列表接口剥离
            d["live_history"] = []
            # 可视化批量任务的 base64 预览图较大（约几百 KB），列表接口剥离；
            # 前端展示走详情接口（/train/tasks/{id}）取回
            if isinstance(d.get("result"), dict) and \
                    d["result"].get("preview_images"):
                d["result"] = {k: v for k, v in d["result"].items()
                               if k != "preview_images"}
            out.append(d)
        return out

    def shutdown(self, timeout=10):
        """服务退出时调用：通知运行中任务停止并限时等待（降低写坏 checkpoint 概率）"""
        self._shutting_down = True
        with self._queue_lock:
            queued = [entry[0] for entry in self._training_queue]
            self._training_queue.clear()
            for t in queued:
                t.stop_event.set()
                t.log("服务关闭，排队任务已取消")
                self._finish_task(t, "stopped")
            self._refresh_queue_positions_locked()
        with self._lock:
            tasks = [t for t in self._tasks.values()
                     if t.status in ("running", "queued", "pending")]
        if not tasks:
            return
        print(f"[shutdown] 通知 {len(tasks)} 个运行中任务停止...")
        for t in tasks:
            t.stop_event.set()
        deadline = time.time() + max(1, timeout)
        for t in tasks:
            th = getattr(t, "_thread", None)
            if th is None or not th.is_alive():
                continue
            th.join(timeout=max(0.1, deadline - time.time()))
        print("[shutdown] 任务已停止（超时未结束的将被强制终止，checkpoint 为原子写入）")

    def get(self, task_id):
        with self._lock:
            t = self._tasks.get(task_id)
        if t is not None:
            return t
        self.scan_disk_tasks()
        with self._disk_lock:
            return self._disk_tasks.get(task_id)

    def stop(self, task_id):
        task = self.get(task_id)
        if task is None:
            return False
        if task.status not in ("running", "queued", "pending"):
            return False
        # A queued training task can be removed immediately; an active one
        # receives the stop signal and its runner advances the queue.
        if task.status == "queued" and self._is_training_kind(task.kind):
            with self._queue_lock:
                for i, (queued_task, _fn, _args) in enumerate(
                        self._training_queue):
                    if queued_task is task:
                        del self._training_queue[i]
                        task.stop_event.set()
                        task.log("排队期间收到停止请求，已取消")
                        self._finish_task(task, "stopped")
                        self._refresh_queue_positions_locked()
                        return True
        task.stop_event.set()
        task.log("收到暂停请求...")
        return True

    def delete(self, task_id, artifacts=False):
        """删除任务记录与日志（训练产物默认保留；artifacts=True 时一并删除）"""
        task = self.get(task_id)
        if task is None:
            return False
        if task.status in ("running", "queued", "pending"):
            return False
        with self._lock:
            self._tasks.pop(task_id, None)
        with self._disk_lock:
            self._disk_tasks.pop(task_id, None)
        save_dir = Path(task.save_dir)
        try:
            if artifacts:
                shutil.rmtree(save_dir, ignore_errors=True)
            else:
                (save_dir / "train_log.txt").unlink(missing_ok=True)
                # 隐藏标记：磁盘扫描不再重建该任务（权重等产物保留）
                (save_dir / ".deleted").write_text("", encoding="utf-8")
        except OSError:
            pass
        return True

    def read_log(self, task_id, tail=200):
        task = self.get(task_id)
        if task is None or not task.log_file.exists():
            return ""
        lines = task.log_file.read_text(
            encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:])


# 全局单例
task_manager = TaskManager()
