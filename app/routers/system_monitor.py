# -*- coding: utf-8 -*-
"""系统状态路由：CPU/内存/磁盘/网络（psutil）+ GPU（torch/NVML）实时监控

psutil / pynvml / torch 均为可选依赖，缺哪个降级哪个；
网络速率由前端按相邻两次轮询的字节差计算
"""
import platform
import threading
import time
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# torch 可选（系统监控不绑定训练框架）
try:
    import torch
except ImportError:
    torch = None
def _torch_ok():
    return torch is not None and torch.cuda.is_available()

def _torch_device_count():
    return torch.cuda.device_count() if _torch_ok() else 0

def _torch_device_name(i):
    return torch.cuda.get_device_name(i) if _torch_ok() else f"GPU {i}"

def _torch_mem_info(i):
    if torch is not None and hasattr(torch.cuda, "mem_get_info"):
        return torch.cuda.mem_get_info(i)
    return None

def _torch_version():
    return torch.__version__ if torch is not None else "not installed"

router = APIRouter(prefix="/api/system", tags=["system"])

# psutil（可选依赖）
try:
    import psutil
    # 预热 CPU 采样基准（psutil 首次调用无基准返回 0）
    psutil.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None, percpu=True)
except Exception:
    psutil = None

# NVML（可选：GPU 利用率 / 温度）
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    pynvml = None
    _NVML_OK = False


def _collect_gpus():
    """GPU 状态列表（显存设备级；利用率/温度需 NVML）。

    优先只走 NVML（线程安全）：本函数由后台采样线程每秒调用，若在这里
    调 torch.cuda.mem_get_info / get_device_name，会与训练线程并发访问
    CUDA 运行时/分配器，Windows 上可能直接 access violation 带走进程。
    torch 仅在无 NVML 时兜底。
    """
    gpus = []
    count = 0
    if _NVML_OK:
        try:
            count = pynvml.nvmlDeviceGetCount()
        except Exception:
            count = 0
    if not count:
        # 无 NVML（未装 pynvml / 驱动不支持）：退回 torch
        if not _torch_ok():
            return gpus
        try:
            count = _torch_device_count()
        except Exception:
            return gpus
    for i in range(count):
        info = {"index": i, "name": f"GPU {i}",
                "memory_total_mb": None, "memory_used_mb": None,
                "memory_percent": None, "utilization": None,
                "temperature": None}
        if _NVML_OK:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "ignore")
                info["name"] = name
                info["utilization"] = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                info["temperature"] = pynvml.nvmlDeviceGetTemperature(
                    h, pynvml.NVML_TEMPERATURE_GPU)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                info["memory_total_mb"] = round(mem.total / 1048576)
                info["memory_used_mb"] = round(mem.used / 1048576)
                if mem.total:
                    info["memory_percent"] = round(
                        mem.used / mem.total * 100, 1)
            except Exception:
                pass
        # 仅 NVML 拿不到时用 torch 兜底（无 NVML 的机器上才会走到）
        if info["name"] == f"GPU {i}":
            try:
                info["name"] = _torch_device_name(i)
            except Exception:
                pass
        if info["memory_total_mb"] is None:
            try:
                free, total = _torch_mem_info(i)
                info["memory_total_mb"] = round(total / 1048576)
                info["memory_used_mb"] = round((total - free) / 1048576)
                if total:
                    info["memory_percent"] = round(
                        (total - free) / total * 100, 1)
            except Exception:
                pass
        gpus.append(info)
    return gpus


@router.get("/logs")
def backend_logs(tail: int = 200, cursor: int = 0):
    """后台日志缓冲；cursor>0 时仅返回其后的新记录（前端增量跟随用）"""
    from app.core import backend_log
    items = backend_log.get_logs(tail=max(1, min(tail, 1000)),
                                 cursor=max(0, cursor))
    return {"logs": items, "count": len(items)}


@router.delete("/logs")
def clear_backend_logs():
    """清空后台日志缓冲（不影响终端输出）"""
    from app.core import backend_log
    backend_log.clear_logs()
    return {"ok": True}


def _collect_hardware_status():
    """采集一次硬件状态（psutil/torch/NVML；可能被慢磁盘阻塞）。"""
    now = time.time()
    out = {"timestamp": now, "psutil_available": psutil is not None}

    # 系统信息（platform 始终可用；psutil 提供开机时间）
    system = {"hostname": platform.node() or "-",
              "os": f"{platform.system()} {platform.release()}",
              "python": platform.python_version(),
              "torch": _torch_version(),
              "cuda_available": _torch_ok()}
    if psutil is not None:
        try:
            boot = psutil.boot_time()
            system["boot_at"] = datetime.fromtimestamp(boot).strftime(
                "%Y-%m-%d %H:%M:%S")
            system["uptime_hours"] = round((now - boot) / 3600, 1)
        except Exception:
            pass
    out["system"] = system

    if psutil is None:
        out.update({"cpu": None, "memory": None, "disks": [],
                    "network": None,
                    "hint": "未安装 psutil：执行 pip install psutil 后可查看 "
                            "CPU / 内存 / 磁盘 / 网络状态"})
        out["gpus"] = _collect_gpus()
        return out

    # CPU
    cpu = {"percent": psutil.cpu_percent(interval=None),
           "per_core": psutil.cpu_percent(interval=None, percpu=True),
           "physical_cores": psutil.cpu_count(logical=False),
           "logical_cores": psutil.cpu_count(logical=True),
           "freq_mhz": None, "freq_max_mhz": None}
    try:
        freq = psutil.cpu_freq()
        if freq:
            cpu["freq_mhz"] = round(freq.current)
            cpu["freq_max_mhz"] = round(freq.max) if freq.max else None
    except Exception:
        pass
    out["cpu"] = cpu

    # 内存
    vm = psutil.virtual_memory()
    out["memory"] = {
        "total_gb": round(vm.total / 1073741824, 1),
        "used_gb": round(vm.used / 1073741824, 1),
        "available_gb": round(vm.available / 1073741824, 1),
        "percent": round(vm.percent, 1)}

    # 磁盘分区（固定磁盘；无法访问的分区跳过）
    disks = []
    try:
        parts = psutil.disk_partitions(all=False)
    except Exception:
        parts = []
    for part in parts:
        try:
            du = psutil.disk_usage(part.mountpoint)
            disks.append({
                "device": part.device.rstrip("\\"),
                "mountpoint": part.mountpoint, "fs": part.fstype,
                "total_gb": round(du.total / 1073741824, 1),
                "used_gb": round(du.used / 1073741824, 1),
                "percent": round(du.percent, 1)})
        except Exception:
            continue
    out["disks"] = disks

    # 网络（累计字节，速率由前端按相邻两次轮询差值计算）
    try:
        net = psutil.net_io_counters()
        out["network"] = {"bytes_sent": net.bytes_sent,
                          "bytes_recv": net.bytes_recv}
    except Exception:
        out["network"] = None

    out["gpus"] = _collect_gpus()
    return out


# 硬件状态快照：后台线程每秒采样一次，接口只返回缓存。
# 训练期间磁盘/CPU 繁忙时，psutil 调用可能变慢；若在请求线程里现采，
# 轮询请求会堆积并拖垮事件循环，浏览器表现为 "Failed to fetch"。
_hw_lock = threading.Lock()
_hw_snapshot = None
_hw_thread = None
_hw_interval = 1.0


def _hw_sampler_loop():
    global _hw_snapshot
    while True:
        try:
            data = _collect_hardware_status()
        except Exception as exc:
            data = {"timestamp": time.time(), "error": str(exc)}
        with _hw_lock:
            _hw_snapshot = data
        time.sleep(_hw_interval)


def _ensure_hw_sampler():
    global _hw_thread
    with _hw_lock:
        if _hw_thread is not None and _hw_thread.is_alive():
            return
        _hw_thread = threading.Thread(target=_hw_sampler_loop,
                                      name="hw-sampler", daemon=True)
        _hw_thread.start()


@router.get("/hardware")
async def hardware_status():
    """实时硬件状态（读取后台采样快照，避免阻塞训练期间的事件循环）。"""
    global _hw_snapshot
    _ensure_hw_sampler()
    with _hw_lock:
        snap = _hw_snapshot
    if snap is None:
        # 首次访问：交给线程池采集，避免阻塞事件循环
        snap = await run_in_threadpool(_collect_hardware_status)
        with _hw_lock:
            _hw_snapshot = snap
    return snap


# ------------------------- 网络代理（预训练模型下载） -------------------------

class ProxySettings(BaseModel):
    """代理配置（PUT /api/system/proxy 请求体）"""
    url: str = ""            # 代理地址，如 http://127.0.0.1:7890 / socks5://...
    enabled: bool = False    # 启用开关（true 时立即写入进程环境变量）
    no_proxy: str = ""       # 放行列表（默认 localhost,127.0.0.1,::1）
    test_url: str = ""       # 连通性测试地址（默认 https://huggingface.co）


class ProxyTestRequest(BaseModel):
    """代理连通性测试请求体（url 留空 = 测试已保存的代理）"""
    url: str = ""
    test_url: str = ""


class ProxySavedRequest(BaseModel):
    """添加代理到已存列表请求体（POST /api/system/proxy/saved）"""
    url: str = ""


@router.get("/proxy")
def get_proxy():
    """当前代理配置 + 环境变量生效状态 + socks 支持检测"""
    from app.core import proxy
    return proxy.get_status()


@router.put("/proxy")
def set_proxy(s: ProxySettings):
    """保存代理配置并立即应用（写入 proxy.json + 进程环境变量）"""
    from app.core import proxy
    try:
        return proxy.set_proxy(url=s.url, enabled=s.enabled,
                               no_proxy=s.no_proxy, test_url=s.test_url)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/proxy/saved")
def add_proxy_saved(req: ProxySavedRequest):
    """添加代理地址到已存列表（校验 + 去重；首个自动选中）"""
    from app.core import proxy
    try:
        return proxy.add_saved(req.url)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/proxy/saved")
def remove_proxy_saved(url: str = ""):
    """从已存列表移除代理（移除选中项时自动切换/停用）"""
    from app.core import proxy
    if not url.strip():
        raise HTTPException(400, "缺少 url 查询参数")
    return proxy.remove_saved(url)


@router.delete("/proxy")
def clear_proxy():
    """删除代理配置并恢复启动时的原始环境变量"""
    from app.core import proxy
    return proxy.clear_saved()


@router.post("/proxy/test")
def test_proxy(req: ProxyTestRequest):
    """测试代理连通性（可测试尚未保存的地址；返回延迟与状态码）"""
    from app.core import proxy
    return proxy.test_proxy(url=req.url, target=req.test_url)
