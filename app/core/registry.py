# -*- coding: utf-8 -*-
"""模型注册表：内置模型工厂注册 + 自定义模型/损失/优化器动态注册

自定义组件目录（nets/ 下，服务启动时自动扫描注册）：

    custom_models/       PyTorch 自定义模型（人群计数密度网等，torch.nn.Module）
    custom_losses.py    PyTorch 自定义损失（+ custom_losses/ 包目录）
    custom_optimizers.py PyTorch 自定义优化器（+ custom_optimizers/ 包目录）
"""
import importlib
import inspect
from pathlib import Path

# 三条训练路线（自定义模型/损失按此分区，前端按路线加载）
TASK_KEYS = ("crowd_density", "crowd_point", "crowd_box", "crowd_sdg")
_TASK_ALIASES = {
    "crowd_counting": "crowd_density",   # 旧任务 kind / 旧自定义损失标注
    "density": "crowd_density",
    "point": "crowd_point",
    "box": "crowd_box",
    "sdg": "crowd_sdg",
}

CUSTOM_REGISTRY = {}  # name -> {"factory", "desc", "tasks": [route...]}
LOSS_REGISTRY = {}    # name -> {"factory", "desc", "tasks": [route...]}
OPTIMIZER_REGISTRY = {}  # name -> {"factory": (parameters, lr, **kwargs)->optimizer, "desc": str}

_NETS_DIR = Path(__file__).resolve().parents[2] / "nets"


def normalize_task(task):
    """任务名归一化：旧 crowd_counting → crowd_density；其余原样"""
    t = str(task or "").strip()
    return _TASK_ALIASES.get(t, t)


def normalize_tasks(tasks, default=("crowd_density",)):
    """把 tasks 规范为去重后的合法路线列表；None/空 → default。

    未标注 tasks 时默认归入密度图（与旧版自定义模型行为一致）；
    需要其他路线请在注册时显式写 tasks=["crowd_point"] 等。
    """
    if isinstance(tasks, str):
        tasks = [tasks]
    out = []
    for t in (tasks or list(default)):
        t = normalize_task(t)
        if t in TASK_KEYS and t not in out:
            out.append(t)
    return out or list(default)


def register_custom_model(name, desc="", tasks=None):
    """自定义模型注册装饰器。

    tasks: 适用路线，可选 "crowd_density" / "crowd_point" / "crowd_box"
           （可多选；缺省 = crowd_density）。

    输出契约（按路线）：
      crowd_density: forward(x) -> [N,1,H',W'] 密度图
      crowd_point:   forward(x) -> {"pred_logits":[B,M,2],
                                    "pred_points":[B,M,2]}（P2PNet 接口）
      crowd_box:     factory() -> ultralytics 可加载的 .yaml / .pt / 模型名
    """
    def decorator(func):
        if not callable(func):
            raise TypeError("注册对象必须是可调用的工厂函数")
        CUSTOM_REGISTRY[name] = {"factory": func, "desc": desc,
                                 "tasks": normalize_tasks(tasks)}
        return func
    return decorator


def register_custom_loss(name, desc="", tasks=None):
    """自定义损失注册装饰器（factory(**kwargs) -> nn.Module）。

    tasks: 适用路线，缺省 = crowd_density。
      crowd_density: forward(pred[N,1,h,w], gt[N,1,h,w]) -> 标量
      crowd_point:   forward(outputs, targets) -> 标量（作为附加损失项，
                     与算法内置损失相加：P2PNet / CP / CU / CLTR 的 outputs
                     为 {pred_logits, pred_points} 字典，P2R 为
                     forward(pred_density, target_seq)，PET 不支持附加）
      crowd_box:     由 ultralytics 引擎托管，暂不使用自定义损失
    """
    def decorator(func):
        if not callable(func):
            raise TypeError("注册对象必须是可调用的工厂函数")
        LOSS_REGISTRY[name] = {"factory": func, "desc": desc,
                               "tasks": normalize_tasks(tasks)}
        return func
    return decorator


def register_custom_optimizer(name, desc=""):
    """自定义优化器注册装饰器

    factory(parameters, lr, momentum, weight_decay, **kwargs) -> torch.optim.Optimizer
    用法见 nets/custom_optimizers.py
    """
    def decorator(func):
        if not callable(func):
            raise TypeError("注册对象必须是可调用的工厂函数")
        OPTIMIZER_REGISTRY[name] = {"factory": func, "desc": desc}
        return func
    return decorator


# mtime 缓存，没改过的文件跳过 reload（扫描在每次 /tasks、/options 都会跑）
_SCAN_MT = {}


def _scan_file(py: Path, module_name: str):
    """导入或重载一个自定义模块，没改动就跳过，失败打日志返回 None"""
    import sys
    key = str(py)
    try:
        mt = py.stat().st_mtime
    except OSError:
        return None
    unchanged = _SCAN_MT.get(key) == mt and module_name in sys.modules
    if unchanged:
        return sys.modules[module_name]
    _SCAN_MT[key] = mt
    try:
        if module_name in sys.modules:
            mod = importlib.reload(sys.modules[module_name])
        else:
            mod = importlib.import_module(module_name)
        return mod
    except ModuleNotFoundError:
        # nets/ 不在 sys.path 就补上重试一次
        parent = str(Path(py).parents[1])
        if parent not in sys.path:
            sys.path.insert(0, parent)
        try:
            return importlib.import_module(module_name)
        except Exception as e:
            print(f"[registry] 自定义模块导入失败 {py.name}: "
                  f"{type(e).__name__}: {e}")
            _SCAN_MT.pop(key, None)   # 失败不记 mtime，改好了还会重试
            return None
    except Exception as e:
        print(f"[registry] 自定义模块导入失败 {py.name}: "
              f"{type(e).__name__}: {e}")
        _SCAN_MT.pop(key, None)
        return None


def scan_custom_models(custom_dir):
    """扫描自定义组件目录并导入注册（增量，没改动的文件跳过）

    nets/ 需在 sys.path：custom_losses 以顶层模块名导入、custom_models
    以包名导入，二者的搜索根都是 nets/（main.py 的 lifespan 只插入了
    项目根，这里补上，否则自定义损失注册始终失败）。
    """
    import sys
    if str(_NETS_DIR) not in sys.path:
        sys.path.insert(0, str(_NETS_DIR))
    custom_dir = Path(custom_dir)
    if not custom_dir.exists():
        return list(CUSTOM_REGISTRY.keys())
    for py in custom_dir.glob("*.py"):
        # 跳过 __init__ 与所有 example_*.py 模板（复制改名后才会注册）
        if py.name.startswith("_") or py.name.startswith("example_"):
            continue
        mod = _scan_file(py, f"custom_models.{py.stem}")
        if mod is None:
            continue
        # 固定名 build_model 兜底
        fn = getattr(mod, "build_model", None)
        if callable(fn) and py.stem not in CUSTOM_REGISTRY:
            CUSTOM_REGISTRY[py.stem] = {
                "factory": fn, "desc": "build_model 固定入口",
                "tasks": normalize_tasks(None)}
    # 自定义损失目录（nets/custom_losses.py + nets/custom_losses/ 包内 .py）
    loss_mod_dir = custom_dir.parent / "custom_losses"
    loss_file = custom_dir.parent / "custom_losses.py"
    if loss_file.is_file():
        _scan_file(loss_file, "custom_losses")
    if loss_mod_dir.is_dir():
        for py in loss_mod_dir.glob("*.py"):
            if not py.name.startswith("_"):
                _scan_file(py, f"custom_losses.{py.stem}")
    # 自定义优化器（nets/custom_optimizers.py + nets/custom_optimizers/ 包内 .py）
    opt_mod_dir = custom_dir.parent / "custom_optimizers"
    opt_file = custom_dir.parent / "custom_optimizers.py"
    if opt_file.is_file():
        _scan_file(opt_file, "custom_optimizers")
    if opt_mod_dir.is_dir():
        for py in opt_mod_dir.glob("*.py"):
            if not py.name.startswith("_"):
                _scan_file(py, f"custom_optimizers.{py.stem}")
    return list(CUSTOM_REGISTRY.keys())


def ensure_scanned():
    """确保全部自定义组件目录已扫描（服务启动 / options 端点调用，幂等）"""
    return scan_custom_models(_NETS_DIR / "custom_models")


def get_custom_loss(name, task=None, **kwargs):
    """构建自定义损失：name 形如 'custom:focal' 或直接注册名。
    task 给出时会校验该损失是否注册到了对应路线。"""
    key = name.split(":", 1)[1] if ":" in name else name
    meta = LOSS_REGISTRY.get(key)
    if not meta:
        raise KeyError(f"未注册的自定义损失: {key}，可用: {list(LOSS_REGISTRY)}")
    if task and normalize_task(task) not in meta.get("tasks", []):
        raise KeyError(
            f"自定义损失 {key} 未注册到任务 {task}"
            f"（已注册: {meta.get('tasks')}）")
    return meta["factory"](**kwargs)


def get_custom_model(name, num_classes, pretrained=False, task=None, **kwargs):
    """构建自定义模型；task 给出时校验该模型是否注册到了对应路线。"""
    meta = CUSTOM_REGISTRY.get(name)
    if not meta:
        raise KeyError(f"未注册的自定义模型: {name}，可用: {list(CUSTOM_REGISTRY)}")
    if task and normalize_task(task) not in meta.get("tasks", []):
        raise KeyError(
            f"自定义模型 {name} 未注册到任务 {task}"
            f"（已注册: {meta.get('tasks')}）")
    factory = meta["factory"]
    sig = inspect.signature(factory)
    params = {}
    if "num_classes" in sig.parameters:
        params["num_classes"] = num_classes
    if "pretrained" in sig.parameters:
        params["pretrained"] = pretrained
    params.update(kwargs)
    return factory(**params)


def custom_models_for(task):
    """某路线可用的自定义模型：[{name, factory, desc, tasks}]"""
    task = normalize_task(task)
    return [{"name": k, **v} for k, v in CUSTOM_REGISTRY.items()
            if task in v.get("tasks", [])]


def custom_losses_for(task):
    """某路线可用的自定义损失：[{name, factory, desc, tasks}]"""
    task = normalize_task(task)
    return [{"name": k, **v} for k, v in LOSS_REGISTRY.items()
            if task in v.get("tasks", [])]


def is_custom_model_for(name, task):
    meta = CUSTOM_REGISTRY.get(str(name or ""))
    return bool(meta and normalize_task(task) in meta.get("tasks", []))


def is_custom_loss_for(name, task):
    key = str(name or "").split(":", 1)[-1]
    meta = LOSS_REGISTRY.get(key)
    return bool(meta and normalize_task(task) in meta.get("tasks", []))


def custom_model_tasks(name):
    meta = CUSTOM_REGISTRY.get(str(name or ""))
    return list(meta.get("tasks", [])) if meta else []


def get_custom_model_source(name):
    """自定义头部框模型：factory() 返回 ultralytics 可加载的源（.yaml/.pt/名称）"""
    meta = CUSTOM_REGISTRY.get(str(name or ""))
    if not meta:
        raise KeyError(f"未注册的自定义模型: {name}")
    return meta["factory"]()


def get_custom_optimizer(name, parameters, lr, **kwargs):
    """构建自定义优化器：name 形如 'custom:my_optim' 或直接注册名"""
    key = name.split(":", 1)[1] if ":" in name else name
    if key not in OPTIMIZER_REGISTRY:
        raise KeyError(f"未注册的自定义优化器: {key}，可用: {list(OPTIMIZER_REGISTRY)}")
    return OPTIMIZER_REGISTRY[key]["factory"](parameters, lr, **kwargs)
