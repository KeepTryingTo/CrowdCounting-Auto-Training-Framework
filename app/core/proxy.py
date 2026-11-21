# -*- coding: utf-8 -*-
"""网络代理配置（预训练权重下载用）

配置存项目根 proxy.json，启动时加载，enabled 时写进环境变量
HTTP_PROXY / HTTPS_PROXY / ALL_PROXY / NO_PROXY（大小写各一套）。
requests / urllib / torch.hub 均识别这些变量，DDP 子进程继承父进程
环境同样生效。NO_PROXY 放行本机回环，浏览器与后端间的请求不走代理。
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

# 代理配置文件（项目根，与 main.py 同级）
PROXY_CONFIG_FILE = Path(__file__).resolve().parents[2] / "proxy.json"

# 默认放行本机回环（防止本地 API 请求走代理）
DEFAULT_NO_PROXY = "localhost,127.0.0.1,::1"

# 默认连通性测试地址（HF 是最常被限的资源；界面可改）
DEFAULT_TEST_URL = "https://huggingface.co"

# 默认代理地址（界面输入框 Tab 自动补全的默认值）
DEFAULT_PROXY_URL = "http://127.0.0.1:7890"

# 受管理的环境变量键（大小写各一套；清除时恢复进程启动时的原始值）
_MANAGED_KEYS = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                 "http_proxy", "https_proxy", "all_proxy", "no_proxy"]

# 进程启动时的原始环境值快照（import 本模块时记录，用于恢复）
_ORIG_ENV = {k: os.environ.get(k) for k in _MANAGED_KEYS}


# ------------------------- 配置读写 -------------------------

def load() -> dict:
    """读取 proxy.json；无文件 / 损坏时返回空配置"""
    empty = {"enabled": False, "url": "", "saved": [],
             "no_proxy": DEFAULT_NO_PROXY,
             "test_url": DEFAULT_TEST_URL, "updated_at": None}
    try:
        raw = PROXY_CONFIG_FILE.read_text(encoding="utf-8")
        cfg = json.loads(raw)
        if not isinstance(cfg, dict):
            return empty
        out = dict(empty)
        out["enabled"] = bool(cfg.get("enabled", False))
        out["url"] = str(cfg.get("url") or "").strip()
        saved = cfg.get("saved")
        if isinstance(saved, list):
            out["saved"] = [str(u).strip() for u in saved if str(u).strip()]
        out["no_proxy"] = str(cfg.get("no_proxy") or "").strip() \
            or DEFAULT_NO_PROXY
        out["test_url"] = str(cfg.get("test_url") or "").strip() \
            or DEFAULT_TEST_URL
        out["updated_at"] = cfg.get("updated_at")
        return out
    except Exception:
        return empty


def _save(cfg: dict):
    cfg = dict(cfg)
    cfg["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    PROXY_CONFIG_FILE.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------- 应用 / 清除 -------------------------

def _normalize_url(url: str) -> str:
    """规范化代理地址：允许省略协议（默认 http://）"""
    url = (url or "").strip()
    if not url:
        return ""
    if "://" not in url:
        url = "http://" + url
    return url


def _validate_url(url: str):
    """校验代理地址格式；非法时抛 ValueError（支持 http/https/socks5/socks5h）"""
    from urllib.parse import urlparse
    p = urlparse(url)
    if p.scheme not in ("http", "https", "socks5", "socks5h"):
        raise ValueError(
            f"不支持的代理协议: {p.scheme or '(空)'}"
            f"（支持 http:// 、https:// 、socks5:// 、socks5h://）")
    if not p.hostname or not p.port:
        raise ValueError(
            f"代理地址需包含主机与端口，如 http://127.0.0.1:7890（当前: {url}）")


def _restore_orig_env():
    """恢复进程启动时的原始代理环境变量（未设置过的键删除）"""
    for k in _MANAGED_KEYS:
        v = _ORIG_ENV.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _apply_env(cfg: dict):
    """将配置写入进程环境变量（覆盖旧的受管理键）"""
    _restore_orig_env()
    if not (cfg.get("enabled") and cfg.get("url")):
        return
    url = cfg["url"]
    no_proxy = cfg.get("no_proxy") or DEFAULT_NO_PROXY
    for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
              "ALL_PROXY", "all_proxy"):
        os.environ[k] = url
    for k in ("NO_PROXY", "no_proxy"):
        # 保留启动时已有的 NO_PROXY 附加项
        orig = _ORIG_ENV.get(k)
        merged = no_proxy if not orig else f"{no_proxy},{orig}"
        os.environ[k] = merged


def init():
    """服务启动时调用：加载已保存配置，enabled 时应用默认代理"""
    cfg = load()
    if cfg.get("enabled") and cfg.get("url"):
        _apply_env(cfg)
        print(f"[proxy] 已应用网络代理: {cfg['url']}"
              f"（配置: {PROXY_CONFIG_FILE.name}）")
    else:
        _restore_orig_env()


# ------------------------- 对外操作 -------------------------

def set_proxy(url: str, enabled: bool, no_proxy: str = "",
              test_url: str = "") -> dict:
    """保存并立即应用/停用。url 非空时顺手加进 saved 列表"""
    url = _normalize_url(url)
    if enabled and not url:
        raise ValueError("启用代理时必须填写代理地址（如 http://127.0.0.1:7890）")
    if enabled and url:
        _validate_url(url)
    cfg = load()
    cfg["url"] = url
    cfg["enabled"] = bool(enabled and url)
    if url and url not in cfg["saved"]:
        cfg["saved"] = [url] + cfg["saved"]
    if no_proxy.strip():
        cfg["no_proxy"] = no_proxy.strip()
    if test_url.strip():
        cfg["test_url"] = test_url.strip()
    _save(cfg)
    _apply_env(cfg)
    return get_status()


def add_saved(url: str) -> dict:
    """加进 saved 列表，去重；列表原本为空时自动选中（不启用）"""
    url = _normalize_url(url)
    if not url:
        raise ValueError("请填写代理地址（如 http://127.0.0.1:7890）")
    _validate_url(url)
    cfg = load()
    added = url not in cfg["saved"]
    if added:
        cfg["saved"] = [url] + cfg["saved"]
    if not cfg["url"]:
        cfg["url"] = url  # 首个添加的代理自动选中（不启用）
    _save(cfg)
    _apply_env(cfg)
    return dict(get_status(), added=added)


def remove_saved(url: str) -> dict:
    """从 saved 移除。删的是选中项就切到剩下的第一个，启用中则直接切换"""
    url = _normalize_url(url)
    cfg = load()
    if url in cfg["saved"]:
        cfg["saved"] = [u for u in cfg["saved"] if u != url]
    if url == cfg["url"]:
        cfg["url"] = cfg["saved"][0] if cfg["saved"] else ""
        cfg["enabled"] = bool(cfg["enabled"] and cfg["url"])
    _save(cfg)
    _apply_env(cfg)
    return get_status()


def clear_saved() -> dict:
    """删除配置文件并恢复原始环境变量（回到未配置状态）"""
    try:
        PROXY_CONFIG_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    _restore_orig_env()
    return get_status()


def get_status() -> dict:
    """当前配置 + 环境变量实际生效状态 + socks 支持检测 + 默认代理"""
    cfg = load()
    env_https = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    return {
        "config": cfg,
        "config_file": str(PROXY_CONFIG_FILE),
        "active": bool(cfg.get("enabled") and cfg.get("url")
                       and env_https == cfg["url"]),
        "env_https_proxy": env_https or "",
        "env_no_proxy": os.environ.get("NO_PROXY")
                        or os.environ.get("no_proxy") or "",
        "socks_supported": _socks_supported(),
        "default_url": DEFAULT_PROXY_URL,
        "managed_keys": _MANAGED_KEYS,
    }


def _socks_supported() -> bool:
    """socks5 代理需要 requests + pysocks（requests[socks]）"""
    try:
        import socks  # noqa: F401
        return True
    except ImportError:
        return False


def test_proxy(url: str = "", target: str = "", timeout: int = 10) -> dict:
    """测连通性。requests 优先（支持 socks5），没有就 urllib 兜底"""
    cfg = load()
    url = _normalize_url(url) or cfg.get("url") or ""
    target = (target or "").strip() or cfg.get("test_url") or DEFAULT_TEST_URL
    out = {"ok": False, "url": url, "target": target,
           "latency_ms": None, "status_code": None, "via": "", "error": ""}
    if not url:
        out["error"] = "未提供代理地址（先填写或先保存代理配置）"
        return out
    try:
        _validate_url(url)
    except ValueError as e:
        out["error"] = str(e)
        return out

    scheme = url.split("://")[0].lower()
    if scheme.startswith("socks") and not _socks_supported():
        out["error"] = ("socks 代理测试需要 pysocks："
                        "pip install requests[socks]（或 pip install pysocks）")
        return out

    # requests（支持 socks5）
    try:
        import requests
        t0 = time.time()
        try:
            r = requests.get(target, proxies={"http": url, "https": url},
                             timeout=timeout, allow_redirects=False)
            out.update(ok=r.status_code < 500, status_code=r.status_code,
                       latency_ms=int((time.time() - t0) * 1000),
                       via="requests")
            if r.status_code >= 500:
                out["error"] = f"目标返回 HTTP {r.status_code}"
        except requests.exceptions.RequestException as e:
            out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return out
    except ImportError:
        pass

    # urllib 兜底（不支持 socks）
    if scheme.startswith("socks"):
        out["error"] = "当前环境无 requests，无法测试 socks 代理"
        return out
    import urllib.request
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": url, "https": url}))
    t0 = time.time()
    try:
        with opener.open(target, timeout=timeout) as resp:
            out.update(ok=resp.status < 500, status_code=resp.status,
                       latency_ms=int((time.time() - t0) * 1000),
                       via="urllib")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out
