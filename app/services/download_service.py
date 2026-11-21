# -*- coding: utf-8 -*-
"""数据集下载服务：从 URL 下载 zip 数据集到 uploads/datasets/

支持：
- 任意可直接下载的 zip URL
- Google Drive（自动转换为直链格式 + 处理大文件确认页）
- Dropbox（自动确保 dl=1 参数）
- OneDrive / SharePoint（自动确保 download=1 参数）
- 预设数据集下载链接（常用人群计数数据集官网）
- 下载进度实时反馈（task.progress）
- 下载完成后自动解压到 uploads/datasets/<名称>/ 与上传数据集共用目录
- 已有同名数据集会被覆盖（被运行中任务占用的拒绝）
"""
import os
import re
import shutil
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

# ---------------------------------------------------------------------------
# 常用人群计数数据集下载链接预设（前端下拉框直接选用）
# 链接依据各数据集官方发布渠道核对（参考 CSDN 博客《人群计数中常用数据集
# 的总结以及使用方式（Python/PyTorch）》整理的官方下载地址，逐条验证可达）：
# - UCF-QNRF：官网 zip 直链（可直接下载）
# - JHU-CROWD++：官网 crowd-counting.com（Dropbox / Google Drive / OneDrive
#   三种渠道均为分享文件夹 / 需浏览器交互，无法程序化直链下载）
# - NWPU-Crowd：官网主页（数据在其 OneDrive 文件夹，浏览器打开下载）
# - ShanghaiTech A/B：DataFountain（原官网 Google Sites 已需 Google 登录）
# - UCF_CC_50：官网数据集页（官方发布为 .rar，解压后需重新打包 zip 上传）
# 注意：仅 url 指向「可直接下载的 zip」时下载才会成功；标注「需手动获取直链」
# 的条目指向官网介绍页（需注册/登录后获取直链，或手动下载后走上传），
# 直接点下载会被网页检测拦截并给出指引。
# ---------------------------------------------------------------------------
DATASET_DOWNLOAD_LINKS = [
    {"key": "qnrf_official", "label": "UCF-QNRF (官网直链，可直接下载)",
     "url": "https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip",
     "size": "~4.3 GB", "desc": "1535 张高分辨率高密度 / 1.25M 点标注"},
    {"key": "jhu_official", "label": "JHU-CROWD++（官网，需手动获取直链）",
     "url": "https://www.crowd-counting.com/",
     "size": "~3.4 GB",
     "desc": "4372 张 / 1.51M 点标注，含头部框 / 天气 / 场景标签；"
             "官网提供 Dropbox / Google Drive / OneDrive 三种下载渠道"},
    {"key": "nwpu_official", "label": "NWPU-Crowd（官网，需手动获取直链）",
     "url": "https://gjy3035.github.io/NWPU-Crowd-Sample-Code/",
     "size": "~8 GB",
     "desc": "5109 张 / 2.13M 点标注，含框标注 / 亮度 / 场景；"
             "数据在官网提供的 OneDrive 文件夹（浏览器打开下载）"},
    {"key": "shha_official",
     "label": "ShanghaiTech Part A（DataFountain，需登录后手动下载）",
     "url": "https://www.datafountain.cn/datasets/5670",
     "size": "~170 MB",
     "desc": "482 张 / 241677 点标注（高密度场景）；原官网 Google Sites "
             "已需 Google 登录，A / B 在同一 DataFountain 页面"},
    {"key": "shb_official",
     "label": "ShanghaiTech Part B（DataFountain，需登录后手动下载）",
     "url": "https://www.datafountain.cn/datasets/5670",
     "size": "~60 MB",
     "desc": "716 张 / 88488 点标注（低密度街道场景）；与 Part A 同一 "
             "DataFountain 页面"},
    {"key": "ucf50_official", "label": "UCF_CC_50（官网，需手动获取直链）",
     "url": "https://www.crcv.ucf.edu/data/ucf-cc-50/",
     "size": "~14 MB",
     "desc": "50 张极高密度 / 63974 点标注；官方发布为 .rar 包"
             "（解压后请重新打包为 zip 再上传）"},
]


# ---------------------------------------------------------------------------
# URL 预处理：自动转换为直链格式
# ---------------------------------------------------------------------------

def _normalize_url(url: str) -> str:
    """将常见云存储分享链接转换为直链下载格式。

    - Google Drive: open?id= / file/d/xxx/view → uc?export=download&id=xxx
    - Dropbox: 确保 dl=1（dl=0 是预览页）
    - OneDrive/SharePoint: 确保 download=1
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path
    qs = parse_qs(parsed.query)

    # ---- Google Drive ----
    if "drive.google.com" in host or "docs.google.com" in host:
        file_id = ""
        # 格式1: /file/d/<ID>/view
        m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", path)
        if m:
            file_id = m.group(1)
        # 格式2: ?id=<ID> 或 open?id=<ID>
        if not file_id and "id" in qs:
            file_id = qs["id"][0]
        # 格式3: /uc?export=download&id=<ID>（已是直链）
        if not file_id and "id" in qs:
            file_id = qs["id"][0]
        if file_id:
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        # 无法提取 ID → 返回原 URL
        return url

    # ---- Dropbox ----
    if "dropbox.com" in host:
        # 确保 dl=1（dl=0 或无 dl 参数均为预览页）
        new_qs = dict(qs)
        new_qs["dl"] = ["1"]
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path,
                          parsed.params, urlencode(new_qs, doseq=True),
                           parsed.fragment))

    # ---- OneDrive / SharePoint ----
    if "sharepoint.com" in host or "1drv.ms" in host or "onedrive.live.com" in host:
        # 确保 download=1
        if "download" not in qs:
            new_qs = dict(qs)
            new_qs["download"] = ["1"]
            return urlunparse((parsed.scheme, parsed.netloc, parsed.path,
                               parsed.params,
                               urlencode(new_qs, doseq=True),
                               parsed.fragment))
        return url

    return url


# ---------------------------------------------------------------------------
# Google Drive 大文件确认页处理
# ---------------------------------------------------------------------------

def _extract_gdrive_confirm_token(html_content: bytes):
    """从 Google Drive 确认页 HTML 中提取 confirm token / uuid。

    Google Drive 对大文件（>100MB）会显示病毒扫描确认页，
    页面中包含一个表单，需要带上 confirm token 和 uuid 重试。
    """
    try:
        text = html_content.decode("utf-8", errors="ignore")
    except Exception:
        return None

    # 方式1：表单 action + hidden input
    # <form id="download-form" action="https://drive.usercontent.google.com/download" ...
    #   <input type="hidden" name="id" value="xxx">
    #   <input type="hidden" name="export" value="download">
    #   <input type="hidden" name="confirm" value="t">
    #   <input type="hidden" name="uuid" value="xxx">
    confirm = None
    uuid_val = None

    # 方式2：JavaScript 变量（新版 GD）
    m = re.search(r'name="confirm"\s+value="([^"]+)"', text)
    if m:
        confirm = m.group(1)
    m = re.search(r'name="uuid"\s+value="([^"]+)"', text)
    if m:
        uuid_val = m.group(1)

    # 方式3：直接搜索 confirm=t 模式
    if not confirm:
        m = re.search(r'confirm=([a-zA-Z0-9_-]+)', text)
        if m:
            confirm = m.group(1)

    # 方式4：搜索 action URL 中的参数
    m = re.search(r'action="(https://drive\.usercontent\.google\.com/download\?[^"]+)"', text)
    if m:
        # 直接返回完整的重试 URL
        return ("url", m.group(1))

    if confirm:
        return ("params", {"confirm": confirm, "uuid": uuid_val})
    return None


# ---------------------------------------------------------------------------
# 主下载函数
# ---------------------------------------------------------------------------

def run_download(task, params: dict):
    """后台任务：从 URL 下载数据集 zip 并解压到 uploads/datasets/

    params:
      url:       下载链接（支持 Google Drive / Dropbox / OneDrive / 直链 zip）
      name:      数据集名（留空自动从 URL 推断）
      upload_dir: uploads/datasets/ 目录路径
    """
    import urllib.request
    import urllib.error

    url = params["url"]
    name = params.get("name") or ""
    upload_dir = Path(params["upload_dir"])

    # ---- 1) URL 预处理（自动转直链） ----
    original_url = url
    url = _normalize_url(url)
    if url != original_url:
        task.log(f"URL 已自动转换为直链格式: {url}")
    task.log(f"开始下载: {url}")

    # ---- 2) 名称推断 ----
    if not name:
        parsed = urlparse(url)
        base = Path(parsed.path).name or "downloaded_dataset"
        name = base.rsplit(".", 1)[0] if base.endswith(".zip") else base
        # Google Drive 的 uc 路径没有名称，用 id
        if "drive.google.com" in url and name in ("download", "uc"):
            qs = parse_qs(parsed.query)
            if "id" in qs:
                name = f"gdrive_{qs['id'][0][:16]}"
        name = re.sub(r"[^\w\-]", "_", name)[:60] or "dataset"
    # 同名数据集不覆盖：解压时自动追加 _1 / _2 … 后缀区分。
    # 临时 zip 加随机后缀，避免并发下载同名数据集写同一个文件
    tmp_zip = upload_dir / f"_tmp_dl_{uuid.uuid4().hex[:8]}_{name}.zip"
    task.log(f"数据集名: {name}（若同名已存在，将自动追加 _1/_2 后缀）")

    # ---- 4) 下载（含 HTML 检测 + Google Drive 确认页重试） ----
    MAX_RETRIES = 3
    chunk_size = 1024 * 512   # 512 KB
    USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36")

    def _attempt_download(dl_url, extra_params=None):
        """单次下载尝试，返回 (resp, is_zip_first_bytes)。

        is_zip_first_bytes: 前几个字节是否为 zip magic（PK\\x03\\x04）
        """
        if extra_params:
            # 合并额外参数到 URL
            p = urlparse(dl_url)
            qs = parse_qs(p.query)
            qs.update(extra_params)
            dl_url = urlunparse((p.scheme, p.netloc, p.path,
                                 p.params, urlencode(qs, doseq=True),
                                 p.fragment))
        req = urllib.request.Request(dl_url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Encoding": "identity",  # 禁用 gzip，方便检测 magic bytes
        })
        resp = urllib.request.urlopen(req, timeout=120)
        return resp

    def _download_to_file(resp, filepath):
        """流式下载到文件 + 进度更新"""
        total = int(resp.headers.get("Content-Length", 0) or 0)
        content_type = resp.headers.get("Content-Type", "").lower()
        task.log(f"Content-Type: {content_type}" +
                 (f" | 文件大小: {total / 1048576:.1f} MB" if total else
                  " | 文件大小: 未知"))
        downloaded = 0
        first_bytes = None
        with open(filepath, "wb") as f:
            while True:
                if task.stop_event.is_set():
                    filepath.unlink(missing_ok=True)
                    raise StopIteration()
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                if first_bytes is None:
                    first_bytes = chunk[:8]
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    task.progress = min(96.0, downloaded / total * 96.0)
                else:
                    task.progress = min(90.0, downloaded / (100 * 1048576))
        return first_bytes, downloaded, content_type, total

    last_error = None
    current_url = url

    for attempt in range(MAX_RETRIES):
        if task.stop_event.is_set():
            tmp_zip.unlink(missing_ok=True)
            raise StopIteration()
        try:
            task.log(f"下载尝试 {attempt + 1}/{MAX_RETRIES}: {current_url[:100]}")
            resp = _attempt_download(current_url)
            first_bytes, downloaded, content_type, total = \
                _download_to_file(resp, tmp_zip)

            # ---- 检查是否为有效 zip ----
            is_zip = (first_bytes is not None and
                      len(first_bytes) >= 4 and
                      first_bytes[:2] == b"PK")
            if is_zip:
                task.log(f"下载完成: {downloaded / 1048576:.1f} MB，"
                         "确认为 zip 文件")
                break

            # ---- 不是 zip：可能是 HTML（Google Drive 确认页 / 404 页面） ----
            # 读取已下载内容的前几 KB 检测
            with open(tmp_zip, "rb") as f:
                head = f.read(8192)
            head_str = ""
            try:
                head_str = head.decode("utf-8", errors="ignore")
            except Exception:
                pass

            is_html = (head_str.lstrip().startswith("<") or
                       "text/html" in content_type)

            if is_html:
                tmp_zip.unlink(missing_ok=True)
                task.log(f"收到 HTML 页面（非 zip），尝试解析重定向...")

                # Google Drive 确认页处理
                if "drive.google.com" in current_url or \
                        "drive.usercontent.google.com" in current_url:
                    result = _extract_gdrive_confirm_token(head)
                    if result:
                        kind, data = result
                        if kind == "url":
                            # 直接拿到了完整的重试 URL
                            current_url = data
                            task.log(f"Google Drive 确认页 → 重试: "
                                     f"{data[:80]}")
                            continue
                        elif kind == "params":
                            # 需要带 confirm/uuid 参数重试
                            current_url = _normalize_url(original_url)
                            extra = data
                            task.log(f"Google Drive 确认 token: {data}")
                            # 手动构建带参数的 URL
                            p = urlparse(current_url)
                            qs = parse_qs(p.query)
                            for k, v in extra.items():
                                if v:
                                    qs[k] = [v]
                            current_url = urlunparse(
                                (p.scheme, p.netloc, p.path, p.params,
                                 urlencode(qs, doseq=True), p.fragment))
                            continue

                # 非 Google Drive 或无法解析 → 给出详细错误
                # 检查是否为 404
                if "404" in head_str or "not found" in head_str.lower():
                    raise ValueError(
                        f"URL 返回 404（文件不存在）: {current_url[:80]}。"
                        f"请检查链接是否正确。")

                # 检查是否需要登录
                if "sign in" in head_str.lower() or \
                        "login" in head_str.lower():
                    raise ValueError(
                        f"URL 需要登录才能访问: {current_url[:80]}。"
                        f"请先在浏览器中登录后获取直链下载地址。")

                # 通用 HTML 错误
                raise ValueError(
                    f"URL 返回的是网页（HTML）而非 zip 文件。"
                    f"Content-Type: {content_type}。"
                    f"\n常见原因与解决方案："
                    f"\n  ① Google Drive 大文件需确认下载 —— 已自动尝试，如仍失败请手动获取直链"
                    f"\n  ② 链接为数据集介绍页而非 zip 直链 —— 请在浏览器中下载数据集 zip 后通过「上传数据集」使用"
                    f"\n  ③ 链接需要登录/权限 —— 请将分享设置改为「任何人都可查看」"
                    f"\n原始链接: {original_url[:80]}")
            else:
                # 不是 zip 也不是 HTML → 未知格式
                tmp_zip.unlink(missing_ok=True)
                magic_hex = " ".join(f"{b:02x}" for b in (first_bytes or b""))
                raise ValueError(
                    f"下载的文件不是 zip 格式（magic bytes: {magic_hex}，"
                    f"Content-Type: {content_type}）。"
                    f"请确认链接指向 .zip 文件。")

        except StopIteration:
            raise
        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            last_error = e
            tmp_zip.unlink(missing_ok=True)
            if attempt < MAX_RETRIES - 1:
                task.log(f"下载出错（第 {attempt + 1} 次）: {e}，重试...")
                continue
            else:
                raise RuntimeError(
                    f"下载失败（已重试 {MAX_RETRIES} 次）: {last_error}。"
                    f"\n请检查："
                    f"\n  ① 链接是否可直接在浏览器中下载 zip"
                    f"\n  ② 网络是否可达（可配置「代理设置」）"
                    f"\n  ③ Google Drive / Dropbox 分享是否设为「任何人可查看」")
    else:
        # for-else：所有重试都未 break（均收到 HTML 但无法处理）
        tmp_zip.unlink(missing_ok=True)
        raise ValueError(
            f"多次尝试后仍无法获取 zip 文件。"
            f"请手动在浏览器中下载数据集 zip 后通过「上传数据集」使用。"
            f"\n原始链接: {original_url[:80]}")

    # ---- 5) 验证 zip 并解压 ----
    task.progress = 97.0
    task.log("正在解压...")
    try:
        zf = zipfile.ZipFile(tmp_zip)
    except zipfile.BadZipFile:
        tmp_zip.unlink(missing_ok=True)
        raise ValueError("文件损坏或不是有效 zip（下载可能不完整）")
    # 同名数据集不覆盖：原子占位唯一目录（name / name_1 / name_2 …）
    from app.core.dataset_naming import unique_dataset_name
    dest_dir = unique_dataset_name(upload_dir, name)
    name = dest_dir.name          # 实际目录名（可能带 _1 / _2 后缀）
    task.log(f"保存目录: {dest_dir}")
    try:
        with zf:
            infos = zf.infolist()
            if len(infos) > 200000:
                raise ValueError(f"zip 条目数 {len(infos)} 超过上限 200000，已拒绝")
            total_unc = sum(i.file_size for i in infos)
            if total_unc > 50 * 1024 ** 3:
                raise ValueError(f"解压后总量 {total_unc / 1024**3:.1f} GB "
                                 f"超过 50 GB 上限")
            zf.extractall(dest_dir)
    except Exception:
        # 含 BadZipFile / 磁盘错误：清理解压了一半的目录，避免残留脏数据
        shutil.rmtree(dest_dir, ignore_errors=True)
        tmp_zip.unlink(missing_ok=True)
        raise
    finally:
        tmp_zip.unlink(missing_ok=True)

    # ---- 6) 单一顶层目录下探（跳过辅助项，冲突名加后缀） ----
    entries = [p for p in dest_dir.iterdir()
               if not p.name.startswith("__")]
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

    # ---- 6.5) 常用数据集自动统一为 Train/Images + GT(.txt) 格式 ----
    # 命中 ShanghaiTech/QNRF/NWPU/UCF_CC_50 布局时原地转换，
    # 与自定义数据集格式统一（转换成功后自动清理原始目录与标注、
    # 仅保留统一结构；未命中/已统一/JHU 原生可读则跳过，
    # 转换失败不影响下载）
    unified = None
    try:
        from app.services.dataset_unify import unify_dataset_if_common
        unified = unify_dataset_if_common(
            dest_dir, log=task.log,
            progress=lambda f: setattr(task, "progress", 97.0 + 3.0 * f),
            stop_event=task.stop_event)
    except StopIteration:
        raise
    except Exception as e:
        task.log(f"[unify] 自动统一格式未执行: {e}")

    # ---- 7) 统计 ----
    task.progress = 100.0
    task.log("解压完成，正在统计...")
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    n_img = sum(1 for p in dest_dir.rglob("*")
                if p.suffix.lower() in img_exts and p.is_file())
    n_txt = sum(1 for p in dest_dir.rglob("*.txt") if p.is_file())
    n_npy = sum(1 for p in dest_dir.rglob("*.npy") if p.is_file())
    n_mat = sum(1 for p in dest_dir.rglob("*.mat") if p.is_file())

    task.result = {
        "name": name, "path": str(dest_dir),
        "num_images": n_img, "num_gt_txt": n_txt,
        "num_gt_npy": n_npy, "num_gt_mat": n_mat,
        "unified": unified,
    }
    if unified:
        task.log(f"已自动统一为框架标准格式（{unified['preset']} 布局："
                 f"Train {unified['train']} / Test {unified['test']} 张，"
                 "GT → .txt 点标注），可直接训练")
    task.log(f"下载完成: {name} | 图像 {n_img} | GT: txt {n_txt} / "
             f"npy {n_npy} / mat {n_mat}")
    task.log(f"保存到: {dest_dir}（可直接在训练模式的数据集下拉框中选用）")
