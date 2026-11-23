# -*- coding: utf-8 -*-
"""常用人群计数数据集 → 框架统一格式自动转换

下载 / 上传解压完成后自动探测常用数据集布局（ShanghaiTech A/B、
UCF-QNRF、NWPU-Crowd、UCF_CC_50），原地转换为统一格式：

    dataset_root/
    ├── Train/
    │   ├── Images/       *.jpg / *.png
    │   └── GT/           *.txt（每行 "x y" 点标注）
    └── Test/
        ├── Images/
        └── GT/

GT 统一为 .txt 点标注（各数据集标注的无损共同形式）：密度图路线训练时
按 σ 现场生成密度图，点预测路线直接读取，两条路线通用。

设计要点：
- 已是统一格式 / 框架原生可读（JHU-CROWD++ 的 train/images+gt 布局
  即原生可读，且与转换目标目录在 Windows 上同名冲突，明确跳过）/
  未命中任何预设 → 不做任何改动（返回 None）
- 转换成功后自动清理原始数据：数据集仅保留 Train/ Test/ 统一结构
  （原图硬链接进 Images/ 不受删除影响；转换失败 / 中止不清理，
  可安全重试；重复执行幂等——已统一即跳过）
- 图像硬链接（同盘不占额外空间，失败回退复制）；GT 由点标注重写为 .txt
- Windows 文件系统大小写不敏感：探测按「特征目录组合」判定，避免
  train/images+gt（JHU 原生 / 已统一）被误判为 QNRF 等布局
"""
import os
import random
import shutil
from pathlib import Path

import numpy as np

# 转换后的统一格式说明（前端「下载/上传」面板展示用）
UNIFIED_TREE = (
    "dataset_root/\n"
    "├── Train/\n"
    "│   ├── Images/       # *.jpg / *.png\n"
    "│   └── GT/           # *.txt（每行 \"x y\" 点标注，两路线通用）或\n"
    "│                     #   *.npy（密度图，sum≈人数，密度图路线直用；\n"
    "│                     #   上传时勾选「GT → .npy」即可自动生成；成功\n"
    "│                     #   转换的源 .txt/.mat 删除，失败/无配对保留）\n"
    "└── Test/\n"
    "    ├── Images/\n"
    "    └── GT/\n"
    "\n"
    "转换成功后自动删除已知原始目录与标注，仅保留本统一结构；未知文件保留\n"
    "（原图硬链接进 Images/，删除原始路径不影响副本）"
)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _is_dir_ci(root: Path, *names) -> bool:
    """大小写不敏感的目录存在检查（Windows 上 Path 本身即不区分）"""
    d = root
    for n in names:
        d = d / n
        if not d.is_dir():
            return False
    return True


def _has_image_subdir(split_dir: Path) -> bool:
    return any((split_dir / n).is_dir()
               for n in ("Images", "images", "imgs", "img"))


def _has_gt_subdir(split_dir: Path) -> bool:
    return any((split_dir / n).is_dir()
               for n in ("GT", "gt", "npys", "masks", "labels"))


def _dir_has_direct_files(d: Path, exts) -> bool:
    try:
        return any(p.is_file() and p.suffix.lower() in exts
                   for p in d.iterdir())
    except OSError:
        return False


def _dir_has_ann_mats(d: Path) -> bool:
    try:
        return any(p.is_file() and p.name.endswith("_ann.mat")
                   for p in d.iterdir())
    except OSError:
        return False


def detect_preset(root: Path):
    """探测数据集根目录命中的常用数据集预设 key。

    返回 'shha' / 'shb' / 'qnrf' / 'nwpu' / 'ucf50' 之一；
    已统一 / 框架原生可读（含 JHU train|val|test 的 images+gt 布局）/ 未知
    布局返回 None（调用方跳过转换）。

    注：JHU-CROWD++ 的 train/val|test + images/gt 布局框架可原生读取
    （GT 6 列取前 2 列即点坐标），且其 train/ 目录与转换目标
    Train/ 在 Windows 大小写不敏感文件系统上为同一目录，原地转换会
    覆盖原始 GT —— 因此 JHU 明确不做转换（val 划分由训练管线
    val_ratio 从 train 切分补足）。
    """
    root = Path(root)

    # --- 已统一 / 框架原生可读守卫：Train(或 train) 下有 Images + GT
    #     子目录（含 JHU 布局 train/images+gt）→ 不动 ---
    for split in ("Train", "train"):
        sp = root / split
        if sp.is_dir() and _has_image_subdir(sp) and _has_gt_subdir(sp):
            return None

    # --- ShanghaiTech A/B：part_A|part_B/train_data，或 zip 下探后的
    #     根级 train_data/images ---
    if _is_dir_ci(root, "part_A", "train_data"):
        return "shha"
    if _is_dir_ci(root, "part_B", "train_data"):
        return "shb"
    if _is_dir_ci(root, "train_data", "images"):
        return "shha"      # 下探后无 part 包裹（解析器有同名回退路径）

    # --- UCF-QNRF：Train/ 目录下图像与 *_ann.mat 直接混放 ---
    for split in ("Train", "train"):
        sp = root / split
        if sp.is_dir() and not _has_image_subdir(sp) \
                and _dir_has_direct_files(sp, _IMG_EXTS) \
                and _dir_has_ann_mats(sp):
            return "qnrf"

    # --- NWPU-Crowd：images/ + mats/（或 jsons/）---
    if _is_dir_ci(root, "images") and (_is_dir_ci(root, "mats") or
                                       _is_dir_ci(root, "jsons")):
        return "nwpu"

    # --- UCF_CC_50：根目录下图像与 *_ann.mat 直接混放（50 对）---
    if _dir_has_direct_files(root, _IMG_EXTS) and _dir_has_ann_mats(root):
        return "ucf50"

    return None


def _link_or_copy(src: Path, dst: Path):
    """硬链接（同盘零占用）→ 失败回退复制；目标已存在则跳过"""
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))


def _write_points_txt(gt_path: Path, points):
    """点数组 → GT .txt（每行 "x y"，保留 2 位小数；空点 → 空文件）"""
    lines = []
    for p in np.asarray(points, dtype=np.float32).reshape(-1, 2):
        lines.append(f"{float(p[0]):.2f} {float(p[1]):.2f}")
    gt_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                       encoding="utf-8")


def _unique_stem(stem: str, used: set) -> str:
    """同名图像冲突时加后缀（保持图像与 GT 主名一致，配对不受影响）"""
    if stem not in used:
        return stem
    i = 1
    while f"{stem}_{i}" in used:
        i += 1
    return f"{stem}_{i}"


_SOURCE_DIR_NAMES = {
    "part_a", "part_b", "train_data", "test_data", "images", "imgs",
    "jsons", "mats", "ground_truth", "ground-truth", "__macosx",
}
_SOURCE_FILE_NAMES = {"train.txt", "val.txt", "test.txt"}


def _is_known_source(p: Path) -> bool:
    """是否为已知原始数据集布局项（用于安全清理，未知文件默认保留）。"""
    low = p.name.lower()
    if p.is_dir():
        return low in _SOURCE_DIR_NAMES
    if low in _SOURCE_FILE_NAMES:
        return True
    return p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".mat")


def _cleanup_originals(root: Path, pre_existing: set, log=print):
    """统一转换成功后删除原始数据，数据集仅保留 Train/ Test/ 统一结构。

    - 根目录中 Train/Test 之外的【已知原始布局】条目删除（part_A、images、
      mats、jsons、train.txt 划分表、__MACOSX 等）；readme 等未知文件保留
    - 转换前即存在的 Train/Test 目录（QNRF：原图与 *_ann.mat 直接
      混放于 Train/，统一结构 Images/ GT/ 建于其内）只保留其 Images/
      GT/ 与未知文件，已知原始标注删除；若无 Images/GT 子目录
      （大小写不同名的原始 train/）则保留目录，避免误删
    - 本次新建的 Train/Test（不在快照中）整体保留

    统一结构中的图像是原始图像的硬链接（同盘零占用），删除原始路径
    不影响 Images/ 中的副本；单个条目删除失败仅记录（如被占用），
    不中断。返回 (删除文件数, 删除目录数)。
    """
    n_files = 0
    n_dirs = 0

    def _rm(p: Path):
        nonlocal n_files, n_dirs
        try:
            if p.is_dir():
                n_files += sum(1 for q in p.rglob("*") if q.is_file())
                shutil.rmtree(p)
                n_dirs += 1
            else:
                p.unlink(missing_ok=True)
                n_files += 1
        except OSError as e:
            log(f"[unify] 清理跳过 {p.name}: {e}")

    for entry in list(root.iterdir()):
        low = entry.name.lower()
        if entry.is_dir() and low in ("train", "test"):
            if low not in pre_existing:
                continue                 # 本次新建的统一目录，整体保留
            has_unified = any(s.is_dir() and s.name.lower() in ("images", "gt")
                              for s in entry.iterdir())
            if not has_unified:
                # 未建入统一结构的同名 split 目录：保留而不是整目录删除，
                # 避免误删转换未覆盖的用户数据（如大小写不同的 train/）。
                log(f"[unify] 保留未转换的 split 目录: {entry.name}")
                continue
            # QNRF 布局：清掉与统一结构混放的原始文件（原图 / *_ann.mat）；
            # 未知说明文件（README 等）保留，避免误删用户资料
            for sub in list(entry.iterdir()):
                if sub.is_dir() and sub.name.lower() in ("images", "gt"):
                    continue
                if _is_known_source(sub):
                    _rm(sub)
                else:
                    log(f"[unify] 保留非原始标注文件: {sub.name}")
        elif _is_known_source(entry):
            _rm(entry)
        else:
            log(f"[unify] 保留未知文件/目录: {entry.name}")
    return n_files, n_dirs


def unify_dataset(root: Path, preset: str, log=print, progress=None,
                  stop_event=None, convert_txt=True):
    """把命中的常用数据集原地转换为统一 Train/Images + GT 布局。

    convert_txt=True（默认）：GT 统一重写为 .txt 点标注；
    convert_txt=False：仅统一目录结构，GT 保留原始标注格式
    （.mat 仍为 .mat / .txt 仍为 .txt，文件名对齐图像主名）。
    返回统计 dict；stop_event 置位时抛 StopIteration（后台任务停止）。
    """
    from app.services.preprocess_service import _parse_dataset

    root = Path(root)
    if preset == "ucf50":
        # UCF_CC_50 无官方划分：种子确定性 80/20 切分（40 train / 10 test）
        items = _parse_dataset(root, preset)
        pool = items["train"] + items["test"]
        rng = random.Random(42)
        rng.shuffle(pool)
        n_train = max(1, round(len(pool) * 0.8))
        split_items = {"train": pool[:n_train], "test": pool[n_train:]}
    else:
        split_items = _parse_dataset(root, preset)
        if preset == "nwpu":
            # NWPU test 划分无标注（评测服务器持有）：仅转换 train+val，
            # 评估由训练管线按 val_ratio 从 Train 确定性切分
            n_no_ann = sum(1 for it in split_items["test"]
                           if not len(it["points"]))
            if n_no_ann:
                log(f"[unify] NWPU test 划分 {len(split_items['test'])} 张"
                    " 无标注（官方评测集），已跳过；评估将按"
                    " val_ratio 从 Train 切分")
                split_items["test"] = []

    total = sum(len(v) for v in split_items.values())
    if total == 0:
        raise ValueError(f"按预设 {preset} 未解析到任何样本，跳过转换")

    # 转换前的根目录快照：区分「本次新建的统一 Train/Test」与「原始布局
    # 自带的 Train/Test」（QNRF 的原图与 *_ann.mat 直接混放于 Train/，
    # 统一结构 Images/GT 建于其内 —— 清理时只删混放文件）
    pre_existing = {p.name.lower() for p in root.iterdir()}
    # 未完成标记：转换中途停止/异常时保留；训练侧 _crowd_dirs 会拒绝
    # 读取带该标记的数据集，避免静默使用“只转换了一半”的 Train/Test。
    marker = root / ".unify_incomplete"
    marker.write_text("unify in progress\n", encoding="utf-8")

    convert_txt = bool(convert_txt)
    stats = {"preset": preset, "train": 0, "test": 0, "gt_txt": 0,
             "gt_kept": 0, "empty_gt": 0}
    idx = 0
    for split, sitems in split_items.items():
        split_dir = root / ("Train" if split == "train" else "Test")
        imgs_dir = split_dir / "Images"
        gt_dir = split_dir / "GT"
        imgs_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        used = set()
        for item in sitems:
            if stop_event is not None and stop_event.is_set():
                raise StopIteration()
            idx += 1
            if progress:
                progress(idx / total)
            img = item["img"]
            stem = _unique_stem(img.stem, used)
            used.add(stem)
            _link_or_copy(img, imgs_dir / f"{stem}{img.suffix}")
            pts = item["points"]
            if convert_txt:
                _write_points_txt(gt_dir / f"{stem}.txt", pts)
                stats["gt_txt"] += 1
            else:
                src_gt = item.get("gt")
                if src_gt is not None and Path(src_gt).is_file():
                    # 保留原始标注格式：<图像主名><原后缀>（.mat/.txt/…）
                    _link_or_copy(
                        Path(src_gt),
                        gt_dir / f"{stem}{Path(src_gt).suffix.lower()}")
                    stats["gt_kept"] += 1
                else:
                    # 无原始标注文件（合成数据）→ 回退写 .txt
                    _write_points_txt(gt_dir / f"{stem}.txt", pts)
                    stats["gt_txt"] += 1
            stats[split] += 1
            if not len(pts):
                stats["empty_gt"] += 1
            if idx % 100 == 0 or idx == total:
                log(f"[unify] {preset} → 统一格式：{idx}/{total}")

    # 转换全部成功 → 清理原始数据（失败 / 停止不会走到这里，原始数据
    # 完整、可安全重试；重试时已存在的目标跳过，幂等）
    cleaned_files, cleaned_dirs = _cleanup_originals(root, pre_existing,
                                                     log=log)
    stats["cleaned_files"] = cleaned_files
    stats["cleaned_dirs"] = cleaned_dirs
    marker.unlink(missing_ok=True)

    log(f"[unify] 完成：Train {stats['train']} / Test {stats['test']} 张，"
        f"GT .txt {stats['gt_txt']} 个 / 保留原格式 {stats['gt_kept']} 个"
        f"（其中空标注 {stats['empty_gt']} 个，即 0 人图像）")
    log(f"[unify] 原始数据已清理（文件 {cleaned_files} 个 / 目录 "
        f"{cleaned_dirs} 个），数据集仅保留统一结构：\n" + UNIFIED_TREE)
    return stats


def unify_dataset_if_common(root, log=print, progress=None, stop_event=None,
                            convert_txt=True):
    """探测 + 转换入口：命中常用数据集布局时转换为统一格式，否则不动。

    convert_txt=True：GT 统一重写为 .txt；False：保留原始标注格式
    （目录仍统一为 Train/Images + GT/）。
    返回统计 dict（含 preset/train/test）；未命中返回 None。
    供下载数据集（后台任务）与上传数据集（线程池）解压完成后调用。
    """
    root = Path(root)
    preset = detect_preset(root)
    if not preset:
        return None
    gt_desc = ("GT/*.txt（统一转写）" if convert_txt
               else "GT/（保留原始标注格式 .mat/.txt）")
    log(f"[unify] 检测到常用数据集布局（{preset}），"
        f"自动转换为统一格式（Train/Images + {gt_desc}）...")
    return unify_dataset(root, preset, log=log, progress=progress,
                         stop_event=stop_event, convert_txt=convert_txt)


# ---------------------------------------------------------------------------
# 上传后处理后台任务：GT 点标注 → .npy 密度图（删原 .txt/.mat）+ 可选 .png
# ---------------------------------------------------------------------------

def _find_image_for_gt(gt_path: Path, img_dir: Path):
    """为 GT 文件定位配对图像（同名 .txt 直接配；.mat 兼容带标记前缀）"""
    from app.services.crowd_counting import _find_image_for_mat
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for e in exts:
        p = img_dir / (gt_path.stem + e)
        if p.exists():
            return p
    if gt_path.suffix.lower() == ".mat":
        return _find_image_for_mat(gt_path, img_dir)
    return None


def _save_heat_png(np_path: Path, out_png: Path, annotate=True):
    """单张 .npy 密度图 → 伪彩色热力图 PNG（与 run_npy2png heat 模式一致）"""
    import matplotlib.cm as cm
    from PIL import Image, ImageDraw, ImageFont
    arr = np.load(np_path)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    d = arr.astype(np.float64) - float(arr.min())
    mx = d.max()
    if mx > 1e-8:
        d = d / mx
    heat = (cm.jet(d)[..., :3] * 255).astype(np.uint8)
    img = Image.fromarray(heat)
    if annotate:
        try:
            font = ImageFont.truetype("arial.ttf", max(14, img.height // 24))
        except OSError:
            try:
                font = ImageFont.truetype("segoeui.ttf",
                                          max(14, img.height // 24))
            except OSError:
                font = ImageFont.load_default()
        label = f"count~{float(arr.sum()):.1f}"
        draw = ImageDraw.Draw(img)
        tb = draw.textbbox((6, 6), label, font=font)
        draw.rectangle([tb[0] - 4, tb[1] - 4, tb[2] + 4, tb[3] + 4],
                       fill=(0, 0, 0))
        draw.text((6, 6), label, fill=(255, 255, 0), font=font)
    img.save(out_png)


def run_upload_postprocess(task, params: dict):
    """上传后处理后台任务（上传时勾选「GT → .npy」/「可视化 .png」触发）。

     params:
       dataset_dir: 数据集根目录
       mat_to_npy:  GT 点标注（.txt/.mat）→ .npy 密度图；仅删除
                    「已成功生成对应 .npy」的源标注。找不到配对图像、
                    解析失败或未生成 .npy 的标注会保留，避免数据丢失。
       sigma:       密度图高斯 sigma
       adaptive:    k 近邻自适应 sigma
       to_txt:      将 .mat 点标注转换为 .txt 点标注（成功后删除源 .mat）
       keep_src:    转为 .npy 时保留源 .txt/.mat（同时要求 .txt 输出时使用）
       vis_png:     转换完成后将 GT .npy 可视化为 .png
       vis_dir:     PNG 保存目录（留空 → <数据集>/vis_png/<Train|Test>）

    结果（task.result）：txt_converted / npy_converted / mat_deleted /
    txt_deleted / png_generated / vis_dir
    """
    from app.services.crowd_counting import (_crowd_dirs, _load_points_gt,
                                             make_density_map)

    root = Path(params["dataset_dir"])
    mat_to_npy = bool(params.get("mat_to_npy"))
    to_txt = bool(params.get("to_txt"))
    keep_src = bool(params.get("keep_src"))
    vis_png = bool(params.get("vis_png"))
    sigma = float(params.get("sigma", 2.0))
    adaptive = bool(params.get("adaptive", False))
    vis_dir = (params.get("vis_dir") or "").strip()

    if not root.is_dir():
        raise ValueError(f"数据集目录不存在: {root}")
    task.log(f"上传后处理 | 数据集: {root.name} | GT→.npy: {mat_to_npy} | "
             f"GT→.txt: {to_txt} | σ={sigma}（自适应={adaptive}）| "
             f"可视化: {vis_png}"
             + (f" | PNG 目录: {vis_dir}" if vis_dir else ""))
    if not mat_to_npy and not to_txt and not vis_png:
        task.log("无后处理选项，直接完成")
        task.progress = 100.0
        task.result = {"txt_converted": 0, "npy_converted": 0,
                       "mat_deleted": 0, "txt_deleted": 0,
                       "mat_kept": 0, "png_generated": 0}
        return

    # ---- 定位各 split 的 Images / GT 目录 ----
    splits = []
    for train_flag, name in ((True, "Train"), (False, "Test")):
        img_dir, gt_dir = _crowd_dirs(root, train_flag)
        if img_dir is not None and gt_dir is not None:
            splits.append((name, img_dir, gt_dir))
    if not splits:
        raise ValueError("未找到 Train/Test + Images/GT 目录结构"
                          f"（数据集应先统一为标准格式）: {root}")

    stats = {"txt_converted": 0, "npy_converted": 0, "mat_deleted": 0,
             "txt_deleted": 0, "mat_kept": 0, "png_generated": 0,
             "vis_dir": ""}

    # ---- 阶段零：.mat 点标注 → .txt 点标注（成功后删除源 .mat） ----
    if to_txt:
        mats = []
        for _, _, gt_dir in splits:
            mats += [p for p in sorted(gt_dir.iterdir())
                     if p.suffix.lower() == ".mat" and p.is_file()]
        task.total_epochs = len(mats)
        task.log(f"[post] 待转换 .mat → .txt: {len(mats)} 个")
        done = 0
        for g in mats:
            if task.stop_event.is_set():
                raise StopIteration()
            done += 1
            task.epoch = done
            task.progress = done / max(1, len(mats)) * 15.0
            try:
                img_p = _find_image_for_gt(g, _img_dir_of(splits, g))
                stem = img_p.stem if img_p is not None else g.stem
                txt_p = g.parent / (stem + ".txt")
                if not txt_p.exists():
                    pts = np.asarray(_load_points_gt(g), np.float32) \
                        .reshape(-1, 2)
                    if pts.size == 0:
                        task.log(f"[post] 跳过 {g.name}：未解析到点坐标"
                                 "（保留源标注）")
                        continue
                    np.savetxt(str(txt_p), pts, fmt="%.2f")
                    stats["txt_converted"] += 1
                if txt_p.exists():
                    g.unlink(missing_ok=True)
                    stats["mat_deleted"] += 1
                if done % 100 == 0 or done == len(mats):
                    task.log(f"[post] {done}/{len(mats)} 已转 .txt")
            except StopIteration:
                raise
            except Exception as e:
                task.log(f"[post] 转 .txt 失败 {g.name}: {e}（保留源标注）")
        task.log(f"[post] .mat → .txt 完成：生成 {stats['txt_converted']} 个，"
                 f"删除 .mat {stats['mat_deleted']} 个")

    # ---- 阶段一：GT 点标注（.txt/.mat）→ .npy 密度图，删除源文件 ----
    if mat_to_npy:
        gt_files = []
        for _, img_dir, gt_dir in splits:
            gt_files += [p for p in sorted(gt_dir.iterdir())
                         if p.suffix.lower() in (".txt", ".mat") and p.is_file()]
        task.total_epochs = len(gt_files)
        task.log(f"[post] 待转换 GT（.txt/.mat → .npy）: {len(gt_files)} 个")
        p_base = 15.0 if to_txt else 0.0
        p_span = 40.0 if to_txt else 50.0
        done = 0
        for g in gt_files:
            if task.stop_event.is_set():
                raise StopIteration()
            done += 1
            task.epoch = done
            task.progress = p_base + done / max(1, len(gt_files)) * p_span
            try:
                img_p = _find_image_for_gt(g, _img_dir_of(splits, g))
                if img_p is None:
                    task.log(f"[post] 跳过 {g.name}：未找到配对图像（保留源标注）")
                    continue
                # 已有同名 .npy 时不覆盖（保留用户已有密度图）
                npy_p = g.parent / (img_p.stem + ".npy")
                if not npy_p.exists():
                    from PIL import Image as _Image
                    with _Image.open(img_p) as im:
                        w, h = im.size
                    pts = _load_points_gt(g)
                    density = make_density_map(h, w, pts, sigma=sigma,
                                               adaptive=adaptive)
                    np.save(npy_p, density)
                    stats["npy_converted"] += 1
                # 只有确认 .npy 已存在时才删除源标注，避免转换失败丢标注
                if not npy_p.exists():
                    task.log(f"[post] 保留 {g.name}：未生成有效 .npy")
                    continue
                if keep_src:
                    continue
                g.unlink(missing_ok=True)
                if g.suffix.lower() == ".mat":
                    stats["mat_deleted"] += 1
                else:
                    stats["txt_deleted"] += 1
                if done % 100 == 0 or done == len(gt_files):
                    task.log(f"[post] {done}/{len(gt_files)} 已转换")
            except StopIteration:
                raise
            except Exception as e:
                task.log(f"[post] 失败 {g.name}: {e}（保留源标注）")
        # 不再全盘删除未转换的 .mat：失败/无配对图像的标注必须保留，
        # 否则用户无法重试。仅统计遗留的 .mat 并在日志中提示。
        leftover_mats = [p for p in root.rglob("*.mat") if p.is_file()]
        stats["mat_kept"] = len(leftover_mats)
        if leftover_mats:
            task.log(f"[post] 保留未转换/未配对 .mat {len(leftover_mats)} 个"
                     "（未删除，请检查配对图像或格式后重试）")
        task.log(f"[post] GT → .npy 完成：生成 {stats['npy_converted']} 个，"
                 f"删除 .txt {stats['txt_deleted']} / .mat {stats['mat_deleted']} 个，"
                 f"保留 .mat {stats['mat_kept']} 个")

    # ---- 阶段二：.npy 密度图 → .png 可视化（保存目录可选） ----
    if vis_png:
        png_base = 55.0 if (to_txt and mat_to_npy) else \
            50.0 if mat_to_npy else 15.0 if to_txt else 0.0
        png_done = 0
        for split_name, img_dir, gt_dir in splits:
            npys = sorted(gt_dir.glob("*.npy"))
            if not npys:
                continue
            base = Path(vis_dir) if vis_dir else (root / "vis_png")
            out_dir = base / split_name
            out_dir.mkdir(parents=True, exist_ok=True)
            stats["vis_dir"] = stats["vis_dir"] or str(base)
            task.log(f"[post] 可视化 {split_name}: {len(npys)} 张 → {out_dir}")
            for np_path in npys:
                if task.stop_event.is_set():
                    raise StopIteration()
                png_done += 1
                task.epoch = png_done
                task.progress = png_base + png_done / max(1, len(npys)) * \
                    (100.0 - png_base)
                out_png = out_dir / (np_path.stem + ".png")
                if not out_png.exists():
                    _save_heat_png(np_path, out_png, annotate=True)
                    stats["png_generated"] += 1
                if png_done % 50 == 0 or png_done == len(npys):
                    task.log(f"[post] 可视化进度 {png_done}")
        task.log(f"[post] 可视化完成：生成 {stats['png_generated']} 张 PNG"
                 + (f"，保存到 {stats['vis_dir']}" if stats["vis_dir"] else ""))

    task.progress = 100.0
    task.result = stats
    task.log(f"上传后处理完成：.txt 生成 {stats['txt_converted']} 个 | "
             f".npy {stats['npy_converted']} 个 | "
             f".mat 删除 {stats['mat_deleted']} 个 / 保留 {stats['mat_kept']} 个 | "
             f".txt 删除 {stats['txt_deleted']} 个 | "
             f".png {stats['png_generated']} 张")


def _img_dir_of(splits, gt_path):
    """按 GT 文件所在目录反查其所属 split 的 Images 目录"""
    for _, img_dir, gt_dir in splits:
        if gt_path.parent == gt_dir:
            return img_dir
    return splits[0][1] if splits else gt_path.parent


def _strip_format_suffix(name: str) -> str:
    """去掉已有的 _npy / _txt / _mat 后缀，避免重复叠加。"""
    low = name.lower()
    for suf in ("_npy", "_txt", "_mat"):
        if low.endswith(suf):
            return name[:-len(suf)]
    return name


def _clone_dataset_tree(src: Path, dst: Path):
    """把数据集复制到新目录（同盘优先硬链接，失败回退复制）。

    硬链接与原数据集共享文件内容，但删除/新增文件互不影响，
    因此后续只在副本中改写 GT 标注，原数据集保持不变。
    """
    for cur, _dirs, files in os.walk(src):
        cur_p = Path(cur)
        rel = cur_p.relative_to(src)
        out_dir = dst if rel == Path(".") else dst / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            s = cur_p / f
            d = out_dir / f
            if d.exists():
                continue
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)


def run_dataset_convert(task, params: dict):
    """把已上传数据集转换为新数据集（原数据集保留）。

    params:
      dataset_dir:   源数据集根目录
      output_format: "txt"（点标注）或 "npy"（密度图）；None 表示仅做 png
      sigma/adaptive: 生成 .npy 密度图参数
      vis_png:       将 .npy 密度图可视化为 .png
      vis_dir:       PNG 保存目录（留空 → 输出数据集/vis_png/<Train|Test>）

    新数据集名称自动追加 ``_txt`` / ``_npy`` 后缀；同名已存在时自动
    追加 ``_1`` / ``_2`` 唯一序号。
    """
    from app.services.crowd_counting import (_crowd_dirs, _load_points_gt,
                                             make_density_map)
    from app.core.dataset_naming import unique_dataset_name

    source = Path(params["dataset_dir"])
    fmt = (params.get("output_format") or "").strip().lower()
    if fmt not in ("txt", "npy", ""):
        raise ValueError(f"不支持的输出格式: {fmt}")
    vis_png = bool(params.get("vis_png"))
    sigma = float(params.get("sigma", 2.0))
    adaptive = bool(params.get("adaptive", False))
    vis_dir = (params.get("vis_dir") or "").strip()

    if not source.is_dir():
        raise ValueError(f"数据集目录不存在: {source}")
    if not fmt and not vis_png:
        raise ValueError("请至少选择 .txt / .npy 转换或 .png 可视化")

    stats = {"txt_converted": 0, "npy_converted": 0, "mat_deleted": 0,
             "txt_deleted": 0, "npy_deleted": 0, "png_generated": 0,
             "new_dataset": "", "new_path": "", "vis_dir": ""}

    target = source
    if fmt:
        base_name = _strip_format_suffix(source.name)
        new_name = f"{base_name}_{fmt}"
        target = unique_dataset_name(source.parent, new_name)
        task.log(f"[conv] 源数据集保持不变: {source}")
        task.log(f"[conv] 新建 {fmt} 数据集: {target}")
        try:
            _clone_dataset_tree(source, target)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        stats["new_dataset"] = target.name
        stats["new_path"] = str(target)

    try:
        splits = []
        for train_flag, split_name in ((True, "Train"), (False, "Test")):
            img_dir, gt_dir = _crowd_dirs(target, train_flag)
            if img_dir is not None and gt_dir is not None:
                splits.append((split_name, img_dir, gt_dir))
        if not splits:
            raise ValueError("未找到 Train/Test + Images/GT 目录结构"
                              f"（数据集应先统一为标准格式）: {target}")

        # ---- .mat → .txt（仅在新数据集中操作） ----
        if fmt == "txt":
            mats = []
            for _, _, gt_dir in splits:
                mats += [p for p in sorted(gt_dir.iterdir())
                         if p.suffix.lower() == ".mat" and p.is_file()]
            task.total_epochs = len(mats)
            task.log(f"[conv] 待转换 .mat → .txt: {len(mats)} 个")
            for i, g in enumerate(mats, 1):
                if task.stop_event.is_set():
                    raise StopIteration()
                task.progress = i / max(1, len(mats)) * 40.0
                try:
                    img_p = _find_image_for_gt(g, _img_dir_of(splits, g))
                    stem = img_p.stem if img_p is not None else g.stem
                    txt_p = g.parent / (stem + ".txt")
                    if not txt_p.exists():
                        pts = np.asarray(_load_points_gt(g), np.float32) \
                            .reshape(-1, 2)
                        if pts.size == 0:
                            task.log(f"[conv] 跳过 {g.name}：未解析到点坐标")
                            continue
                        np.savetxt(str(txt_p), pts, fmt="%.2f")
                        stats["txt_converted"] += 1
                    if txt_p.exists():
                        g.unlink(missing_ok=True)
                        stats["mat_deleted"] += 1
                except StopIteration:
                    raise
                except Exception as e:
                    task.log(f"[conv] 转 .txt 失败 {g.name}: {e}（保留源 .mat）")
            # 输出 _txt 数据集：移除副本中的 .npy，保持标注格式单一
            for _, _, gt_dir in splits:
                for p in gt_dir.glob("*.npy"):
                    p.unlink(missing_ok=True)
                    stats["npy_deleted"] += 1
            txt_total = sum(1 for _, _, gt_dir in splits
                            for p in gt_dir.glob("*.txt") if p.is_file())
            if txt_total == 0:
                raise ValueError("源数据集未找到可转换为 .txt 的点标注"
                                  "（.mat/.txt），请检查标注格式")
            task.log(f"[conv] .txt 数据集就绪：{txt_total} 个点标注文件")

        # ---- 点标注 → .npy 密度图（仅在新数据集中操作） ----
        if fmt == "npy":
            gt_files = []
            for _, _, gt_dir in splits:
                gt_files += [p for p in sorted(gt_dir.iterdir())
                             if p.suffix.lower() in (".txt", ".mat")
                             and p.is_file()]
            task.total_epochs = len(gt_files)
            task.log(f"[conv] 待转换 GT → .npy: {len(gt_files)} 个")
            for i, g in enumerate(gt_files, 1):
                if task.stop_event.is_set():
                    raise StopIteration()
                task.progress = i / max(1, len(gt_files)) * 60.0
                try:
                    img_p = _find_image_for_gt(g, _img_dir_of(splits, g))
                    if img_p is None:
                        task.log(f"[conv] 跳过 {g.name}：未找到配对图像")
                        continue
                    npy_p = g.parent / (img_p.stem + ".npy")
                    if not npy_p.exists():
                        from PIL import Image as _Image
                        with _Image.open(img_p) as im:
                            w, h = im.size
                        pts = _load_points_gt(g)
                        density = make_density_map(h, w, pts, sigma=sigma,
                                                   adaptive=adaptive)
                        np.save(npy_p, density)
                        stats["npy_converted"] += 1
                    if not npy_p.exists():
                        task.log(f"[conv] 保留 {g.name}：未生成有效 .npy")
                        continue
                    g.unlink(missing_ok=True)
                    if g.suffix.lower() == ".mat":
                        stats["mat_deleted"] += 1
                    else:
                        stats["txt_deleted"] += 1
                except StopIteration:
                    raise
                except Exception as e:
                    task.log(f"[conv] 转 .npy 失败 {g.name}: {e}（保留源标注）")
            npy_total = sum(1 for _, _, gt_dir in splits
                            for p in gt_dir.glob("*.npy") if p.is_file())
            if npy_total == 0:
                raise ValueError("源数据集未找到可用于生成密度图的点标注"
                                  "（.txt/.mat），请检查标注格式")
            task.log(f"[conv] .npy 数据集就绪：{npy_total} 张密度图")

        # ---- .npy → .png 可视化 ----
        if vis_png:
            png_done = 0
            npys_all = []
            for split_name, _, gt_dir in splits:
                for p in sorted(gt_dir.glob("*.npy")):
                    npys_all.append((split_name, p))
            for i, (split_name, np_path) in enumerate(npys_all, 1):
                if task.stop_event.is_set():
                    raise StopIteration()
                png_done += 1
                task.progress = 70.0 + png_done / max(1, len(npys_all)) * 30.0
                base = Path(vis_dir) if vis_dir else (target / "vis_png")
                out_dir = base / split_name
                out_dir.mkdir(parents=True, exist_ok=True)
                stats["vis_dir"] = stats["vis_dir"] or str(base)
                out_png = out_dir / (np_path.stem + ".png")
                if not out_png.exists():
                    _save_heat_png(np_path, out_png, annotate=True)
                    stats["png_generated"] += 1
                if i % 50 == 0 or i == len(npys_all):
                    task.log(f"[conv] 可视化进度 {i}/{len(npys_all)}")
            if not npys_all:
                task.log("[conv] 未找到 .npy 密度图，跳过 PNG 可视化")
            else:
                task.log(f"[conv] 可视化完成：生成 {stats['png_generated']} 张 PNG"
                         + (f"，保存到 {stats['vis_dir']}"
                            if stats["vis_dir"] else ""))
    except Exception:
        # 新建数据集失败时清理副本；源数据集始终不受影响
        if target != source:
            shutil.rmtree(target, ignore_errors=True)
            stats["new_dataset"] = ""
            stats["new_path"] = ""
        raise

    task.progress = 100.0
    task.result = stats
    task.log(f"数据集转换完成：新数据集 {stats['new_dataset'] or '(未新建)'} | "
             f".txt {stats['txt_converted']} 个 | .npy {stats['npy_converted']} 个 | "
             f".png {stats['png_generated']} 张 | 原数据集已保留")
