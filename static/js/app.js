/* 人群计数自动训练和推理框架 - 前端逻辑（中英文界面，见 i18n.js） */
"use strict";

/* i18n 兜底：i18n.js 加载失败时保持基础可用 */
if (!window.DLATF_I18N) {
  window.DLATF_I18N = {
    t: (k) => String(k).split(".").pop(),
    tf: (k, ...args) => String(k),
    setLang: () => {}, applyI18n: () => {}, lang: "zh",
  };
}
const t = window.DLATF_I18N.t;
const tf = window.DLATF_I18N.tf;

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const state = {
  tasks: [],          // 子任务清单（含模型列表）
  currentTask: null,
  uploads: { models: [], datasets: [], trained_weights: [] },
  trainTaskId: null,
  pollTimer: null,
  testTaskId: null,
  testPollTimer: null,
  inferImageFiles: [],   // 批量推理：多文件
  uploadedWeightName: null,
  frameworks: [],         // 后端框架可用性（当前仅 PyTorch）
  currentFramework: "pytorch",
  uploadZipFile: null,    // 「上传数据集」面板待上传的 zip 文件
};

/* ---------------- 通用 ---------------- */

function toast(msg, type = "info", ms = 3800) {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch (_) { /* ignore */ }
  if (!res.ok) {
    throw new Error((data && data.detail) ? detailText(data.detail) : `HTTP ${res.status}`);
  }
  return data;
}

function detailText(d) {
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map(x => (x.msg || JSON.stringify(x))).join("; ");
  return JSON.stringify(d);
}

function esc(s) {
  return String(s === undefined || s === null ? "" : s).replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* 带上传进度条的 POST（XMLHttpRequest 实现） */
function uploadWithProgress(url, fd, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch (_) { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new Error((data && data.detail) ? detailText(data.detail) : `HTTP ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error(t("msg.networkError")));
    xhr.send(fd);
  });
}

/* ---------------- 初始化 ---------------- */

async function init() {
  // 先绑定界面事件（不依赖后端 API），保证任何情况下交互可用
  bindHeaderControls();
  bindGlobalNav();
  bindModeSwitch();
  bindTaskSelect();
  bindTrainControls();
  bindDatasetDeleteButtons();
  bindUploadPanel();
  bindPreprocessPanel();
  bindVisualizePanel();
  bindInferControls();
  bindDeployControls();
  bindTestControls();
  bindOverflowTooltip();
  // 再拉取元数据（PyTorch 可用时加载任务清单）
  try {
    await loadFrameworks();
    const ptOK = state.frameworks.find(f => f.key === "pytorch")?.available !== false;
    const tasks = [];
    if (ptOK) { tasks.push(loadTasks()); tasks.push(loadOptions()); }
    tasks.push(loadUploads());
    tasks.push(loadProxyConfig());
    await Promise.all(tasks);
    refreshFooterInfo();
    if (!ptOK) {
      console.info("PyTorch not available — crowd counting endpoints skipped");
    }
  } catch (e) {
    console.error(e);
    toast(tf("msg.loadFailed", e.message), "error", 10000);
    // 下拉框给出可见的错误提示（而不是空白）
    const errOption = `<option value="">${esc(t("msg.errOption"))}</option>`;
    for (const sel of ["#cvTaskSelect", "#trainModel", "#inferModel",
                       "#trainDevice", "#inferDevice", "#inferWeightSelect"]) {
      const el = $(sel);
      if (el && !el.options.length) el.innerHTML = errOption;
    }
  }
  applyUrlHash();
  // 预渲染学习版图（不等用户点击，确保内容始终在 DOM 中）
  try { renderLearningMap(); } catch (e) { console.warn("预渲染学习版图:", e); }
}

  /* URL hash 直达：#infer 推理模式 / #train 训练模式；
    #crowd（或 #cv）/ #upload / #hardware / #proxy / #metrics / #learning / #author 面板切换 */
  function applyUrlHash() {
    const h = decodeURIComponent(location.hash.replace("#", ""));
    if (["infer", "train", "test"].includes(h)) {
      const btn = document.querySelector(`.mode-btn[data-mode="${h}"]`);
      if (btn) btn.click();
    }
    // #crowd 与二级导航按钮的 data-domain="cv" 对齐
    const dom = h === "crowd" ? "cv" : h;
    if (["cv", "upload", "preprocess", "visualize", "hardware", "proxy",
         "metrics", "learning", "author"].includes(dom)) {
      const btn = document.querySelector(`.nav-btn[data-domain="${dom}"]`);
      if (btn) btn.click();
    }
  }

async function loadTasks() {
  const tasks = await api("/api/crowd/tasks");
  state.tasks = tasks;
  rebuildCvTaskSelect();
  if (state.tasks.length) selectTask(state.tasks[0].key);
  refreshTestTasks();
}

function rebuildCvTaskSelect() {
  const sel = $("#cvTaskSelect");
  if (!sel || !state.tasks.length) return;
  const cur = state.currentTask ? state.currentTask.key : state.tasks[0].key;
  sel.innerHTML = state.tasks.map(tk =>
    `<option value="${tk.key}">${esc(taskLabel(tk))}</option>`).join("");
  sel.value = cur;
}

async function loadOptions() {
  const opt = await api("/api/crowd/options");
  fillSelect("#trainDevice", [{ id: "cpu", label: "CPU" }],
    (d) => d.id, (d) => d.label);
  // 设备列表以 options 返回为准
  const devs = opt.devices || [{ id: "cpu", label: "CPU" }];
  fillSelect("#trainDevice", devs, (d) => d.id, (d) => d.label);
  fillSelect("#inferDevice", devs, (d) => d.id, (d) => d.label);
  state.lossesByTask = opt.losses || {};
  state.schedulersByTask = opt.schedulers || {};
  // 优化器：内置 + 自定义（custom:* 前缀）从 options 动态填充
  if (opt.optimizers && opt.optimizers.length) {
    fillSelect("#optimizer", opt.optimizers, o => o, o => o);
  }
  // 记录 GPU 数（updateTrainModelUI 按「模型×卡数」控制多 GPU 选项显隐）
  state.gpuCount = opt.gpu_count || 0;
  // 多 GPU 模式选择（gpu_count > 1 时显示）
  const mgBox = document.getElementById("fieldMultiGpu");
  const showMg = (opt.gpu_count || 0) > 1;
  if (mgBox) mgBox.style.display = showMg ? "" : "none";
  // options 与任务清单并发加载：若任务已先渲染，用最新 losses/GPU 数刷新
  refreshTestDevices();
  updateTrainModelUI();
}

async function loadUploads() {
  state.uploads = await api("/api/uploads");
  // 数据集下拉
  fillSelect("#datasetSelect",
    [{ value: "" }, ...state.uploads.datasets],
    (d) => d.name || "", (d) => d.name || t("common.select"));
  // 单域泛化：目标数据集 / 测试数据集（同一份已上传数据集清单）
  for (const sel of ["#targetDatasetSelect", "#sdgTestDataset"]) {
    if ($(sel)) {
      const keep = $(sel).value;
      fillSelect(sel, [{ value: "" }].concat(state.uploads.datasets),
        (d) => d.name || "", (d) => d.name || t("common.select"));
      if (keep) $(sel).value = keep;
    }
  }
  // 单域泛化测试：训练产物权重（best/last）
  if ($("#sdgTestWeight")) {
    const wOpts = [{ name: "", path: "" },
                   ...((state.uploads && state.uploads.trained_weights) || [])];
    fillSelect("#sdgTestWeight", wOpts, (m) => m.path || m.name,
      (m) => m.name ? `${m.name}（${m.path || ""}）` : t("common.dontUse"));
  }
  // 上传权重按格式分流：PyTorch 检查点（训练/导出可用）与 ONNX（仅推理）
  const isOnnx = (m) => /\.onnx$/i.test(m.name || "");
  const ptModels = (state.uploads.models || []).filter(m => !isOnnx(m));
  const onnxUploads = (state.uploads.models || []).filter(m => isOnnx(m));
  const onnxOpts = [
    ...(state.uploads.onnx_models || []).map(m => ({
      ...m, label: `[ONNX] ${m.run || m.name} (${m.size_mb}MB)` })),
    ...onnxUploads.map(m => ({
      ...m, label: `[ONNX] ${m.name} (${m.size_mb}MB)` })),
  ];
  // 初始权重：上传的 PyTorch 权重 + 历史训练产物（.pt/.pth；onnx 无法用于
  // 训练）。训练产物可直接作为初始化权重微调，或勾选「续训模式」恢复中断训练
  const initOpts = [{ name: "", path: "" },
                    ...(state.uploads.trained_weights || [])
                      .map(m => ({ ...m, trained: true })),
                    ...ptModels];
  const iwKeepTrain = $("#trainInitWeight") ? $("#trainInitWeight").value : "";
  fillSelect("#trainInitWeight", initOpts, (m) => m.path || m.name,
    (m) => m.name
      ? `${m.trained ? `[${t("train.trainedWeightTag")}] ` : ""}${m.name}` +
        (m.size_mb ? ` (${m.size_mb}MB)` : "")
      : t("common.dontUse"));
  if (iwKeepTrain && $("#trainInitWeight") &&
      [...$("#trainInitWeight").options].some(o => o.value === iwKeepTrain)) {
    $("#trainInitWeight").value = iwKeepTrain;
  }
  // 推理权重（训练产物 + 上传权重 + ONNX 部署产物/上传）：value 用完整路径；
  // ONNX 条目走 onnxruntime 推理（导出产物验证 / 外部模型）
  const wOpts = [{ name: "", path: "" },
                 ...state.uploads.trained_weights, ...ptModels,
                 ...onnxOpts];
  const iwSel = $("#inferWeightSelect");
  const iwKeep = iwSel ? iwSel.value : "";
  fillSelect("#inferWeightSelect", wOpts, (m) => m.path || m.name,
    (m) => m.label || m.name || t("common.dontUse"));
  if (iwKeep && iwSel && [...iwSel.options].some(o => o.value === iwKeep))
    iwSel.value = iwKeep;
  // PyTorch-only 权重集（训练初始权重之外的其余选择器共用）
  const wOptsPt = [{ name: "", path: "" },
                   ...state.uploads.trained_weights, ...ptModels];
  // 蒸馏教师权重（头部框路线）：与推理权重同源（训练产物 + 上传），
  // 保留当前选择以便训练完成刷新列表时不丢配置
  const twSel = $("#distillTeacherWeight");
  if (twSel) {
    const keep = twSel.value;
    fillSelect("#distillTeacherWeight", wOptsPt, (m) => m.path || m.name,
      (m) => m.name || t("common.dontUse"));
    if (keep && [...twSel.options].some(o => o.value === keep))
      twSel.value = keep;
  }
  // 部署模式权重（ONNX 导出源）：与推理权重同源（不含 onnx —— onnx 无法
  // 再导出），保留当前选择
  const dwSel = $("#deployWeightSelect");
  if (dwSel) {
    const keep = dwSel.value;
    fillSelect("#deployWeightSelect", wOptsPt, (m) => m.path || m.name,
      (m) => m.name || t("common.dontUse"));
    if (keep && [...dwSel.options].some(o => o.value === keep))
      dwSel.value = keep;
  }
  updateInferHints();
  updateDatasetDeleteBtns();
  refreshTestWeights();
  refreshTestDatasets();
}

/* 已上传数据集删除按钮（训练卡片数据集选择） */
const DATASET_DELETE_BTNS = [
  ["#btnDeleteDataset", "#datasetSelect"],
];

function updateDatasetDeleteBtns() {
  for (const [btnId, selId] of DATASET_DELETE_BTNS) {
    const btn = $(btnId), sel = $(selId);
    if (btn && sel) btn.disabled = !sel.value;
  }
}

function bindDatasetDeleteButtons() {
  for (const [btnId, selId] of DATASET_DELETE_BTNS) {
    const btn = $(btnId), sel = $(selId);
    if (!btn || !sel) continue;
    btn.addEventListener("click", () => deleteUploadDataset(selId));
    sel.addEventListener("change", () => { btn.disabled = !sel.value; });
  }
}

/* 删除已上传的数据集（服务器 uploads/datasets/<name>，二次确认后不可恢复） */
async function deleteUploadDataset(selectId) {
  const sel = $(selectId);
  const name = sel && sel.value;
  if (!name) return;
  const ds = ((state.uploads && state.uploads.datasets) || [])
    .find(d => d.name === name);
  const detail = ds && ds.num_images > 0
    ? " " + tf("msg.dsDeleteDetail", ds.num_images) : "";
  if (!window.confirm(`${t("msg.confirmDeleteDataset")}\n${name}${detail}`))
    return;
  try {
    await api(`/api/upload/dataset/${encodeURIComponent(name)}`,
              { method: "DELETE" });
    toast(tf("msg.datasetDeleted", name), "success");
    sel.value = "";
    if (selectId === "#datasetSelect") {
      $("#datasetStats").textContent = "";
      $("#datasetPath").value = "";
      await loadUploads();
    }
    updateDatasetDeleteBtns();
  } catch (e) {
    toast(tf("msg.deleteFailed", e.message), "error", 6000);
  }
}

/* ---------------- 「上传数据集」全局模块 ---------------- */

/* 拖拽 / 点击选择 zip，实时显示待上传文件名 */
function bindUploadPanel() {
  const dz = $("#uploadDropzone");
  if (!dz) return;
  dz.addEventListener("click", () => $("#uploadZipFile").click());
  dz.addEventListener("dragover", (e) => {
    e.preventDefault(); dz.classList.add("drag");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setUploadZip(f);
  });
  $("#uploadZipFile").addEventListener("change", (e) => {
    if (e.target.files.length) setUploadZip(e.target.files[0]);
  });
  $("#btnUploadDs").addEventListener("click", uploadDatasetGlobal);
  $("#btnDsRefresh").addEventListener("click", loadDatasetPanel);
  // ---- 数据集结构浏览（逐级懒加载） ----
  const treeSel = $("#dsTreeSelect");
  if (treeSel) {
    treeSel.addEventListener("change", () => loadDatasetTree(treeSel.value));
  }
  const treeRefresh = $("#btnDsTreeRefresh");
  if (treeRefresh) {
    // 刷新数据集列表同时重载当前数据集根目录（fillDsTreeSelect 会保留选择）
    treeRefresh.addEventListener("click", () => loadDatasetPanel());
  }
  const treeCollapse = $("#btnDsTreeCollapse");
  if (treeCollapse) {
    treeCollapse.addEventListener("click", () => {
      if (treeSel && treeSel.value) loadDatasetTree(treeSel.value);
    });
  }
  const treeView = $("#dsTreeView");
  if (treeView) {
    treeView.addEventListener("click", (e) => {
      const row = e.target.closest(".ds-row.ds-dir");
      if (row) toggleDatasetNode(row);
    });
  }
  // ---- 上传转换选项联动：勾选 .mat → .npy 时显示 sigma/自适应参数 ----
  const upMat = $("#upMatToNpy");
  if (upMat) {
    upMat.addEventListener("change", () => {
      const p = $("#upNpyParams");
      if (p) p.style.display = upMat.checked ? "" : "none";
    });
  }
  // 勾选可视化时显示 PNG 保存目录
  const upVis = $("#upVisPng");
  if (upVis) {
    upVis.addEventListener("change", () => {
      const d = $("#upVisDirField");
      if (d) d.style.display = upVis.checked ? "" : "none";
    });
  }
  // ---- ④ GT 密度图转换与可视化工具（自训练卡片迁移至此）----
  $("#btnMat2Density").addEventListener("click", startMat2Density);
  $("#btnNpyVis").addEventListener("click", startNpyVis);
  $("#npyVisMode").addEventListener("change", e => {
    const ov = e.target.value === "overlay";
    const a = $("#npyVisAlphaField"), g = $("#npyVisImgDirField");
    if (a) a.style.display = ov ? "" : "none";
    if (g) g.style.display = ov ? "" : "none";
  });
  // ---- 下载区域 ----
  const dlPreset = $("#dlPresetLink");
  if (dlPreset) {
    dlPreset.addEventListener("change", async (e) => {
      const link = state.dlLinks &&
        state.dlLinks.find(l => l.key === e.target.value);
      if (link) {
        $("#dlUrl").value = link.url;
        const info = $("#dlPresetInfo");
        if (info) {
          info.style.display = "";
          info.textContent = `${link.desc} | ${link.size}`;
        }
        if (!$("#dlName").value) $("#dlName").value = link.key;
      }
      // 选中常用数据集 → 展示目录格式要求（与「数据读取预处理」同源）
      // 与转换后的统一格式（xxx_official → xxx 预设 key）
      await showDlFormatHint((e.target.value || "").replace(/_official$/, ""));
    });
  }
  const dlBtn = $("#btnDownloadDs");
  if (dlBtn) dlBtn.addEventListener("click", downloadDatasetGlobal);
}

/* 下载面板：常用数据集格式要求 + 统一格式提示（复用预处理模块预设清单） */
async function showDlFormatHint(presetKey) {
  const card = $("#dlTreeCard"), tree = $("#dlTree"), unified = $("#dlUnifiedTree");
  if (!card || !tree) return;
  if (unified) unified.textContent = t("upload.convTree");
  if (!presetKey) { card.style.display = "none"; return; }
  try {
    if (!state.ppPresets || !state.ppPresets.length) {
      state.ppPresets = await api("/api/crowd/preprocess/datasets");
    }
    const preset = state.ppPresets.find(p => p.key === presetKey);
    tree.textContent = preset ? (preset.tree || "") : "";
    card.style.display = preset ? "" : "none";
  } catch (_) {
    card.style.display = "none";
  }
}

function setUploadZip(file) {
  if (!file.name.toLowerCase().endsWith(".zip")) {
    return toast(t("msg.pickZip"), "error");
  }
  state.uploadZipFile = file;
  const st = $("#uploadStatus");
  if (st) st.textContent = file.name + "（" +
    (file.size / 1048576).toFixed(1) + " MB）";
}

/* 上传（带进度条）；成功后刷新面板列表与训练卡片的下拉并自动选中 */
async function uploadDatasetGlobal() {
  const f = state.uploadZipFile;
  if (!f) return toast(t("msg.pickZip"), "error");
  const fd = new FormData();
  fd.append("file", f);
  // 上传转换选项（常用数据集自动统一后：GT → .npy 密度图 + 可视化 .png）
  const upMat = $("#upMatToNpy");
  const upVis = $("#upVisPng");
  const matToNpy = !!(upMat && upMat.checked);
  const visPng = !!(upVis && upVis.checked);
  const upConv = $("#upConvertTxt");
  fd.append("convert_txt", upConv ? !!upConv.checked : true);
  fd.append("mat_to_npy", matToNpy);
  fd.append("sigma", ($("#upMatSigma") && $("#upMatSigma").value) || 2);
  fd.append("adaptive", !!($("#upMatAdaptive") && $("#upMatAdaptive").checked));
  fd.append("vis_png", visPng);
  fd.append("vis_dir", ($("#upVisDir") && $("#upVisDir").value.trim()) || "");
  const btn = $("#btnUploadDs");
  const wrap = $("#uploadProgressWrap");
  const bar = $("#uploadProgressBar");
  const st = $("#uploadStatus");
  btn.disabled = true;
  btn.textContent = tf("msg.uploading", 0);
  if (wrap) { wrap.style.display = ""; if (bar) bar.style.width = "0%"; }
  try {
    const r = await uploadWithProgress("/api/upload/dataset", fd, (p) => {
      btn.textContent = tf("msg.uploading", p);
      if (bar) bar.style.width = p + "%";
    });
    toast(tf("msg.uploadDsDone", r.name, r.num_images, r.num_gt_txt,
             r.num_gt_npy, r.num_gt_mat), "success", 6000);
    // 常用数据集自动统一格式提示（与下载数据集一致）
    if (r.unified && r.unified.preset) {
      toast(tf("upload.msg.unified", r.unified.preset,
               r.unified.train, r.unified.test), "success", 10000);
    }
    if (st) st.textContent = "";
    state.uploadZipFile = null;
    $("#uploadZipFile").value = "";
    await loadUploads();          // 联动训练卡片下拉
    $("#datasetSelect").value = r.name;
    showDatasetStats(r.name);
    await loadDatasetPanel();     // 刷新本面板列表
    selectDatasetTree(r.name);    // 结构浏览器定位到新数据集
    // 后台转换任务（GT → .npy + 可选可视化）：轮询进度
    if (r.post_task_id) {
      pollUploadPostTask(r.post_task_id, r.name);
    }
  } catch (e) {
    toast(tf("msg.uploadFailed", e.message), "error", 8000);
    if (st)
      st.textContent = t("msg.uploadFailedShort");
  } finally {
    btn.disabled = false;
    btn.textContent = t("upload.btn");
    if (wrap) setTimeout(() => { wrap.style.display = "none"; }, 1200);
  }
}

/* 上传后处理任务轮询：GT → .npy 转换（删原 .txt/.mat）+ 可选 .png 可视化 */
function pollUploadPostTask(taskId, dsName) {
  const st = $("#uploadStatus");
  if (st) st.textContent = tf("upload.msg.postStarted", dsName);
  const timer = setInterval(async () => {
    let task;
    try { task = await api(`/api/crowd/train/tasks/${taskId}`); }
    catch (_) { return; }
    if (st) st.textContent =
      `${statusLabel(task.status, task.kind)} · ${dsName} · ` +
      `${Math.round(task.progress || 0)}%` +
      (task.error ? ` | ${task.error}` : "");
    if (["completed", "failed", "stopped"].includes(task.status)) {
      clearInterval(timer);
      if (task.status === "completed" && task.result) {
        const r = task.result;
        toast(tf("upload.msg.postDone", r.npy_converted || 0,
                 r.mat_deleted || 0, r.png_generated || 0),
              "success", 10000);
        if (st) st.textContent = tf("upload.msg.postDone",
          r.npy_converted || 0, r.mat_deleted || 0, r.png_generated || 0);
        loadUploads();            // 刷新 GT 统计（.npy 数量变化）
        loadDatasetPanel();
      } else if (task.status !== "stopped" && st) {
        st.textContent = task.error || t("status.failed");
      }
    }
  }, 2000);
}

/* 渲染已上传数据集表格 */
async function loadDatasetPanel() {
  const tbody = $("#dsTableBody");
  if (!tbody) return;
  // 并行拉取数据集列表与下载链接预设
  let up, dlLinks;
  try {
    [up, dlLinks] = await Promise.all([
      api("/api/uploads"),
      api("/api/download/dataset/links").catch(() => null),
    ]);
  } catch (e) { return; }
  state.uploads = up;   // 与训练卡片共享同一份状态
  state.dlLinks = dlLinks || [];

  // 填充下载链接预设下拉
  const dlSel = $("#dlPresetLink");
  if (dlSel && state.dlLinks.length) {
    dlSel.innerHTML = `<option value="">${esc(t("upload.selectPreset"))}</option>` +
      state.dlLinks.map(l =>
        `<option value="${esc(l.key)}">${esc(l.label)}（${esc(l.size)}）</option>`).join("");
  }

  const dss = up.datasets || [];
  fillDsTreeSelect(dss);
  if (!dss.length) {
    tbody.innerHTML = `<tr><td colspan="5" style="color:var(--text-dim)">` +
      esc(t("upload.empty")) + `</td></tr>`;
    return;
  }
  tbody.innerHTML = dss.map(d => {
    const gt = d.num_gt_txt || d.num_gt_npy || d.num_gt_mat
      ? `${d.num_gt_txt || 0} / ${d.num_gt_npy || 0} / ${d.num_gt_mat || 0}`
      : "—";
    return `<tr>
      <td>${esc(d.name)}</td>
      <td>${d.num_images}</td>
      <td style="font-family:monospace;font-size:12px">${esc(gt)}</td>
      <td style="font-size:11.5px;color:var(--text-dim);max-width:240px;
        overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
        title="${esc(d.path)}">${esc(d.path)}</td>
      <td style="white-space:nowrap">
        <button class="btn small"
        onclick="downloadDatasetGlobal('${esc(d.name)}')">${esc(t("upload.downloadBtn"))}</button>
        <button class="btn small" style="color:var(--red);margin-left:6px"
        onclick="deleteDatasetGlobal('${esc(d.name)}')">${esc(t("upload.deleteBtn"))}</button></td>
    </tr>`;
  }).join("");
}

/* ---------------- 数据集结构浏览（逐级懒加载） ---------------- */

const _dsTreeState = { dataset: "" };
let _dsTreeReq = 0;   // 请求令牌：快速切换数据集时丢弃过期响应

function fillDsTreeSelect(dss) {
  const sel = $("#dsTreeSelect");
  if (!sel) return;
  const keep = sel.value;
  sel.innerHTML = `<option value="">${esc(t("upload.structSelect"))}</option>` +
    dss.map(d =>
      `<option value="${esc(d.name)}">${esc(d.name)}</option>`).join("");
  if (keep && dss.some(d => d.name === keep)) sel.value = keep;
  else if (dss.length) sel.value = dss[0].name;
  loadDatasetTree(sel.value || "");
}

/* 上传/下载完成后，让结构浏览器直接定位到新数据集 */
function selectDatasetTree(name) {
  const sel = $("#dsTreeSelect");
  if (!sel || !name) return;
  if ([...sel.options].some(o => o.value === name)) {
    sel.value = name;
    loadDatasetTree(name);
  }
}

async function loadDatasetTree(name) {
  const box = $("#dsTreeView");
  if (!box) return;
  const token = ++_dsTreeReq;
  _dsTreeState.dataset = name || "";
  if (!name) {
    box.innerHTML =
      `<div class="ds-tree-empty">${esc(t("upload.structEmpty"))}</div>`;
    return;
  }
  box.innerHTML =
    `<div class="ds-tree-empty">${esc(t("upload.structLoading"))}</div>`;
  try {
    const r = await api(`/api/upload/dataset/tree?name=${
      encodeURIComponent(name)}&path=`);
    if (token !== _dsTreeReq) return;   // 已切到别的数据集，丢弃旧响应
    box.innerHTML = "";
    const root = document.createElement("div");
    root.className = "ds-root";
    root.textContent = name;
    box.appendChild(root);
    const wrap = document.createElement("div");
    renderDsEntries(wrap, r.entries || [], 0);
    box.appendChild(wrap);
    if (!(r.entries || []).length) {
      const empty = document.createElement("div");
      empty.className = "ds-tree-empty";
      empty.textContent = t("upload.structNoFiles");
      box.appendChild(empty);
    }
    if (r.truncated) {
      const tip = document.createElement("div");
      tip.className = "ds-empty";
      tip.textContent = t("upload.structTruncated");
      box.appendChild(tip);
    }
  } catch (e) {
    if (token !== _dsTreeReq) return;
    box.innerHTML = `<div class="ds-tree-empty">${
      esc(tf("upload.structError", e.message))}</div>`;
  }
}

function renderDsEntries(box, entries, depth) {
  box.innerHTML = "";
  for (const e of entries) {
    const node = document.createElement("div");
    node.className = "ds-node";
    const row = document.createElement("div");
    row.className = "ds-row" + (e.type === "dir" ? " ds-dir" : " ds-file");
    row.dataset.path = e.path || "";
    row.dataset.type = e.type || "file";
    row.dataset.depth = String(depth);
    row.style.paddingLeft = `${6 + depth * 18}px`;
    const tw = document.createElement("span");
    tw.className = "ds-twisty";
    tw.textContent = e.type === "dir" ? "▸" : "";
    const nm = document.createElement("span");
    nm.className = "ds-name";
    nm.textContent = e.name || "";
    const meta = document.createElement("span");
    meta.className = "ds-meta";
    meta.textContent = e.type === "dir"
      ? t("upload.structDir") : fmtBytes(e.size);
    row.append(tw, nm, meta);
    node.appendChild(row);
    if (e.type === "dir") {
      const ch = document.createElement("div");
      ch.className = "ds-children";
      ch.hidden = true;
      node.appendChild(ch);
    }
    box.appendChild(node);
  }
}

async function toggleDatasetNode(row) {
  const node = row.parentElement;
  if (!node) return;
  let ch = null;
  for (const c of node.children) {
    if (c.classList && c.classList.contains("ds-children")) { ch = c; break; }
  }
  if (!ch) return;
  const tw = row.querySelector(".ds-twisty");
  if (!ch.dataset.loaded) {
    ch.dataset.loaded = "1";
    row.classList.add("loading");
    try {
      const depth = parseInt(row.dataset.depth || "0", 10) || 0;
      const r = await api(`/api/upload/dataset/tree?name=${
        encodeURIComponent(_dsTreeState.dataset)}&path=${
        encodeURIComponent(row.dataset.path || "")}`);
      renderDsEntries(ch, r.entries || [], depth + 1);
      if (!(r.entries || []).length) {
        const empty = document.createElement("div");
        empty.className = "ds-empty";
        empty.style.paddingLeft = `${24 + (depth + 1) * 18}px`;
        empty.textContent = t("upload.structEmptyDir");
        ch.appendChild(empty);
      }
      if (r.truncated) {
        const tip = document.createElement("div");
        tip.className = "ds-empty";
        tip.style.paddingLeft = `${24 + (depth + 1) * 18}px`;
        tip.textContent = t("upload.structTruncated");
        ch.appendChild(tip);
      }
    } catch (e) {
      ch.dataset.loaded = "";
      row.classList.remove("loading");
      toast(tf("upload.structError", e.message), "error");
      return;
    }
    row.classList.remove("loading");
  }
  ch.hidden = !ch.hidden;
  if (tw) tw.textContent = ch.hidden ? "▸" : "▾";
}

/* 下载数据集（后台任务 + 进度轮询） */
async function downloadDatasetGlobal() {
  const url = $("#dlUrl").value.trim();
  if (!url) return toast(t("upload.msg.dlUrlRequired"), "error");
  const name = $("#dlName").value.trim();
  const btn = $("#btnDownloadDs");
  const st = $("#dlStatus");
  const wrap = $("#dlProgressWrap");
  const bar = $("#dlProgressBar");
  btn.disabled = true;
  btn.textContent = t("msg.starting");
  if (wrap) { wrap.style.display = ""; if (bar) bar.style.width = "0%"; }
  if (st) st.textContent = t("upload.msg.dlConnecting");
  try {
    const qs = new URLSearchParams({ url: url });
    if (name) qs.set("name", name);
    const r = await api(`/api/download/dataset?${qs}`, { method: "POST" });
    const tid = r.task_id;
    // 轮询进度
    const timer = setInterval(async () => {
      try {
        const task = await api(`/api/crowd/train/tasks/${tid}`);
        if (bar) bar.style.width = (task.progress || 0) + "%";
        if (st) st.textContent =
          `${statusLabel(task.status, task.kind)} ${Math.round(task.progress || 0)}%`;
        if (["completed", "failed", "stopped"].includes(task.status)) {
          clearInterval(timer);
          btn.disabled = false;
          btn.textContent = t("upload.dlBtn");
          if (task.status === "completed" && task.result) {
            const res = task.result;
            toast(tf("upload.msg.dlDone", res.name, res.num_images),
                  "success", 8000);
            if (st) st.textContent = tf("upload.msg.dlDone",
              res.name, res.num_images);
            // 常用数据集自动统一格式提示
            if (res.unified && res.unified.preset) {
              toast(tf("upload.msg.unified", res.unified.preset,
                       res.unified.train, res.unified.test),
                    "success", 10000);
            }
            await loadUploads();
            $("#datasetSelect").value = res.name;
            showDatasetStats(res.name);
            await loadDatasetPanel();
            selectDatasetTree(res.name);
          } else if (task.status === "failed") {
            toast(tf("msg.startFailed", task.error || ""), "error", 10000);
            if (st) st.textContent = task.error || t("status.failed");
          }
        }
      } catch (_) { /* 静默 */ }
    }, 2000);
  } catch (e) {
    btn.disabled = false;
    btn.textContent = t("upload.dlBtn");
    if (st) st.textContent = "";
    toast(tf("msg.startFailed", e.message), "error", 8000);
  }
}

/* 面板内删除（带运行中占用提示，后端会拦截） */
async function deleteDatasetGlobal(name) {
  if (!window.confirm(`${t("msg.confirmDeleteDataset")}\n${name}`)) return;
  try {
    await api(`/api/upload/dataset/${encodeURIComponent(name)}`,
              { method: "DELETE" });
    toast(tf("msg.datasetDeleted", name), "success");
    await loadUploads();
    await loadDatasetPanel();
  } catch (e) {
    toast(tf("msg.deleteFailed", e.message), "error", 6000);
  }
}

/* 下载已上传数据集（服务端打包为 zip，含统一后的 Images/GT 结构） */
function downloadDatasetGlobal(name) {
  if (!name) return;
  toast(tf("msg.datasetDownloading", name), "success", 3000);
  const a = document.createElement("a");
  a.href = `/api/upload/dataset/${encodeURIComponent(name)}/download`;
  a.download = `${name}.zip`;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/* ---------------- 「数据读取预处理」全局模块 ---------------- */

let _ppBound = false;

function bindPreprocessPanel() {
  if (_ppBound) return;
  _ppBound = true;
  const presetSel = $("#ppPreset");
  if (!presetSel) return;
  presetSel.addEventListener("change", () => {
    const preset = state.ppPresets && state.ppPresets.find(
      p => p.key === presetSel.value);
    if (preset) {
      const treeEl = $("#ppTree");
      if (treeEl) treeEl.textContent = preset.tree || "";
      const card = $("#ppTreeCard");
      if (card) card.style.display = preset.tree ? "" : "none";
    }
  });
  const srcSel = $("#ppDatasetSelect");
  if (srcSel) {
    srcSel.addEventListener("change", (e) => {
      if (e.target.value) $("#ppDatasetPath").value = "";
    });
  }
  const pathInp = $("#ppDatasetPath");
  if (pathInp) {
    pathInp.addEventListener("input", (e) => {
      if (e.target.value) $("#ppDatasetSelect").value = "";
    });
  }
  const bsSel = $("#ppBoxSource");
  if (bsSel) {
    bsSel.addEventListener("change", (e) => {
      const v = e.target.value;
      const ad = $("#ppAdaptiveParams");
      const yo = $("#ppYoloParams");
      if (ad) ad.style.display = v === "adaptive" ? "" : "none";
      if (yo) yo.style.display = v === "yolo_detect" ? "" : "none";
    });
  }
  const btn = $("#btnPreprocess");
  if (btn) btn.addEventListener("click", startPreprocess);
}

async function loadPreprocessPanel() {
  // 拉取数据集预设清单 + 已上传数据集列表
  try {
    const presets = await api("/api/crowd/preprocess/datasets");
    state.ppPresets = presets;
    const sel = $("#ppPreset");
    if (sel && presets.length) {
      sel.innerHTML = presets.map(p =>
        `<option value="${esc(p.key)}">${esc(p.label)}（${p.annotation === "box" ? "含框" : "纯点"}标注）</option>`).join("");
      // 默认选中第一项并显示 tree
      const ev = new Event("change");
      sel.dispatchEvent(ev);
    }
  } catch (e) { /* 静默 */ }
  try {
    const up = await api("/api/uploads");
    const sel = $("#ppDatasetSelect");
    if (sel) {
      const cur = sel.value;
      const dss = [{ name: "" }, ...up.datasets];
      fillSelect("#ppDatasetSelect", dss, d => d.name || "",
        d => d.name || t("common.select"));
      if (cur) sel.value = cur;
    }
    // 填充 YOLO 权重下拉（crowd_box 训练产物）
    const yw = $("#ppYoloWeight");
    if (yw) {
      const wts = [{ name: "", path: "" },
        ...up.trained_weights.filter(w => w.name.endsWith(".pt"))];
      fillSelect("#ppYoloWeight", wts, w => w.path || w.name,
        w => w.name || t("common.dontUse"));
    }
  } catch (e) { /* 静默 */ }
}

async function startPreprocess() {
  const preset = $("#ppPreset") ? $("#ppPreset").value : "custom_txt";
  const dataset = ($("#ppDatasetSelect") && $("#ppDatasetSelect").value) ||
    ($("#ppDatasetPath") ? $("#ppDatasetPath").value.trim() : "");
  if (!dataset) return toast(t("msg.dsRequired"), "error");
  const boxSource = $("#ppBoxSource") ? $("#ppBoxSource").value : "adaptive";
  const yoloWeight = $("#ppYoloWeight") ? $("#ppYoloWeight").value : "";
  if (boxSource === "yolo_detect" && !yoloWeight) {
    return toast(t("preprocess.msg.yoloWeightRequired"), "error");
  }

  const qs = new URLSearchParams({
    dataset_path: dataset,
    preset: preset,
    box_source: boxSource,
    yolo_weight: yoloWeight,
    yolo_conf: $("#ppYoloConf") ? $("#ppYoloConf").value : "0.25",
    adaptive_k: $("#ppAdaptiveK") ? $("#ppAdaptiveK").value : "4",
    adaptive_scale: $("#ppAdaptiveScale") ? $("#ppAdaptiveScale").value : "0.4",
    adaptive_default: $("#ppAdaptiveDefault") ? $("#ppAdaptiveDefault").value : "20",
    save_dir: $("#ppSaveDir") ? $("#ppSaveDir").value.trim() : "",
  });
  const btn = $("#btnPreprocess");
  btn.disabled = true;
  btn.textContent = t("msg.starting");
  const st = $("#ppStatus");
  if (st) st.textContent = t("preprocess.msg.running");
  const log = $("#ppLog");
  if (log) { log.style.display = ""; log.textContent = ""; }
  const wrap = $("#ppProgressWrap");
  if (wrap) wrap.style.display = "";
  try {
    const r = await api(`/api/crowd/preprocess?${qs}`, { method: "POST" });
    // 轮询任务进度
    const tid = r.task_id;
    const timer = setInterval(async () => {
      try {
      const task = await api(`/api/crowd/train/tasks/${tid}`);
      const bar = $("#ppProgressBar");
      if (bar) bar.style.width = (task.progress || 0) + "%";
      if (st) st.textContent =
      `${statusLabel(task.status, task.kind)} ${task.epoch}/${task.total_epochs}`;
      const lr = await api(`/api/crowd/train/tasks/${tid}/log?tail=15`);
        if (log) log.textContent = lr.log || "";
        if (["completed", "failed", "stopped"].includes(task.status)) {
          clearInterval(timer);
          btn.disabled = false;
          btn.textContent = t("preprocess.start");
          if (task.status === "completed" && task.result) {
            const res = task.result;
            toast(t("preprocess.msg.done", res.processed, res.skipped,
                     res.boxes_generated), "success", 8000);
            if (st) st.textContent = t("preprocess.msg.done",
              res.processed, res.skipped, res.boxes_generated)
              + (res.save_dir ? ` | 输出: ${res.save_dir}` : "");
            // 输出目录与源数据集同级（<数据集名>_box），刷新数据集清单
            try { await loadUploads(); } catch (_) { /* ignore */ }
            try { await loadPreprocessPanel(); } catch (_) { /* ignore */ }
          } else if (task.status === "failed") {
            toast(tf("msg.startFailed", task.error || ""), "error", 8000);
            if (st) st.textContent = task.error || t("status.failed");
          }
        }
      } catch (_) { /* 静默 */ }
    }, 2000);
  } catch (e) {
    btn.disabled = false;
    btn.textContent = t("preprocess.start");
    if (st) st.textContent = "";
    toast(tf("msg.startFailed", e.message), "error", 8000);
  }
}
function fillSelect(sel, items, valueFn, labelFn) {
  const el = $(sel);
  if (!el) return;
  el.innerHTML = items.map(it =>
    `<option value="${esc(valueFn(it))}">${esc(labelFn(it))}</option>`).join("");
}

/* 语言切换后刷新静态下拉首项文案（不重新拉取后端数据） */
function refreshStaticOptions() {  const pairs = [
    ["#datasetSelect", "common.select"],
    ["#trainInitWeight", "common.dontUse"],
    ["#inferWeightSelect", "common.dontUse"],
    ["#distillTeacherWeight", "common.dontUse"],
    ["#deployWeightSelect", "common.dontUse"],
  ];
  for (const [sel, key] of pairs) {
    const el = $(sel);
    if (el && el.options.length && !el.options[0].value) {
      el.options[0].textContent = t(key);
    }
  }
}

function refreshFooterInfo() {
  api("/api/devices").then(d => {
    $("#footerInfo").textContent =
      `PyTorch ${d.torch_version} | CUDA: ${d.cuda_available ? t("cuda.yes") : t("cuda.no")}`;
  }).catch(() => {});
}

/* ---------------- 头部控件：语言切换 / 深度学习框架切换 ---------------- */

function applyTheme(name, persist = true) {
  const allowed = ["dark", "light", "midnight", "solarized"];
  const theme = allowed.includes(String(name)) ? String(name) : "dark";
  document.documentElement.setAttribute("data-theme", theme);
  if (persist) {
    try { localStorage.setItem("dlatf_theme", theme); } catch (e) { /* ignore */ }
  }
  const sel = $("#themeSelect");
  if (sel) sel.value = theme;
}

function bindHeaderControls() {
  const themeSel = $("#themeSelect");
  if (themeSel) {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    themeSel.value = current;
    themeSel.addEventListener("change", () => applyTheme(themeSel.value));
  }
  const langBtn = $("#langToggle");
  if (langBtn) {
    langBtn.addEventListener("click", () => {
      window.DLATF_I18N.setLang(
        window.DLATF_I18N.lang === "zh" ? "en" : "zh");
    });
  }
  updateLangBtn();
}

function updateLangBtn() {
  const btn = $("#langToggle");
  if (btn) btn.textContent = window.DLATF_I18N.lang === "zh" ? "EN" : "中";
}

async function loadFrameworks() {
  let data = null;
  try { data = await api("/api/frameworks"); } catch (_) { /* 离线兜底 */ }
  state.frameworks = (data && data.frameworks && data.frameworks.length)
    ? data.frameworks
    : [{ key: "pytorch", label: "PyTorch", available: true }];
  state.currentFramework = (data && data.current) || "pytorch";
  // 若默认框架不可用，切换到第一个可用框架
  const defFw = state.frameworks.find(f => f.key === state.currentFramework);
  if (!defFw || !defFw.available) {
    const alt = state.frameworks.find(f => f.available);
    if (alt) state.currentFramework = alt.key;
  }
  renderFrameworkNav();
  renderSubNav(state.currentFramework);
  const domains = fwDomains(state.currentFramework);
  if (domains.length) selectSubModule(domains[0].key, state.currentFramework);
}

/* 语言切换回调（由 i18n.js setLang 触发）：静态文案由 applyI18n 处理，
   这里刷新全部动态渲染内容 */
window.__onLangChange = function () {
  updateLangBtn();
  refreshStaticOptions();
  // 暂存当前下拉选择，重渲染后恢复
  const keep = {};
  for (const sel of ["#trainModel", "#inferModel", "#lossSelect", "#scheduler",
                     "#distillTeacherWeight", "#distillTeacherModel"]) {
    const el = $(sel);
    if (el) keep[sel] = el.value;
  }
  rebuildCvTaskSelect();
  if (state.currentTask) selectTask(state.currentTask.key);
  for (const [sel, val] of Object.entries(keep)) {
    const el = $(sel);
    if (el && val && [...el.options].some(o => o.value === val)) el.value = val;
  }
  refreshFooterInfo();
  refreshTaskHistory();
  updateInferHints();
  updateTrainModelUI();
  renderFrameworkNav();
  renderSubNav(state.currentFramework);
  renderLearningMap();  // 学习版图标题跟随语言
  loadDatasetPanel();   // 数据集面板表格文案跟随语言
  if (hwState.lastData) renderHardware(hwState.lastData);  // 硬件面板按新语言重渲染
  if (_mvBuilt) initMetricsPanel(true);   // 指标可视化卡片文案跟随语言
  loadProxyConfig();  // 代理状态行按新语言重渲染（重新拉取后端状态）
  loadPreprocessPanel();  // 预处理面板文案跟随语言
};

/* ---------------- 领域/模式切换 ---------------- */

/* ---------------- 两级导航：框架一级 + 领域二级 ---------------- */

/* PyTorch 领域子模块（本框架专注人群计数） */
const PT_DOMAINS = [
  { key: "cv", labelKey: "nav.cv", defaultLabel: "人群计数", panel: "panel-cv",
    hash: "crowd" },
];

function fwDomains(fwKey) {
  return PT_DOMAINS;
}

/* 一级导航：渲染框架按钮（框架 + 全局模块同层级） */
function renderFrameworkNav() {
  const box = $("#fwNav");
  if (!box) return;
  box.innerHTML = state.frameworks.map(fw => {
    const active = fw.key === state.currentFramework;
    const badge = fw.available ? "" :
      `<span class="fw-badge">${esc(t("fw.plannedBadge"))}</span>`;
    return `<button type="button" class="fw-btn${active ? " active" : ""}${fw.available ? "" : " planned"}"
      data-framework="${esc(fw.key)}">${esc(fw.label)}${badge}</button>`;
  }).join("");
  box.querySelectorAll(".fw-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const fw = state.frameworks.find(x => x.key === btn.dataset.framework);
      if (!fw) return;
      state.currentFramework = fw.key;
      selectFramework(fw.key);
      if (!fw.available) {
        toast(t("fw.comingSoon"), "info", 4000);
      }
    });
  });
}

/* 选择框架：更新一级高亮 + 重新渲染二级领域 + 重新加载当前领域数据 */
function selectFramework(fwKey) {
  state.currentFramework = fwKey;
  renderFrameworkNav();
  renderSubNav(fwKey);
  // 自动激活第一个领域（人群计数）
  const domains = fwDomains(fwKey);
  if (domains.length) selectSubModule(domains[0].key, fwKey);
}

/* 二级导航：渲染领域按钮 */
function renderSubNav(fwKey) {
  const nav = $("#subNav");
  if (!nav) return;
  const fwLabel = (state.frameworks.find(f => f.key === fwKey) || {}).label || "";
  let html = `<span class="sub-label">${esc(fwLabel)} ▸</span>`;
  html += fwDomains(fwKey).map(domain => {
    const label = t(domain.labelKey);
    const dLabel = label === domain.labelKey ? domain.defaultLabel : label;
    return `<button class="nav-btn" data-domain="${esc(domain.key)}">${esc(dLabel)}</button>`;
  }).join("");
  nav.innerHTML = html;
  nav.querySelectorAll(".nav-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      // 始终使用 state.currentFramework（防止闭包 fwKey 过期）
      selectSubModule(btn.dataset.domain, state.currentFramework);
    });
  });
}

/* 选择领域子模块：激活对应面板 */
function selectSubModule(domainKey, fwKey) {
  const domain = fwDomains(fwKey || state.currentFramework)
    .find(d => d.key === domainKey);
  if (!domain) return;
  // 高亮二级按钮
  const nav = $("#subNav");
  if (nav) {
    nav.querySelectorAll(".nav-btn").forEach(b =>
      b.classList.toggle("active", b.dataset.domain === domainKey));
  }
  // 取消全局导航高亮
  $$("#globalNav .nav-btn").forEach(b => b.classList.remove("active"));
  // 隐藏所有面板，激活目标面板
  $$(".panel").forEach(p => p.classList.remove("active"));
  const panel = document.getElementById(domain.panel);
  if (panel) panel.classList.add("active");
  // 切走时停掉硬件/日志轮询
  stopHwPolling();
  stopBlogPolling();
  // 刷新模型列表
  refreshCvModels(fwKey || state.currentFramework);
}

/* 刷新人群计数模型列表 */
async function refreshCvModels(fwKey) {
  try {
    const tasks = await api("/api/crowd/tasks");
    state.tasks = tasks;
    rebuildCvTaskSelect();
    if (state.currentTask) selectTask(state.currentTask.key);
  } catch (e) { /* 静默 */ }
}

/* 绑定全局导航（系统状态 / 代理 / 指标 / 学习版图 / 个人主页） */
function bindGlobalNav() {
  $$("#globalNav .nav-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      // 高亮全局导航按钮，取消二级按钮高亮
      $$("#globalNav .nav-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      $$("#subNav .nav-btn").forEach(b => b.classList.remove("active"));
      const domain = btn.dataset.domain;
      $$(".panel").forEach(p => p.classList.remove("active"));
      $(`#panel-${domain}`).classList.add("active");
      if (domain === "upload") loadDatasetPanel();
      if (domain === "preprocess") loadPreprocessPanel();
      if (domain === "visualize") loadVisualizePanel();
      if (domain === "learning") renderLearningMap();
      if (domain === "proxy") loadProxyConfig();
      if (domain === "metrics") initMetricsPanel();
      if (domain === "hardware") { startHwPolling(); startBlogPolling(); }
      else { stopHwPolling(); stopBlogPolling(); }
    });
  });
}

function bindModeSwitch() {
  $$(".mode-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      // 切换作用域限定在按钮所属面板内
      const panel = btn.closest(".panel") || document;
      panel.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      panel.querySelectorAll(".mode-panel").forEach(p => p.classList.remove("active"));
      const target = panel.querySelector(`#mode-${btn.dataset.mode}`);
      if (target) target.classList.add("active");
    });
  });
}

function bindTaskSelect() {
  $("#cvTaskSelect").addEventListener("change", (e) => selectTask(e.target.value));
}

/* ---------------- 独立测试模式 ---------------- */

const TESTABLE_TASKS = ["crowd_sdg", "crowd_density", "crowd_point"];

function refreshTestTasks() {
  const sel = $("#testTaskSelect");
  if (!sel) return;
  const keep = sel.value;
  const list = (state.tasks || []).filter(tk => TESTABLE_TASKS.includes(tk.key));
  fillSelect("#testTaskSelect", list, tk => tk.key, tk => taskLabel(tk));
  if (keep && list.some(tk => tk.key === keep)) sel.value = keep;
  else if (list.length) sel.value = list[0].key;
  refreshTestModels();
}

function refreshTestModels() {
  const sel = $("#testModel");
  if (!sel) return;
  const taskKey = ($("#testTaskSelect") && $("#testTaskSelect").value) || "";
  const tk = (state.tasks || []).find(x => x.key === taskKey);
  const models = (tk && (tk.models || tk.train_models)) || [];
  const keep = sel.value;
  fillSelect("#testModel", models, m => m.name, m => modelOptionLabel(m));
  if (keep && models.some(m => m.name === keep)) sel.value = keep;
}

function refreshTestWeights() {
  const sel = $("#testWeightSelect");
  if (!sel) return;
  const isOnnx = (m) => /\.onnx$/i.test(m.name || "");
  const pt = ((state.uploads && state.uploads.models) || [])
    .filter(m => !isOnnx(m));
  const opts = [{ name: "", path: "" },
    ...((state.uploads && state.uploads.trained_weights) || []), ...pt];
  const keep = sel.value;
  fillSelect("#testWeightSelect", opts, m => m.path || m.name,
    m => m.name || t("common.dontUse"));
  if (keep && [...sel.options].some(o => o.value === keep)) sel.value = keep;
}

function refreshTestDatasets() {
  const sel = $("#testDatasetSelect");
  if (!sel) return;
  const keep = sel.value;
  fillSelect("#testDatasetSelect",
    [{ name: "", path: "" }, ...((state.uploads && state.uploads.datasets) || [])],
    d => d.name || "", d => d.name || t("common.select"));
  if (keep && [...sel.options].some(o => o.value === keep)) sel.value = keep;
}

function refreshTestDevices() {
  const src = $("#trainDevice");
  const dst = $("#testDevice");
  if (!src || !dst) return;
  const keep = dst.value;
  dst.innerHTML = src.innerHTML;
  if (keep && [...dst.options].some(o => o.value === keep)) dst.value = keep;
}

function bindTestControls() {
  if (!$("#btnRunTest")) return;
  $("#testTaskSelect").addEventListener("change", refreshTestModels);
  $("#testDatasetSelect").addEventListener("change", (e) => {
    if (e.target.value) $("#testDatasetPath").value = "";
  });
  $("#testDatasetPath").addEventListener("input", (e) => {
    if (e.target.value) $("#testDatasetSelect").value = "";
  });
  $("#btnUploadTestWeight").addEventListener("click",
    () => $("#testWeightFile").click());
  $("#testWeightFile").addEventListener("change", () =>
    uploadWeightToSelect("#testWeightFile", "#btnUploadTestWeight",
                         "#testWeightSelect", "test.uploadWeightBtn"));
  $("#btnRunTest").addEventListener("click", runTest);
  $("#btnStopTest").addEventListener("click", stopTest);
  refreshTestTasks();
  refreshTestWeights();
  refreshTestDatasets();
  refreshTestDevices();
}

function stopTestPolling() {
  if (state.testPollTimer) {
    clearInterval(state.testPollTimer);
    state.testPollTimer = null;
  }
}

async function runTest() {
  const taskKey = $("#testTaskSelect").value;
  const model = $("#testModel").value;
  const weight = $("#testWeightSelect").value;
  const dataset = $("#testDatasetSelect").value ||
    ($("#testDatasetPath").value || "").trim();
  if (!model) return toast(t("test.msg.needModel"), "error");
  if (!weight) return toast(t("test.msg.needWeight"), "error");
  if (!dataset) return toast(t("test.msg.needDataset"), "error");
  const btn = $("#btnRunTest");
  btn.disabled = true;
  btn.textContent = t("msg.starting");
  $("#btnStopTest").disabled = true;
  $("#testMetrics").innerHTML = "";
  $("#testLog").textContent = "";
  const body = {
    task: taskKey, model, weight_path: weight, dataset_path: dataset,
    split: $("#testSplit").value, patch_size: +($("#testPatch").value || 0),
    device: $("#testDevice").value || "cpu",
  };
  try {
    const r = await api("/api/crowd/test", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.testTaskId = r.task_id;
    $("#btnStopTest").disabled = false;
    toast(tf("test.msg.started", r.run_name), "success");
    stopTestPolling();
    state.testPollTimer = setInterval(pollTest, 1500);
    pollTest();
  } catch (e) {
    toast(tf("msg.startFailed", e.message), "error", 8000);
    btn.disabled = false;
    btn.textContent = t("test.run");
  }
}

async function pollTest() {
  const id = state.testTaskId;
  if (!id) return stopTestPolling();
  let task;
  try { task = await api(`/api/crowd/train/tasks/${id}`); }
  catch (_) { return; }
  const pct = Math.round(task.progress || 0);
  $("#testProgress").style.width = pct + "%";
  $("#testStatus").innerHTML =
    `${esc(t("label.status"))}<b>${statusLabel(task.status, task.kind)}</b> · ${pct}%` +
    (task.error ? ` · <span style="color:var(--red)">${esc(task.error)}</span>` : "");
  try {
    const lr = await api(`/api/crowd/train/tasks/${id}/log?tail=80`);
    if (lr && lr.log) $("#testLog").textContent = lr.log;
  } catch (_) { /* ignore */ }
  if (!["completed", "failed", "stopped"].includes(task.status)) return;
  stopTestPolling();
  $("#btnRunTest").disabled = false;
  $("#btnRunTest").textContent = t("test.run");
  $("#btnStopTest").disabled = true;
  if (task.status === "completed" && task.result) {
    const res = task.result;
    const rmse = res.rmse !== undefined ? res.rmse : res.mse;
    $("#testMetrics").innerHTML =
      `<div class="stat"><b>${esc(res.mae)}</b><span>MAE</span></div>` +
      `<div class="stat"><b>${esc(rmse)}</b><span>RMSE</span></div>` +
      `<div class="stat"><b>${esc(res.num_images)}</b><span>${esc(t("test.samples"))}</span></div>` +
      `<div class="stat"><b>${esc(res.model || "")}</b><span>${esc(t("test.model"))}</span></div>`;
    toast(tf("test.msg.done", res.mae, rmse), "success", 8000);
    await loadUploads();
  } else if (task.status === "failed") {
    toast(`${t("msg.trainFailed")}: ${task.error || ""}`, "error", 8000);
  }
}

async function stopTest() {
  const id = state.testTaskId;
  if (!id) return;
  try {
    await api(`/api/crowd/train/tasks/${id}/stop`, { method: "POST" });
    toast(t("test.msg.stopping"), "info");
  } catch (e) {
    toast(e.message, "error", 6000);
  }
}

/* 任务标签/描述：优先使用前端词典（跟随界面语言），未知 key 回退后端文案 */
function taskLabel(tk) {
  const v = t("tasklabel." + tk.key);
  return v === ("tasklabel." + tk.key) ? (tk.label || tk.key) : v;
}

function taskDesc(tk) {
  const v = t("taskdesc." + tk.key);
  return v === ("taskdesc." + tk.key) ? (tk.desc || "") : v;
}

/* 人群计数训练超参默认值（切换子任务时重置表单；img_size = 训练裁剪尺寸） */
const TRAIN_DEFAULTS_BASE = {
  batchSize: 2, numWorkers: 0, epochs: 30, earlyStopping: 0, evalEvery: 1, valRatio: 0.2, seed: 42,
  amp: false, augment: true, pretrained: true,
  optimizer: "adam",
};
const TRAIN_DEFAULTS = {
  crowd_density: { ...TRAIN_DEFAULTS_BASE,
    lr: 0.00001, scheduler: "cosine",
    loss: "mse", imgSize: 384 },
  // 点预测损失由具体算法内置，默认按首个模型 p2pnet 给；
  // updateTrainModelUI 会按所选模型（p2r/cp/cu）纠正
  crowd_point: { ...TRAIN_DEFAULTS_BASE,
    lr: 0.0001, scheduler: "cosine",
    loss: "p2pnet", imgSize: 1024 },
  crowd_box: { ...TRAIN_DEFAULTS_BASE,
    epochs: 50, batchSize: 4, lr: 0.01, optimizer: "sgd",
    scheduler: "cosine", loss: "yolo", imgSize: 640 },
  // 单域泛化（DCCUS / LLB_Count / MPCount）：MPCount AdamW 1e-3，其余 Adam 1e-5
  crowd_sdg: { ...TRAIN_DEFAULTS_BASE,
    epochs: 100, batchSize: 1, lr: 0.00001, optimizer: "adam",
    scheduler: "cosine", loss: "dccus", imgSize: 320 },
  crowd_counting: { ...TRAIN_DEFAULTS_BASE,   // 兼容旧任务 kind
    lr: 0.00001, scheduler: "cosine",
    loss: "mse", imgSize: 384 },
};

/* 输入/裁剪分辨率可选值：全路线统一 256 ~ 2048 区间（按算法对齐约束筛选） */
const RESOLUTION_PRESETS = {
  density: [256, 320, 384, 448, 512, 640, 768, 1024, 1280, 1536, 2048],
  p2pnet: [256, 384, 512, 640, 768, 1024, 1280, 1536, 2048],
  p2r: [256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048],
  cp: [256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048],  // 64 倍数（patch 网格）
  cu: [256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048],  // 64 倍数（patch 网格）
  pet: [256, 384, 512, 640, 768, 1024, 1280, 1536, 2048],
  cltr: [256, 384, 512, 640, 768, 1024, 1280, 1536, 2048],
  sdg: [256, 320, 384, 448, 512, 640, 768, 1024, 1280, 1536, 2048],
  raz: [128, 192, 256, 320, 384, 512, 640, 768, 1024],
  d2c: [256, 320, 384, 448, 512, 640, 768, 1024, 1280, 1536],
  box: [256, 320, 416, 480, 512, 640, 768, 960, 1024, 1280, 1536, 2048],
};
const RESOLUTION_DEFAULTS = {
  density: 384, p2pnet: 1024, p2r: 256, cp: 256, cu: 256, pet: 256, cltr: 256,
  sdg: 320, raz: 256, d2c: 384, box: 640,
};

function resolutionKey(taskKey, model) {
  if (taskKey === "crowd_box") return "box";
  if (taskKey === "crowd_sdg") return "sdg";
  if (model === "pet") return "pet";
  if (model === "cltr") return "cltr";
  if (model === "raz") return "raz";
  if (model === "d2c") return "d2c";
  if (taskKey === "crowd_point") {
    return ["p2pnet", "p2r", "cp", "cu"].includes(model) ? model : "p2pnet";
  }
  return "density";
}

function resolutionLabel(v) {
  return Number(v) > 0 ? `${v} × ${v}` : t("train.origRes");
}

/* 人群计数的技术路线（后端子任务 key） */
const CROWD_ROUTE_KEYS = ["crowd_counting", "crowd_density", "crowd_point",
                          "crowd_box"];
const isCrowdTask = (key) => CROWD_ROUTE_KEYS.includes(key);

let appliedDefaultsTask = null;
let _lastTrainModel = null;   // 训练模型上一次值（分辨率仅在模型/任务变化时重置）
let _lastTrainTask = null;

/* 各优化器推荐默认超参，跟后端 optim_builders.OPTIMIZER_DEFAULTS 一致 */
const OPTIMIZER_DEFAULTS = {
  adam:     { momentum: 0.9,  weightDecay: 0.0 },
  sgd:      { momentum: 0.9,  weightDecay: 0.0005 },
  adamw:    { momentum: 0.9,  weightDecay: 0.01 },
  rmsprop:  { momentum: 0.9,  weightDecay: 0.0 },
  adagrad:  { momentum: 0.0,  weightDecay: 0.0 },
  adadelta: { momentum: 0.0,  weightDecay: 0.0 },
  adamax:   { momentum: 0.9,  weightDecay: 0.0 },
  asgd:     { momentum: 0.0,  weightDecay: 0.0001 },
};
/* 动量只对 SGD/RMSprop 系生效（其余禁用动量框） */
const MOMENTUM_OPTIMS = new Set(["sgd", "momentum", "rmsprop"]);

/* 按优化器填默认动量/权重衰减；动量输入对所有优化器开放编辑
   （Adam 系内部不使用动量，悬停提示说明）。
   emptyMomDefault 是优化器为空值时的动量回填 */
function applyOptimizerDefaults(optEl, momEl, wdEl, emptyMomDefault) {
  if (!optEl) return;
  const name = optEl.value;
  const d = OPTIMIZER_DEFAULTS[name];
  if (momEl) {
    if (d) momEl.value = d.momentum;
    else if (emptyMomDefault !== undefined) momEl.value = emptyMomDefault;
    momEl.disabled = false;
    momEl.title = (d && !MOMENTUM_OPTIMS.has(name)) ? t("hint.momNA") : "";
  }
  if (wdEl && d) wdEl.value = d.weightDecay;
}

/* 给训练表单绑优化器切换联动 */
function bindOptimizerDefaults(optSel, momSel, wdSel, emptyMomDefault) {
  if (!optSel) return;
  optSel.addEventListener("change", () =>
    applyOptimizerDefaults(optSel, momSel, wdSel, emptyMomDefault));
}

/* 数值输入，空/非法回退默认；0 是合法值 */
function numOr(el, dflt) {
  if (!el) return dflt;
  const v = parseFloat(el.value);
  return Number.isFinite(v) ? v : dflt;
}

/* 范围校验，不合法直接抛错 */
function numInRange(el, min, max, name) {
  const v = parseFloat(el && el.value);
  if (!Number.isFinite(v) || v < min || v > max) {
    throw new Error(tf("msg.numRange", name, min, max));
  }
  return v;
}

/* 将某领域的默认超参数填入训练表单（仅任务变化时调用，避免覆盖用户修改） */
function applyTrainDefaults(key) {
  const d = TRAIN_DEFAULTS[key];
  if (!d) return;
  const numFields = { epochs: "#epochs", batchSize: "#batchSize",
    lr: "#lr", weightDecay: "#weightDecay", imgSize: "#imgSize",
    numWorkers: "#numWorkers", earlyStopping: "#earlyStopping",
    evalEvery: "#evalEvery", valRatio: "#valRatio",
    seed: "#seed" };
  for (const [k, sel] of Object.entries(numFields)) {
    const el = $(sel);
    if (el && d[k] !== undefined) el.value = d[k];
  }
  const selFields = { optimizer: "#optimizer", scheduler: "#scheduler",
    loss: "#lossSelect" };
  for (const [k, sel] of Object.entries(selFields)) {
    const el = $(sel);
    if (el && d[k] !== undefined &&
        [...el.options].some(o => o.value === String(d[k]))) {
      el.value = d[k];
    }
  }
  if (d.pretrained !== undefined) $("#trainPretrained").checked = d.pretrained;
  if (d.augment !== undefined) $("#augment").checked = d.augment;
  // 任务默认优化器定下后联动默认动量/权重衰减
  applyOptimizerDefaults($("#optimizer"), $("#momentum"), $("#weightDecay"));
  appliedDefaultsTask = key;
}

function selectTask(key) {
  const task = state.tasks.find(x => x.key === key);
  if (!task) return;
  state.currentTask = task;
  $("#cvTaskDesc").textContent = taskDesc(task);

  // 训练可用性
  $("#trainUnavailable").style.display = task.trainable ? "none" : "block";
  $("#trainCards").style.display = task.trainable ? "" : "none";

  // 模型下拉：训练与推理共用同一清单（带能力标签）
  const inferModels = task.models || [];
  const trainModels = (task.train_models && task.train_models.length)
    ? task.train_models : inferModels;
  fillSelect("#trainModel", trainModels, m => m.name,
    m => `${m.name}${m.desc ? " — " + m.desc : ""}`);
  fillSelect("#inferModel", inferModels, m => m.name, m => modelOptionLabel(m));

  // 损失函数 / 调度器（按任务；旧 key 回退到密度图配置）
  const losses = (state.lossesByTask && (state.lossesByTask[key] ||
               state.lossesByTask["crowd_density"])) || ["mse"];
  fillSelect("#lossSelect", losses, l => l, l => lossName(l));
  const scheds = (state.schedulersByTask && (state.schedulersByTask[key] ||
                state.schedulersByTask["crowd_density"])) ||
    ["step", "cosine"];
  fillSelect("#scheduler", scheds, s => s, s => schedulerName(s));

  // 该领域默认超参数（仅任务变化时重置，语言切换等重复调用不覆盖）
  if (appliedDefaultsTask !== key) applyTrainDefaults(key);

  // 按技术路线显隐：
  //  - 点预测（crowd_point）：无需密度图工具/裁剪尺寸，显示置信度阈值
  //  - 头部框（crowd_box）：无需密度图工具，显示标注格式与 YOLO 增强，
  //    imgSize 作为 YOLO 输入尺寸保留显示，conf 为检测置信度阈值
  //  - 密度图（crowd_density）：显示专属参数与工具（GT 预先转换为 .npy
  //    密度图；.txt 点标注仍可直接训练，σ 固定用后端默认 8——界面
  //    sigma / k 近邻自适应参数已按需移除）
  const isPoint = key === "crowd_point";
  const isBox = key === "crowd_box";
  // 空值防御：元素缺失时跳过显隐（防止版本混搭/漏改 HTML 导致页面崩溃）
  const show = (sel, visible) => {
    const el = $(sel);
    if (el) el.style.display = visible ? "" : "none";
  };
show("#fieldConf", isPoint || isBox);
// 分辨率（输入/裁剪尺寸）对所有路线可选：P2PNet 支持原始分辨率，
// P2R/CP/CU 为裁剪尺寸，密度图为 crop，YOLO 为输入尺寸
show("#fieldImgSize", true);
  show("#fieldAnnoFormat", isBox);
  show("#fieldYoloAug", isBox);
  show("#fieldYoloDistill", isBox);
  if (isBox) fillDistillTeacherSelects();

  // 调度器专属参数显隐（warmup_cosine/t0/poly 按选择切换）
  const sched = $("#scheduler") ? $("#scheduler").value : "cosine";
  show("#fieldWarmupEpochs", sched === "warmup_cosine");
  show("#fieldT0Epochs", sched === "warm_restarts");
  show("#fieldPolyPower", sched === "poly");

  // 数据集提示（按路线）
  $("#datasetHint").textContent =
    t(isPoint ? "ds.crowd_point"
      : isBox ? "ds.crowd_box"
        : (key === "crowd_sdg" ? "ds.crowd_sdg" : "ds.crowd_density"));

  // 预训练默认值：按模型能力决定（见 updateInferHints）
  updateInferHints();
  // 按所选模型刷新损失函数与分辨率选项（并给默认值）
  updateTrainModelUI();
}

/* 头部框路线：填充知识蒸馏教师选项
   - 教师权重 = 训练产物 + 上传权重（value 为完整路径；推荐同数据集训练产物）
   - 教师模型 = 官方 COCO 预训练名单（排除 RT-DETR / 自定义 / 无官方权重的 v6） */
function fillDistillTeacherSelects() {
  const task = state.currentTask;
  if (!task || task.key !== "crowd_box") return;
  const keepW = $("#distillTeacherWeight") ? $("#distillTeacherWeight").value : "";
  const wOpts = [{ name: "", path: "" },
                 ...((state.uploads && state.uploads.trained_weights) || []),
                 ...((state.uploads && state.uploads.models) || [])];
  fillSelect("#distillTeacherWeight", wOpts, m => m.path || m.name,
    m => m.name || t("common.dontUse"));
  const wSel = $("#distillTeacherWeight");
  if (wSel && keepW && [...wSel.options].some(o => o.value === keepW))
    wSel.value = keepW;
  const tms = (task.train_models || task.models || []).filter(m =>
    m.name && !m.name.startsWith("rtdetr") && m.name !== "yolo_custom" &&
    !m.name.startsWith("yolov6"));
  fillSelect("#distillTeacherModel", [{ name: "", desc: "" }, ...tms],
    m => m.name, m => m.name + (m.desc ? " — " + m.desc : ""));
}

/* ---------------- 推理模式适配 ---------------- */

function modelOptionLabel(m) {
  const parts = [m.name];
  const isCustom = String(m.type || "").startsWith("custom");
  if (m.need_weight) parts.push(t("tag.needWeight"));
  else if (m.pretrained_local) parts.push(t("tag.pretrainedLocal"));
  else if (m.pretrained) parts.push(t("tag.pretrainedOnline"));
  else if (isCustom) parts.push(t("tag.custom"));
  if (m.desc && (!m.need_weight || isCustom)) parts.push(`— ${m.desc}`);
  return parts.join(" ");
}

function currentInferModelMeta() {
  const task = state.currentTask;
  if (!task) return null;
  const name = $("#inferModel").value;
  return (task.models || []).find(m => m.name === name) || { name };
}

function updateInferHints() {
  const task = state.currentTask;
  const meta = currentInferModelMeta();
  if (!task || !meta) return;
  const hint = $("#inferModeHint");
  const box = $("#inferPretrainedBox");
  const weightSelect = $("#inferWeightSelect");
  if (!hint || !box || !weightSelect) return;  // 版本混搭防御
  // 模型切换时同步官方默认阈值：CLTR 使用 0.35，其余恢复 0.5
  const modelName = meta.name || "";
  if (state._lastInferModel !== modelName) {
    state._lastInferModel = modelName;
    const thrEl = $("#confThreshold");
    if (thrEl) {
      if (modelName === "cltr") thrEl.value = "0.35";
    if (modelName === "raz") thrEl.value = "0.3";
      else if (String(thrEl.value) === "0.35") thrEl.value = "0.5";
    }
  }
  const weightSel = weightSelect.value;
  const selOpt = weightSelect.selectedOptions && weightSelect.selectedOptions[0];
  const weightLabel = selOpt ? selOpt.textContent : "";
  const hasWeight = !!weightSel;

  // ONNX 模型：走 onnxruntime 推理（模型/设备/预训练选项不参与），
  // 异步探测类型后展示识别结论
  if (/\.onnx$/i.test(weightSel || "")) {
    box.style.display = "none";
    $("#inferPretrained").checked = false;
    hint.className = "infer-mode-hint";
    hint.textContent = t("hint.onnxProbing");
    refreshOnnxInferHint(weightSel, hint);
    return;
  }

  // 预训练开关适配
  if (hasWeight) {
    box.style.display = "none";
    $("#inferPretrained").checked = false;
    hint.className = "infer-mode-hint ok";
    hint.textContent = tf("hint.withWeight", weightLabel, meta.name);
    return;
  }
  // 需权重：显式标注 need_weight，或模型无预训练能力（pretrained === false）
  if (meta.need_weight || meta.pretrained === false) {
    box.style.display = "none";
    $("#inferPretrained").checked = false;
    hint.className = "infer-mode-hint warn";
    hint.textContent = tf("hint.needWeight", meta.name);
    return;
  }
  // 有预训练能力
  box.style.display = "";
  $("#inferPretrained").checked = true;
  if (meta.pretrained_local) {
    hint.className = "infer-mode-hint ok";
    hint.textContent = taskKindHint(task.key, true);
  } else {
    hint.className = "infer-mode-hint";
    hint.textContent = taskKindHint(task.key, false);
  }
}

function taskKindHint(taskKey, local) {
  switch (taskKey) {
    case "crowd_counting":
    case "crowd_density":
      return t("hint.kind.crowd_density");
    case "crowd_point":
      return t("hint.kind.crowd_point");
    case "crowd_box":
      return t("hint.kind.crowd_box");
    default:
      return "";
  }
}

/* ONNX 权重选中 → 调识别端点展示模型类型 / 输入尺寸（与部署面板同款体验）；
   选择已切换时丢弃过期响应，防止覆盖新提示 */
async function refreshOnnxInferHint(wp, hint) {
  const stillCurrent = () => {
    const sel = $("#inferWeightSelect");
    return sel && sel.value === wp;
  };
  try {
    const fmt = await api("/api/crowd/onnx/infer_format", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ onnx_path: wp }),
    });
    if (!stillCurrent()) return;
    const routeTag = { crowd_density: "deploy.tagDensity",
                       crowd_point: "deploy.tagPoint",
                       crowd_box: "deploy.tagBox" }[fmt.route] || "";
    hint.className = "infer-mode-hint ok";
    hint.innerHTML =
      `<b>${t("hint.onnxReady")}</b>${t(routeTag)} ${esc(fmt.model_label)}` +
      ` · ${t("deploy.defaultSize")} ${fmt.imgsz}` +
      (fmt.multiple ? `（${t("deploy.multiple")} ${fmt.multiple}）` : "") +
      (fmt.dynamic_batch ? ` · ${t("deploy.dynamicOn")}` : "") +
      `<br>${t("deploy.source")}: ${esc(t("deploy.source." + (fmt.source || "")) || fmt.source || "")}` +
      ` · ${esc(t("hint.onnxIgnore"))}`;
  } catch (e) {
    if (!stillCurrent()) return;
    hint.className = "infer-mode-hint warn";
    hint.textContent = tf("msg.onnxFormatFailed", e.message);
  }
}

function lossName(l) {
  const m = {
    mse: t("loss.mse"), mae: t("loss.mae"),
    ssim: t("loss.ssim"), mse_ssim: t("loss.mse_ssim"),
    // 点预测 / 头部框：算法内置损失（点预测默认项，可叠加自定义附加损失）
    p2pnet: t("loss.p2pnet"), p2r: t("loss.p2r"),
    cp: t("loss.cp"), cu: t("loss.cu"), pet: t("loss.pet"),
    cltr: t("loss.cltr"), yolo: t("loss.yolo"),
    raz: t("loss.raz"), d2c: t("loss.d2c"),
    // 单域泛化
    dccus: t("loss.dccus"), llb: t("loss.llb"), mpcount: t("loss.mpcount"),
  };
  return m[l] || l;
}

/* ---------------- 损失分项权重（默认值 = 各算法论文/源码设置） ---------------- */
const LOSS_WEIGHT_SPECS = {
  density_mse_ssim: [["mse_w", 0.5], ["ssim_w", 0.5]],
  density_dmcount: [["ot_w", 0.1], ["tv_w", 0.01], ["count_w", 1.0]],
  p2pnet: [["ce_w", 1.0], ["points_w", 0.0002],
           ["cost_class", 1.0], ["cost_point", 0.05]],
  pet: [["ce_w", 1.0], ["points_w", 5.0],
        ["cost_class", 1.0], ["cost_point", 0.05]],
  cltr: [["ce_w", 2.0], ["points_w", 5.0], ["giou_w", 2.0],
         ["cost_class", 2.0], ["cost_point", 5.0]],
  p2r: [["factor", 1.0]],
  sdg_dccus: [["den_w", 1.0], ["mask_w", 100.0],
              ["sim_w", 0.001], ["orth_w", 1.0]],
  sdg_llb: [["den_w", 1.0], ["local_w", 10.0], ["bound_w", 10.0],
            ["sim_w", 1.0], ["orth_w", 1.0]],
  sdg_mpcount: [["den_w", 1.0], ["cls_w", 10.0], ["con_w", 10.0]],
  yolo: [["box_w", 7.5], ["cls_w", 0.5], ["dfl_w", 1.5]],
};

function lossWeightSpec(taskKey, model, loss) {
  if (taskKey === "crowd_sdg") {
    if (model === "sdg_mpcount") return "sdg_mpcount";
    if (model === "sdg_llb") return "sdg_llb";
    return "sdg_dccus";
  }
  if (taskKey === "crowd_box") return "yolo";
  if (taskKey === "crowd_density") {
    if (loss === "mse_ssim") return "density_mse_ssim";
    if (loss === "dmcount") return "density_dmcount";
    return null;
  }
  if (model === "p2r") return "p2r";
  if (model === "pet") return "pet";
  if (model === "cltr") return "cltr";
  if (model === "raz" || model === "d2c") return null;
  return "p2pnet";   // p2pnet / cp / cu / 自定义点模型（P2PNet 匹配损失）
}

function renderLossWeights(taskKey, model, loss) {
  const box = $("#lossWeights");
  if (!box) return;
  const specKey = lossWeightSpec(taskKey, model, loss);
  const spec = (specKey && LOSS_WEIGHT_SPECS[specKey]) || [];
  const items = spec.slice();
  if (String(loss || "").startsWith("custom:") &&
      (taskKey === "crowd_point" || taskKey === "crowd_sdg")) {
    items.push(["aux_w", 1.0]);
  }
  if (!items.length) {
    box.innerHTML = `<span class="hint">${esc(t("train.lossWeightsNone"))}</span>`;
    return;
  }
  box.innerHTML = items.map(([k, d]) => `
    <label class="lw-item" title="${esc(t("lw." + k))}">
      <span>${esc(t("lw." + k))}</span>
      <input type="number" step="0.0001" min="0" data-lw="${esc(k)}" value="${d}">
    </label>`).join("");
}

function collectLossWeights() {
  const out = {};
  document.querySelectorAll("#lossWeights input[data-lw]").forEach(el => {
    const v = parseFloat(el.value);
    if (Number.isFinite(v)) out[el.dataset.lw] = v;
  });
  return out;
}

function schedulerName(s) {
  const m = {
    cosine: t("sched.cosine"), step: t("sched.step"),
    linear: t("sched.linear"),
    multistep: t("sched.multistep"), poly: t("sched.poly"),
    exponential: t("sched.exponential"),
    warmup_cosine: t("sched.warmup_cosine"),
    warm_restarts: t("sched.warm_restarts"),
    plateau: t("sched.plateau"), none: t("sched.none"),
  };
  return m[s] || s;
}

/* ---------------- 训练 ---------------- */

function bindTrainControls() {
  $("#btnUploadDataset").addEventListener("click", uploadDataset);
  $("#datasetSelect").addEventListener("change", (e) => {
    if (e.target.value) $("#datasetPath").value = "";
    showDatasetStats(e.target.value);
  });
  $("#datasetPath").addEventListener("input", (e) => {
    if (e.target.value) { $("#datasetSelect").value = ""; $("#datasetStats").textContent = ""; }
  });
$("#btnStartTrain").addEventListener("click", startTrain);
$("#btnStopTrain").addEventListener("click", stopTrain);
// 切优化器时联动填默认动量/权重衰减
bindOptimizerDefaults($("#optimizer"), $("#momentum"), $("#weightDecay"));
// 初始按当前优化器预填一次
applyOptimizerDefaults($("#optimizer"), $("#momentum"), $("#weightDecay"));
// 切换损失时刷新分项权重输入（默认值随算法/损失变化）
if ($("#lossSelect")) {
  $("#lossSelect").addEventListener("change", () => {
    renderLossWeights(_lastTrainTask, $("#trainModel").value,
                      $("#lossSelect").value);
  });
}
// 调度器切换时联动显示专属参数
  $("#scheduler").addEventListener("change", e => {
    const v = e.target.value;
    const show2 = (sel, on) => {
      const el = $(sel);
      if (el) el.style.display = on ? "" : "none";
    };
    show2("#fieldWarmupEpochs", v === "warmup_cosine");
    show2("#fieldT0Epochs", v === "warm_restarts");
    show2("#fieldPolyPower", v === "poly");
  });
  $("#trainModel").addEventListener("change", updateTrainModelUI);
  // 训练初始权重上传 + 断点续训 + 运行目录恢复
  $("#btnUploadTrainWeight").addEventListener("click",
    () => $("#trainWeightFile").click());
  $("#trainWeightFile").addEventListener("change", uploadTrainWeight);
  $("#btnUploadTrainRun").addEventListener("click", uploadTrainRun);
  // 单域泛化：仅测试（跨域泛化性能评估）
  if ($("#btnSdgTest"))
    $("#btnSdgTest").addEventListener("click", runSdgTest);
  refreshTaskHistory();
}

/* ---------------- 单域泛化 · 仅测试（跨域 MAE/MSE） ---------------- */

async function runSdgTest() {
  const weight = $("#sdgTestWeight") ? $("#sdgTestWeight").value : "";
  const dataset = $("#sdgTestDataset") ? $("#sdgTestDataset").value : "";
  const st = $("#sdgTestStatus"), log = $("#sdgTestLog"),
    out = $("#sdgTestResult"), btn = $("#btnSdgTest");
  if (!weight) return toast(t("sdg.msg.needWeight"), "error");
  if (!dataset) return toast(t("sdg.msg.needDataset"), "error");
  btn.disabled = true;
  btn.textContent = t("msg.starting");
  if (st) st.textContent = t("msg.starting");
  if (out) out.innerHTML = "";
  try {
    const body = {
      weight_path: weight,
      dataset_path: dataset,
      model: $("#trainModel") ? $("#trainModel").value : "sdg_dccus",
      sdg_backbone: ($("#sdgBackbone") && $("#sdgBackbone").value) || null,
      sdg_test_patch: +(($("#sdgTestPatchOnly") &&
                         $("#sdgTestPatchOnly").value) || 0),
      device: ($("#trainDevice") && $("#trainDevice").value) || "cpu",
    };
    const r = await api("/api/crowd/sdg/test", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const timer = setInterval(async () => {
      try {
        const task = await api(`/api/crowd/train/tasks/${r.task_id}`);
        if (st) st.textContent =
          `${statusLabel(task.status, task.kind)} ${Math.round(task.progress || 0)}%`;
        try {
          const lr = await api(`/api/crowd/train/tasks/${r.task_id}/log?tail=60`);
          if (log && lr) log.textContent = lr.log || "";
        } catch (_) { /* ignore */ }
        if (["completed", "failed", "stopped"].includes(task.status)) {
          clearInterval(timer);
          btn.disabled = false;
          btn.textContent = t("sdg.testRun");
          if (task.status === "completed" && task.result) {
            const res = task.result;
            if (out) {
              out.innerHTML =
                `<div class="stat"><b>${res.mae}</b><span>MAE</span></div>` +
                `<div class="stat"><b>${res.rmse !== undefined ? res.rmse : res.mse}</b><span>RMSE</span></div>` +
                `<div class="stat"><b>${res.num_images}</b><span>样本数</span></div>` +
                `<div class="stat"><b>${esc(res.model || "")}</b><span>模型</span></div>`;
            }
            toast(tf("sdg.msg.done", res.mae,
                     res.rmse !== undefined ? res.rmse : res.mse),
                  "success", 8000);
            await loadUploads();
          } else if (task.status !== "completed") {
            toast(`${t("msg.trainFailed")}: ${task.error || ""}`, "error", 8000);
          }
        }
      } catch (_) { /* retry next tick */ }
    }, 1500);
  } catch (e) {
    toast(`测试启动失败：${e.message}`, "error", 6000);
    btn.disabled = false;
    btn.textContent = t("sdg.testRun");
  }
}

/* 根据当前选择的训练模型适配 UI：
   - 全部点预测模型（P2PNet / P2R / Consistent-Point / CU / PET / CLTR）：
     优化器、学习率、调度器、损失函数均可编辑，默认值对齐各算法论文配置
     （Adam(AdamW) + 论文学习率 + 论文损失；半监督三件套 LR 不衰减）
   - 损失：默认对应论文损失，可追加注册到 crowd_point 的自定义附加损失；
     PET 的损失在模型内部计算（点查询 + 四叉树），不可替换
   - YOLO（头部框路线）：优化器/调度器/损失由 ultralytics 引擎托管，
     学习率（lr0）与权重衰减生效，显示 YOLO 专属增强参数 */
function updateTrainModelUI() {
  const task = state.currentTask;
  if (!task) return;
  const model = $("#trainModel").value || "";
  // 自定义点模型（非 p2r/cp/cu/pet/cltr）走 P2PNet 基线
  const isCustomPoint = task.key === "crowd_point" &&
    !["p2pnet", "p2r", "cp", "cu", "pet", "cltr", "raz", "d2c"].includes(model);
  const isP2P = model === "p2pnet" || isCustomPoint;
  const isP2R = model === "p2r";
  const isCP = model === "cp";
  const isCU = model === "cu";
  const isPET = model === "pet";
  const isCLTR = model === "cltr";
  const isRAZ = model === "raz";
  const isD2C = model === "d2c";
  const razStageField = $("#fieldRazStage");
  if (razStageField) razStageField.style.display = isRAZ ? "" : "none";
  const isSemi = isP2R || isCP || isCU;  // 半监督点预测：显示标注比例
  const isYolo = task.key === "crowd_box";
  // 仅模型/任务变化时重置算法相关默认值（语言切换等重复渲染保留用户选择）
  const changed = (_lastTrainModel !== model || _lastTrainTask !== task.key);

  // ---- 优化器 / 学习率 / 调度器：点预测全部可编辑，默认对齐论文配置 ----
  // （P2PNet Adam 1e-4 / P2R Adam 5e-5 / CP、CU、CLTR Adam 1e-4 /
  //   PET AdamW 1e-4；半监督三件套论文 LR 在训练周期内不衰减 → none）
  const MODEL_PAPER = {
    p2pnet: { optimizer: "adam", lr: 0.0001, scheduler: "cosine" },
    raz:    { optimizer: "adamw", lr: 0.0001, scheduler: "cosine" },
    d2c:    { optimizer: "adam", lr: 0.00001, scheduler: "cosine" },
    p2r:    { optimizer: "adam", lr: 0.00005, scheduler: "cosine" },
    cp:     { optimizer: "adam", lr: 0.0001, scheduler: "cosine" },
    cu:     { optimizer: "adam", lr: 0.0001, scheduler: "cosine" },
    pet:    { optimizer: "adamw", lr: 0.0001, scheduler: "cosine" },
    cltr:   { optimizer: "adam", lr: 0.0001, scheduler: "cosine" },
  };
  if (task.key === "crowd_point" && changed) {
    const paper = MODEL_PAPER[model] ||
      { optimizer: "adam", lr: 0.0001, scheduler: "cosine" };
    const optSel = $("#optimizer");
    if (optSel &&
        [...optSel.options].some(o => o.value === paper.optimizer)) {
      // 仅切换优化器选项；动量/权重衰减保留用户当前值（不套用推荐默认，
      // 避免覆盖各算法论文的 weight_decay 设定）
      optSel.value = paper.optimizer;
    }
    if ($("#lr")) $("#lr").value = paper.lr;
    const schSel = $("#scheduler");
    if (schSel &&
        [...schSel.options].some(o => o.value === paper.scheduler)) {
      schSel.value = paper.scheduler;
    }
  }

  // ---- 损失函数：显示当前算法实际使用的损失（论文默认），并选定默认值 ----
  // 点预测：默认 = 该算法论文损失；可追加注册到 crowd_point 的自定义附加损失
  const lossSel = $("#lossSelect");
  if (lossSel) {
    const pointLosses = ((state.lossesByTask &&
      state.lossesByTask.crowd_point) || ["p2pnet"]);
    const builtinPointLosses = ["p2pnet", "raz", "d2c"];
    const customs = pointLosses.filter(
      l => !builtinPointLosses.includes(String(l)));
    if (task.key === "crowd_point") {
      if (model === "raz" || model === "d2c") {
        const builtin = model === "raz" ? "raz" : "d2c";
        const opts = [builtin, ...customs]
          .filter((v, i, a) => a.indexOf(v) === i);
        const cur = lossSel.value;
        fillSelect("#lossSelect", opts, l => l, l => lossName(l));
        lossSel.value = (!changed && opts.map(String).includes(cur))
          ? cur : builtin;
      } else if (model === "pet") {
        // PET：模型内点查询损失 + 四叉树损失（模型内部计算，不可替换）
        fillSelect("#lossSelect", ["pet"], l => l, l => lossName(l));
        lossSel.value = "pet";
      } else if (["cltr", "p2r", "cp", "cu"].includes(model)) {
        // 半监督/点定位：默认论文损失 + 自定义附加损失
        const opts = [model, ...customs]
          .filter((v, i, a) => a.indexOf(v) === i);
        const cur = lossSel.value;
        fillSelect("#lossSelect", opts, l => l, l => lossName(l));
        lossSel.value = (!changed && opts.map(String).includes(cur))
          ? cur : model;
      } else {
        // P2PNet / 自定义点模型：内置匹配损失 + 注册到 crowd_point 的
        // 自定义附加损失（默认 p2pnet）
        const opts = ["p2pnet", ...customs]
          .filter((v, i, a) => a.indexOf(v) === i);
        const cur = lossSel.value;
        fillSelect("#lossSelect", opts, l => l, l => lossName(l));
        lossSel.value = (!changed && opts.includes(cur)) ? cur : "p2pnet";
      }
    } else if (task.key === "crowd_sdg") {
      // 单域泛化：默认 = 算法内置损失（论文设置）；可追加注册到
      // crowd_sdg 的自定义附加损失（与内置损失相加）
      const builtin = model === "sdg_llb" ? "llb"
        : (model === "sdg_mpcount" ? "mpcount" : "dccus");
      const customs = ((state.lossesByTask &&
        state.lossesByTask.crowd_sdg) || [])
        .filter(l => !["dccus", "llb", "mpcount"].includes(String(l)));
      const opts = [builtin, ...customs];
      const cur = lossSel.value;
      fillSelect("#lossSelect", opts, l => l, l => lossName(l));
      lossSel.value = (!changed && opts.map(String).includes(cur))
        ? cur : builtin;
      // 算法默认超参（模型变化时重置：MPCount AdamW 1e-3 / 其余 Adam 1e-5）
      if (changed) {
        const isMp = model === "sdg_mpcount";
        const optSel = $("#optimizer");
        const optName = isMp ? "adamw" : "adam";
        if (optSel && [...optSel.options].some(o => o.value === optName))
          optSel.value = optName;
        if ($("#lr")) $("#lr").value = isMp ? 0.001 : 0.00001;
        const schSel = $("#scheduler");
        const schName = "cosine";
        if (schSel && [...schSel.options].some(o => o.value === schName))
          schSel.value = schName;
        if ($("#sdgBackbone") && !isMp) $("#sdgBackbone").value = "convNeXt_tiny";
      }
    } else if (task.key === "crowd_box") {
      // YOLO 检测损失由引擎内置；保留可选项入口，支持后续注册的自定义损失
      const opts = (state.lossesByTask && state.lossesByTask.crowd_box) ||
        ["yolo"];
      const cur = lossSel.value;
      fillSelect("#lossSelect", opts, l => l, l => lossName(l));
      lossSel.value = opts.map(String).includes(cur) ? cur : "yolo";
    } else {
      const opts = (state.lossesByTask && state.lossesByTask[task.key]) ||
        (state.lossesByTask && state.lossesByTask.crowd_density) || ["mse"];
      const cur = lossSel.value;
      fillSelect("#lossSelect", opts, l => l, l => lossName(l));
      lossSel.value = opts.map(String).includes(cur) ? cur : "mse";
    }
  }

  // ---- 输入/裁剪分辨率：按算法给可选值与默认值 ----
  const resKey = resolutionKey(task.key, model);
  const resOpts = RESOLUTION_PRESETS[resKey] || RESOLUTION_PRESETS.density;
  const resSel = $("#imgSize");
  if (resSel) {
    const cur = resSel.value;
    fillSelect("#imgSize", resOpts, v => String(v),
      v => resolutionLabel(v));
    const valid = resOpts.map(String).includes(String(cur));
    resSel.value = (!changed && valid)
      ? String(cur) : String(RESOLUTION_DEFAULTS[resKey]);
  }
  renderLossWeights(task.key, model, lossSel ? lossSel.value : "");
  _lastTrainModel = model;
  _lastTrainTask = task.key;

  const setCtrl = (sel, disabled, hintKey) => {
    const el = $(sel);
    if (el) {
      el.disabled = disabled;
      el.title = disabled ? t(hintKey) : "";
    }
  };
  // 优化器 / 动量 / 调度器 / 学习率：全部路线均可编辑
  // （YOLO 由 ultralytics 引擎训练，参数会映射为 optimizer / momentum /
  //   cos_lr / lr0 传入；其余路线走框架自建优化器与调度器）
  for (const sel of ["#optimizer", "#momentum", "#scheduler"]) {
    setCtrl(sel, false, "");
  }
  setCtrl("#lr", false, "");
  // 损失：全部路线均可选择（默认沿用原论文/引擎设置）
  //  - 密度图/点预测：内置默认 + 已注册自定义附加损失
  //  - 单域泛化：算法内置默认 + 注册到 crowd_sdg 的自定义附加损失
  //  - YOLO：引擎内置 box/cls/dfl（含注册项入口）
  //  - PET：模型内部计算（仅内置项，可查看但不可替换）
  setCtrl("#lossSelect", false, "");
  const _lossEl = $("#lossSelect");
  if (_lossEl) {
    _lossEl.title = isYolo ? t("hint.yoloLoss")
      : (isPET ? t("hint.petManaged")
      : (task.key === "crowd_sdg" ? t("hint.sdgLoss") : ""));
  }
  // 单域泛化：骨干/伪域/迭代/切块参数与目标数据集（仅该路线显示）
  const isSdg = task.key === "crowd_sdg";
  for (const sel of [["#fieldTargetDataset", isSdg],
                     ["#fieldSdgParams", isSdg],
                     ["#sdgTestCard", isSdg],
                     ["#fieldSdgBackbone", isSdg &&
                        model !== "sdg_mpcount"],
                     ["#fieldSdgClusters", isSdg &&
                        model !== "sdg_mpcount"],
                     ["#fieldSdgIters", isSdg &&
                        model !== "sdg_mpcount"]]) {
    const el = $(sel[0]);
    if (el) el.style.display = sel[1] ? "" : "none";
  }
  // 半监督点预测：显示「有标注样本比例」与各自说明
  const p2rField = $("#fieldP2rLabelRatio");
  if (p2rField) p2rField.style.display = isSemi ? "" : "none";
  const p2rHint = $("#p2rHint");
  if (p2rHint) p2rHint.style.display = isP2R ? "" : "none";
  const cpHint = $("#cpHint");
  if (cpHint) cpHint.style.display = isCP ? "" : "none";
  const cuHint = $("#cuHint");
  if (cuHint) cuHint.style.display = isCU ? "" : "none";
  // 点预测模型 / YOLO 暂不支持框架级多 GPU 并行（YOLO 由 ultralytics 引擎原生调度）
  const mgBox = document.getElementById("fieldMultiGpu");
  if (mgBox) {
    mgBox.style.display = ((isP2P || isSemi || isYolo || isPET || isCLTR || isRAZ || isD2C) ||
      !state.gpuCount || state.gpuCount < 2) ? "none" : "";
  }
}

/* ---------------- .mat -> GT 密度图工具 ---------------- */

let matTaskId = null;
let matPollTimer = null;

async function startMat2Density() {
  const matDir = $("#matDir").value.trim();
  const saveDir = $("#matSaveDir").value.trim();
  if (!matDir) return toast(t("msg.matDirRequired"), "error");
  if (!saveDir) return toast(t("msg.matSaveRequired"), "error");
  const qs = new URLSearchParams({
    mat_dir: matDir, save_dir: saveDir,
    img_dir: $("#matImgDir").value.trim(),
    sigma: $("#matSigma").value, adaptive: $("#matAdaptive").checked,
  });
  $("#btnMat2Density").disabled = true;
  $("#btnMat2Density").textContent = t("msg.starting");
  try {
    const r = await api(`/api/crowd/mat2density?${qs}`, { method: "POST" });
    matTaskId = r.task_id;
    toast(t("msg.matStarted"), "success");
    $("#matToolStatus").textContent = t("msg.matRunning");
    if (matPollTimer) clearInterval(matPollTimer);
    matPollTimer = setInterval(pollMatTask, 1500);
  } catch (e) {
    toast(tf("msg.startFailed", e.message), "error", 6000);
  } finally {
    $("#btnMat2Density").disabled = false;
    $("#btnMat2Density").textContent = t("mat.convert");
  }
}

async function pollMatTask() {
  if (!matTaskId) { clearInterval(matPollTimer); matPollTimer = null; return; }
  let task;
  try { task = await api(`/api/crowd/train/tasks/${matTaskId}`); }
  catch (_) { return; }
      $("#matToolStatus").textContent =
      tf("msg.matStatus", statusLabel(task.status, task.kind), task.epoch,
        task.total_epochs, Math.round(task.progress)) +
    (task.error ? ` | ${task.error}` : "");
  if (["completed", "failed", "stopped"].includes(task.status)) {
    clearInterval(matPollTimer); matPollTimer = null; matTaskId = null;
    if (task.status === "completed" && task.result) {
      const r = task.result;
      toast(tf("msg.matDone", r.converted, r.skipped, r.failed, r.save_dir),
            "success", 6000);
      $("#matToolStatus").textContent =
        tf("msg.matDone", r.converted, r.skipped, r.failed, r.save_dir);
    }
  }
}

/* ---------------- .npy 密度图 → PNG 可视化工具 ---------------- */

let npyVisTaskId = null;
let npyVisPollTimer = null;

async function startNpyVis() {
  const dir = $("#npyVisDir").value.trim();
  if (!dir) return toast(t("msg.npyDirRequired"), "error");
  const mode = $("#npyVisMode").value;
  if (mode === "overlay" && !$("#npyVisImgDir").value.trim()) {
    return toast(t("npyvis.msg.imgDirRequired"), "error");
  }
  const qs = new URLSearchParams({
    npy_dir: dir,
    save_dir: $("#npyVisSaveDir").value.trim(),
    mode: mode,
    annotate: $("#npyVisAnnotate").checked,
    img_dir: $("#npyVisImgDir").value.trim(),
    alpha: $("#npyVisAlpha").value,
  });
  $("#btnNpyVis").disabled = true;
  $("#btnNpyVis").textContent = t("msg.starting");
  try {
    const r = await api(`/api/crowd/npy2png?${qs}`, { method: "POST" });
    npyVisTaskId = r.task_id;
    toast(t("msg.npyVisStarted"), "success");
    $("#npyVisStatus").textContent = t("msg.npyVisRunning");
    if (npyVisPollTimer) clearInterval(npyVisPollTimer);
    npyVisPollTimer = setInterval(pollNpyVisTask, 1500);
  } catch (e) {
    toast(tf("msg.startFailed", e.message), "error", 6000);
  } finally {
    $("#btnNpyVis").disabled = false;
    $("#btnNpyVis").textContent = t("npyvis.btn");
  }
}

async function pollNpyVisTask() {
  if (!npyVisTaskId) { clearInterval(npyVisPollTimer); npyVisPollTimer = null; return; }
  let task;
  try { task = await api(`/api/crowd/train/tasks/${npyVisTaskId}`); }
  catch (_) { return; }
      $("#npyVisStatus").textContent =
      tf("msg.matStatus", statusLabel(task.status, task.kind), task.epoch,
        task.total_epochs, Math.round(task.progress)) +
    (task.error ? ` | ${task.error}` : "");
  if (["completed", "failed", "stopped"].includes(task.status)) {
    clearInterval(npyVisPollTimer); npyVisPollTimer = null; npyVisTaskId = null;
    if (task.status === "completed" && task.result) {
      const r = task.result;
      toast(tf("msg.npyVisDone", r.visualized, r.skipped, r.failed, r.save_dir),
            "success", 6000);
      $("#npyVisStatus").textContent =
        tf("msg.npyVisDone", r.visualized, r.skipped, r.failed, r.save_dir);
    }
  }
}

async function uploadDataset() {
  const f = $("#datasetFile").files[0];
  if (!f) return toast(t("msg.pickZip"), "error");
  const fd = new FormData();
  fd.append("file", f);
  $("#btnUploadDataset").disabled = true;
  $("#btnUploadDataset").textContent = tf("msg.uploading", 0);
  try {
    const r = await uploadWithProgress("/api/upload/dataset", fd, (p) => {
      $("#btnUploadDataset").textContent = tf("msg.uploading", p);
    });
    toast(tf("msg.dsUploaded", r.num_images, r.num_classes), "success");
    await loadUploads();
    $("#datasetSelect").value = r.name;
    updateDatasetDeleteBtns();
    showDatasetStats(r.name);
    $("#datasetPath").value = "";
  } catch (e) {
    toast(tf("msg.uploadFailed", e.message), "error");
  } finally {
    $("#btnUploadDataset").disabled = false;
    $("#btnUploadDataset").textContent = t("train.uploadDatasetBtn");
  }
}

function showDatasetStats(name) {
  const ds = state.uploads.datasets.find(d => d.name === name);
  if (!ds) { $("#datasetStats").textContent = ""; return; }
  $("#datasetStats").textContent =
    tf("msg.dsSelected", ds.name, ds.num_images, ds.num_classes) +
    (ds.num_gt_txt ? tf("msg.gtTxt", ds.num_gt_txt) : "") +
    (ds.num_gt_npy ? tf("msg.gtNpy", ds.num_gt_npy) : "");
}

function gatherTrainParams() {
  const task = state.currentTask;
  const dataset = $("#datasetSelect").value || $("#datasetPath").value.trim();
  if (!dataset) throw new Error(t("msg.dsRequired"));
  // 数值校验（lr 下限 1e-8：后端要求 > 0，填 0 会被 422 拒绝）
  numInRange($("#epochs"), 1, 10000, "Epochs");
  numInRange($("#batchSize"), 1, 1024, "Batch Size");
  numInRange($("#lr"), 0.00000001, 10, t("train.lr"));
  numInRange($("#weightDecay"), 0, 1, t("train.weightDecay"));
  numInRange($("#momentum"), 0, 0.999, t("train.momentum"));
  // 知识蒸馏（仅头部框路线）：需要教师权重或教师模型二选一
  const isBox = task.key === "crowd_box";
  const distillOn = isBox && $("#yoloDistill") && $("#yoloDistill").checked;
  const distillTeacherW = (distillOn && $("#distillTeacherWeight")
    && $("#distillTeacherWeight").value) || "";
  const distillTeacherM = (distillOn && $("#distillTeacherModel")
    && $("#distillTeacherModel").value) || "";
  if (distillOn) {
    numInRange($("#distillLambda"), 0, 100, t("train.distillLambda"));
    if (!distillTeacherW && !distillTeacherM)
      throw new Error(t("msg.distillNeedTeacher"));
  }
  return {
    task: task.key,
    model: $("#trainModel").value,
    dataset_path: dataset,
    epochs: +$("#epochs").value,
    batch_size: +$("#batchSize").value,
    lr: +$("#lr").value,
    weight_decay: +$("#weightDecay").value,
    optimizer: $("#optimizer").value,
    momentum: numOr($("#momentum"), 0.9),
    scheduler: $("#scheduler").value,
    loss: $("#lossSelect").value,
    loss_weights: collectLossWeights(),
    img_size: +$("#imgSize").value,
    num_workers: +$("#numWorkers").value,
    pretrained: $("#trainPretrained").checked,
    weight_path: $("#trainInitWeight").value || null,
    resume_weight: ($("#resumeFromWeight") && $("#resumeFromWeight").checked &&
                    $("#trainInitWeight").value) || null,
    device: $("#trainDevice").value,
    multi_gpu_mode: (document.getElementById("fieldMultiGpu") &&
                     document.getElementById("fieldMultiGpu").style.display !== "none")
      ? ($("#multiGpuMode").value || null) : null,
    run_name: $("#runName").value.trim() || null,
    early_stopping: +$("#earlyStopping").value,
    eval_every: +(($("#evalEvery") && $("#evalEvery").value) || 1),
    val_ratio: +(($("#valRatio") && $("#valRatio").value) || 0.2),
    seed: +$("#seed").value,
    augment: $("#augment").checked,
    augment_strength: numOr($("#augmentStrength"), 1.0),
    warmup_epochs: +(($("#warmupEpochs") && $("#warmupEpochs").value) || 5),
    t0_epochs: +(($("#t0Epochs") && $("#t0Epochs").value) || 10),
    poly_power: +(($("#polyPower") && $("#polyPower").value) || 0.9),
    label_ratio: numOr($("#p2rLabelRatio"), 1.0) || 1.0,
    annotation_format: ($("#annotationFormat") &&
                        $("#annotationFormat").value) || "yolo",
    yolo_mosaic: +$("#yoloMosaic").value,
    yolo_scale: +$("#yoloScale").value,
    yolo_mixup: +$("#yoloMixup").value,
    yolo_copy_paste: +$("#yoloCopyPaste").value,
    // 单域泛化（仅 crowd_sdg 生效）
    target_dataset_path: ($("#targetDatasetSelect") &&
                          $("#targetDatasetSelect").value) || null,
    sdg_backbone: ($("#sdgBackbone") && $("#sdgBackbone").value) || null,
    sdg_num_clusters: +(($("#sdgNumClusters") &&
                         $("#sdgNumClusters").value) || 4),
    sdg_iters: +(($("#sdgIters") && $("#sdgIters").value) || 20),
    sdg_test_patch: +(($("#sdgTestPatch") &&
                       $("#sdgTestPatch").value) || 0),
    raz_stage: ($("#razStage") && $("#razStage").value) || "all",
    distill: distillOn,
    distill_teacher_weight: distillTeacherW || null,
    distill_teacher_model: distillTeacherM || null,
    distill_lambda: numOr($("#distillLambda"), 6.0),
  };
}

async function startTrain() {
  // 断点续训模式：必须选择/上传中断的 last.pth
  if ($("#resumeFromWeight") && $("#resumeFromWeight").checked &&
      !$("#trainInitWeight").value) {
    return toast(t("msg.resumeNeedWeight"), "error", 6000);
  }
  let params;
  try { params = gatherTrainParams(); }
  catch (e) { return toast(e.message, "error"); }

  $("#btnStartTrain").disabled = true;
  $("#btnStartTrain").textContent = t("msg.starting");
  try {
    const r = await api("/api/crowd/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    state.trainTaskId = r.task_id;
    toast(tf("msg.trainStarted", r.run_name), "success");
    $("#btnStopTrain").disabled = false;
    $("#trainLog").textContent = "";
    refreshTaskHistory();
    startPolling();
  } catch (e) {
    toast(tf("msg.startFailed", e.message), "error", 6000);
  } finally {
    $("#btnStartTrain").disabled = false;
    $("#btnStartTrain").textContent = t("train.start");
  }
}

function startPolling() {
  stopPollingTimer();
  state.pollTimer = setInterval(pollTrainStatus, 1500);
  pollTrainStatus();
}

function stopPollingTimer() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || !isFinite(Number(seconds))) {
    return "—";
  }
  let total = Math.max(0, Math.round(Number(seconds)));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

async function pollTrainStatus() {
  const id = state.trainTaskId;
  if (!id) return stopPollingTimer();
  let task;
  try { task = await api(`/api/crowd/train/tasks/${id}`); }
  catch (_) { return; }

  const queueHint = (task.status === "queued" && task.queue_position)
    ? ` · ${esc(tf("status.queuePos", task.queue_position))}` : "";
  const timing = [];
  if (task.started_at && task.elapsed_seconds !== null &&
      task.elapsed_seconds !== undefined) {
    timing.push(esc(tf("train.elapsed", formatDuration(task.elapsed_seconds))));
  }
  if (task.status === "running" && task.eta_seconds !== null &&
      task.eta_seconds !== undefined) {
    timing.push(esc(tf("train.eta", formatDuration(task.eta_seconds))));
  }
  const timingHint = timing.length ? ` · ${timing.join(" · ")}` : "";
  $("#trainStatus").innerHTML =
    `${esc(t("label.status"))}<b>${statusLabel(task.status, task.kind)}</b>${queueHint} · Epoch ${task.epoch}/${task.total_epochs}${timingHint}` +
    (task.error ? ` · <span style="color:var(--red)">${esc(task.error)}</span>` : "");
  $("#trainProgress").style.width = (task.progress || 0) + "%";

  const m = task.metrics || {};
  $("#trainMetrics").innerHTML = Object.entries(m).map(([k, v]) =>
    `<div class="metric"><div class="k">${metricName(k)}</div><div class="v">${v}</div></div>`
  ).join("");
  renderLiveTrainCharts(task);

  // 训练中实时刷新日志（含逐 batch 的 loss 行），完成后由 viewTaskLog 接管
  if (["queued", "running"].includes(task.status)) {
    api(`/api/crowd/train/tasks/${id}/log?tail=30`).then(r => {
      if (!r || !r.log) return;
      const el = $("#trainLog");
      if (!el) return;
      el.textContent = r.log;
      el.scrollTop = el.scrollHeight;
    }).catch(() => {});
  }

  if (["completed", "failed", "stopped"].includes(task.status)) {
    stopPollingTimer();
    $("#btnStopTrain").disabled = true;
    refreshTaskHistory();
    await loadUploads();  // 刷新训练产物权重列表
    if (task.status === "completed" && task.result && task.result.save_dir) {
      // 训练产物下载链接（日志/曲线/指标图）
      toast(tf("msg.trainDoneSaved", statusLabel(task.status)), "success");
    } else {
      toast(tf("msg.trainDone", statusLabel(task.status)),
            task.status === "completed" ? "success" : "error");
    }
  }
}

/* ---- 训练图表：损失曲线 + 验证指标曲线（SVG 渲染，可下载 PNG） ---- */
const METRIC_COLORS = ["#f59e0b", "#a78bfa", "#34d399", "#f472b6", "#fb7185", "#38bdf8"];

function hasNum(h, k) {
  const v = h[k];
  return v !== undefined && v !== null && v !== "" && isFinite(Number(v));
}

/* 折线图 SVG；series: [{key, color, label}]，x 轴取 h.epoch */
function lineChartSvg(history, series, xKey = "epoch") {
  const rows = history.filter(h => h[xKey] !== undefined && series.some(s => hasNum(h, s.key)));
  if (!rows.length) return "";
  const W = 640, H = 180, PAD = 36;
  const xs = rows.map(h => Number(h[xKey]));
  const allVals = series.flatMap(s => rows.filter(h => hasNum(h, s.key)).map(h => Number(h[s.key])));
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...allVals), yMax = Math.max(...allVals);
  const sx = e => PAD + (e - xMin) / Math.max(1e-8, xMax - xMin) * (W - PAD - 8);
  const sy = v => H - PAD - (v - yMin) / Math.max(1e-8, yMax - yMin) * (H - PAD - 12);
  const cs = getComputedStyle(document.documentElement);
  const chartBg = (cs.getPropertyValue("--bg-card2") || "#212a35").trim();
  const chartBorder = (cs.getPropertyValue("--border") || "#2d3948").trim();
  const chartText = (cs.getPropertyValue("--text-dim") || "#8b9bab").trim();
  let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" style="width:100%;height:auto">`;
  svg += `<rect x="0" y="0" width="${W}" height="${H}" fill="${chartBg}"/>`;
  svg += `<line x1="${PAD}" y1="${H - PAD}" x2="${W - 8}" y2="${H - PAD}" stroke="${chartBorder}"/>`;
  svg += `<line x1="${PAD}" y1="12" x2="${PAD}" y2="${H - PAD}" stroke="${chartBorder}"/>`;
  for (const s of series) {
    const pts = rows.filter(h => hasNum(h, s.key));
    if (pts.length > 1) {
      const pl = pts.map(h => `${sx(Number(h[xKey])).toFixed(1)},${sy(h[s.key]).toFixed(1)}`).join(" ");
      svg += `<polyline fill="none" stroke="${s.color}" stroke-width="2" points="${pl}"/>`;
    }
    // 数据点圆点（点数不多时绘制，单点也可见）
    if (pts.length <= 60) {
      for (const h of pts) {
        svg += `<circle cx="${sx(Number(h[xKey])).toFixed(1)}" cy="${sy(h[s.key]).toFixed(1)}" r="2.5" fill="${s.color}"/>`;
      }
    }
  }
  svg += `<text x="${PAD}" y="${H - 8}" fill="${chartText}" font-size="11">${xMin}</text>`;
  svg += `<text x="${W - 20}" y="${H - 8}" fill="${chartText}" font-size="11">${xMax}</text>`;
  svg += `<text x="4" y="${H - PAD + 4}" fill="${chartText}" font-size="10">${yMin.toPrecision(3)}</text>`;
  svg += `<text x="4" y="16" fill="${chartText}" font-size="10">${yMax.toPrecision(3)}</text>`;
  // 图例（右上角竖排，指标图可能有 3+ 条曲线）
  series.forEach((s, i) => {
    const ly = 16 + i * 16;
    svg += `<line x1="${W - 118}" y1="${ly - 4}" x2="${W - 104}" y2="${ly - 4}" stroke="${s.color}" stroke-width="2"/>`;
    svg += `<text x="${W - 100}" y="${ly}" fill="${chartText}" font-size="11">${esc(s.label)}</text>`;
  });
  svg += `</svg>`;
  return svg;
}

/* 单个图块：标题 + 下载按钮 + SVG */
function chartBlockHtml(svg, titleKey, fname) {
  return `<div class="chart-block">
    <div class="chart-head">
      <span class="chart-title">${esc(t(titleKey))}</span>
      <button type="button" class="btn small chart-dl" onclick="downloadChart(this, '${fname}')">${esc(t("chart.download"))}</button>
    </div>
    ${svg}
  </div>`;
}

/* history → 损失图 + 验证指标图 HTML（无数据返回 ""） */
function buildTrainChartsHtml(history, fnamePrefix) {
  if (!history || !history.length) return "";
  const lossSeries = [
    { key: "train_loss", color: "#3b82f6", label: t("metric.train_loss") },
    { key: "val_loss", color: "#22d3ee", label: t("metric.val_loss") },
    { key: "loss", color: "#3b82f6", label: t("metric.train_loss") },
  ].filter(s => history.some(h => hasNum(h, s.key)));
  // 验证指标：损失 / epoch / lr 以外的数值字段（val_mae 等）
  const skip = new Set(["epoch", "train_loss", "val_loss", "loss", "lr"]);
  const metricKeys = [...new Set(history.flatMap(h => Object.keys(h)))]
    .filter(k => !skip.has(k) && history.some(h => hasNum(h, k)));
  let html = "";
  const lossSvg = lineChartSvg(history, lossSeries);
  if (lossSvg) html += chartBlockHtml(lossSvg, "chart.loss", `${fnamePrefix}loss.png`);
  if (metricKeys.length) {
    const mSeries = metricKeys.map((k, i) =>
      ({ key: k, color: METRIC_COLORS[i % METRIC_COLORS.length], label: metricName(k) }));
    const mSvg = lineChartSvg(history, mSeries);
    if (mSvg) html += chartBlockHtml(mSvg, "chart.metrics", `${fnamePrefix}metrics.png`);
  }
  return html;
}

/* 渲染进容器；返回是否有内容 */
function renderTrainCharts(box, history, fnamePrefix) {
  if (!box) return false;
  const html = buildTrainChartsHtml(history, fnamePrefix);
  if (!html) { box.style.display = "none"; return false; }
  box.style.display = "";
  box.innerHTML = html;
  return true;
}

/* 图表下载按钮：SVG → 2x PNG */
function downloadChart(btn, fname) {
  const block = btn.closest(".chart-block");
  const svg = block && block.querySelector("svg");
  if (!svg) return;
  const clone = svg.cloneNode(true);
  const vb = (svg.getAttribute("viewBox") || "0 0 640 180").split(/\s+/).map(Number);
  const w = vb[2] || 640, h = vb[3] || 180;
  clone.removeAttribute("style");
  clone.setAttribute("width", w);
  clone.setAttribute("height", h);
  const xml = new XMLSerializer().serializeToString(clone);
  const url = URL.createObjectURL(new Blob([xml], { type: "image/svg+xml;charset=utf-8" }));
  const img = new Image();
  img.onload = () => {
    const scale = 2;
    const canvas = document.createElement("canvas");
    canvas.width = w * scale; canvas.height = h * scale;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = (getComputedStyle(document.documentElement).getPropertyValue("--bg-card2") || "#212a35").trim();
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(url);
    const a = document.createElement("a");
    a.download = fname || "chart.png";
    a.href = canvas.toDataURL("image/png");
    a.click();
  };
  img.onerror = () => { URL.revokeObjectURL(url); toast(t("chart.exportFail"), "error"); };
  img.src = url;
}

/* 实时损失 + 指标走向图（随任务 history 更新） */
function renderLossChart(history) {
  renderTrainCharts($("#trainChart"), history, `task${state.trainTaskId || ""}_`);
}

/* 训练中实时曲线：逐 step 训练损失（持续刷新）+ 逐 epoch 验证指标 */
function renderLiveTrainCharts(task) {
  const box = $("#trainChart");
  if (!box) return;
  const prefix = `task${state.trainTaskId || ""}_`;
  const live = Array.isArray(task.live_history) ? task.live_history : [];
  const history = Array.isArray(task.history) ? task.history : [];

  if (!live.length) {
    // 无逐 step 数据（旧任务 / 未开始）：回退为按 epoch 的完整曲线
    if (!renderTrainCharts(box, history, prefix)) box.style.display = "none";
    return;
  }
  let html = "";
  // 1) 实时训练损失（逐 step，随 batch 持续刷新）
  const liveSvg = lineChartSvg(live, [
    { key: "train_loss", color: "#3b82f6", label: t("metric.train_loss") }
  ], "step");
  if (liveSvg) {
    html += chartBlockHtml(liveSvg, "chart.lossLive", `${prefix}loss_live.png`);
  }
  // 2) 验证指标（逐 epoch）：val_loss / val_mae 等（排除训练损失）
  const skip = new Set(["epoch", "train_loss", "loss", "lr"]);
  const keys = [...new Set(history.flatMap(h => Object.keys(h)))]
    .filter(k => !skip.has(k) && history.some(h => hasNum(h, k)));
  if (keys.length) {
    const series = keys.map((k, i) =>
      ({ key: k, color: METRIC_COLORS[i % METRIC_COLORS.length], label: metricName(k) }));
    const svg = lineChartSvg(history, series, "epoch");
    if (svg) html += chartBlockHtml(svg, "chart.metrics", `${prefix}metrics.png`);
  }
  if (!html) { box.style.display = "none"; return; }
  box.style.display = "";
  box.innerHTML = html;
}

/* 历史任务曲线回看（训练完成后 / 刷新页面后仍可查看并下载） */
function showTaskChart(taskId) {
  api(`/api/crowd/train/tasks/${taskId}`).then(task => {
    if (!renderTrainCharts($("#trainChart"), task.history || [], `task${taskId}_`)) {
      return toast(t("chart.noData"), "error");
    }
    $("#trainChart").scrollIntoView({ behavior: "smooth", block: "center" });
  }).catch(e => toast(e.message, "error"));
}

function metricName(k) {
  const m = { train_loss: t("metric.train_loss"), val_loss: t("metric.val_loss"),
              val_mae: t("metric.val_mae") };
  return m[k] || k;
}

async function stopTrain() {
  if (!state.trainTaskId) return;
  return stopTrainTask(state.trainTaskId);
}

async function stopTrainTask(id) {
  if (!id) return;
  try {
    await api(`/api/crowd/train/tasks/${id}/stop`, { method: "POST" });
    toast(t("msg.stopSent"), "success");
    refreshTaskHistory();
  } catch (e) { toast(e.message, "error"); }
}

async function refreshTaskHistory() {
  let tasks;
  try { tasks = await api("/api/crowd/train/tasks"); }
  catch (_) { return; }
  const tbody = $("#taskHistoryBody");
  if (!tasks.length) {
    tbody.innerHTML = `<tr><td colspan="7" style="color:var(--text-dim)">${esc(t("msg.noTasks"))}</td></tr>`;
    return;
  }
  tbody.innerHTML = tasks.map(x => {
    // 通用最优指标显示（各任务键名不同）
    let best = "—";
    if (x.best && x.best.epoch !== undefined) {
      const key = ["val_mae", "val_loss"]
        .find(k => x.best[k] !== undefined);
      if (key) best = `${key}=${x.best[key]} (epoch ${x.best.epoch})`;
    }
    // 权重列只显示文件名（含模型名前缀，如 mobilenet_v2_best.pth），
    // 完整路径放入 title 提示
    const weightFull = (x.result && x.result.best_weight) || "";
    const weight = weightFull ? String(weightFull).split(/[\\/]/).pop() : "—";
    const dl = (kind) => `/api/crowd/train/tasks/${x.id}/download?kind=${kind}`;
    const ops = [`<button class="btn small" onclick="viewTaskLog('${x.id}')">${esc(t("history.log"))}</button>`,
                 `<a class="btn small anchor-btn" href="${dl("log")}">${esc(t("history.dlLog"))}</a>`];
    if (["running", "queued", "pending"].includes(x.status)) {
      ops.unshift(`<button class="btn small" style="color:var(--red)" onclick="stopTrainTask('${x.id}')">${esc(t("train.stop"))}</button>`);
    }
    // 曲线回看（有 history 数据时展示，可下载 PNG）
    if (Array.isArray(x.history) && x.history.length) {
      ops.push(`<button class="btn small" onclick="showTaskChart('${x.id}')">${esc(t("chart.view"))}</button>`);
    }
    if (x.result && x.result.best_weight) {
      ops.push(`<a class="btn small anchor-btn" href="${dl("best")}">${esc(t("history.dlBest"))}</a>`);
    }
    if (x.resumable) {
      ops.push(`<button class="btn small" onclick="resumeTrain('${x.id}')">${esc(t("history.resume"))}</button>`);
    }
    ops.push(`<button class="btn small" style="color:var(--red)" onclick="deleteTrainTask('${x.id}')">${esc(t("history.delete"))}</button>`);
    const queueTag = (x.status === "queued" && x.queue_position)
      ? ` <span class="chip" style="font-size:10.5px;padding:1px 8px">#${x.queue_position}</span>`
      : "";
    return `<tr>
      <td>${esc(x.name)}${x.historical ? ` <span class="chip" style="font-size:10.5px;padding:1px 8px">${esc(t("history.diskTag"))}</span>` : ""}</td>
      <td>${esc(x.kind)}</td>
      <td><span class="badge ${x.status}">${statusLabel(x.status, x.kind)}</span>${queueTag}</td>
      <td>${x.epoch}/${x.total_epochs} (${Math.round(x.progress)}%)</td>
      <td>${best}</td>
      <td title="${esc(weightFull)}" style="font-size:11.5px;color:var(--text-dim);max-width:260px;overflow:hidden;text-overflow:ellipsis">${esc(weight)}</td>
      <td>${ops.join(" ")}</td>
    </tr>`;
  }).join("");
}

function statusLabel(s, kind) {
  // 运行中状态按任务类型区分文案：下载 → 下载中，ONNX → 导出中，
  // .mat 转换 → 转换中，.npy 可视化 / 模型可视化 → 可视化中，
  // 预处理 → 预处理中；训练任务（无 kind / crowd_*）保持「训练中」
  if (s === "running" && kind) {
    const km = {
      download: t("status.dlRunning"),
      onnx: t("status.onnxRunning"),
      mat2density: t("status.matRunning"),
      npy2png: t("status.npyRunning"),
      preprocess: t("status.prepRunning"),
      visualize: t("status.visRunning"),
      postprocess: t("status.postRunning"),
      test: t("status.testRunning"),
    };
    if (km[kind]) return km[kind];
  }
  const m = { running: t("status.running"), completed: t("status.completed"),
              failed: t("status.failed"), stopped: t("status.stopped"),
              pending: t("status.pending"), queued: t("status.queued") };
  return m[s] || s;
}

async function viewTaskLog(id) {
  state._lastViewedLogTaskId = id;
  try {
    const r = await api(`/api/crowd/train/tasks/${id}/log?tail=200`);
    $("#trainLog").textContent = r.log || t("msg.noLog");
    $("#trainLog").scrollTop = $("#trainLog").scrollHeight;
  } catch (e) { toast(e.message, "error"); }
}

/* 刷新历史日志（重新拉取最后查看的任务） */
async function refreshTaskLog() {
  if (!state._lastViewedLogTaskId) return;
  viewTaskLog(state._lastViewedLogTaskId);
}

/* 断点续训：从历史/已暂停任务恢复继续训练（复用原参数与运行目录） */
async function resumeTrain(id) {
  try {
    const r = await api("/api/crowd/train/resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: id }),
    });
    state.trainTaskId = r.task_id;
    toast(tf("msg.resumeStarted", r.run_name), "success", 6000);
    $("#btnStopTrain").disabled = false;
    $("#trainLog").textContent = "";
    refreshTaskHistory();
    startPolling();
  } catch (e) {
    toast(tf("msg.resumeFailed", e.message), "error", 6000);
  }
}

/* 删除任务记录与日志（可选同时删除训练产物） */
async function deleteTrainTask(id) {
  // 先问产物再问记录，任何一步取消都整个中止
  const artifacts = window.confirm(t("msg.confirmDeleteArtifacts"));
  if (!window.confirm(t("msg.confirmDelete"))) return;
  try {
    await api(`/api/crowd/train/tasks/${id}${artifacts ? "?artifacts=true" : ""}`,
              { method: "DELETE" });
    toast(t("msg.deleteDone"), "success");
    refreshTaskHistory();
  } catch (e) {
    toast(tf("msg.deleteFailed", e.message), "error", 6000);
  }
}

/* ---------------- 推理 ---------------- */

function bindInferControls() {
  const dz = $("#dropzone");
  dz.addEventListener("click", () => $("#inferImage").click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    if (e.dataTransfer.files.length) setInferImages(e.dataTransfer.files);
  });
  $("#inferImage").addEventListener("change", (e) => {
    if (e.target.files.length) setInferImages(e.target.files);
  });
  const inferFolder = $("#inferFolder");
  if (inferFolder) {
    inferFolder.addEventListener("change", (e) => {
      if (e.target.files.length) setInferImages(e.target.files);
    });
  }
  $("#btnUploadWeight").addEventListener("click", uploadWeight);
  $("#btnInfer").addEventListener("click", runInference);
  $("#btnCameraDetect").addEventListener("click", openCVCamera);
  $("#inferModel").addEventListener("change", updateInferHints);
  $("#inferWeightSelect").addEventListener("change", updateInferHints);
}

function setInferImage(file) {
  if (!file.type.startsWith("image/")) return toast(t("msg.pickImage"), "error");
  state.inferImageFiles = [file];
  renderInferPreviews();
}

function setInferImages(files) {
  const imgs = Array.from(files).filter(f => f.type.startsWith("image/"));
  if (!imgs.length) return toast(t("msg.pickImage"), "error");
  state.inferImageFiles = imgs;
  renderInferPreviews();
}

function renderInferPreviews() {
  const box = $("#imagePreview");
  if (!state.inferImageFiles.length) { box.innerHTML = ""; return; }
  if (state.inferImageFiles.length === 1) {
    const url = URL.createObjectURL(state.inferImageFiles[0]);
    box.innerHTML = `<img src="${url}" alt="preview">`;
  } else {
    box.innerHTML = `<p class="hint">${esc(tf("msg.batchSelected", state.inferImageFiles.length))}</p>` +
      state.inferImageFiles.slice(0, 5).map(f =>
        `<img src="${URL.createObjectURL(f)}" alt="${esc(f.name)}" style="max-height:120px;margin:4px;border-radius:8px;">`).join("");
  }
}

async function uploadWeight() {
  return uploadWeightToSelect("#inferWeightFile", "#btnUploadWeight",
                              "#inferWeightSelect", "infer.uploadWeightBtn",
                              r => { updateInferHints(); });
}

/* 通用权重上传：上传 → 刷新列表 → 自动选中新权重（训练初始权重 / 推理权重共用） */
async function uploadWeightToSelect(fileId, btnId, selectId, btnLabelKey, after) {
  const f = $(fileId).files[0];
  if (!f) return toast(t("msg.pickWeight"), "error");
  const fd = new FormData();
  fd.append("file", f);
  const btn = $(btnId);
  btn.disabled = true;
  btn.textContent = tf("msg.uploading", 0);
  try {
    const r = await uploadWithProgress("/api/upload/model", fd, (p) => {
      btn.textContent = tf("msg.uploading", p);
    });
    state.uploadedWeightName = r.name;
    toast(tf("msg.weightUploaded", r.size_mb), "success");
    await loadUploads();
    $(selectId).value = r.path;  // 自动选中新上传的权重
    if (after) after(r);
  } catch (e) {
    toast(tf("msg.uploadFailed", e.message), "error");
  } finally {
    btn.disabled = false;
    btn.textContent = t(btnLabelKey);
  }
}

async function uploadTrainWeight() {
  return uploadWeightToSelect("#trainWeightFile", "#btnUploadTrainWeight",
                              "#trainInitWeight", "train.uploadWeightBtn");
}

/* ---------------- 部署模式：ONNX 导出 + 片段代码 ---------------- */

function bindDeployControls() {
  const btn = $("#btnDeployUploadWeight");
  if (!btn) return;   // 版本混搭防御
  btn.addEventListener("click", () => $("#deployWeightFile").click());
  $("#deployWeightFile").addEventListener("change", () =>
    uploadWeightToSelect("#deployWeightFile", "#btnDeployUploadWeight",
                         "#deployWeightSelect", "infer.uploadWeightBtn",
                         () => refreshDeployFormat()));
  $("#deployWeightSelect").addEventListener("change", refreshDeployFormat);
  $("#btnOnnxExport").addEventListener("click", startOnnxExport);
}

/* 权重选择变化 → 调格式推断端点，渲染模型类型 + 默认尺寸 + YOLO 专属参数 */
async function refreshDeployFormat() {
  const sel = $("#deployWeightSelect");
  const info = $("#deployFormatInfo");
  if (!sel || !info) return;
  const wp = sel.value;
  state.deployFormat = null;
  const exportBtn = $("#btnOnnxExport");
  const resetExportBtn = () => {
    if (!exportBtn) return;
    exportBtn.disabled = false;
    exportBtn.title = "";
  };
  if (!wp) { info.style.display = "none"; resetExportBtn(); return; }
  info.style.display = "";
  info.className = "infer-mode-hint";
  info.textContent = t("deploy.probing");
  try {
    const fmt = await api("/api/crowd/onnx/format", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ weight_path: wp }),
    });
    state.deployFormat = fmt;
    const routeTag = { crowd_density: "deploy.tagDensity",
                       crowd_point: "deploy.tagPoint",
                       crowd_box: "deploy.tagBox" }[fmt.route] || "";
    info.className = "infer-mode-hint ok";
    const cls = fmt.classes && fmt.classes.length
      ? ` · ${fmt.classes.length} ${t("deploy.classes")}` : "";
    info.innerHTML =
      `<b>${t(routeTag)}</b> ${esc(fmt.model_label)} · ` +
      t("deploy.defaultSize") + ` ${fmt.imgsz}` +
      (fmt.multiple ? `（${t("deploy.multiple")} ${fmt.multiple}）` : "") +
      cls + (fmt.distilled ? ` · ${t("deploy.distilled")}` : "") +
      `<br>${t("deploy.source")}: ${esc(t("deploy.source." + (fmt.source || "")) || fmt.source || "")}`;
    // 不支持导出的点预测模型（CLTR / 自定义点模型）：前端直接禁用并提示；
    // PET 虽后端有明确提示，也一并禁用，避免用户误以为可导出
    const unsupported = fmt.route === "crowd_point" &&
      !["p2pnet", "p2r", "cp", "cu"].includes(String(fmt.model || ""));
    if (unsupported) {
      info.className = "infer-mode-hint warn";
      info.innerHTML += `<br>${esc(t("deploy.unsupported"))}`;
      if (exportBtn) {
        exportBtn.disabled = true;
        exportBtn.title = t("deploy.unsupported");
      }
    } else {
      resetExportBtn();
    }
    // 输入边长为 0（自动）时回填建议默认；YOLO 专属参数显隐
    const szEl = $("#deployImgsz");
    if (szEl && (+szEl.value === 0 || szEl.dataset.auto === "1"))
      { szEl.value = fmt.imgsz; szEl.dataset.auto = "1"; }
    const isBox = fmt.route === "crowd_box";
    const showEl = (id, on) => { const el = $(id); if (el) el.style.display = on ? "" : "none"; };
    showEl("#fieldDeployHalf", isBox);
    showEl("#fieldDeployNms", isBox);
  } catch (e) {
    info.className = "infer-mode-hint warn";
    info.textContent = tf("msg.deployFormatFailed", e.message);
    resetExportBtn();
  }
}

async function startOnnxExport() {
  const wp = $("#deployWeightSelect") ? $("#deployWeightSelect").value : "";
  if (!wp) return toast(t("msg.deployWeightRequired"), "error");
  const fmt = state.deployFormat;
  if (fmt && fmt.route === "crowd_point" &&
      !["p2pnet", "p2r", "cp", "cu"].includes(String(fmt.model || ""))) {
    return toast(t("deploy.unsupported"), "error", 8000);
  }
  if (fmt && fmt.multiple) {
    const sz = +$("#deployImgsz").value;
    if (sz > 0 && sz % fmt.multiple !== 0)
      return toast(tf("msg.deploySizeAlign", fmt.multiple), "error", 6000);
  }
  const body = {
    weight_path: wp,
    imgsz: +$("#deployImgsz").value || 0,
    opset: +$("#deployOpset").value,
    dynamic_batch: $("#deployDynamic").checked,
    simplify: $("#deploySimplify").checked,
    half: $("#deployHalf").checked,
    nms: $("#deployNms").checked,
    conf_threshold: numOr($("#deployConfThr"), 0.5),
  };
  const btn = $("#btnOnnxExport");
  btn.disabled = true;
  btn.textContent = t("msg.starting");
  try {
    const r = await api("/api/crowd/onnx/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.deployTaskId = r.task_id;
    toast(tf("msg.deployStarted", r.run_name), "success");
    $("#deployResult").innerHTML =
      `<p class="hint"><span class="loading"></span> ${t("deploy.exporting")}</p>`;
    startDeployPolling();
  } catch (e) {
    toast(tf("msg.startFailed", e.message), "error", 6000);
  } finally {
    btn.disabled = false;
    btn.textContent = t("deploy.start");
  }
}

function startDeployPolling() {
  stopDeployPolling();
  state.deployPollTimer = setInterval(pollDeployTask, 1500);
  pollDeployTask();
}

function stopDeployPolling() {
  if (state.deployPollTimer) { clearInterval(state.deployPollTimer); state.deployPollTimer = null; }
}

async function pollDeployTask() {
  const id = state.deployTaskId;
  if (!id) return stopDeployPolling();
  let task;
  try { task = await api(`/api/crowd/train/tasks/${id}`); }
  catch (_) { return; }
  if (["completed", "failed", "stopped"].includes(task.status)) {
    stopDeployPolling();
    if (task.status !== "completed")
          return toast(task.error || statusLabel(task.status, task.kind), "error", 8000);
    renderDeployResult(task);
    toast(t("msg.deployDone"), "success");
  }
}

/* 导出结果：下载链接 + 按模型类型生成的部署片段代码（一键复制） */
function renderDeployResult(task) {
  const r = task.result || {};
  const meta = r.meta || {};
  const box = $("#deployResult");
  if (!box) return;
  if (!r.onnx_url) {
    box.innerHTML = `<p class="hint" style="color:var(--red)">${esc(task.error || "export failed")}</p>`;
    return;
  }
  const routeTag = { crowd_density: "deploy.tagDensity",
                     crowd_point: "deploy.tagPoint",
                     crowd_box: "deploy.tagBox" }[r.route] || "";
  const snippet = r.snippet || "";
  box.innerHTML = `
    <div class="infer-mode-hint ok" style="margin-bottom:8px">
      <b>${t("deploy.doneTitle")}</b> ${t(routeTag)} ${esc(r.model || "")}
      · ${r.size_mb} MB · opset ${meta.opset} · ${meta.dynamic_batch ? t("deploy.dynamicOn") : "batch=1"}
    </div>
    <p style="margin:4px 0">
      <a class="btn small" href="${r.onnx_url}" download="model.onnx" style="margin-top:0">⬇ model.onnx</a>
      <a class="btn small" href="${r.snippet_url}" download="snippet.py" style="margin-top:0">⬇ snippet.py</a>
      <button class="btn small" id="btnCopySnippet" style="margin-top:0">${t("deploy.copySnippet")}</button>
      <button class="btn small primary" id="btnOnnxTestInfer" style="margin-top:0">${t("deploy.testInfer")}</button>
    </p>
    <p class="hint" style="margin:4px 0">${t("deploy.usage")}<br>
      <code>python snippet.py &lt;image&gt;${r.route === "crowd_point" && r.model === "p2r" ? "" : " [conf]"}</code>
    </p>
    <pre id="deploySnippet" style="max-height:420px;overflow:auto;background:var(--bg-soft,#11162a);border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:12px;font-size:12px;line-height:1.5;margin:8px 0 0">${esc(snippet)}</pre>`;
  const copyBtn = $("#btnCopySnippet");
  if (copyBtn) copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(snippet);
      toast(t("msg.copied"), "success");
    } catch (e) {
      const ta = document.createElement("textarea");
      ta.value = snippet;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      toast(t("msg.copied"), "success");
    }
  });
  // 用刚导出的产物推理：切到推理模式并自动选中该 onnx（验证部署链路）
  const testBtn = $("#btnOnnxTestInfer");
  if (testBtn) testBtn.addEventListener("click", () =>
    gotoInferWithOnnx(r.onnx_path));
}

/* 跳到推理模式并选中指定 onnx（导出完成后的「用此模型推理」入口） */
async function gotoInferWithOnnx(onnxPath) {
  if (!onnxPath) return;
  await loadUploads();   // 刚导出的产物需刷新列表后才会出现在下拉中
  const sel = $("#inferWeightSelect");
  if (!sel || ![...sel.options].some(o => o.value === onnxPath))
    return toast(t("msg.onnxNotListed"), "error", 6000);
  sel.value = onnxPath;
  // 同面板内切到推理模式（mode-btn 绑定已由 bindModeSwitch 完成）
  const btn = document.querySelector("#panel-cv .mode-btn[data-mode='infer']");
  if (btn) btn.click();
  updateInferHints();
  toast(t("msg.onnxSelected"), "success", 5000);
}

/* 恢复中断训练：上传运行目录 zip → 恢复为历史任务（随后点击「继续训练」） */
async function uploadTrainRun() {
  const f = $("#trainRunZipFile").files[0];
  if (!f) return toast(t("msg.pickZip"), "error");
  const fd = new FormData();
  fd.append("file", f);
  const btn = $("#btnUploadTrainRun");
  btn.disabled = true;
  btn.textContent = tf("msg.uploading", 0);
  const status = $("#trainRestoreStatus");
  try {
    const r = await uploadWithProgress("/api/crowd/train/upload_run", fd, p => {
      btn.textContent = tf("msg.uploading", p);
    });
    status.textContent = tf("msg.runRestored", r.run_name) +
      (r.resumable ? "" : " " + t("msg.runNoWeight"));
    toast(tf("msg.runRestored", r.run_name), "success", 6000);
    refreshTaskHistory();
  } catch (e) {
    status.textContent = "";
    toast(tf("msg.uploadFailed", e.message), "error", 6000);
  } finally {
    btn.disabled = false;
    btn.textContent = t("train.restoreBtn");
  }
}

async function runInference() {
  if (!state.inferImageFiles.length) return toast(t("msg.pickInferImage"), "error");
  const task = state.currentTask;
  const model = $("#inferModel").value;
  const meta = currentInferModelMeta();
  const weightSel = $("#inferWeightSelect").value;
  const usePretrained = $("#inferPretrained").checked;

  // 基于模型能力的校验
  if (meta && (meta.need_weight || meta.pretrained === false) && !weightSel) {
    return toast(tf("msg.noPretrained", model), "error", 6000);
  }
  if (!weightSel && !usePretrained) {
    return toast(t("msg.weightOrPretrained"), "error", 5000);
  }

  const fd = new FormData();
  for (const f of state.inferImageFiles) fd.append("images", f);
  fd.append("task", task.key);
  fd.append("model", model);
  fd.append("weight_path", weightSel);
  fd.append("pretrained", usePretrained);
  fd.append("device", $("#inferDevice").value);
  fd.append("conf_threshold", $("#confThreshold").value);
  fd.append("save_result", $("#saveResult").checked);
  const inferSaveDir = $("#inferSaveDir") ? $("#inferSaveDir").value.trim() : "";
  if (inferSaveDir) fd.append("save_dir", inferSaveDir);

  const btn = $("#btnInfer");
  btn.disabled = true;
  btn.textContent = tf("msg.uploading", 0);
  const nImgs = state.inferImageFiles.length;
  $("#inferResult").innerHTML = `<p class="hint"><span class="loading"></span> ${
    nImgs > 1 ? esc(tf("msg.batchInferring", nImgs)) : esc(t("msg.modelInferring"))
  }${esc(t("msg.pleaseWait"))}</p>`;
  try {
    const r = await uploadWithProgress("/api/crowd/infer", fd, p => {
      // 上传阶段显示进度，上传完成后显示处理中
      btn.textContent = p >= 100 ? t("msg.inferring") : tf("msg.uploading", p);
    });
    if (r.batch) renderBatchResult(r);
    else renderInferResult(task.key, r);
    if ($("#saveResult").checked) await loadUploads();
  } catch (e) {
    $("#inferResult").innerHTML =
      `<p style="color:var(--red)">${esc(tf("msg.inferFailed", e.message))}</p>`;
    toast(tf("msg.inferFailed", e.message), "error", 6000);
  } finally {
    btn.disabled = false;
    btn.textContent = t("infer.start");
  }
}

/* 打开摄像头实时计数弹窗（复用当前面板的模型 / 权重 / 阈值） */
function openCVCamera() {
  const task = state.currentTask;
  if (!task) return;
  const model = $("#inferModel").value;
  if (!model) return toast(t("camera.noModel"), "error");
  const wp = $("#inferWeightSelect").value;
  const usePretrained = $("#inferPretrained").checked;
  if (wp && /\.onnx$/i.test(wp)) {
    return toast(t("msg.cameraNoOnnx"), "error", 5000);
  }
  if (!wp && !usePretrained) {
    return toast(t("msg.weightOrPretrained"), "error", 5000);
  }
  openCameraModal({
    framework: "pytorch",
    task: task.key,
    taskLabel: task.label || task.key,
    model, weightPath: wp,
    pretrained: usePretrained,
    device: $("#inferDevice").value,
    conf: parseFloat($("#confThreshold").value || "0.5"),
  });
}

function renderBatchResult(r) {
  let html = `<h4 style="margin:6px 0">${esc(tf("res.batchDone", r.num_images))}</h4>`;
  if (r.preview_images && r.preview_images.length) {
    html += `<p class="hint">${esc(tf("msg.batchPreviewLimit", r.preview_images.length, r.num_images))}</p>`;
    html += `<div class="result-images">` + r.preview_images.map(p =>
      `<figure><img src="data:image/jpeg;base64,${p.vis_b64}" alt="vis">` +
      `<figcaption>${esc(p.image || "")} ? count=${esc(String(p.count))}</figcaption></figure>`
    ).join("") + `</div>`;
  }
  if (r.zip_url) {
    html += `<div class="saved-url">${esc(t("res.download"))}` +
      `<a class="btn small anchor-btn" href="${r.zip_url}" target="_blank">ZIP</a>` +
      `<span style="margin-left:8px">${esc(r.zip_url)}</span></div>`;
  }
  const shown = (r.results || []).slice(0, 5);
  html += `<table class="topk-table"><thead><tr><th>${esc(t("th.file"))}</th><th>${esc(t("th.summary"))}</th></tr></thead><tbody>`;
  for (const it of shown) {
    let summary = "";
    if (it.results && it.results.length) {
      // Top-K 全部显示（最多 3 项），而非仅第 1 名
      summary = it.results.slice(0, 3).map(x =>
        `${x.class} (${(x.score * 100).toFixed(1)}%)`).join(" · ") +
        (it.results.length > 3 ? " …" : "");
    } else if (it.count !== undefined) {
      summary = tf("summary.count", it.count);
    } else if (it.num_faces !== undefined) {
      summary = tf("summary.faces", it.num_faces);
    } else if (it.num_detected !== undefined) {
      summary = tf("summary.objects", it.num_detected);
    } else if (it.class_stats) {
      const top = Object.entries(it.class_stats).sort((a, b) => b[1] - a[1])[0];
      summary = top ? `${top[0]} ${(top[1] * 100).toFixed(1)}%` : "—";
    }
    html += `<tr><td>${esc(it.filename || "")}</td><td>${esc(summary)}</td></tr>`;
  }
  html += `</tbody></table>`;
  $("#inferResult").innerHTML = html;
}

function renderInferResult(taskKey, r) {
  const box = $("#inferResult");
  let html = "";

  if (isCrowdTask(taskKey)) {
    html += `<div class="result-count">${esc(tf("res.count", r.count))}</div>`;
    if (r.onnx) {
      // ONNX Runtime 推理结果（部署产物验证 / 上传的 onnx 模型）
      html += `<p class="hint" style="text-align:center">${esc(t("res.onnxEngine"))}</p>`;
    }
    if (r.point_based) {
      // 点预测模式（P2PNet / P2R / Consistent-Point / CU）：人数 + 预测点可视化（红点）
      // P2R 的点由 logit>0 判定（与训练期评估口径一致），无置信度阈值
      if (r.model === "p2r") {
        html += `<p class="hint" style="text-align:center">P2RLoss · ${
          esc(t("res.p2pPoints"))}</p>`;
      } else if (r.model === "cp") {
        html += `<p class="hint" style="text-align:center">Consistent-Point · ${
          esc(t("res.p2pPoints"))} · conf ${r.conf_threshold}</p>`;
      } else if (r.model === "cu") {
        html += `<p class="hint" style="text-align:center">CU · ${
          esc(t("res.p2pPoints"))} · conf ${r.conf_threshold}</p>`;
      } else if (r.model === "pet") {
        html += `<p class="hint" style="text-align:center">PET · ${
          esc(t("res.p2pPoints"))} · conf ${r.conf_threshold}</p>`;
      } else if (r.model === "cltr") {
        html += `<p class="hint" style="text-align:center">CLTR · ${
          esc(t("res.p2pPoints"))} · conf ${r.conf_threshold}</p>`;
      } else if (r.model === "raz") {
        html += `<p class="hint" style="text-align:center">RAZ · density ${
          r.density_count ?? "-"} / loc ${r.loc_count ?? "-"} / fused ${
          r.count} · rounds ${r.raz_rounds ?? 0}</p>`;
      } else {
        html += `<p class="hint" style="text-align:center">P2PNet · ${
          esc(t("res.p2pPoints"))} · conf ${r.conf_threshold}</p>`;
      }
      html += imagesHtml([[t("res.p2pVis"), r.visualization]]);
    } else if (r.box_based) {
      // 头部框模式（YOLO）：人头数 + 检测框可视化 + 各类计数
      const chips = Object.entries(r.summary || {})
        .map(([c, n]) => `<span class="chip">${esc(c)} × <b>${n}</b></span>`)
        .join("");
      if (chips) html += `<div class="summary-chips">${chips}</div>`;
      html += `<p class="hint">${esc(tf("res.detected", r.num_detected, r.conf_threshold))}</p>`;
      html += imagesHtml([[t("res.boxVis"), r.visualization]]);
    } else {
      html += `<p class="hint" style="text-align:center">${esc(t("res.densitySize"))}${r.density_size.join(" × ")} · ${esc(t("res.model"))}${esc(r.model)}</p>`;
      html += imagesHtml([[t("res.densityOverlay"), r.visualization], [t("res.density"), r.density_map]]);
    }
  }
  if (r.saved_url) {
    html += `<div class="saved-url">${esc(t("res.saved"))}<a href="${r.saved_url}" target="_blank">${r.saved_url}</a></div>`;
  }
  state.lastInferVis = r.visualization
    ? { b64: r.visualization, name: (r.filename || "inference") + "_vis.png" }
    : null;
  if (state.lastInferVis) {
    html += `<div class="actions"><button class="btn small" onclick="downloadLastInferVis()">${esc(t("res.downloadVis"))}</button></div>`;
  }
  box.innerHTML = html;
}

function downloadBase64(b64, filename) {
  if (!b64) return;
  const a = document.createElement("a");
  a.href = `data:image/png;base64,${b64}`;
  a.download = filename || "visualization.png";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function downloadLastInferVis() {
  if (state.lastInferVis) downloadBase64(state.lastInferVis.b64, state.lastInferVis.name);
}

function downloadLastVis() {
  if (state.lastVis) downloadBase64(state.lastVis.b64, state.lastVis.name);
}

function imagesHtml(items) {
  return `<div class="result-images">` + items.map(([cap, b64]) =>
    `<figure><img src="data:image/png;base64,${b64}" alt="${esc(cap)}"><figcaption>${esc(cap)}</figcaption></figure>`
  ).join("") + `</div>`;
}

/* ---------------- 启动 ---------------- */

// 周期刷新任务历史（有任务运行时）
setInterval(() => {
  if (state.trainTaskId) refreshTaskHistory();
}, 10000);

/* ---------------- 系统状态（硬件实时监控） ---------------- */

const hwState = {
  timer: null,      // 轮询定时器
  lastData: null,   // 最近一次状态（语言切换时重渲染）
  lastNet: null,    // 上次网络字节计数（计算速率）
  rates: null,      // {down, up} 字节/秒
};

function hwColorClass(p) {
  return p >= 85 ? "busy" : p >= 60 ? "warn" : "ok";
}

function hwBar(percent) {
  const p = Math.max(0, Math.min(100, percent || 0));
  return `<div class="hw-bar"><div class="hw-bar-fill ${hwColorClass(p)}" style="width:${p}%"></div></div>`;
}

function fmtBytes(b) {
  if (b == null) return "-";
  const gb = b / 1073741824;
  if (gb >= 1) return gb.toFixed(2) + " GB";
  const mb = b / 1048576;
  if (mb >= 1) return mb.toFixed(1) + " MB";
  return (b / 1024).toFixed(1) + " KB";
}

function fmtRate(bps) {
  return bps == null ? "—" : fmtBytes(bps) + "/s";
}

function fmtUptime(hours) {
  if (hours == null) return "—";
  const d = Math.floor(hours / 24), h = Math.round(hours % 24);
  return (d > 0 ? d + t("hw.day") + " " : "") + h + t("hw.hour");
}

async function loadHwStatus() {
  const box = $("#hwContent");
  try {
    const d = await api("/api/system/hardware");
    // 网络速率：相邻两次轮询的字节差 ÷ 时间差
    const now = Date.now();
    if (d.network && hwState.lastNet && hwState.lastNet.ts) {
      const dt = (now - hwState.lastNet.ts) / 1000;
      if (dt > 0.2) {
        hwState.rates = {
          down: Math.max(0, (d.network.bytes_recv - hwState.lastNet.recv) / dt),
          up: Math.max(0, (d.network.bytes_sent - hwState.lastNet.sent) / dt),
        };
      }
    }
    if (d.network) {
      hwState.lastNet = { ts: now, sent: d.network.bytes_sent,
                          recv: d.network.bytes_recv };
    }
    hwState.lastData = d;
    renderHardware(d);
  } catch (e) {
    if (box) box.innerHTML =
      `<p style="color:var(--red)">${esc(tf("msg.hwLoadFailed", e.message))}</p>`;
  }
}

function startHwPolling() {
  stopHwPolling();
  loadHwStatus();   // 进入面板/切换间隔立即刷新一次
  if ($("#hwLive") && !$("#hwLive").checked) return;
  const iv = +($("#hwInterval") ? $("#hwInterval").value : 2000);
  hwState.timer = setInterval(loadHwStatus, iv);
}

function stopHwPolling() {
  if (hwState.timer) { clearInterval(hwState.timer); hwState.timer = null; }
}

/* ---------------- 网络代理（独立代理设置面板） ---------------- */

/* 默认代理（输入框预填 / Tab 补全默认值）与常见本地代理端口候选 */
const PROXY_DEFAULT_URL = "http://127.0.0.1:7890";
const PROXY_PRESETS = [
  "http://127.0.0.1:7890",     // Clash 默认
  "http://127.0.0.1:7897",     // Clash Verge 默认
  "http://127.0.0.1:10809",    // v2rayN HTTP
  "socks5://127.0.0.1:10808",  // v2rayN SOCKS5
  "http://127.0.0.1:1080",     // 通用本地端口
];

const proxyState = { saved: [], cfg: {} };

async function loadProxyConfig() {
  try {
    const d = await api("/api/system/proxy");
    proxyState.saved = (d.config && d.config.saved) || [];
    proxyState.cfg = d.config || {};
    if ($("#proxyTestUrl") && !$("#proxyTestUrl").value)
      $("#proxyTestUrl").value = proxyState.cfg.test_url
        || "https://huggingface.co";
    // 输入框预填默认代理（已加入列表的不重复预填）
    const input = $("#proxyUrlInput");
    if (input && !input.value.trim()) {
      const def = d.default_url || PROXY_DEFAULT_URL;
      if (!proxyState.saved.includes(def)) input.value = def;
    }
    renderProxyPanel(d);
  } catch (_) { /* 接口不可用时静默（面板仍可填写） */ }
}

function proxyAutocompletePool() {
  /* Tab 补全候选：内置预设 + 已存列表（去重、保持顺序） */
  return [...new Set([...PROXY_PRESETS, ...proxyState.saved])];
}

function proxyInputTabComplete(e) {
  /* Tab 自动补全：空 → 默认代理；已输入 → 补全第一个前缀匹配候选 */
  const input = $("#proxyUrlInput");
  const v = input.value.trim();
  let fill = null;
  if (!v) fill = PROXY_DEFAULT_URL;
  else fill = proxyAutocompletePool().find(p => p !== v && p.startsWith(v)) || null;
  if (fill) {
    e.preventDefault();  // 补全时阻止 Tab 焦点跳出输入框
    input.value = fill;
  }
}

function renderProxyPanel(d) {
  /* 渲染整个面板：输入提示 + 列表 + 启用复选框 + 状态行 */
  const hint = $("#proxyInputHint");
  if (hint) hint.innerHTML = esc(tf("proxy.inputHint", PROXY_DEFAULT_URL));
  renderProxyList(d);
  renderProxyStatus(d);
}

function renderProxyList(d) {
  const box = $("#proxyList");
  if (!box) return;
  const cfg = (d && d.config) || proxyState.cfg || {};
  const saved = cfg.saved || [];
  if (!saved.length) {
    box.innerHTML = `<div class="hint" style="margin:0">${esc(t("proxy.emptyList"))}</div>`;
  } else {
    box.innerHTML = saved.map(u => {
      const sel = u === cfg.url;
      return `<div class="proxy-row${sel ? " selected" : ""}" data-url="${esc(u)}">
        <input type="radio" name="proxyChoice" ${sel ? "checked" : ""}
               title="${esc(t("proxy.select"))}">
        <span class="proxy-url" title="${esc(u)}">${esc(u)}</span>
        <button class="btn small proxy-test" style="margin-top:0">${esc(t("proxy.test"))}</button>
        <button class="btn small danger proxy-del" style="margin-top:0">${esc(t("proxy.del"))}</button>
      </div>`;
    }).join("");
    box.querySelectorAll(".proxy-row").forEach(row => {
      row.addEventListener("click", e => {
        if (e.target.closest("button")) return;
        selectProxyUrl(row.dataset.url);
      });
    });
    box.querySelectorAll(".proxy-del").forEach(btn => {
      btn.addEventListener("click", () =>
        removeProxyUrl(btn.closest(".proxy-row").dataset.url));
    });
    box.querySelectorAll(".proxy-test").forEach(btn => {
      btn.addEventListener("click", () =>
        testProxyConn(btn.closest(".proxy-row").dataset.url, btn));
    });
  }
  if ($("#proxyEnabled")) {
    $("#proxyEnabled").checked = !!cfg.enabled;
    $("#proxyEnabled").disabled = !cfg.url;  // 列表无选中项时禁用开关
  }
}

function renderProxyStatus(d) {
  const box = $("#proxyStatus");
  if (!box) return;
  const cfg = d.config || {};
  const parts = [];
  if (cfg.url) {
    const on = d.active;
    parts.push(`<span style="color:var(--${on ? "green" : "red"})">●</span> ` +
      `${esc(cfg.url)} — ${on ? esc(t("proxy.active")) : esc(t("proxy.inactive"))}`);
    if (cfg.updated_at) parts.push(esc(tf("proxy.updatedAt", cfg.updated_at)));
    if (!d.socks_supported && /^socks/i.test(cfg.url || ""))
      parts.push(esc(t("proxy.socksHint")));
  } else {
    parts.push(esc(t("proxy.none")));
  }
  box.innerHTML = parts.join(" &nbsp;|&nbsp; ");
}

function _syncProxyState(d) {
  proxyState.saved = (d.config && d.config.saved) || [];
  proxyState.cfg = d.config || {};
  renderProxyPanel(d);
}

async function applyProxySettings(msgFn) {
  /* 勾选/取消启用 */
  const url = (proxyState.cfg || {}).url || "";
  const enabled = $("#proxyEnabled").checked;
  if (enabled && !url) {
    $("#proxyEnabled").checked = false;   // 勾选启用但无选中代理:回退开关
    toast(t("msg.proxyUrlRequired"), "error");
    return null;
  }
  try {
    const d = await api("/api/system/proxy", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, enabled, no_proxy: "",
                             test_url: $("#proxyTestUrl").value.trim() }),
    });
    _syncProxyState(d);
    if (msgFn) toast(msgFn(d), "success");
    return d;
  } catch (e) {
    toast(tf("msg.proxySaveFailed", e.message), "error", 8000);
    return null;
  }
}

async function addProxyUrl() {
  /* 「添加」按钮 / 回车：把输入框地址加入已存列表（空输入 → 默认代理） */
  const input = $("#proxyUrlInput");
  let url = (input.value || "").trim();
  if (!url) {
    url = PROXY_DEFAULT_URL;
    input.value = url;
  }
  try {
    const d = await api("/api/system/proxy/saved", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    input.value = "";
    _syncProxyState(d);
    toast(d.added ? t("proxy.msgAdded") : t("proxy.msgExists"),
          d.added ? "success" : "info");
  } catch (e) {
    toast(tf("msg.proxySaveFailed", e.message), "error", 8000);
  }
}

async function removeProxyUrl(url) {
  /* 列表行「删除」：从已存列表移除（后端处理选中项切换/停用） */
  if (!confirm(tf("proxy.confirmRemove", url))) return;
  try {
    const d = await api(
      `/api/system/proxy/saved?url=${encodeURIComponent(url)}`,
      { method: "DELETE" });
    _syncProxyState(d);
    toast(t("proxy.msgRemoved"), "success");
  } catch (e) {
    toast(tf("msg.proxySaveFailed", e.message), "error", 8000);
  }
}

async function selectProxyUrl(url) {
  /* 点行切换选中，启用中就直接切到新地址 */
  if (url === (proxyState.cfg || {}).url) return;
  try {
    const d = await api("/api/system/proxy", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, enabled: !!proxyState.cfg.enabled,
                             no_proxy: "",
                             test_url: $("#proxyTestUrl").value.trim() }),
    });
    _syncProxyState(d);
  } catch (e) {
    toast(tf("msg.proxySaveFailed", e.message), "error", 8000);
  }
}

async function clearProxyConfig() {
  /* 「清除全部配置」：删除 proxy.json 并恢复直连 */
  if (!confirm(t("proxy.confirmClear"))) return;
  try {
    const d = await api("/api/system/proxy", { method: "DELETE" });
    proxyState.saved = [];
    proxyState.cfg = d.config || {};
    if ($("#proxyUrlInput")) $("#proxyUrlInput").value = "";
    renderProxyPanel(d);
    toast(t("msg.proxyCleared"), "success");
  } catch (e) {
    toast(tf("msg.proxySaveFailed", e.message), "error");
  }
}

async function testProxyConn(url, btn) {
  /* 列表行「测试」：连通性测试（目标地址取「测试地址」输入框） */
  btn.disabled = true;
  btn.textContent = t("proxy.testing");
  try {
    const d = await api("/api/system/proxy/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, test_url: $("#proxyTestUrl").value.trim() }),
    });
    if (d.ok) {
      toast(tf("msg.proxyTestOk", d.status_code, d.latency_ms), "success", 6000);
    } else {
      toast(tf("msg.proxyTestFail", d.error || "unknown"), "error", 10000);
    }
  } catch (e) {
    toast(tf("msg.proxyTestFail", e.message), "error", 10000);
  } finally {
    btn.disabled = false;
    btn.textContent = t("proxy.test");
  }
}

/* ---------------- 评价指标可视化（纯前端 SVG + CSS） ---------------- */

let _mvBuilt = false;

function mvFrac(a, b) {
  return `<span class="mv-frac"><span class="num">${a}</span><span class="den">${b}</span></span>`;
}
function mvSqrt(x) {
  return `<span class="mv-sqrt">√<span class="mvsq">${x}</span></span>`;
}
/* 种子随机，示例数据可复现 */
function mvRand(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function mvSvg(w, h, inner) {
  return `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">${inner}</svg>`;
}
const mvLn = (x1, y1, x2, y2, s, wd, dash) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${s}" stroke-width="${wd || 1}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
const mvTxt = (x, y, s, fill, size, anchor) =>
  `<text x="${x}" y="${y}" fill="${fill || "#8b9bab"}" font-size="${size || 11}"${anchor ? ` text-anchor="${anchor}"` : ""}>${s}</text>`;
const mvCir = (cx, cy, r, fill, stroke) =>
  `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}"${stroke ? ` stroke="${stroke}" stroke-width="2"` : ""}/>`;
const mvRect = (x, y, w, h, fill, stroke, sw, op, dash) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}"${stroke ? ` stroke="${stroke}" stroke-width="${sw || 1.5}"${dash ? ` stroke-dasharray="${dash}"` : ""}` : ""}${op !== undefined ? ` fill-opacity="${op}"` : ""}/>`;
const mvPoly = (pts, stroke, fill, wd, op) =>
  `<polyline points="${pts}" fill="${fill || "none"}" stroke="${stroke}" stroke-width="${wd || 2}"${op !== undefined ? ` fill-opacity="${op}"` : ""}/>`;

/* 0~1 方形坐标区 */
function mvAxes(w, pad) {
  const px = v => pad + v * (w - 2 * pad);
  const py = v => w - pad - v * (w - 2 * pad);
  let inner = "";
  for (let g = 0; g <= 4; g++) {
    inner += mvLn(px(g / 4), py(0), px(g / 4), py(1), "#2d3948", 1);
    inner += mvLn(px(0), py(g / 4), px(1), py(g / 4), "#2d3948", 1);
    inner += mvTxt(px(g / 4), py(0) + 14, (g / 4).toFixed(2), "#8b9bab", 10, "middle");
    if (g > 0) inner += mvTxt(px(0) - 6, py(g / 4) + 3, (g / 4).toFixed(2), "#8b9bab", 10, "end");
  }
  return { px, py, inner };
}
function mvSlider(id, min, max, step, val, label) {
  return `<div class="mv-ctl"><label>${label} <b id="${id}Val"></b></label>` +
    `<input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></div>`;
}
function mvBtn(id, text, on) {
  return `<button type="button" class="btn small${on ? " on" : ""}" id="${id}">${text}</button>`;
}
function mvStat(label, val) {
  return `<span class="mv-stat">${label} <b>${val}</b></span>`;
}

/* ===== MAE / RMSE ===== */
const MV_MAE = {
  id: "mae", tagKey: "mv.sec.crowd",
  nameKey: "mv.mae.name", introKey: "mv.mae.intro",
  formula:
    `MAE = ${mvFrac("1", "N")}·Σ|ŷ − y|　` +
    `RMSE = ${mvSqrt(`${mvFrac("1", "N")}·Σ(ŷ − y)<sup>2</sup>`)}`,
  build(canvas, ctrl, read) {
    const rnd = mvRand(9);
    const gts = [12, 45, 3, 128, 27, 66, 512, 19, 88, 34];
    const noise = gts.map(g => (rnd() - .5) * g * .12);
    const OUT = 6;
    const draw = bias => {
      const preds = gts.map((g, i) => Math.max(0, Math.round(g + noise[i] + bias)));
      const errs = preds.map((p, i) => p - gts[i]);
      const mae = errs.reduce((a, e) => a + Math.abs(e), 0) / errs.length;
      const rmse = Math.sqrt(errs.reduce((a, e) => a + e * e, 0) / errs.length);
      const W = 330, H = 240, PL = 14, rowH = (H - 24) / gts.length;
      const maxSqrt = Math.sqrt(Math.max(...gts.map((g, i) => Math.max(g, preds[i]))));
      const bl = v => PL + Math.sqrt(v) / maxSqrt * (W - PL - 78);
      let inner = "";
      gts.forEach((g, i) => {
        const y = 6 + i * rowH, isOut = i === OUT;
        inner += mvRect(PL, y, bl(g) - PL, rowH - 8, "#475569", null, 0, .8);
        inner += mvRect(PL, y + rowH - 8, bl(preds[i]) - PL, 4,
          isOut ? "#eab308" : "#22d3ee");
        inner += mvTxt(W - 74, y + rowH - 10,
          `${gts[i]}→${preds[i]}（${errs[i] >= 0 ? "+" : ""}${errs[i]}）`,
          isOut ? "#eab308" : "#8b9bab", 9.5);
      });
      canvas.innerHTML = mvSvg(W, H, inner);
      const noOut = errs.filter((_, i) => i !== OUT);
      read.innerHTML =
        mvStat("MAE", mae.toFixed(1)) +
        mvStat("RMSE", rmse.toFixed(1)) +
        mvStat("MAE(去掉第7张)", (noOut.reduce((a, e) => a + Math.abs(e), 0) / noOut.length).toFixed(1)) +
        mvStat("RMSE(去掉第7张)", Math.sqrt(noOut.reduce((a, e) => a + e * e, 0) / noOut.length).toFixed(1)) +
        `<span class="mv-stat">灰条=GT　<b style="color:#22d3ee">━</b> 预测　<b style="color:#eab308">━</b> 密集人群离群</span>`;
    };
    ctrl.innerHTML = mvSlider("mvMaeBias", -30, 30, 1, 0, t("mv.ctrl.bias"));
    const slider = ctrl.querySelector("#mvMaeBias");
    const sync = () => {
      const v = parseFloat(slider.value);
      ctrl.querySelector("#mvMaeBiasVal").textContent = (v > 0 ? "+" : "") + v;
      draw(v);
    };
    slider.addEventListener("input", sync);
    sync();
  },
};

/* ===== 人群计数指标（MAE / RMSE） ===== */
const MV_SECTIONS = [
  { labelKey: "mv.sec.crowd", metrics: [MV_MAE] },
];

function mvBuildCard(m) {
  const card = document.createElement("div");
  card.className = "metric-card";
  card.innerHTML =
    `<div class="mv-head"><h4>${esc(t(m.nameKey))}</h4>` +
    `<span class="mv-tag">${esc(t(m.tagKey))}</span></div>` +
    `<div class="mv-body"><div class="mv-doc">` +
    `<div class="mv-formula">${m.formula}</div>` +
    `<p class="mv-intro">${esc(t(m.introKey))}</p></div>` +
    `<div class="mv-viz"><div class="mv-canvas" id="mvCanvas_${m.id}"></div>` +
    `<div class="mv-ctrl" id="mvCtrl_${m.id}"></div>` +
    `<div class="mv-readout" id="mvRead_${m.id}"></div></div></div>`;
  m.build(
    card.querySelector(`#mvCanvas_${m.id}`),
    card.querySelector(`#mvCtrl_${m.id}`),
    card.querySelector(`#mvRead_${m.id}`));
  return card;
}

function initMetricsPanel(force) {
  const box = document.getElementById("metricsPanelBody");
  if (!box) return;
  if (_mvBuilt && !force) return;
  _mvBuilt = true;
  box.innerHTML = "";
  for (const sec of MV_SECTIONS) {
    const el = document.createElement("div");
    el.className = "mv-section";
    el.innerHTML = `<h3>${esc(t(sec.labelKey))}</h3>`;
    const grid = document.createElement("div");
    grid.className = "mv-grid";
    el.appendChild(grid);
    box.appendChild(el);
    for (const m of sec.metrics) grid.appendChild(mvBuildCard(m));
  }
}

/* ---------------- 后台日志（stdout/stderr 实时捕获） ---------------- */

const blogState = {
  timer: null,          // 轮询定时器
  lastId: 0,            // 增量游标（上次已读记录 id）
  lines: [],            // 已拉取的日志行（上限截断）
  maxLines: 2000,       // 界面保留行数上限
  pollInterval: 10000,  // 轮询间隔（10 秒，避免频繁 DOM 重绘卡顿）
  lastUpdate: null,     // 上次刷新时间
};

async function loadBackendLogs() {
  const box = $("#backendLogBox");
  if (!box) return;
  try {
    const q = new URLSearchParams({ tail: 200, cursor: blogState.lastId });
    const r = await api(`/api/system/logs?${q}`);
    if (r.logs && r.logs.length) {
      blogState.lines.push(...r.logs);
      if (blogState.lines.length > blogState.maxLines)
        blogState.lines = blogState.lines.slice(-blogState.maxLines);
    }
    blogState.lastUpdate = new Date();
    renderBackendLogs();
    updateBlogTimestamp();
  } catch (e) { /* 轮询失败静默（下轮重试） */ }
}

function updateBlogTimestamp() {
  const el = $("#blogLastUpdate");
  if (el && blogState.lastUpdate) {
    el.textContent = blogState.lastUpdate.toLocaleTimeString();
  }
}

function startBlogPolling() {
  stopBlogPolling();
  loadBackendLogs();
  blogState.timer = setInterval(loadBackendLogs, blogState.pollInterval);
}

function stopBlogPolling() {
  if (blogState.timer) { clearInterval(blogState.timer); blogState.timer = null; }
}

function renderBackendLogs() {
  const box = $("#backendLogBox");
  if (!box) return;
  const stream = $("#blogStream") ? $("#blogStream").value : "all";
  const kw = ($("#blogFilter") ? $("#blogFilter").value : "").trim().toLowerCase();
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  let html = "";
  for (const it of blogState.lines) {
    if (stream !== "all" && it.stream !== stream) continue;
    if (kw && !String(it.text).toLowerCase().includes(kw)) continue;
    const cls = it.stream === "err" ? "blog-err" : "blog-out";
    html += `<span class="blog-ts">${esc(it.ts)}</span> ` +
            `<span class="${cls}">[${it.stream === "err" ? "ERR" : "OUT"}]</span> ` +
            `${esc(it.text)}\n`;
  }
  box.innerHTML = html || `<span class="blog-ts">${esc(t("blog.empty"))}</span>`;
  const follow = $("#blogFollow") ? $("#blogFollow").checked : true;
  if (follow && atBottom) box.scrollTop = box.scrollHeight;
}

async function clearBackendLogs() {
  if (!window.confirm(t("blog.confirmClear"))) return;
  try {
    await api("/api/system/logs", { method: "DELETE" });
    blogState.lines = [];
    blogState.lastId = 0;
    renderBackendLogs();
    toast(t("blog.cleared"), "success");
  } catch (e) { toast(e.message, "error"); }
}

async function downloadBackendLogs() {
  try {
    const r = await api("/api/system/logs?tail=1000");
    const text = (r.logs || [])
      .map(it => `${it.ts} [${it.stream === "err" ? "ERR" : "OUT"}] ${it.text}`)
      .join("\n");
    const blob = new Blob([text || ""], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    const now = new Date(), p = n => String(n).padStart(2, "0");
    a.download = `backend_log_${now.getFullYear()}${p(now.getMonth() + 1)}` +
                 `${p(now.getDate())}_${p(now.getHours())}${p(now.getMinutes())}.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) { toast(e.message, "error"); }
}

function renderHardware(d) {
  const box = $("#hwContent");
  if (!box) return;
  let html = "";
  if (d.psutil_available === false) {
    html += `<div class="notice">${esc(t("hw.psutilMissing"))}</div>`;
  }
  html += `<div class="hw-grid">`;

  // CPU
  if (d.cpu) {
    const c = d.cpu;
    const cores = (c.per_core || []).map((p, i) =>
      `<div class="hw-core"><div class="hw-core-bar"><div class="${hwColorClass(p)}" style="height:${Math.max(4, Math.min(100, p))}%"></div></div><span>${i}</span></div>`).join("");
    html += `<div class="card">
      <h3>${esc(t("hw.cpu"))}</h3>
      <div class="hw-big">${(c.percent ?? 0).toFixed(1)}<small>%</small></div>
      ${hwBar(c.percent)}
      <div class="hw-kv"><span class="k">${esc(t("hw.physicalCores"))}</span><span>${c.physical_cores ?? "-"} / ${c.logical_cores ?? "-"}</span></div>
      ${c.freq_mhz ? `<div class="hw-kv"><span class="k">${esc(t("hw.freq"))}</span><span>${c.freq_mhz} MHz${c.freq_max_mhz ? " / " + c.freq_max_mhz + " MHz" : ""}</span></div>` : ""}
      ${cores ? `<div class="hw-cores">${cores}</div><p class="hint" style="margin:6px 0 0">${esc(t("hw.perCore"))}</p>` : ""}
    </div>`;
  }

  // 内存
  if (d.memory) {
    const m = d.memory;
    html += `<div class="card">
      <h3>${esc(t("hw.memory"))}</h3>
      <div class="hw-big">${m.used_gb}<small> / ${m.total_gb} GB</small></div>
      ${hwBar(m.percent)}
      <div class="hw-kv"><span class="k">${esc(t("hw.available"))}</span><span>${m.available_gb} GB</span></div>
      <div class="hw-kv"><span class="k">${esc(t("hw.util"))}</span><span>${m.percent}%</span></div>
    </div>`;
  }

  // GPU
  if (d.gpus && d.gpus.length) {
    d.gpus.forEach(g => {
      const memPct = g.memory_percent ?? 0;
      html += `<div class="card">
        <h3>${esc(t("hw.gpu"))} ${g.index} · ${esc(g.name)}</h3>
        <div class="hw-kv"><span class="k">${esc(t("hw.gpuMem"))}</span>
          <span>${g.memory_used_mb ?? "-"} / ${g.memory_total_mb ?? "-"} MB${g.memory_percent != null ? ` (${g.memory_percent}%)` : ""}</span></div>
        ${hwBar(memPct)}
        <div class="hw-kv"><span class="k">${esc(t("hw.gpuUtil"))}</span><span>${g.utilization ?? "-"}%</span></div>
        ${hwBar(g.utilization ?? 0)}
        ${g.temperature != null ? `<div class="hw-kv"><span class="k">${esc(t("hw.temp"))}</span><span>${g.temperature} °C</span></div>` : ""}
      </div>`;
    });
  } else {
    html += `<div class="card"><h3>${esc(t("hw.gpu"))}</h3>
      <p class="hint">${esc(t("hw.noGpu"))}</p></div>`;
  }

  // 磁盘
  (d.disks || []).forEach(k => {
    html += `<div class="card">
      <h3>${esc(t("hw.disk"))} · ${esc(k.device)}</h3>
      <div class="hw-big">${k.used_gb}<small> / ${k.total_gb} GB</small></div>
      ${hwBar(k.percent)}
      <div class="hw-kv"><span class="k">${esc(t("hw.util"))}</span><span>${k.percent}% · ${esc(k.fs)}</span></div>
    </div>`;
  });

  // 网络
  if (d.network) {
    const r = hwState.rates;
    html += `<div class="card">
      <h3>${esc(t("hw.network"))}</h3>
      <div class="hw-kv"><span class="k">↓ ${esc(t("hw.download"))}</span><span>${fmtRate(r && r.down)}</span></div>
      ${hwBar(r ? Math.min(100, r.down / 1250000 * 100) : 0)}
      <div class="hw-kv"><span class="k">↑ ${esc(t("hw.upload"))}</span><span>${fmtRate(r && r.up)}</span></div>
      ${hwBar(r ? Math.min(100, r.up / 1250000 * 100) : 0)}
      <div class="hw-kv"><span class="k">${esc(t("hw.cumulative"))}</span><span>↓ ${fmtBytes(d.network.bytes_recv)} / ↑ ${fmtBytes(d.network.bytes_sent)}</span></div>
    </div>`;
  }

  // 系统信息
  const s = d.system || {};
  html += `<div class="card">
    <h3>${esc(t("hw.system"))}</h3>
    <div class="hw-kv"><span class="k">${esc(t("hw.hostname"))}</span><span>${esc(s.hostname || "-")}</span></div>
    <div class="hw-kv"><span class="k">${esc(t("hw.os"))}</span><span>${esc(s.os || "-")}</span></div>
    ${s.boot_at ? `<div class="hw-kv"><span class="k">${esc(t("hw.bootAt"))}</span><span>${esc(s.boot_at)}</span></div>` : ""}
    ${s.uptime_hours != null ? `<div class="hw-kv"><span class="k">${esc(t("hw.uptime"))}</span><span>${fmtUptime(s.uptime_hours)}</span></div>` : ""}
    <div class="hw-kv"><span class="k">${esc(t("hw.python"))}</span><span>${esc(s.python || "-")}</span></div>
    <div class="hw-kv"><span class="k">${esc(t("hw.torch"))}</span><span>${esc(s.torch || "-")}</span></div>
    <div class="hw-kv"><span class="k">${esc(t("hw.cuda"))}</span><span>${s.cuda_available ? esc(t("hw.yes")) : esc(t("hw.no"))}</span></div>
  </div>`;

  html += `</div>`;
  box.innerHTML = html;
}

/* ---------------- 全局溢出提示 ---------------- */
/* 悬停时检测 scrollWidth > clientWidth，浮层显示被截断的完整文本 */

let _overflowTip = null;

function _hideOverflowTip() {
  if (_overflowTip) { _overflowTip.remove(); _overflowTip = null; }
}

function _showOverflowTip(el) {
  _hideOverflowTip();
  const text = (el.textContent || "").replace(/\s+/g, " ").trim();
  if (!text) return;
  _overflowTip = document.createElement("div");
  _overflowTip.className = "overflow-tip";
  _overflowTip.textContent = text;
  document.body.appendChild(_overflowTip);
  const r = el.getBoundingClientRect();
  const tw = _overflowTip.offsetWidth, th = _overflowTip.offsetHeight;
  const x = Math.max(8, Math.min(r.left, window.innerWidth - tw - 12));
  let y = r.top - th - 8;
  if (y < 8) y = Math.min(r.bottom + 8, window.innerHeight - th - 8);
  _overflowTip.style.left = x + "px";
  _overflowTip.style.top = Math.max(8, y) + "px";
}

function _overflowTruncated(el) {
  if (!(el instanceof Element)) return false;
  const tag = el.tagName;
  // 原生控件自带完整值提示；已有 title 的走浏览器原生提示
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" ||
      tag === "OPTION") return false;
  if (el.title) return false;
  const cs = getComputedStyle(el);
  const ox = cs.overflowX;
  if (ox !== "hidden" && ox !== "clip") return false;
  return el.scrollWidth > el.clientWidth + 1;   // 内容确实被截断
}

function bindOverflowTooltip() {
  document.addEventListener("mouseover", (e) => {
    // 从事件目标向上找第一个被截断的元素（最多 6 层）
    let el = e.target;
    for (let i = 0; el && el !== document.body && i < 6; i++) {
      if (_overflowTruncated(el)) {
        if (!(_overflowTip && _overflowTip._for === el)) {
          _showOverflowTip(el);
          if (_overflowTip) _overflowTip._for = el;
        }
        return;
      }
      el = el.parentElement;
    }
    _hideOverflowTip();
  });
  document.addEventListener("mouseout", (e) => {
    if (!e.relatedTarget) _hideOverflowTip();   // 移出窗口
  });
  window.addEventListener("scroll", _hideOverflowTip, true);
  window.addEventListener("resize", _hideOverflowTip);
}


/* ---- 摄像头实时计数弹窗 ---- */

const camState = { cfg: null, running: false, pollTimer: null };

function cameraApiBase(fw) {
  return "/api/crowd";
}

function openCameraModal(cfg) {
  camState.cfg = cfg;
  const modal = document.getElementById("cameraModal");
  if (!modal) return;
  const sub = document.getElementById("cameraModalSub");
  if (sub) {
    sub.textContent = `PyTorch · ${cfg.taskLabel || cfg.task}`;
  }
  const status = document.getElementById("cameraStatus");
  if (status) status.textContent = "";
  const img = document.getElementById("cameraStreamImg");
  if (img) img.removeAttribute("src");
  const ph = document.getElementById("cameraPlaceholder");
  if (ph) ph.style.display = "";
  modal.hidden = false;
}

function closeCameraModal() {
  stopCamera(true);
  const modal = document.getElementById("cameraModal");
  if (modal) modal.hidden = true;
}

async function startCamera() {
  const cfg = camState.cfg;
  if (!cfg) return;
  const btn = document.getElementById("btnCameraStart");
  const status = document.getElementById("cameraStatus");
  if (btn) btn.disabled = true;
  if (status) status.textContent = t("camera.starting");
  const res = (document.getElementById("camResolution") || {}).value || "";
  const parts = res ? res.split("x").map(Number) : [0, 0];
  const body = {
    task: cfg.task,
    model: cfg.model,
    weight_path: cfg.weightPath || "",
    device: cfg.device || "cpu",
    conf_threshold: isNaN(cfg.conf) ? 0.5 : cfg.conf,
    iou_threshold: isNaN(cfg.iou) ? 0.45 : cfg.iou,
    topk: 3,
    pretrained: !!cfg.pretrained,
    min_neighbors: cfg.minNeighbors || 5,
    camera_index:
      parseInt((document.getElementById("camIndex") || {}).value || "0", 10) || 0,
    width: parts[0] || 0, height: parts[1] || 0,
    lang: (window.DLATF_I18N && window.DLATF_I18N.lang) || "zh",
  };
  try {
    const r = await api(cameraApiBase(cfg.framework) + "/camera/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    camState.running = true;
    const ph = document.getElementById("cameraPlaceholder");
    if (ph) ph.style.display = "none";
    const img = document.getElementById("cameraStreamImg");
    if (img) {
      img.onerror = () => {
        if (!camState.running) return;
        const ph2 = document.getElementById("cameraPlaceholder");
        if (ph2) ph2.style.display = "";
        if (status) status.textContent = t("camera.streamFailed");
      };
      img.src = cameraApiBase(cfg.framework) +
        "/camera/stream?t=" + Date.now();
    }
    if (status) status.textContent = t("camera.running");
    if (camState.pollTimer) clearInterval(camState.pollTimer);
    camState.pollTimer = setInterval(pollCameraStatus, 1000);
  } catch (e) {
    if (status) status.textContent = "";
    toast(tf("camera.startFailed", e.message), "error", 8000);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function stopCamera(silent) {
  const cfg = camState.cfg;
  if (camState.pollTimer) {
    clearInterval(camState.pollTimer);
    camState.pollTimer = null;
  }
  const img = document.getElementById("cameraStreamImg");
  if (img) img.removeAttribute("src");
  const ph = document.getElementById("cameraPlaceholder");
  const status = document.getElementById("cameraStatus");
  if (!cfg || !camState.running) {
    if (!silent && status) status.textContent = t("camera.stopped");
    return;
  }
  camState.running = false;
  try {
    await api(cameraApiBase(cfg.framework) + "/camera/stop", { method: "POST" });
  } catch (_) { /* 静默 */ }
  if (!silent) {
    if (status) status.textContent = t("camera.stopped");
    if (ph) ph.style.display = "";
  }
}

async function pollCameraStatus() {
  const cfg = camState.cfg;
  if (!cfg || !camState.running) return;
  let s;
  try { s = await api(cameraApiBase(cfg.framework) + "/camera/status"); }
  catch (_) { return; }
  const status = document.getElementById("cameraStatus");
  if (!status) return;
  if (!s || !s.session) {
    // 会话已不存在（摄像头断开 / 服务端停止 / 被新会话替换）
    camState.running = false;
    if (camState.pollTimer) {
      clearInterval(camState.pollTimer);
      camState.pollTimer = null;
    }
    const img = document.getElementById("cameraStreamImg");
    if (img) img.removeAttribute("src");
    const ph = document.getElementById("cameraPlaceholder");
    if (ph) ph.style.display = "";
    status.textContent = t("camera.sessionLost");
    return;
  }
  if (s.running) {
    status.textContent = tf("camera.statusFmt",
      t("camera.running"), s.avg_fps, s.frames);
  } else {
    camState.running = false;
    if (camState.pollTimer) {
      clearInterval(camState.pollTimer);
      camState.pollTimer = null;
    }
    const img = document.getElementById("cameraStreamImg");
    if (img) img.removeAttribute("src");
    const ph = document.getElementById("cameraPlaceholder");
    if (ph) ph.style.display = "";
    status.textContent = s.error
      ? tf("camera.stoppedByError", s.error) : t("camera.stopped");
  }
}

/* 模态框按钮绑定（脚本在 body 末尾加载，DOM 已就绪） */
(function bindCameraModal() {
  const startBtn = document.getElementById("btnCameraStart");
  const stopBtn = document.getElementById("btnCameraStop");
  const closeBtn = document.getElementById("btnCameraClose");
  const modal = document.getElementById("cameraModal");
  if (startBtn) startBtn.addEventListener("click", startCamera);
  if (stopBtn) stopBtn.addEventListener("click", () => stopCamera(false));
  if (closeBtn) closeBtn.addEventListener("click", closeCameraModal);
  if (modal) modal.addEventListener("click", e => {
    if (e.target === modal) closeCameraModal();
  });
})();

/* ---------------- 「可视化」全局模块 ---------------- */

let _visBound = false;

function bindVisualizePanel() {
  if (_visBound) return;
  _visBound = true;
  const modeSel = $("#visMode");
  if (!modeSel) return;
  modeSel.addEventListener("change", () => updateVisUI());
  const routeSel = $("#visRoute");
  if (routeSel) routeSel.addEventListener("change", () => updateVisUI());
  const btnSingle = $("#btnVisSingle");
  if (btnSingle) btnSingle.addEventListener("click", visSingle);
  const btnBatch = $("#btnVisBatch");
  if (btnBatch) btnBatch.addEventListener("click", visBatch);
  const folderInput = $("#visFolderFiles");
  if (folderInput) {
    folderInput.addEventListener("change", (e) => {
      state.visFolderFiles = Array.from(e.target.files || [])
        .filter(f => f.type.startsWith("image/"));
      const st = $("#visBatchStatus");
      const dirInput = $("#visImageDir");
      if (dirInput && state.visFolderFiles.length) dirInput.value = "";
      if (st) st.textContent = state.visFolderFiles.length
        ? tf("vis.msg.folderSelected", state.visFolderFiles.length)
        : "";
    });
  }
  const imageDirInput = $("#visImageDir");
  if (imageDirInput) {
    imageDirInput.addEventListener("input", () => {
      if (!imageDirInput.value.trim()) return;
      state.visFolderFiles = [];
      if (folderInput) folderInput.value = "";
    });
  }
}

function updateVisUI() {
  const mode = $("#visMode") ? $("#visMode").value : "label";
  const route = $("#visRoute") ? $("#visRoute").value : "crowd_density";
  const isLabel = mode === "label";
  const isBox = route === "crowd_box";
  const show = (sel, on) => {
    const el = $(sel);
    if (el) el.style.display = on ? "" : "none";
  };
  show("#visLabelFileField", isLabel);
  show("#visLabelFormatField", isLabel && isBox);
  show("#visModelField", !isLabel);
  show("#visWeightField", !isLabel);
  show("#visLabelDirField", isLabel);
  // 设备/置信度阈值仅模型模式有意义；阈值仅点预测与头部框路线使用
  show("#visDeviceField", !isLabel);
  show("#visConfField", !isLabel && route !== "crowd_density");
}

// 标签格式按路线推导：密度图→npy、点预测→mat、头部框→下拉显选（txt/xml）。
// #visLabelFormat 在非头部框路线隐藏，但其残留值不能作为请求参数
function visLabelFormatOf(route) {
  if (route === "crowd_density") return "npy";
  if (route === "crowd_point") return "mat";
  const sel = $("#visLabelFormat");
  return sel ? sel.value : "yolo_txt";
}

async function loadVisualizePanel() {
  updateVisUI();
  // 填充模型下拉（模型推理可视化模式）；选择模型时按其所属路线自动同步
  // 「技术路线」，避免路线×模型错配被后端拒绝
  try {
    const tasks = await api("/api/crowd/tasks");
    const sel = $("#visModel");
    if (sel && tasks.length) {
      sel.innerHTML = tasks.map(t => {
        const models = t.models || [];
        return models.map(m =>
          `<option value="${esc(m.name)}" data-route="${esc(t.key)}">${esc(m.name)} — ${esc(t.label)}</option>`).join("");
      }).join("");
      if (!sel._visRouteSync) {
        sel._visRouteSync = true;
        sel.addEventListener("change", () => {
          const opt = sel.selectedOptions && sel.selectedOptions[0];
          const r = opt && opt.dataset.route;
          const rs = $("#visRoute");
          if (r && rs && rs.value !== r) {
            rs.value = r;
            updateVisUI();
          }
        });
      }
    }
    // 权重下拉：训练产物 + 上传的权重（与推理面板口径一致）
    const up = await api("/api/uploads");
    const wSel = $("#visWeight");
    if (wSel) {
      const wts = [{ name: "", path: "" },
                   ...up.trained_weights, ...up.models];
      fillSelect("#visWeight", wts, w => w.path || w.name,
        w => w.name ? `${w.name} (${w.size_mb}MB)` : t("common.dontUse"));
    }
    // 计算设备下拉（与推理面板共用 /api/crowd/options）
    const opt = await api("/api/crowd/options");
    const devs = (opt && opt.devices) || [{ id: "cpu", label: "CPU" }];
    fillSelect("#visDevice", devs, d => d.id, d => d.label);
  } catch (e) { /* 静默 */ }
}

async function visSingle() {
  const imgFile = $("#visImageFile") && $("#visImageFile").files[0];
  if (!imgFile) return toast(t("vis.msg.imageRequired"), "error");
  const mode = $("#visMode") ? $("#visMode").value : "label";
  const route = $("#visRoute") ? $("#visRoute").value : "crowd_density";
  const formData = new FormData();
  formData.append("mode", mode);
  formData.append("route", route);
  formData.append("image", imgFile);
  if (mode === "label") {
    const lblFile = $("#visLabelFile") && $("#visLabelFile").files[0];
    if (!lblFile) return toast(t("vis.msg.labelRequired"), "error");
    formData.append("label_file", lblFile);
    formData.append("label_format", visLabelFormatOf(route));
  } else {
    const model = $("#visModel") ? $("#visModel").value : "";
    if (!model) return toast(t("vis.msg.modelRequired"), "error");
    formData.append("model", model);
    const wp = $("#visWeight") ? $("#visWeight").value : "";
    if (wp) formData.append("weight_path", wp);
    formData.append("device", $("#visDevice") ? $("#visDevice").value : "cpu");
    formData.append("conf_threshold",
      $("#visConf") ? ($("#visConf").value || "0.5") : "0.5");
  }
  const visSaveSingle = $("#visSaveSingle");
  const visSingleSaveDir = $("#visSingleSaveDir");
  if (visSaveSingle && visSaveSingle.checked) formData.append("save_result", "true");
  if (visSingleSaveDir && visSingleSaveDir.value.trim()) {
    formData.append("save_dir", visSingleSaveDir.value.trim());
  }
  const btn = $("#btnVisSingle");
  btn.disabled = true;
  btn.textContent = t("msg.modelInferring");
  const st = $("#visSingleStatus");
  if (st) st.textContent = "";
  try {
    const r = await uploadWithProgress("/api/crowd/visualize/single", formData,
      (p) => { btn.textContent = tf("msg.uploading", p); });
    // 显示结果
    const gallery = $("#visResultGallery");
    const card = $("#visResultCard");
    if (gallery && card) {
      card.style.display = "";
      let html = `<div class="result-count">${esc(tf("res.count", r.count))}</div>`;
      if (r.visualization) {
        html += `<div class="result-images"><figure>` +
          `<img src="data:image/png;base64,${r.visualization}" alt="vis">` +
          `<figcaption>${esc(r.filename || "")}</figcaption></figure></div>`;
      }
      state.lastVis = r.visualization
        ? { b64: r.visualization, name: (r.filename || "visualization") + "_vis.png" }
        : null;
      if (state.lastVis) {
        html += `<div class="actions"><button class="btn small" onclick="downloadLastVis()">${esc(t("res.downloadVis"))}</button></div>`;
      }
      gallery.innerHTML = html;
    }
    if (st) st.textContent = t("vis.msg.done");
  } catch (e) {
    if (st) st.textContent = "";
    toast(tf("msg.inferFailed", e.message), "error", 8000);
  } finally {
    btn.disabled = false;
    btn.textContent = t("vis.singleBtn");
  }
}

function renderVisBatchResult(res) {
  const pv = res.preview_images || [];
  const gallery = $("#visResultGallery");
  const card = $("#visResultCard");
  if (!gallery || !card) return;
  card.style.display = "";
  let html = `<div class="result-count">${esc(
    tf("vis.msg.previewNote", res.processed || 0, pv.length))}</div>`;
  if (pv.length) {
    html += `<div class="result-images">` + pv.map(p =>
      `<figure><img src="data:image/jpeg;base64,${p.vis_b64}" alt="vis">` +
      `<figcaption>${esc(p.image || "")} ? count=${esc(String(p.count))}</figcaption></figure>`
    ).join("") + `</div>`;
  }
  if (res.zip_url) {
    html += `<div class="saved-url">${esc(t("res.download"))}` +
      `<a class="btn small anchor-btn" href="${res.zip_url}" target="_blank">ZIP</a>` +
      `<span style="margin-left:8px">${esc(res.zip_url)}</span></div>`;
  }
  if (res.save_dir) html += `<div class="saved-url">${esc(t("res.saved"))}${esc(res.save_dir)}</div>`;
  gallery.innerHTML = html;
}

async function visUploadBatch(files, mode, route) {
  if (mode !== "model") {
    return toast(t("vis.msg.folderModelOnly"), "error", 6000);
  }
  const model = $("#visModel") ? $("#visModel").value : "";
  if (!model) return toast(t("vis.msg.modelRequired"), "error");
  const fd = new FormData();
  fd.append("mode", "model");
  fd.append("route", route);
  for (const f of files) fd.append("images", f);
  fd.append("model", model);
  const wp = $("#visWeight") ? $("#visWeight").value : "";
  if (wp) fd.append("weight_path", wp);
  fd.append("device", $("#visDevice") ? $("#visDevice").value : "cpu");
  fd.append("conf_threshold", $("#visConf") ? ($("#visConf").value || "0.5") : "0.5");
  const saveResult = $("#visSaveBatch") ? $("#visSaveBatch").checked : true;
  fd.append("save_result", saveResult ? "true" : "false");
  const saveDir = $("#visSaveDir") ? $("#visSaveDir").value.trim() : "";
  if (saveDir) fd.append("save_dir", saveDir);
  const btn = $("#btnVisBatch");
  btn.disabled = true;
  btn.textContent = t("msg.starting");
  const st = $("#visBatchStatus");
  if (st) st.textContent = t("vis.msg.batchRunning");
  try {
    const res = await uploadWithProgress("/api/crowd/visualize/upload_batch", fd,
      p => { btn.textContent = tf("msg.uploading", p); });
    renderVisBatchResult(res);
    if (st) st.textContent = tf("vis.msg.batchDone", res.processed, res.failed);
    toast(tf("vis.msg.batchDone", res.processed, res.failed), "success", 8000);
  } catch (e) {
    if (st) st.textContent = "";
    toast(tf("msg.inferFailed", e.message), "error", 8000);
  } finally {
    btn.disabled = false;
    btn.textContent = t("vis.batchBtn");
  }
}

async function visBatch() {
  const imageDir = $("#visImageDir") ? $("#visImageDir").value.trim() : "";
  const folderFiles = state.visFolderFiles || [];
  if (!imageDir && !folderFiles.length) return toast(t("vis.msg.imageDirRequired"), "error");
  const mode = $("#visMode") ? $("#visMode").value : "label";
  const route = $("#visRoute") ? $("#visRoute").value : "crowd_density";
  if (folderFiles.length) return visUploadBatch(folderFiles, mode, route);
  const formData = new FormData();
  formData.append("mode", mode);
  formData.append("route", route);
  formData.append("image_dir", imageDir);
  if (mode === "label") {
    const labelDir = $("#visLabelDir") ? $("#visLabelDir").value.trim() : "";
    if (!labelDir) return toast(t("vis.msg.labelDirRequired"), "error");
    formData.append("label_dir", labelDir);
    formData.append("label_format", visLabelFormatOf(route));
  } else {
    const model = $("#visModel") ? $("#visModel").value : "";
    if (!model) return toast(t("vis.msg.modelRequired"), "error");
    formData.append("model", model);
    const wp = $("#visWeight") ? $("#visWeight").value : "";
    if (wp) formData.append("weight_path", wp);
    formData.append("device", $("#visDevice") ? $("#visDevice").value : "cpu");
    formData.append("conf_threshold",
      $("#visConf") ? ($("#visConf").value || "0.5") : "0.5");
  }
  const saveDir = $("#visSaveDir") ? $("#visSaveDir").value.trim() : "";
  if (saveDir) formData.append("save_dir", saveDir);
  const saveBatch = $("#visSaveBatch");
  formData.append("save_result", saveBatch ? String(saveBatch.checked) : "true");

  const btn = $("#btnVisBatch");
  btn.disabled = true;
  btn.textContent = t("msg.starting");
  const st = $("#visBatchStatus");
  const wrap = $("#visProgressWrap");
  const bar = $("#visProgressBar");
  const log = $("#visLog");
  if (wrap) wrap.style.display = "";
  if (log) { log.style.display = ""; log.textContent = ""; }
  if (st) st.textContent = t("vis.msg.batchRunning");
  try {
    const r = await uploadWithProgress("/api/crowd/visualize/batch", formData,
      (p) => { btn.textContent = tf("msg.uploading", p); });
    const tid = r.task_id;
    const timer = setInterval(async () => {
      try {
          const task = await api(`/api/crowd/train/tasks/${tid}`);
          if (bar) bar.style.width = (task.progress || 0) + "%";
          if (st) st.textContent =
          `${statusLabel(task.status, task.kind)} ${task.epoch}/${task.total_epochs}`;
          const lr = await api(`/api/crowd/train/tasks/${tid}/log?tail=10`);
        if (log) log.textContent = lr.log || "";
        if (["completed", "failed", "stopped"].includes(task.status)) {
          clearInterval(timer);
          btn.disabled = false;
          btn.textContent = t("vis.batchBtn");
          if (task.status === "completed" && task.result) {
            const res = task.result;
            toast(tf("vis.msg.batchDone", res.processed, res.failed), "success", 8000);
            if (st) st.textContent = tf("vis.msg.batchDone", res.processed, res.failed);
            renderVisBatchResult(res);
          } else if (task.status === "failed") {
            toast(tf("msg.startFailed", task.error || ""), "error", 10000);
            if (st) st.textContent = task.error || t("status.failed");
          }
        }
      } catch (_) { /* 静默 */ }
    }, 2000);
  } catch (e) {
    btn.disabled = false;
    btn.textContent = t("vis.batchBtn");
    if (st) st.textContent = "";
    toast(tf("msg.startFailed", e.message), "error", 8000);
  }
}

/* ---------------- 学习版图 ---------------- */

// 数据与 Codes/README.md「参考与致谢」保持一致；datasets 为人群计数常用数据集（链接 + 简介）
const LEARNING_MAP_DATA = [
  {
    icon: "👥", titleKey: "learn.crowd", defaultTitle: "人群计数",
    datasets: [
      { type: "dataset", title: "ShanghaiTech (Part A/B)", url: "https://www.datafountain.cn/datasets/5670", desc: "1198 张图、33 万点标注（A 密集 / B 稀疏）；人群计数最常用基准——本框架内置 SHTechA 预训练检查点，.mat 标注可用内置工具直接转密度图（点击直达下载页）", descEn: "1,198 images, 330k point annotations (A dense / B sparse); the most-used benchmark — SHTechA checkpoint bundled, .mat annotations converted with built-in tool (click to download page)" },
      { type: "dataset", title: "UCF-QNRF", url: "https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip", desc: "1535 张高分辨率高密度人群图、约 125 万点标注；评测密集场景 MAE 的困难基准（点击直接下载 zip）", descEn: "1,535 high-resolution dense images, ~1.25M points; hard benchmark for dense MAE (click for direct zip download)" },
      { type: "dataset", title: "NWPU-Crowd", url: "https://gjy3035.github.io/NWPU-Crowd-Sample-Code/", desc: "5109 张图、213 万标注点（含负样本/框标注）；目前规模最大的人群计数基准（点击进入下载页）", descEn: "5,109 images, 2.13M points (incl. negatives/boxes); the largest benchmark (click for download page)" },
      { type: "dataset", title: "UCF_CC_50", url: "https://www.crcv.ucf.edu/data/ucf-cc-50/", desc: "50 张极高密度人群图（单图 1279~4633 人）；小规模困难基准（点击进入官网下载页，官方发布为 .rar 压缩包）", descEn: "50 extremely dense images (1,279-4,633 each); small hard benchmark (click for official download page, distributed as .rar)" },
      { type: "dataset", title: "JHU-CROWD++", url: "https://www.crowd-counting.com/", desc: "4372 张图、150 万标注点、含头部框 + 天气标签 + 遮挡/模糊标注；多样化场景基准（点击进入官网下载页：Dropbox / Google Drive / OneDrive）", descEn: "4,372 images, 1.5M points with head boxes + weather + occlusion/blur tags; diverse-scene benchmark (click for official download page: Dropbox / Google Drive / OneDrive)" },
    ],
    links: [
      { type: "github", title: "CrowdCounting-framework-PyTorch", url: "https://github.com/KeepTryingTo/CrowdCounting-framework-PyTorch", desc: "基于密度图：损失函数 MSE/MAE/SSIM/MMD/OT + 可视化（本框架 nets/crowd/ 与 crowd_losses.py 的移植来源）", descEn: "Density-map based: losses MSE/MAE/SSIM/MMD/OT + visualization (porting source of this framework nets/crowd/ and crowd_losses.py)" },
      { type: "github", title: "CrowdCounting-P2PNet", url: "https://github.com/TencentYilianResearch/CrowdCounting-P2PNet", desc: "基于点预测：P2PNet（CVPR 2022）（本框架 nets/p2pnet/ 的 vendor 来源）", descEn: "Point-based: P2PNet (CVPR 2022) (vendor source of this framework nets/p2pnet/)" },
      { type: "github", title: "Uncertainty-Semi-Supervised-Crowd-Counting", url: "https://github.com/KeepTryingTo/Uncertainty-Semi-Supervised-Crowd-Counting", desc: "CU：Calibrating Uncertainty（ICCV 2023 半监督点预测）——本框架 nets/cu/ 的 vendor 来源", descEn: "CU: Calibrating Uncertainty (ICCV 2023 semi-supervised point-based counting) — vendor source of this framework nets/cu/" },
      { type: "github", title: "gjy3035/Awesome-Crowd-Counting", url: "https://github.com/gjy3035/Awesome-Crowd-Counting", desc: "人群计数论文与代码大全（C-3-Framework 统一评测）", descEn: "Awesome crowd counting papers & code list (C-3-Framework unified evaluation)" },
      { type: "github", title: "cvlab-stonybrook/DM-Count", url: "https://github.com/cvlab-stonybrook/DM-Count", desc: "DM-Count：分布匹配人群计数（计数 + Sinkhorn OT + TV 损失；crowd_losses.py 的 DMCountLoss 完整损失与 OT_Loss 参考实现）", descEn: "DM-Count: distribution-matching crowd counting (counting + Sinkhorn OT + TV losses; reference of DMCountLoss and OT_Loss in crowd_losses.py)" },
      { type: "github", title: "GhostNet (Efficient-AI-Backbones)", url: "https://github.com/huawei-noah/Efficient-AI-Backbones", desc: "GhostNet v1/v2：内置密度网默认骨干（华为诺亚方舟实验室）", descEn: "GhostNet v1/v2: default backbone of the built-in density net (Huawei Noah Ark Lab)" },
    ],
  },
  {
    icon: "▶", titleKey: "learn.videos", defaultTitle: "人群计数 · 视频（Bilibili）",
    links: [
      { type: "video", title: "Bilibili 个人空间 · keep_trying_go（更多人群计数视频）", url: "https://space.bilibili.com/625095571/video", desc: "作者 Bilibili 视频主页；人群计数、目标统计与论文解读视频按主页视频列表查看。", descEn: "Author's Bilibili video space; browse the video list for crowd counting, object counting and paper walkthroughs." },
      { type: "video", title: "人群计数领域近十年论文和代码综述", url: "https://www.bilibili.com/video/BV1yqWg6cEjG", desc: "梳理人群计数近十年代表论文、代码与数据集，适合作为学习路线总览。", descEn: "A survey of representative crowd-counting papers, code and datasets from the last decade; useful as a learning roadmap." },
      { type: "video", title: "基于主动学习的半监督人群计数算法", url: "https://www.bilibili.com/video/BV1hFJt61Ehg", desc: "讲解基于主动学习的半监督人群计数：样本选择、标注效率与训练流程。", descEn: "Active-learning based semi-supervised crowd counting: sample selection, annotation efficiency and training pipeline." },
      { type: "video", title: "基于单域泛化的目标统计 URM 算法详解", url: "https://www.bilibili.com/video/BV16pDjBuE4r", desc: "单域泛化目标统计 URM 论文解读，涉及视觉-语言通用表示与跨域计数。", descEn: "URM for single-domain-generalized object counting, covering vision-language universal representations and cross-domain counting." },
    ],
  },
  {
    icon: "✎", titleKey: "learn.blogs", defaultTitle: "人群计数 · 博客（CSDN）",
    links: [
      { type: "csdn", title: "CSDN 博客主页 · Keep_Trying_Go", url: "https://blog.csdn.net/Keep_Trying_Go?type=blog", desc: "作者 CSDN 博客主页，包含人群计数论文解读、复现实验与工程实践文章。", descEn: "Author's CSDN blog with crowd-counting paper notes, reproductions and engineering posts." },
      { type: "csdn", title: "D2C：Decoupled Two-Stage Crowd Counting 方法与实验", url: "https://blog.csdn.net/Keep_Trying_Go/article/details/170809078", desc: "D2C 解耦两阶段人群计数：PMR 概率图、CMR 计数图、LCG-PPD 定位与实验结果。", descEn: "D2C decoupled two-stage counting: PMR probability map, CMR count map, LCG-PPD localization and experiments." },
      { type: "csdn", title: "RAZ：Recurrent Attentive Zooming 论文方法与实验", url: "https://blog.csdn.net/Keep_Trying_Go/article/details/170776483", desc: "RAZ 循环注意力缩放：Main-Net、RAZ-Net、局部放大与计数/定位融合。", descEn: "RAZ recurrent attentive zooming: Main-Net, RAZ-Net, local zoom and counting/localization fusion." },
      { type: "csdn", title: "PET 人群计数论文方法与实验详解", url: "https://blog.csdn.net/Keep_Trying_Go/article/details/170289534", desc: "PET 点查询四叉树与渐进矩形窗口注意力，覆盖计数、定位及多任务设置。", descEn: "PET point-query quadtree and progressive rectangular-window attention for counting, localization and multi-task settings." },
      { type: "csdn", title: "SAL 尺度感知主动学习人群计数", url: "https://blog.csdn.net/Keep_Trying_Go/article/details/170289502", desc: "SAL 通过尺度查询熵/散度筛选高价值样本，提升半监督标注效率。", descEn: "SAL uses scale-query entropy/divergence to select informative samples for semi-supervised annotation efficiency." },
      { type: "csdn", title: "P2RLoss 论文方法与实验详解", url: "https://blog.csdn.net/Keep_Trying_Go/article/details/170289436", desc: "P2RLoss 点到区域匹配与背景置信度传播，缓解半监督点检测的背景梯度缺失。", descEn: "P2RLoss Point-to-Region matching and background confidence propagation for semi-supervised point detection." },
      { type: "csdn", title: "Consistent-Point 半监督人群计数算法详解", url: "https://blog.csdn.net/Keep_Trying_Go/article/details/169744882", desc: "Consistent-Point 的 PA 位置聚合与 IUC 不确定性校准，解决伪点位置/类别不一致。", descEn: "Consistent-Point PA position aggregation and IUC uncertainty calibration for pseudo-point inconsistency." },
      { type: "csdn", title: "CU：不确定性半监督人群计数算法", url: "https://blog.csdn.net/Keep_Trying_Go/article/details/169653058", desc: "CU 使用累积空间距离 ASM、排序损失与不确定性分支提升伪标签质量。", descEn: "CU uses accumulated spatial distance ASM, ranking loss and an uncertainty branch to improve pseudo labels." },
      { type: "csdn", title: "Crowd-SAM 人群检测论文详解", url: "https://blog.csdn.net/Keep_Trying_Go/article/details/168236939", desc: "Crowd-SAM 使用 DINOv2 前景热力图与自提示实现少样本密集人群标注。", descEn: "Crowd-SAM uses DINOv2 foreground heatmaps and self-prompting for few-shot dense crowd annotation." },
      { type: "csdn", title: "Exclusivity-Guided Mask Learning 半监督人群实例分割与计数", url: "https://blog.csdn.net/Keep_Trying_Go/article/details/167130146", desc: "EDP-SAM 伪掩码、XMask 排他性学习和结构化伪标签用于人群实例分割与计数。", descEn: "EDP-SAM pseudo masks, XMask exclusivity learning and structured pseudo labels for crowd instance segmentation and counting." },
      { type: "csdn", title: "Multi-modal Crowd Counting via a Broker Modality", url: "https://blog.csdn.net/Keep_Trying_Go/article/details/156830311", desc: "跨 RGB/热成像的代理模态生成与两阶段训练，缓解模态差异和空间未对齐。", descEn: "Broker-modality generation and two-stage training for RGB/thermal crowd counting under modality gaps and misalignment." },
      { type: "csdn", title: "URM：单域泛化少样本目标统计", url: "https://blog.csdn.net/Keep_Trying_Go/article/details/156870771", desc: "URM 使用 MaskCLIP 通用视觉表示与语言提示，提升少样本计数的跨域泛化。", descEn: "URM uses MaskCLIP universal visual representations and language prompts for cross-domain few-shot counting." },
      { type: "csdn", title: "CrowdDiff：基于扩散模型的人群计数", url: "https://blog.csdn.net/Keep_Trying_Go/article/details/157175412", desc: "CrowdDiff 将密度图生成建模为去噪扩散过程，缓解背景噪声与密度分布失真。", descEn: "CrowdDiff models density-map generation as denoising diffusion to reduce background noise and density distortion." },
    ],
  },
];


const _LEARN_BADGE_LABELS = { github: "GitHub", csdn: "CSDN", doc: "Doc", video: "Bilibili", local: "Local", dataset: "Dataset" };

function _learnLinksHtml(links) {
  let html = "";
  for (const link of links) {
    const badgeCls = link.type === "github" ? "github" :
                     link.type === "csdn" ? "csdn" :
                     link.type === "doc" ? "doc" :
                     link.type === "video" ? "video" :
                     link.type === "dataset" ? "dataset" : "local";
    const badgeText = _LEARN_BADGE_LABELS[link.type] || link.type;
    const isLink = !!link.url;
    const desc = (window.DLATF_I18N.lang === "en" && link.descEn) ? link.descEn : link.desc;
    const openTag = isLink
      ? `<a class="learn-link" href="${link.url}" target="_blank" rel="noopener">`
      : `<div class="learn-link">`;
    html += `${openTag}
      <span class="link-badge ${badgeCls}">${badgeText}</span>
      <span class="link-info">
        <span class="link-title">${esc(link.title)}</span>
        <span class="link-url">${esc(link.url || "本地参考路径")}</span>
        ${desc ? `<span class="link-desc">${esc(desc)}</span>` : ""}
      </span>
    ${isLink ? "</a>" : "</div>"}`;
  }
  return html;
}

function renderLearningMap() {
  const box = document.getElementById("learningMap");
  if (!box) return;
  let html = "";
  for (const section of LEARNING_MAP_DATA) {
    const title = t(section.titleKey);
    const sTitle = title === section.titleKey ? section.defaultTitle : title;
    html += `<div class="learn-section">`;
    html += `<h3><span class="learn-icon">${section.icon}</span>${esc(sTitle)}</h3>`;
    html += `<div class="learn-links">`;
    if (section.datasets && section.datasets.length) {
      html += `<div class="learn-group-label">${esc(t("learn.datasets"))}</div>`;
      html += _learnLinksHtml(section.datasets);
      html += `<div class="learn-group-label">${esc(t("learn.refs"))}</div>`;
    }
    html += _learnLinksHtml(section.links);
    html += `</div></div>`;
  }
  box.innerHTML = html;
}

/* ---------------- 启动（init 须在模块级 const 定义之后调用，避免 TDZ） ---------------- */
// 环境自检标志：脚本完整执行至此，index.html 的「app.js 加载失败」告警不再触发
window.__DLATF_READY = true;
init().catch(e => {
  console.error(e);
  toast(tf("msg.initFailed", e.message), "error", 8000);
});
