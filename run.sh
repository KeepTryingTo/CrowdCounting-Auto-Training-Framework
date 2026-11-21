#!/usr/bin/env bash
# Crowd Counting Auto Training Framework - Linux / macOS launcher
# 用法:
#   bash run.sh            # 交互式选择环境并启动
#   bash run.sh --check    # 仅检查依赖，不启动服务
# 首次使用可执行: chmod +x run.sh && ./run.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "=============================================================="
echo "   Crowd Counting Auto Training Framework"
echo "   Project: $SCRIPT_DIR"
echo "=============================================================="
echo

_py_ok() {
  # Python >= 3.7
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 7) else 1)' \
    >/dev/null 2>&1
}

find_python() {
  local p c r

  # 1) run_python.txt（首行；去掉 CR 与首尾空白）
  if [ -f "$SCRIPT_DIR/run_python.txt" ]; then
    p="$(head -n 1 "$SCRIPT_DIR/run_python.txt" | tr -d '\r' \
         | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    if [ -n "$p" ] && [ -x "$p" ] && _py_ok "$p"; then
      echo "$p"; return 0
    fi
  fi

  # 2) PATH 中的 python3 / python
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
      p="$(command -v "$c")"
      if _py_ok "$p"; then echo "$p"; return 0; fi
    fi
  done

  # 3) 常见 conda / miniconda / miniforge 位置
  local roots=()
  [ -n "${CONDA_PREFIX:-}" ] && roots+=("$CONDA_PREFIX")
  roots+=("$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" \
          "$HOME/mambaforge" "/opt/conda" "/opt/anaconda3" \
          "/opt/miniconda3" "/usr/local" "/usr")
  for r in "${roots[@]}"; do
    for p in "$r/bin/python3" "$r/bin/python" "$r/python3" "$r/python"; do
      if [ -x "$p" ] && _py_ok "$p"; then echo "$p"; return 0; fi
    done
  done
  return 1
}

BOOT_PY="$(find_python || true)"
if [ -z "$BOOT_PY" ]; then
  echo "  ERROR: No Python 3.7+ found on this machine."
  echo
  echo "  Please install Anaconda/Miniconda or add Python to PATH."
  echo "  Download: https://www.anaconda.com/download"
  exit 1
fi

echo "  Bootstrap Python: $BOOT_PY"

if [ "${1:-}" = "--check" ]; then
  echo "  Running environment check (no server will be started)..."
  echo
  "$BOOT_PY" - <<'PY'
import importlib.util as u
import sys

mods = ["fastapi", "uvicorn", "PIL", "numpy", "tqdm", "scipy",
        "matplotlib", "torch", "torchvision"]
missing = [m for m in mods if u.find_spec(m) is None]
print("  OK: all dependencies present." if not missing
      else "  MISSING: " + ", ".join(missing))
sys.exit(1 if missing else 0)
PY
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo
    echo "  Environment check FAILED. Install with:"
    echo "      pip install -r requirements.txt"
    exit 1
  fi
  echo "  Environment check passed."
  exit 0
fi

echo "  Starting interactive launcher..."
echo
exec "$BOOT_PY" "$SCRIPT_DIR/launcher.py" "$@"
