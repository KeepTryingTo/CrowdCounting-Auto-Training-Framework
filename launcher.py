# -*- coding: utf-8 -*-
"""Interactive launcher: pick a Python env, check packages, run the server."""
import os
import sys
import socket
import subprocess
import shutil
import time

if sys.platform == "win32":
    import msvcrt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRED = ["fastapi", "uvicorn", "PIL", "numpy", "tqdm", "scipy", "matplotlib"]
FRAMEWORKS = ["torch", "torchvision"]
ALL_CHECK = REQUIRED + FRAMEWORKS

HIDE = 0x08000000 if sys.platform == "win32" else 0

# 服务端口与控制台日志（uvicorn 输出 / 启动失败原因均记录于此）
# 8060：避开其他常用 8000 端口程序（本端口会通过 --port 传给 main.py）
SERVER_PORT = 8060
SERVER_LOG = os.path.join(PROJECT_DIR, "logs", "server_console.log")
_server_log_file = None


def clear_screen():
    os.system("cls" if sys.platform == "win32" else "clear")


def check_pkgs(python_path):
    results = {}
    for pkg in ALL_CHECK:
        # torch/torchvision 冷启动导入可能超过 10s（假性 not installed），
        # 框架类放宽到 30s
        timeout = 30 if pkg in FRAMEWORKS else 10
        try:
            r = subprocess.run(
                [python_path, "-c", "import " + pkg],
                capture_output=True, timeout=timeout, creationflags=HIDE)
            results[pkg] = r.returncode == 0
        except Exception:
            results[pkg] = False
    return results


def get_version(python_path):
    try:
        r = subprocess.run(
            [python_path, "-c",
             "import sys;print('{}.{}.{}'.format(*sys.version_info[:3]))"],
            capture_output=True, text=True, timeout=10, creationflags=HIDE)
        return r.stdout.strip()
    except Exception:
        return ""


def discover_envs():
    envs = []
    seen = set()

    def add(path, label):
        path = os.path.normpath(path)
        if path in seen or not os.path.isfile(path):
            return
        seen.add(path)
        envs.append({"path": path, "label": label})

    txt = os.path.join(PROJECT_DIR, "run_python.txt")
    if os.path.isfile(txt):
        try:
            p = open(txt).read().strip()
            if p and os.path.isfile(p):
                add(p, "run_python.txt (recommended)")
        except Exception:
            pass

    conda_roots = []
    if shutil.which("conda"):
        root = os.path.dirname(os.path.dirname(shutil.which("conda")))
        conda_roots.append(root)
    home = os.path.expanduser("~")
    for d in [r"D:\conda3\anaconda3", r"D:\conda3",
              os.path.join(home, "anaconda3"), os.path.join(home, "miniconda3"),
              os.path.join(home, "miniforge3"), os.path.join(home, "mambaforge"),
              r"C:\ProgramData\Anaconda3", r"C:\ProgramData\miniconda3",
              r"C:\Anaconda3", r"D:\Anaconda3", r"E:\Anaconda3",
              "/opt/conda", "/opt/anaconda3", "/opt/miniconda3"]:
        if os.path.isdir(d):
            conda_roots.append(d)

    win = sys.platform == "win32"
    exe = "python.exe" if win else "python"
    cp = os.environ.get("CONDA_PREFIX")
    if cp:
        cp_py = os.path.join(cp, exe) if win else os.path.join(cp, "bin", exe)
        if os.path.isfile(cp_py):
            add(cp_py, "conda: " + os.path.basename(os.path.normpath(cp)))
    for root in conda_roots:
        bp = os.path.join(root, exe) if win else os.path.join(root, "bin", exe)
        if os.path.isfile(bp):
            add(bp, "conda base")
        envs_dir = os.path.join(root, "envs")
        if os.path.isdir(envs_dir):
            for name in sorted(os.listdir(envs_dir)):
                ep = (os.path.join(envs_dir, name, exe) if win
                      else os.path.join(envs_dir, name, "bin", exe))
                if os.path.isfile(ep):
                    add(ep, "conda: " + name)

    for p in [shutil.which("python"), shutil.which("python3")]:
        if p:
            add(p, "system PATH")

    add(sys.executable, "current")
    return envs


def scan_envs(envs):
    """Check version + packages for each environment."""
    total = len(envs)
    for i, env in enumerate(envs):
        print("  [{}/{}] {} ...".format(i + 1, total, env["label"]))
        env["version"] = get_version(env["path"])
        env["pkgs"] = check_pkgs(env["path"])
    clear_screen()


def print_banner():
    clear_screen()
    w = 70
    print("=" * w)
    print("  Crowd Counting Auto Training Framework")
    print("  " + PROJECT_DIR)
    print("=" * w)


def print_env_menu(envs, selected):
    clear_screen()
    w = 70
    print("=" * w)
    print("  Select Python Environment")
    print("  (Up/Down: navigate | Enter: select | Q: quit)")
    print("=" * w)
    for i, env in enumerate(envs):
        pkgs = env.get("pkgs", {})
        ver = env.get("version", "?")
        marks = []
        for fw in FRAMEWORKS:
            marks.append("[+]" if pkgs.get(fw) else "[-]")
        arrow = " >>" if i == selected else "   "
        req_ok, missing = check_required(env)
        req_str = "OK" if req_ok else str(len(missing)) + " missing"
        print("{} {:<22s} py{:<8s} tor:{} tv:{} | req:{}".format(
            arrow, env["label"], ver, marks[0], marks[1], req_str))
    print("-" * w)
    print("  [+] = installed  [-] = not installed")
    print("=" * w)


def check_required(env):
    missing = [p for p in REQUIRED if not env.get("pkgs", {}).get(p, False)]
    return len(missing) == 0, missing


def get_key(timeout=0.1):
    if sys.platform == "win32":
        start = time.time()
        while time.time() - start < timeout:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch == b"\xe0":
                    ch2 = msvcrt.getch()
                    if ch2 == b"H":
                        return "up"
                    elif ch2 == b"P":
                        return "down"
                    return None
                elif ch == b"\r":
                    return "enter"
                else:
                    try:
                        return ch.decode("ascii").lower()
                    except Exception:
                        return None
            time.sleep(0.02)
        return None
    else:
        import select
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return "up" if ch3 == "A" else ("down" if ch3 == "B" else None)
            return None
        if ch in ("\r", "\n"):
            return "enter"
        return ch.lower()


def interactive_select(envs):
    selected = 0
    while True:
        print_env_menu(envs, selected)
        key = get_key(timeout=3600)
        if key == "up":
            selected = (selected - 1) % len(envs)
        elif key == "down":
            selected = (selected + 1) % len(envs)
        elif key == "enter":
            return envs[selected]
        elif key == "q":
            return None
        # other keys: just redraw


def confirm_env(env):
    clear_screen()
    w = 70
    print("=" * w)
    print("  Selected Environment")
    print("=" * w)
    print("  Label:    " + env["label"])
    print("  Python:   " + env["path"])
    print("  Version:  py" + env.get("version", "?"))
    print()
    pkgs = env.get("pkgs", {})
    print("  Framework availability:")
    print("    PyTorch:      " + ("[+] available" if pkgs.get("torch") else "[-] not installed"))
    print("    torchvision:  " + ("[+] available" if pkgs.get("torchvision") else "[-] not installed"))
    print()

    # PyTorch 栈不可用的环境（装了一半/损坏/版本过老）上服务必然启动失败：
    # crowd 路由在 import 期就会拉起 torchvision（见过 MemoryError / ImportError）。
    # 这里强制 y/n 确认，而不是仅提示后直接回车放行
    if not (pkgs.get("torch") and pkgs.get("torchvision")):
        print("  WARNING: PyTorch / torchvision unavailable in this environment.")
        print("           The server REQUIRES them (crowd counting core) and will")
        print("           crash at startup with an import error.")
        try:
            answer = input("  Start server anyway? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer != "y":
            print("  Cancelled. Please re-run and select the recommended")
            print("  environment (run_python.txt / conda: py310).")
            return False
        print()

    ok, missing = check_required(env)
    if not ok:
        print("  WARNING: Missing required packages: " + ", ".join(missing))
        try:
            answer = input("  Install now? (y/n): ").strip().lower()
        except EOFError:
            answer = "n"
        if answer == "y":
            print("  Installing (may take a few minutes)...")
            r = subprocess.run(
                [env["path"], "-m", "pip", "install", "-r",
                 os.path.join(PROJECT_DIR, "requirements.txt")])
            if r.returncode != 0:
                print("  Installation FAILED. Install manually:")
                print('    "{}" -m pip install -r requirements.txt'.format(env["path"]))
                input("  Press Enter to exit...")
                sys.exit(1)
            print("  Installation successful.")
        else:
            print("  Server may not work without required packages.")

    print()
    print("  Press Enter to start server (or Ctrl+C to cancel)...")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        return False
    return True


def port_in_use(port=SERVER_PORT):
    """探一下端口有没有人监听"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False


def find_port_pids(port=SERVER_PORT):
    """返回正在监听该端口的进程 PID 列表（Windows netstat / Linux lsof|ss）"""
    pids = set()
    if sys.platform == "win32":
        try:
            r = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                               capture_output=True, text=True, timeout=10,
                               creationflags=HIDE)
        except Exception:
            return []
        for line in (r.stdout or "").splitlines():
            parts = line.split()
            # TCP  127.0.0.1:8060  0.0.0.0:0  LISTENING  17540
            if len(parts) == 5 and parts[3].upper() == "LISTENING":
                try:
                    if int(parts[1].rsplit(":", 1)[1]) == port:
                        pids.add(int(parts[4]))
                except (ValueError, IndexError):
                    continue
        return sorted(pids)

    # Linux / macOS: lsof 优先，ss 兜底
    try:
        r = subprocess.run(["lsof", "-tiTCP:%d" % port, "-sTCP:LISTEN"],
                           capture_output=True, text=True, timeout=10)
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if line.isdigit():
                pids.add(int(line))
    except Exception:
        pass
    if not pids:
        try:
            import re as _re
            r = subprocess.run(["ss", "-ltnp"], capture_output=True,
                               text=True, timeout=10)
            for line in (r.stdout or "").splitlines():
                if not _re.search(r":%d\b" % port, line):
                    continue
                for m in _re.finditer(r"pid=(\d+)", line):
                    pids.add(int(m.group(1)))
        except Exception:
            pass
    return sorted(pids)


def _proc_desc(pid):
    """查进程名，失败就只显示 PID（Windows tasklist / Linux ps）"""
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                ["tasklist", "/FI", "PID eq %d" % pid, "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10, creationflags=HIDE)
            for line in (r.stdout or "").splitlines():
                line = line.strip()
                if line.startswith('"'):
                    cols = [c.strip('"') for c in line.split('","')]
                    if len(cols) >= 2:
                        return "%s (PID %d)" % (cols[0], pid)
        except Exception:
            pass
        return "PID %d" % pid

    try:
        r = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                           capture_output=True, text=True, timeout=10)
        names = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
        if names:
            return "%s (PID %d)" % (names[0], pid)
    except Exception:
        pass
    return "PID %d" % pid


def ensure_port_free(port=SERVER_PORT):
    """端口被占时列出进程问要不要杀。直接关窗/Ctrl+C 带不走服务子进程，
    下次启动就会撞端口，code 1 退出。"""
    if not port_in_use(port):
        return True
    pids = find_port_pids(port)
    descs = ", ".join(_proc_desc(p) for p in pids) if pids else "unknown process"
    print("  WARNING: port %d is already in use by: %s" % (port, descs))
    print("  (usually a leftover server process from a previous run;")
    print("   a new server would fail to bind and exit immediately)")
    try:
        ans = input("  Terminate it so the new server can start? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"
    if ans != "y":
        return False
    if sys.platform == "win32":
        for pid in pids:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           creationflags=HIDE)
    else:
        import signal
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        time.sleep(1)
        for pid in pids:   # 未退出则强制结束（残留服务进程）
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    for _ in range(10):  # 最多等 5 秒端口释放
        if not port_in_use(port):
            print("  Port %d released." % port)
            return True
        time.sleep(0.5)
    print("  ERROR: port %d is still in use after terminating the process(es)."
          % port)
    return False


def _open_server_log():
    """打开控制台日志，每次启动重写"""
    global _server_log_file
    try:
        os.makedirs(os.path.dirname(SERVER_LOG), exist_ok=True)
        if _server_log_file is not None:
            try:
                _server_log_file.close()
            except Exception:
                pass
        _server_log_file = open(SERVER_LOG, "w", encoding="utf-8",
                                errors="replace")
    except Exception:
        _server_log_file = None
    return _server_log_file


def _print_log_tail(n=15):
    """秒退时把日志尾部打出来看看原因；返回尾部文本供调用方判断错误类型"""
    tail = ""
    try:
        with open(SERVER_LOG, "r", encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip() for l in f if l.strip()]
        if lines:
            tail = "\n".join(lines[-n:])
            print("  Last server console output:")
            for l in lines[-n:]:
                print("    " + l)
    except Exception:
        pass
    return tail


def start_server(python_path):
    """启动服务子进程，输出进 server_console.log；端口没腾出来返回 None"""
    if not ensure_port_free(SERVER_PORT):
        return None
    logf = _open_server_log()
    proc = subprocess.Popen(
        [python_path, "main.py", "--port", str(SERVER_PORT)], cwd=PROJECT_DIR,
        creationflags=HIDE, stdout=logf, stderr=subprocess.STDOUT)
    # 短暂观察是否立即退出（端口绑定失败 / 依赖缺失等），是则展示日志尾部
    time.sleep(2.5)
    if proc.poll() is not None:
        print("  Server exited immediately (code %s)." % proc.returncode)
        _print_log_tail()
        print("  Full log: %s" % SERVER_LOG)
    return proc


def stop_server(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def print_server_status(env, proc):
    w = 70
    print("=" * w)
    print("  Server Running")
    print("-" * w)
    print("  Environment: " + env["label"])
    print("  Python:     " + env["path"])
    print("  URL:        http://127.0.0.1:%d" % SERVER_PORT)
    print("  PID:        " + str(proc.pid))
    print("  Console log: " + SERVER_LOG)
    print("-" * w)
    print("  [r] restart    [e] switch environment    [q] quit")
    print("=" * w)


def manage_server(env):
    proc = start_server(env["path"])
    if proc is None:
        input("  Press Enter to go back...")
        return "switch"

    while True:
        clear_screen()
        print_server_status(env, proc)

        # wait for key, poll server state meanwhile
        key = get_key(timeout=2.0)

        if proc.poll() is not None:
            print()
            print("  Server exited (code {}).".format(proc.returncode))
            tail = _print_log_tail()
            print("  Full log: " + SERVER_LOG)
            # 该环境缺框架/框架损坏时给出明确指引（traceback 只有晦涩的导入错误）
            _pkgs = env.get("pkgs", {})
            if not (_pkgs.get("torch") and _pkgs.get("torchvision")):
                print("  HINT: this environment has no working PyTorch/torchvision")
                print("        (see tor/tv marks in the env menu). Press [e] to switch")
                print("        to the recommended environment (run_python.txt).")
            elif ("torch" in tail) and ("Error" in tail or "error" in tail):
                print("  HINT: the log shows a torch/torchvision import error;")
                print("        press [e] to try another environment.")
            print("  [r] restart  [e] switch env  [q] quit")
            key = get_key(timeout=3600)

        if key == "r":
            print("  Restarting...")
            stop_server(proc)
            proc = start_server(env["path"])
            if proc is None:
                input("  Press Enter to go back...")
                return "switch"
            time.sleep(1)
        elif key == "e":
            stop_server(proc)
            return "switch"
        elif key == "q":
            stop_server(proc)
            print("  Server stopped.")
            return "quit"


def main():
    while True:
        print_banner()
        print()
        print("[1/2] Discovering Python environments...")
        envs = discover_envs()
        if not envs:
            print("  ERROR: No Python found.")
            input("  Press Enter to exit...")
            sys.exit(1)

        print("  Found {} environment(s). Checking packages...".format(len(envs)))
        print()
        scan_envs(envs)

        print("[2/2] Select environment")
        env = interactive_select(envs)
        if env is None:
            print("  Cancelled.")
            sys.exit(0)

        if not confirm_env(env):
            print("  Cancelled.")
            sys.exit(0)

        result = manage_server(env)
        if result == "quit":
            break
        elif result == "switch":
            print("  Going back to environment selection...")
            continue

    print("  Goodbye.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Interrupted. Exiting...")
        sys.exit(0)
