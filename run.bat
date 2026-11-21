@echo off
setlocal EnableExtensions
title Crowd Counting Auto Training Framework
cd /d "%~dp0"
chcp 65001 >nul 2>&1

REM Bootstrap: find any working Python, hand off to launcher.py

echo ==============================================================
echo    Crowd Counting Auto Training Framework
echo    Project: %~dp0
echo ==============================================================
echo.

REM ---- find a bootstrap Python ----
set "BOOT_PY="

REM Try run_python.txt first
REM （for /f 正确剥离 CRLF 的 CR；set /p 会保留 CR 导致 if exist 永远失败）
if exist "%~dp0run_python.txt" (
    for /f "usebackq delims=" %%i in ("%~dp0run_python.txt") do set "CUSTOM_PY=%%i"
    if defined CUSTOM_PY if exist "%CUSTOM_PY%" (
        set "BOOT_PY=%CUSTOM_PY%"
        goto :launch
    )
)

REM Try current PATH
for /f "delims=" %%i in ('where python 2^>nul') do if not defined BOOT_PY (
    "%%i" -c "import sys;sys.exit(0 if sys.version_info>=(3,7) else 1)" >nul 2>&1
    if not errorlevel 1 set "BOOT_PY=%%i"
)

REM Try common conda locations
if not defined BOOT_PY (
    for %%d in ("D:\conda3\anaconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "C:\ProgramData\Anaconda3" "D:\Anaconda3") do (
        if exist "%%~d\python.exe" if not defined BOOT_PY set "BOOT_PY=%%~d\python.exe"
    )
)

REM Try py launcher
if not defined BOOT_PY (
    py -3 -c "print('ok')" >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "BOOT_PY=%%i"
    )
)

if not defined BOOT_PY (
    echo  ERROR: No Python found on this machine.
    echo.
    echo  Please install Anaconda or add Python to your PATH.
    echo  Download: https://www.anaconda.com/download
    echo.
    pause
    exit /b 1
)

:launch
echo  Bootstrap Python: %BOOT_PY%

REM --check: verify dependencies only, do not start the server
if /i "%~1"=="--check" (
    echo  Running environment check ^(no server will be started^)...
    echo.
    "%BOOT_PY%" -c "import importlib.util as u,sys; mods=['fastapi','uvicorn','PIL','numpy','tqdm','scipy','matplotlib','torch','torchvision']; missing=[m for m in mods if u.find_spec(m) is None]; print('  OK: all dependencies present.' if not missing else '  MISSING: '+', '.join(missing)); sys.exit(1 if missing else 0)"
    if errorlevel 1 (
        echo.
        echo  Environment check FAILED. Install with:
        echo      pip install -r requirements.txt
        exit /b 1
    )
    echo  Environment check passed.
    exit /b 0
)

echo  Starting interactive launcher...
echo.
"%BOOT_PY%" "%~dp0launcher.py"

if errorlevel 1 (
    echo.
    echo  Launcher exited with an error.
    pause
) else (
    echo.
    echo  Launcher finished.
)
exit /b %errorlevel%
