@echo off
rem ===== One-click environment setup for a new PC =====
rem 1) Install Python 3.12 first (https://www.python.org/downloads/windows/, check "Add python.exe to PATH")
rem 2) Double-click this file. It creates .venv next to this file and installs everything (~3 GB download).
rem Caches (pip / HuggingFace / torch) go to .cache next to this file, not to the C drive.
setlocal
cd /d %~dp0
set PIP_CACHE_DIR=%~dp0.cache\pip
set HF_HOME=%~dp0.cache\huggingface
set TORCH_HOME=%~dp0.cache\torch

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] python not found. Install Python 3.12 and check "Add python.exe to PATH", then run again.
  pause & exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Python %PYVER% found.

if not exist .venv (
  echo Creating virtual environment .venv ...
  python -m venv .venv || (echo [ERROR] venv creation failed & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo Installing PyTorch (CUDA 12.4 build; works on CPU-only PCs too) ...
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 || (echo [ERROR] torch install failed & pause & exit /b 1)

echo.
echo Installing the rest ...
python -m pip install -r requirements.txt || (echo [ERROR] requirements install failed & pause & exit /b 1)

echo.
echo Checking GPU ...
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo.
echo Creating sample data (optional, for testing) ...
cd anomaly
python make_sample_data.py
python make_sample_cls_data.py
python make_sample_seg_data.py
cd ..

echo.
echo ===== Setup complete. Run anomaly\start_ui.bat to open the UI. =====
pause
