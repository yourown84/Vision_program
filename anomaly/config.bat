@echo off
rem ===== Site settings (edit this file on each PC) =====
rem Manager server address. On the server PC itself this is http://127.0.0.1:9000
rem On a line PC, put the server PC's IP, e.g. http://192.168.0.10:9000
set MANAGER_URL=http://127.0.0.1:9000

rem Name of this PC as shown in the manager dashboard (default: computer name)
set AGENT_ID=%COMPUTERNAME%

rem Port of the local inspection API server
set API_PORT=8000

rem Force CPU with "cpu" (for PCs without an NVIDIA GPU); leave empty for auto
set INSPECT_DEVICE=

rem ===== Common (do not edit). Paths are relative to the project folder, so the folder can live anywhere =====
set PYTHONIOENCODING=utf-8
set TRUST_REMOTE_CODE=1
set HF_HOME=%~dp0..\.cache\huggingface
set TORCH_HOME=%~dp0..\.cache\torch
set GRADIO_TEMP_DIR=%~dp0..\.cache\gradio
