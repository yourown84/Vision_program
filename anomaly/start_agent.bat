@echo off
chcp 65001 >nul
cd /d %~dp0
call "%~dp0config.bat"
set LOCAL_API=http://127.0.0.1:%API_PORT%
echo Starting line-PC agent. id=%AGENT_ID%  manager=%MANAGER_URL%  local api=%LOCAL_API%
echo (start_api.bat must be running on this PC as well)
..\.venv\Scripts\python.exe agent.py
pause
