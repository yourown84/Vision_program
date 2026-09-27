@echo off
chcp 65001 >nul
cd /d %~dp0
call "%~dp0config.bat"
echo Starting inspection API server at http://127.0.0.1:%API_PORT%  (docs: http://127.0.0.1:%API_PORT%/docs)
echo Results are reported to manager: %MANAGER_URL%
echo Close this window to stop the server.
..\.venv\Scripts\python.exe server.py
pause
