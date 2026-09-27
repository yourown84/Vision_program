@echo off
chcp 65001 >nul
cd /d %~dp0
call "%~dp0config.bat"
echo Starting manager server. Dashboard: http://127.0.0.1:9000/  (other PCs: http://THIS-PC-IP:9000/)
echo Close this window to stop.
..\.venv\Scripts\python.exe manager.py
pause
