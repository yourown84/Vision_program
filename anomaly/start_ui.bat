@echo off
chcp 65001 >nul
cd /d %~dp0
call "%~dp0config.bat"
echo Starting inspection UI. A browser window opens in a moment. Close this window to stop.
..\.venv\Scripts\python.exe app.py
pause
