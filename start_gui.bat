@echo off
setlocal
cd /d "%~dp0"

echo Starting OCR GUI...
echo Make sure the backend is running: start_server.bat
echo.

set "PYTHON=py"
where py >nul 2>&1
if errorlevel 1 set "PYTHON=python"

if exist ".venv\Scripts\activate.bat" call "%~dp0.venv\Scripts\activate.bat"

"%PYTHON%" gui.py
pause
