@echo off
setlocal
cd /d "%~dp0"

echo Starting OCR Core Service (Engineer A)...
echo API docs: http://127.0.0.1:8000/docs
echo.

set "PYTHON=py"
where py >nul 2>&1
if errorlevel 1 set "PYTHON=python"

if exist ".venv\Scripts\activate.bat" call "%~dp0.venv\Scripts\activate.bat"

"%PYTHON%" -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
