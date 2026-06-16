@echo off
setlocal
cd /d "%~dp0"

echo Starting OCR Service + LINE Webhook...
echo API docs : http://127.0.0.1:8000/docs
echo Webhook  : http://127.0.0.1:8000/callback
echo.
echo Use ngrok to expose port 8000, then set LINE Webhook to:
echo   https://YOUR-NGROK-URL/callback
echo.

set "PYTHON=py"
where py >nul 2>&1
if errorlevel 1 set "PYTHON=python"

if exist ".venv\Scripts\activate.bat" call "%~dp0.venv\Scripts\activate.bat"

where ngrok >nul 2>&1
if not errorlevel 1 (
    echo Starting ngrok in a new window...
    start "ngrok" cmd /k ngrok http 8000
)

"%PYTHON%" -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
