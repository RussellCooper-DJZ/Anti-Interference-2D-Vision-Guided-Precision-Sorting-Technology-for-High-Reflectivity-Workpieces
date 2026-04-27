@echo off
chcp 65001 >nul
echo ==========================================
echo   AGEANet FastAPI Backend Starter
echo ==========================================

REM 激活虚拟环境（如果存在）
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM 启动服务
echo [INFO] Starting API Server on http://localhost:8000
echo [INFO] API docs: http://localhost:8000/docs
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

pause
