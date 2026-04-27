@echo off
chcp 65001 >nul
title AGEANet Launcher
setlocal enabledelayedexpansion

set "PROJECT_ROOT=%~dp0"
set "VISION_DIR=%PROJECT_ROOT%Anti-Interference-2D-Vision"
set "GUI_DIR=%PROJECT_ROOT%AntiInterference2D.GUI"
set "API_PORT=8000"
set "HEALTH_URL=http://localhost:%API_PORT%/api/v1/health"

echo ==========================================
echo   AGEANet One-Click Launcher
echo   FastAPI Backend + C# WPF GUI
echo ==========================================
echo.

:: Activate virtual environment
echo [INFO] Looking for Python venv...
if exist "%VISION_DIR%\.venv_win\Scripts\activate.bat" (
    echo [INFO] Activating .venv_win
    call "%VISION_DIR%\.venv_win\Scripts\activate.bat"
) else if exist "%VISION_DIR%\.venv\Scripts\activate.bat" (
    echo [INFO] Activating .venv
    call "%VISION_DIR%\.venv\Scripts\activate.bat"
) else (
    echo [WARN] No venv found, using system Python
)

:: Check dependencies
echo [INFO] Checking FastAPI dependencies...
python -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Dependencies missing, installing...
    pip install fastapi uvicorn[standard] python-multipart pillow
)

:: Start FastAPI backend
echo [INFO] Starting FastAPI backend on port %API_PORT%...
cd /d "%VISION_DIR%"
start "AGEANet-Backend" cmd /c "python -m uvicorn api.server:app --host 0.0.0.0 --port %API_PORT%"

:: Wait for backend ready
echo [INFO] Waiting for backend to be ready...
set /a retries=0
:wait_loop
set /a retries+=1
if !retries! gtr 30 (
    echo [ERROR] Backend startup timeout
    pause
    exit /b 1
)

powershell -Command "try { $r=Invoke-RestMethod -Uri '%HEALTH_URL%' -TimeoutSec 2; if($r.status -eq 'ok'){ exit 0 } } catch{} exit 1" >nul 2>&1
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_loop
)
echo [OK] Backend is ready at %HEALTH_URL%

:: Start C# WPF GUI
echo [INFO] Starting C# WPF GUI...
cd /d "%GUI_DIR%"

where dotnet >nul 2>&1
if errorlevel 1 (
    echo [ERROR] dotnet CLI not found. Please install .NET 8 SDK.
    pause
    exit /b 1
)

echo [INFO] Building WPF project...
dotnet build
if errorlevel 1 (
    echo [ERROR] Build failed
    pause
    exit /b 1
)

echo [OK] Build success, launching GUI...
dotnet run --no-build

:: Cleanup
echo [INFO] Shutting down backend...
taskkill /FI "WINDOWTITLE eq AGEANet-Backend" /F >nul 2>&1
echo [OK] Backend stopped

echo.
pause
