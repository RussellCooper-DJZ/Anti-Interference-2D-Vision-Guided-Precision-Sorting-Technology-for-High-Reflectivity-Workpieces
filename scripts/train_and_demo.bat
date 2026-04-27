@echo off
title FLARE Training System

echo ========================================
echo FLARE High-Reflectivity Model Training
echo ========================================
echo.

cd /d "%~dp0..\Anti-Interference-2D-Vision"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.12+
    pause
    exit /b 1
)

REM Check dependencies
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Select operation:
echo [1] Train (50 epochs)
echo [2] Resume training
echo [3] Start demo server
echo [4] Evaluate model
echo [5] Exit
echo.

set /p choice=Enter choice (1-5):

if "%choice%"=="1" (
    echo.
    echo [TRAINING] Starting 50 epochs...
    python training/train.py --synth_dir ./dataset --epochs 50 --batch_size 8 --lr 1e-4 --save_dir ./checkpoints --base_ch 32
)

if "%choice%"=="2" (
    echo.
    echo [TRAINING] Resuming from checkpoint...
    python training/train.py --synth_dir ./dataset --epochs 50 --batch_size 8 --lr 1e-4 --resume --save_dir ./checkpoints --base_ch 32
)

if "%choice%"=="3" (
    echo.
    echo [DEMO] Starting server...
    echo Please visit: http://localhost:8501
    python -m streamlit run demo_streamlit.py --server.port 8501
)

if "%choice%"=="4" (
    echo.
    echo [EVAL] Running evaluation...
    python training/evaluate.py --checkpoint checkpoints/best.pth --data dataset_merged --output ./eval_results --base_ch 32
)

if "%choice%"=="5" (
    exit /b 0
)

echo.
echo Done!
pause
