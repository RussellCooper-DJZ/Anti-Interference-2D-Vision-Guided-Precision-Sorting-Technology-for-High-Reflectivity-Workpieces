@echo off
title FLARE Training (New Edge Head)

cd /d "%~dp0"

echo ========================================
echo FLARE Training - Simplified Edge Detection
echo ========================================
echo.

echo Using settings:
echo   - Epochs: 50
echo   - Batch size: 2
echo   - Learning rate: 3e-5
echo   - Base channels: 64
echo   - Dataset: dataset
echo.

python -m training.train --synth_dir ./dataset --epochs 50 --batch_size 2 --lr 3e-5 --save_dir ./checkpoints --base_ch 64

pause
