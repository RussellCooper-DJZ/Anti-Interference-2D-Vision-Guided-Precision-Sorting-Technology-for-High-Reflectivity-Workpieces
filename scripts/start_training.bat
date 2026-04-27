@echo off
title FLARE Training

cd /d "%~dp0..\Anti-Interference-2D-Vision"

echo Starting FLARE training (50 epochs)...
echo.

python training/train.py --synth_dir ./dataset --epochs 50 --batch_size 8 --lr 1e-4 --save_dir ./checkpoints --base_ch 32

pause
