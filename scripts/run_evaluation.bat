@echo off
title FLARE Evaluation

cd /d "%~dp0..\Anti-Interference-2D-Vision"

echo Running model evaluation...
echo.

python training/evaluate.py --checkpoint checkpoints/best.pth --data dataset_merged --output ./eval_results --base_ch 32

echo.
echo Results saved to ./eval_results
pause
