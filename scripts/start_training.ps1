# start_training.ps1 — Windows PowerShell 训练启动脚本
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VisionDir = Join-Path $ProjectRoot "Anti-Interference-2D-Vision"
Set-Location $VisionDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AGEANet FLARE Training" -ForegroundColor Cyan
Write-Host "  OS: Windows" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 优先使用 launch.py
if (Test-Path "launch.py") {
    Write-Host "使用统一启动器 (launch.py)..." -ForegroundColor Green
    python launch.py --mode training
    exit
}

# 回退：手动启动
Write-Host "启动训练 (50 epochs)..." -ForegroundColor Yellow
python training/train.py --synth_dir ./dataset --epochs 50 --batch_size 8 --lr 1e-4 --save_dir ./checkpoints --base_ch 32
