# start_demo.ps1 — Windows PowerShell 启动脚本
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VisionDir = Join-Path $ProjectRoot "Anti-Interference-2D-Vision"
Set-Location $VisionDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AGEANet FLARE Demo Server" -ForegroundColor Cyan
Write-Host "  OS: Windows" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 优先使用 launch.py 自动管理环境
if (Test-Path "launch.py") {
    Write-Host "使用统一启动器 (launch.py)..." -ForegroundColor Green
    python launch.py --mode demo
    exit
}

# 回退：手动启动
Write-Host "启动 Streamlit 演示界面..." -ForegroundColor Yellow
Write-Host "请访问: http://localhost:8501" -ForegroundColor Yellow
Write-Host "按 Ctrl+C 停止" -ForegroundColor Yellow
Write-Host ""

python -m streamlit run demo_streamlit.py --server.port 8501
