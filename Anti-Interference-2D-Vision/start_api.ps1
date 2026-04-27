# start_api.ps1 — 启动 FastAPI 后端服务
# 供 C# WPF 上位机调用

$ErrorActionPreference = "Stop"

# 激活虚拟环境（如果存在）
$VenvPath = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & $VenvPath
}

# 检查依赖
Write-Host "Checking dependencies..." -ForegroundColor Cyan
python -c "import fastapi, uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing FastAPI dependencies..." -ForegroundColor Yellow
    pip install fastapi uvicorn[standard] python-multipart pillow
}

# 启动服务
Write-Host "Starting AGEANet API Server on http://localhost:8000" -ForegroundColor Green
Write-Host "API docs: http://localhost:8000/docs" -ForegroundColor Green
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
