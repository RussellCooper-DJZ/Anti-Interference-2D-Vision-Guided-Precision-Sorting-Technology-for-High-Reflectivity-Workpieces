#Requires -Version 5.1
<#
.SYNOPSIS
    AGEANet One-Click Launcher
.DESCRIPTION
    Starts Python FastAPI backend (in a new window) and C# WPF GUI.
    The backend runs in its own window so you can see errors directly.
.USAGE
    .\start_all.ps1
#>

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$VisionDir   = Join-Path $ProjectRoot "Anti-Interference-2D-Vision"
$GuiDir      = Join-Path $ProjectRoot "AntiInterference2D.GUI"
$ApiPort     = 8000
$HealthUrl   = "http://localhost:$ApiPort/api/v1/health"

function Write-Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Error($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

Write-Info "AGEANet One-Click Launcher"
Write-Info "=========================="

# Step 1: Find Python executable in venv
Write-Info "Looking for Python in venv..."
$PythonPaths = @(
    (Join-Path $VisionDir ".venv_win\Scripts\python.exe"),
    (Join-Path $VisionDir ".venv\Scripts\python.exe"),
    "python"
)
$PythonExe = $null
foreach ($p in $PythonPaths) {
    if (Test-Path $p) {
        $PythonExe = $p
        break
    }
}
if (-not $PythonExe) {
    Write-Error "Python not found. Please install Python and create a venv."
    pause
    exit 1
}
Write-Ok "Found Python: $PythonExe"

# Step 2: Check dependencies
Write-Info "Checking dependencies..."
& $PythonExe -c "import fastapi, uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Dependencies missing, installing..."
    & $PythonExe -m pip install fastapi uvicorn[standard] python-multipart pillow
}

# Step 3: Start backend in a NEW WINDOW so errors are visible
Write-Info "Starting FastAPI backend in a new window..."
$BackendCmd = "cd /d `"$VisionDir`" && `"$PythonExe`" -m uvicorn api.server:app --host 0.0.0.0 --port $ApiPort"
Start-Process cmd.exe -ArgumentList "/c title AGEANet-Backend && $BackendCmd && pause" -WindowStyle Normal

# Step 4: Wait for backend ready
Write-Info "Waiting for backend (port $ApiPort)..."
$MaxRetries = 30
$Ready = $false
for ($i = 1; $i -le $MaxRetries; $i++) {
    try {
        $resp = Invoke-RestMethod -Uri $HealthUrl -Method GET -TimeoutSec 2 -ErrorAction Stop
        if ($resp.status -eq "ok") {
            $Ready = $true
            break
        }
    } catch {
        Write-Host "." -NoNewline -ForegroundColor DarkGray
    }
    Start-Sleep -Seconds 1
}
Write-Host ""

if (-not $Ready) {
    Write-Error "Backend did not start. Check the 'AGEANet-Backend' window for errors."
    Write-Error "Common fixes:"
    Write-Error "  1. Install dependencies: pip install fastapi uvicorn python-multipart pillow"
    Write-Error "  2. Check port $ApiPort is not in use"
    Write-Error "  3. Verify api/server.py has no import errors"
    pause
    exit 1
}
Write-Ok "Backend is ready at $HealthUrl"

# Step 5: Start WPF GUI
Write-Info "Starting C# WPF GUI..."
if (-not (Test-Path $GuiDir)) {
    Write-Error "WPF project not found: $GuiDir"
    pause
    exit 1
}

Set-Location $GuiDir

$dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
if (-not $dotnet) {
    Write-Error "dotnet CLI not found. Please install .NET 8 SDK from https://dotnet.microsoft.com/download"
    pause
    exit 1
}

try {
    Write-Info "Building WPF project..."
    dotnet build | ForEach-Object { Write-Host "[BUILD] $_" -ForegroundColor DarkGray }
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }

    Write-Ok "Build success, launching GUI..."
    dotnet run --no-build
} catch {
    Write-Error $_
    pause
    exit 1
}

Write-Info "Done. Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
