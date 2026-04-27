#Requires -Version 5.1
<#
.SYNOPSIS
    Start only the C# WPF GUI (assumes backend is already running).
#>

$ErrorActionPreference = "Stop"
$GuiDir = Join-Path $PSScriptRoot "AntiInterference2D.GUI"

if (-not (Test-Path $GuiDir)) {
    Write-Host "[ERROR] WPF project not found: $GuiDir" -ForegroundColor Red
    pause
    exit 1
}

Set-Location $GuiDir

$dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
if (-not $dotnet) {
    Write-Host "[ERROR] dotnet CLI not found. Please install .NET 8 SDK." -ForegroundColor Red
    pause
    exit 1
}

Write-Host "[INFO] Building WPF..." -ForegroundColor Cyan
dotnet build
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Build failed" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "[OK] Launching GUI..." -ForegroundColor Green
dotnet run --no-build
