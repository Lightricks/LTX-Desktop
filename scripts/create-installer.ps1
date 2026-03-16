# create-installer.ps1
# Runs electron-builder to produce the installer (exe).
# This is the ONLY build stage that needs code-signing secrets.
#
# Expects the frontend to be built and python-embed to be ready.
# See local-build.ps1 for the convenience wrapper that runs all stages.

param(
    [switch]$Unpack,
    [string]$Publish = "",
    [ValidateSet("cuda", "rocm")]
    [string]$GpuBackend = "cuda"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$ReleaseDir = Join-Path $ProjectDir "release"

Set-Location $ProjectDir

# Verify prerequisites
if (-not (Test-Path "dist") -or -not (Test-Path "dist-electron")) {
    Write-Host "ERROR: Frontend not built. Run local-build.ps1 or 'npm run build:frontend' first." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "python-embed")) {
    Write-Host "ERROR: Python environment not found. Run local-build.ps1 or prepare-python.ps1 first." -ForegroundColor Red
    exit 1
}

# Set artifact name suffix for GPU variant (read by electron-builder.yml template)
$env:LTX_ARTIFACT_SUFFIX = if ($GpuBackend -eq "rocm") { "-ROCm" } else { "" }

# Build with electron-builder
if ($Unpack) {
    Write-Host "Packaging unpacked app (fast mode)..." -ForegroundColor Yellow
    pnpm exec electron-builder --win --dir
} else {
    Write-Host "Packaging installer..." -ForegroundColor Yellow
    $PublishArgs = @()
    if ($Publish -ne "") {
        $PublishArgs = @("--publish", $Publish)
    }
    pnpm exec electron-builder --win @PublishArgs
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build!" -ForegroundColor Red
    exit 1
}

# Summary
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

if ($Unpack) {
    $UnpackedDir = Join-Path $ReleaseDir "win-unpacked"
    $ExePath = Join-Path $UnpackedDir "LTX Desktop.exe"
    Write-Host "`nUnpacked app ready!" -ForegroundColor Cyan
    Write-Host "Run: $ExePath" -ForegroundColor Cyan
    Write-Host "`nTip: Just restart the app after code changes - no rebuild needed!" -ForegroundColor Green
} else {
    $Installer = Get-ChildItem -Path $ReleaseDir -Filter "*.exe" | Where-Object { $_.Name -like "*Setup*" } | Select-Object -First 1
    if ($Installer) {
        $InstallerSize = [math]::Round($Installer.Length / 1MB, 2)
        Write-Host "`nInstaller: $($Installer.Name)" -ForegroundColor Cyan
        Write-Host "Size: $InstallerSize MB" -ForegroundColor Cyan
        Write-Host "Location: $($Installer.FullName)" -ForegroundColor Cyan
    }
}

Write-Host "`nNote: AI models (~150GB) will be downloaded on first run." -ForegroundColor Yellow
