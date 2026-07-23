# Reset the dev env to a clean state (deps removed + reinstalled). Run `pnpm dev` after.

$ErrorActionPreference = "Stop"

function Ok($msg) { Write-Host "✓ $msg" -ForegroundColor Green }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

Write-Host "Removing build artifacts and dependencies..."
foreach ($dir in @("node_modules", "dist", "dist-electron", "release", "backend\.venv")) {
    if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
}
Get-ChildItem -Path . -Filter *.tsbuildinfo -Recurse -File |
    Where-Object { $_.FullName -notmatch '\\node_modules\\' } |
    Remove-Item -Force
Get-ChildItem -Path backend -Filter __pycache__ -Recurse -Directory |
    Remove-Item -Recurse -Force
Ok "workspace cleaned"

Write-Host "`nReinstalling Node dependencies..."
pnpm install
if ($LASTEXITCODE -ne 0) { Write-Host "✗ pnpm install failed" -ForegroundColor Red; exit 1 }
Ok "pnpm install complete"

Write-Host "`nReinstalling Python backend venv..."
Set-Location (Join-Path $ProjectDir "backend")
uv sync --extra dev
if ($LASTEXITCODE -ne 0) { Write-Host "✗ uv sync failed" -ForegroundColor Red; exit 1 }
Ok "uv sync complete"

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  Clean complete! Run the app with:  pnpm dev" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
