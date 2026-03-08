# Windows development setup for LTX Desktop

$ErrorActionPreference = "Stop"

function Ok($msg)   { Write-Host "✓ $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "✗ $msg" -ForegroundColor Red; exit 1 }

# ── Pre-checks ──────────────────────────────────────────────────────
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Fail "node not found — install Node.js 18+ from https://nodejs.org/"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Fail "uv not found — install with: powershell -ExecutionPolicy ByPass -c 'irm https://astral.sh/uv/install.ps1 | iex'"
}
Ok "node $(node -v)"
Ok "uv   $(uv --version)"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# ── pnpm install ────────────────────────────────────────────────────
if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
    Write-Host "pnpm not found — installing via corepack..."
    corepack enable
    if ($LASTEXITCODE -ne 0) {
        Write-Host "corepack failed, trying npm install..."
        npm install -g pnpm@10.30.3
        if ($LASTEXITCODE -ne 0) { Fail "Could not install pnpm" }
    }
    Ok "pnpm installed"
}

Write-Host "`nInstalling Node dependencies..."
Set-Location $ProjectDir
pnpm install
if ($LASTEXITCODE -ne 0) { Fail "pnpm install failed" }
Ok "pnpm install complete"

# ── uv sync ─────────────────────────────────────────────────────────
Write-Host "`nSetting up Python backend venv..."
Set-Location (Join-Path $ProjectDir "backend")
uv sync --extra dev
if ($LASTEXITCODE -ne 0) { Fail "uv sync failed" }
Ok "uv sync complete"

# Verify torch + CUDA
Write-Host "`nVerifying PyTorch CUDA support..."
try {
    & .venv\Scripts\python.exe -c "import torch; cuda=torch.cuda.is_available(); print(f'CUDA available: {cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}') if cuda else None"
} catch {
    Write-Host "  Could not verify PyTorch — this is OK if setup is still downloading." -ForegroundColor DarkYellow
}

# ── ffmpeg check ────────────────────────────────────────────────────
Write-Host ""
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    $ffmpegVer = (ffmpeg -version 2>&1 | Select-Object -First 1)
    Ok "ffmpeg found: $ffmpegVer"
} else {
    Write-Host "⚠  ffmpeg not found — install with: winget install ffmpeg" -ForegroundColor Yellow
    Write-Host "   (imageio-ffmpeg bundled binary will be used as fallback)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  Setup complete! Run the app with:  pnpm dev" -ForegroundColor Cyan
Write-Host "  Debug mode (with debugpy):         pnpm dev:debug" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
