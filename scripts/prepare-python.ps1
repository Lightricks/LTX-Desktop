# prepare-python.ps1
# Downloads embedded Python and installs all dependencies for distribution.
#
# Dependencies are read from uv.lock (via `uv export`) — pyproject.toml is the
# single source of truth. No hardcoded dependency lists.
#
# Prerequisites:
#   - uv must be installed (https://docs.astral.sh/uv/)

param(
    [string]$PythonVersion = (Get-Content "$PSScriptRoot\..\backend\.python-version" -Raw).Trim(),
    [string]$OutputDir = "python-embed",
    [ValidateSet("cuda", "rocm")]
    [string]$GpuBackend = "cuda"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LTX Video - Python Environment Setup" -ForegroundColor Cyan
Write-Host "  GPU Backend: $GpuBackend" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# AMD ROCm Windows wheels only support Python 3.12
if ($GpuBackend -eq "rocm") {
    $PythonVersion = "3.12.10"
    Write-Host "ROCm build: forcing Python $PythonVersion (AMD requirement)" -ForegroundColor Yellow
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $ProjectDir "backend"
$OutputPath = Join-Path $ProjectDir $OutputDir
$TempDir = Join-Path $env:TEMP "ltx-python-build"

# Python embed URL
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"

# PyTorch CUDA index (must match the index in pyproject.toml)
$PyTorchCudaIndex = "https://download.pytorch.org/whl/cu128"

# ROCm wheel base URL (AMD repo.radeon.com)
$RocmBase = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2"
$RocmSdkWheels = @(
    "$RocmBase/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl",
    "$RocmBase/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl",
    "$RocmBase/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl",
    "$RocmBase/rocm-7.2.0.dev0.tar.gz"
)
$RocmTorchWheels = @(
    "$RocmBase/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl",
    "$RocmBase/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl",
    "$RocmBase/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl"
)

# ============================================================
# Step 1: Verify prerequisites
# ============================================================
Write-Host "`nStep 1: Verifying prerequisites..." -ForegroundColor Yellow

# Check uv is available
$UvExe = Get-Command uv -ErrorAction SilentlyContinue
if (-not $UvExe) {
    # Fall back to uv in the dev venv
    $VenvUv = Join-Path $BackendDir ".venv\Scripts\uv.exe"
    if (Test-Path $VenvUv) {
        $UvExe = $VenvUv
    } else {
        Write-Host "ERROR: uv not found. Install it: https://docs.astral.sh/uv/" -ForegroundColor Red
        exit 1
    }
}
Write-Host "uv: $UvExe" -ForegroundColor Green

# ============================================================
# Step 2: Generate requirements.txt from uv.lock
# ============================================================
Write-Host "`nStep 2: Generating requirements.txt from uv.lock..." -ForegroundColor Yellow

$RequirementsFile = Join-Path $BackendDir "requirements-dist.txt"

# Export pinned deps, excluding the project itself (--no-emit-project)
& uv export --frozen --no-hashes --no-editable --no-emit-project `
    --no-header --no-annotate `
    --project $BackendDir `
    > $RequirementsFile

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: uv export failed!" -ForegroundColor Red
    exit 1
}

# For ROCm builds: strip out CUDA-specific packages that will be replaced
# by ROCm wheels (torch/torchvision/torchaudio) or are CUDA-only (sageattention, triton-windows)
if ($GpuBackend -eq "rocm") {
    $CudaOnlyPrefixes = @("torch==", "torchaudio==", "torchvision==", "sageattention==", "triton-windows==", "triton==")
    $Filtered = Get-Content $RequirementsFile | Where-Object {
        $line = $_.ToLower().TrimStart()
        -not ($CudaOnlyPrefixes | Where-Object { $line.StartsWith($_) })
    }
    Set-Content -Path $RequirementsFile -Value $Filtered
    $RemovedCount = (Get-Content $RequirementsFile | Where-Object { $_ -match "^\S" }).Count
    Write-Host "ROCm build: filtered to $RemovedCount dependencies (removed CUDA-only packages)" -ForegroundColor Yellow
}

$DepCount = (Get-Content $RequirementsFile | Where-Object { $_ -match "^\S" }).Count
Write-Host "Exported $DepCount dependencies from uv.lock" -ForegroundColor Green

# ============================================================
# Step 3: Prepare directories
# ============================================================
Write-Host "`nStep 3: Preparing directories..." -ForegroundColor Yellow

if (Test-Path $OutputPath) {
    Write-Host "Removing existing python-embed directory..."
    Remove-Item -Recurse -Force $OutputPath
}

if (Test-Path $TempDir) {
    Remove-Item -Recurse -Force $TempDir
}

New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

# ============================================================
# Step 4: Download and extract embedded Python
# ============================================================
Write-Host "`nStep 4: Downloading Python $PythonVersion embeddable..." -ForegroundColor Yellow

$PythonZip = Join-Path $TempDir "python-embed.zip"
Invoke-WebRequest -Uri $PythonUrl -OutFile $PythonZip -UseBasicParsing
Write-Host "Downloaded Python embeddable package"

Expand-Archive -Path $PythonZip -DestinationPath $OutputPath -Force
Write-Host "Extracted to $OutputPath"

# ============================================================
# Step 5: Enable pip in embedded Python
# ============================================================
Write-Host "`nStep 5: Enabling pip in embedded Python..." -ForegroundColor Yellow

$PthFile = Get-ChildItem -Path $OutputPath -Filter "python*._pth" | Select-Object -First 1
if ($PthFile) {
    $PthContent = Get-Content $PthFile.FullName
    $PthContent = $PthContent -replace "^#import site", "import site"
    $PthContent += "`nLib\site-packages"
    Set-Content -Path $PthFile.FullName -Value $PthContent
    Write-Host "Modified $($PthFile.Name) to enable pip and site-packages"
}

$GetPipPath = Join-Path $TempDir "get-pip.py"
Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPipPath -UseBasicParsing

$PythonExe = Join-Path $OutputPath "python.exe"
& $PythonExe $GetPipPath --no-warn-script-location
Write-Host "pip installed" -ForegroundColor Green

# Install setuptools and wheel (needed to build some dependencies from source)
& $PythonExe -m pip install setuptools wheel --no-warn-script-location --quiet
Write-Host "setuptools and wheel installed" -ForegroundColor Green

# ============================================================
# Step 6: Install all dependencies from requirements.txt
# ============================================================
Write-Host "`nStep 6: Installing dependencies from requirements.txt..." -ForegroundColor Yellow

# Use uv for installation — it natively handles uv_build backends and is faster.
# Falls back to pip if uv is not available (local dev without uv).
$UvAvailable = Get-Command uv -ErrorAction SilentlyContinue
if ($GpuBackend -eq "cuda") {
    if ($UvAvailable) {
        & uv pip install -r $RequirementsFile `
            --extra-index-url $PyTorchCudaIndex `
            --index-strategy unsafe-best-match `
            --python $PythonExe
    } else {
        & $PythonExe -m pip install -r $RequirementsFile `
            --extra-index-url $PyTorchCudaIndex `
            --no-warn-script-location --quiet
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: pip install failed!" -ForegroundColor Red
        exit 1
    }
} else {
    # ROCm: install non-torch deps first (no CUDA index), then ROCm SDK and PyTorch
    Write-Host "  Installing non-CUDA dependencies..." -ForegroundColor Cyan
    if ($UvAvailable) {
        & uv pip install -r $RequirementsFile --python $PythonExe
    } else {
        & $PythonExe -m pip install -r $RequirementsFile --no-warn-script-location --quiet
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: pip install failed!" -ForegroundColor Red
        exit 1
    }

    Write-Host "  Installing ROCm SDK components..." -ForegroundColor Cyan
    & $PythonExe -m pip install --no-cache-dir @RocmSdkWheels --no-warn-script-location
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: ROCm SDK install failed!" -ForegroundColor Red
        exit 1
    }

    Write-Host "  Installing ROCm PyTorch..." -ForegroundColor Cyan
    & $PythonExe -m pip install --no-cache-dir @RocmTorchWheels --no-warn-script-location
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: ROCm PyTorch install failed!" -ForegroundColor Red
        exit 1
    }
}
Write-Host "All dependencies installed" -ForegroundColor Green

# ============================================================
# Step 7: Copy Python headers for Triton/SageAttention JIT
# ============================================================
Write-Host "`nStep 7: Copying Python development files for Triton JIT..." -ForegroundColor Yellow

if ($GpuBackend -eq "rocm") {
    Write-Host "  Skipping (Triton JIT not required for ROCm builds)" -ForegroundColor DarkGray
} else {

# Ensure the exact Python version is available via uv, then copy headers/libs
& uv python install "$PythonVersion" --quiet
$UvPython = & uv python find "$PythonVersion" 2>$null
if ($UvPython) {
    # Ask Python itself for its prefix — reliable regardless of install layout
    $UvPrefix = & $UvPython -c "import sys; print(sys.prefix)"
    Write-Host "  Using uv-managed Python at: $UvPrefix"

    $IncludeSrc = Join-Path $UvPrefix "Include"
    if (-not (Test-Path $IncludeSrc)) { $IncludeSrc = Join-Path $UvPrefix "include" }
    $IncludeDst = Join-Path $OutputPath "Include"
    if (Test-Path $IncludeSrc) {
        Copy-Item -Path $IncludeSrc -Destination $IncludeDst -Recurse -Force
        Write-Host "  Copied Include folder (Python headers)"
    } else {
        Write-Host "ERROR: No Include folder found at $UvPrefix" -ForegroundColor Red
        exit 1
    }

    $LibsSrc = Join-Path $UvPrefix "libs"
    $LibsDst = Join-Path $OutputPath "libs"
    if (Test-Path $LibsSrc) {
        Copy-Item -Path $LibsSrc -Destination $LibsDst -Recurse -Force
        Write-Host "  Copied libs folder (Python libraries)"
    } else {
        Write-Host "ERROR: No libs folder found at $UvPrefix" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "ERROR: Could not find Python $PythonVersion via uv" -ForegroundColor Red
    exit 1
}
} # end if ($GpuBackend -ne "rocm")

# ============================================================
# Step 8: Clean up
# ============================================================
Write-Host "`nStep 8: Cleaning up..." -ForegroundColor Yellow

# Remove pip cache
$PipCachePaths = @(
    (Join-Path $OutputPath "Lib\site-packages\pip\_vendor\cachecontrol\caches"),
    (Join-Path $OutputPath "Lib\site-packages\pip\cache"),
    (Join-Path $OutputPath "Scripts\pip*cache*")
)
foreach ($cachePath in $PipCachePaths) {
    if (Test-Path $cachePath) {
        Remove-Item -Recurse -Force $cachePath -ErrorAction SilentlyContinue
        Write-Host "  Removed cache: $cachePath"
    }
}
Get-ChildItem -Path (Join-Path $OutputPath "Lib\site-packages") -Directory -Filter "pip-*" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Remove __pycache__ and .pyc
Get-ChildItem -Path $OutputPath -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path $OutputPath -Filter "*.pyc" | Remove-Item -Force

# Clean up temp directory and generated requirements file
Remove-Item -Recurse -Force $TempDir
Remove-Item -Force $RequirementsFile -ErrorAction SilentlyContinue

# ============================================================
# Step 9: Verify
# ============================================================
Write-Host "`nStep 9: Verifying installation..." -ForegroundColor Yellow

$TestScript = @"
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    hip = getattr(torch.version, 'hip', None)
    if hip:
        print(f'Backend: AMD ROCm/HIP {hip}')
        print(f'GPU available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'Device: {torch.cuda.get_device_name(0)}')
    else:
        print(f'Backend: NVIDIA CUDA {torch.version.cuda}')
        print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'PyTorch import failed: {e}')
try:
    import fastapi
    print(f'FastAPI: {fastapi.__version__}')
except ImportError as e:
    print(f'FastAPI import failed: {e}')
try:
    import diffusers
    print(f'Diffusers: {diffusers.__version__}')
except ImportError as e:
    print(f'Diffusers import failed: {e}')
try:
    from ltx_pipelines import distilled
    print(f'ltx-pipelines: OK')
except ImportError as e:
    print(f'ltx-pipelines: FAILED - {e}')
"@

$TestScript | & $PythonExe -

# Calculate size
$Size = (Get-ChildItem -Path $OutputPath -Recurse | Measure-Object -Property Length -Sum).Sum
$SizeGB = [math]::Round($Size / 1GB, 2)

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Python environment ready!" -ForegroundColor Green
Write-Host "  Location: $OutputPath" -ForegroundColor Green
Write-Host "  Size: $SizeGB GB" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
