# run-validation-suite.ps1
# End-to-end validation for local Windows builds.

param(
    [switch]$SkipBackendTests,
    [switch]$SkipInstallerBuild,
    [switch]$SkipStartupValidation
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [string]$Command
    )

    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & powershell -NoProfile -ExecutionPolicy Bypass -Command $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
}

Invoke-Step -Name "Typecheck (TS)" -Command "pnpm typecheck:ts"
Invoke-Step -Name "Typecheck (Py)" -Command "pnpm typecheck:py"

if (-not $SkipBackendTests) {
    Invoke-Step -Name "Backend tests" -Command "pnpm backend:test"
}

Invoke-Step -Name "Build unpacked app" -Command "pnpm build:fast:win"

if (-not $SkipInstallerBuild) {
    Invoke-Step -Name "Build installer" -Command "pnpm build:win"
}

if (-not $SkipStartupValidation) {
    Invoke-Step -Name "Startup validation with blocked port 8000" -Command "pnpm validate:startup:win"
}

Write-Host "`nValidation suite completed successfully." -ForegroundColor Green
