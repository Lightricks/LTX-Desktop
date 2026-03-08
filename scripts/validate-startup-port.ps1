# validate-startup-port.ps1
# Validates Electron backend startup when port 8000 is intentionally occupied.

param(
    [string]$ExePath = "release\\win-unpacked\\LTX Desktop.exe",
    [int]$PreferredPort = 8100,
    [int]$StartupWaitSeconds = 35,
    [switch]$KillExisting = $true
)

$ErrorActionPreference = "Stop"

function Assert-FileExists {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "Required file not found: $Path"
    }
}

function Get-LatestSessionLog {
    $logsDir = Join-Path $env:LOCALAPPDATA "LTXDesktop\\logs"
    if (-not (Test-Path $logsDir)) {
        return $null
    }
    return Get-ChildItem -Path $logsDir -Filter "session_*.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

$resolvedExePath = Resolve-Path $ExePath | Select-Object -ExpandProperty Path
Assert-FileExists $resolvedExePath

if ($KillExisting) {
    Get-Process | Where-Object { $_.ProcessName -like "LTX Desktop*" } | ForEach-Object {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

$listener = $null
$appProcess = $null

try {
    Write-Host "[1/4] Reserving port $PreferredPort on 127.0.0.1..." -ForegroundColor Yellow
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse("127.0.0.1"), $PreferredPort)
    $listener.Start()

    Write-Host "[2/4] Starting app with LTX_PORT=${PreferredPort}: $resolvedExePath" -ForegroundColor Yellow
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $resolvedExePath
    $startInfo.WorkingDirectory = [System.IO.Path]::GetDirectoryName($resolvedExePath)
    $startInfo.UseShellExecute = $false
    $startInfo.EnvironmentVariables["LTX_PORT"] = [string]$PreferredPort
    $appProcess = [System.Diagnostics.Process]::Start($startInfo)

    Write-Host "[3/4] Waiting $StartupWaitSeconds seconds for backend startup..." -ForegroundColor Yellow
    Start-Sleep -Seconds $StartupWaitSeconds

    Write-Host "[4/4] Inspecting latest session log..." -ForegroundColor Yellow
    $latestLog = Get-LatestSessionLog
    if ($null -eq $latestLog) {
        throw "No session log found under %LOCALAPPDATA%\\LTXDesktop\\logs"
    }

    $logPath = $latestLog.FullName
    Write-Host "Using log: $logPath" -ForegroundColor Cyan
    $logText = Get-Content -Path $logPath -Raw

    $sawPortFallback = $logText -match "Backend port \d+ unavailable, switching to \d+"
    $sawStartupError = $logText -match "Python backend exited during startup with code"
    $sawBackendReady = ($logText -match "Application startup complete") -or ($logText -match "Uvicorn running") -or ($logText -match "Server running on")

    if (-not $sawPortFallback) {
        throw "Validation failed: did not find fallback-port log line."
    }

    if ($sawStartupError) {
        throw "Validation failed: backend still exited during startup."
    }

    if (-not $sawBackendReady) {
        throw "Validation failed: did not find backend-ready log line."
    }

    Write-Host "Validation passed: startup succeeded with port-8000 conflict." -ForegroundColor Green
    exit 0
}
catch {
    Write-Host "Validation failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    if ($appProcess -and -not $appProcess.HasExited) {
        Stop-Process -Id $appProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($listener) {
        $listener.Stop()
    }
}
