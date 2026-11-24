<#
.SYNOPSIS
  Launch the perf dashboard on the Windows 5090 box (disconnect-safe: keeps the
  box awake). Delegates to run_dashboard.py, which starts the server and opens a
  browser; the URL is printed too.

.PARAMETER Port
  Dashboard port (default 8750, localhost only).

.EXAMPLE
  .\run_dashboard.ps1
  .\run_dashboard.ps1 -Port 8760
#>
param(
  [int]$Port = 8750
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# SetThreadExecutionState flags. Use decimal literals so PowerShell doesn't parse
# 0x80000003 as a negative Int32 (which fails to cast to the UInt32 parameter).
$ES_CONTINUOUS        = [uint32]2147483648   # 0x80000000
$ES_SYSTEM_REQUIRED   = [uint32]1            # 0x00000001
$ES_DISPLAY_REQUIRED  = [uint32]2            # 0x00000002
$keepAwake = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED)

# Keep-awake is best-effort — never let it block the launch.
$havePower = $false
try {
  Add-Type -Name Power -Namespace Win32 -MemberDefinition @'
  [System.Runtime.InteropServices.DllImport("kernel32.dll")]
  public static extern uint SetThreadExecutionState(uint esFlags);
'@ -ErrorAction SilentlyContinue
  [void][Win32.Power]::SetThreadExecutionState($keepAwake)
  $havePower = $true
} catch {
  Write-Warning "keep-awake unavailable ($($_.Exception.Message)); continuing."
}

try {
  if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv run python run_dashboard.py --port $Port
  } else {
    python run_dashboard.py --port $Port
  }
}
finally {
  if ($havePower) { try { [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) } catch {} }
}
