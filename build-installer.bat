@echo off
setlocal EnableDelayedExpansion

echo.
echo   _   _______  __  ____            _    _
echo  ^| ^| ^|_   _\ \/ / ^|  _ \  ___  ___^| ^| _^| ^|_ ___  _ __
echo  ^| ^|   ^| ^|  \  /  ^| ^| ^| ^|/ _ \/ __^| ^|/ / __/ _ \^| '_ \
echo  ^| ^|___^| ^|  /  \  ^| ^|_^| ^|  __/\__ \   ^<^| ^|^| (_) ^| ^|_) ^|
echo  ^|_____^|_^| /_/\_\ ^|____/ \___^|^|___/_^|\_\\__\___/^| .__/
echo                                                  ^|_^|
echo   One-Click Installer Build
echo.

:: ============================================================
:: Check prerequisites
:: ============================================================

echo Checking prerequisites...
echo.

where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js not found.
    echo         Download and install from: https://nodejs.org/
    echo.
    goto :fail
)
for /f "tokens=*" %%v in ('node -v') do echo [OK] Node.js %%v

where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Git not found.
    echo         Download and install from: https://git-scm.com/
    echo.
    goto :fail
)
for /f "tokens=*" %%v in ('git --version') do echo [OK] %%v

:: ============================================================
:: Install pnpm if missing
:: ============================================================

where pnpm >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo Installing pnpm...
    call npm install -g pnpm
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to install pnpm.
        goto :fail
    )
)
for /f "tokens=*" %%v in ('pnpm --version') do echo [OK] pnpm %%v

:: ============================================================
:: Install uv if missing
:: ============================================================

where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo Installing uv (Python package manager)...
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to install uv.
        goto :fail
    )
    :: Refresh PATH so uv is available
    for /f "tokens=*" %%p in ('powershell -Command "[Environment]::GetEnvironmentVariable('Path','User')"') do set "PATH=%%p;%PATH%"
)
for /f "tokens=*" %%v in ('uv --version') do echo [OK] %%v

echo.
echo All prerequisites satisfied.
echo.

:: ============================================================
:: Run the full build
:: ============================================================

echo Starting full installer build...
echo This will download ~10GB of dependencies and may take a while.
echo.

powershell -ExecutionPolicy ByPass -File "%~dp0scripts\local-build.ps1"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Build failed. Check the output above for details.
    goto :fail
)

:: ============================================================
:: Success
:: ============================================================

echo.
echo =========================================================
echo   BUILD COMPLETE!
echo   Your installer is in the release\ folder.
echo =========================================================
echo.

if exist "%~dp0release" (
    explorer "%~dp0release"
)

pause
exit /b 0

:fail
echo.
echo Build failed. Please fix the errors above and try again.
pause
exit /b 1
