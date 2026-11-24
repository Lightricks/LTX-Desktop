@echo off
echo Setting up Visual Studio environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

REM Required for PyTorch C++ extension builds
set DISTUTILS_USE_SDK=1

echo Checking compiler...
where cl.exe
if errorlevel 1 (
    echo ERROR: cl.exe not found after environment setup
    exit /b 1
)

echo.
echo DISTUTILS_USE_SDK=%DISTUTILS_USE_SDK%
echo.
echo Installing SageAttention 2.2.0 from GitHub...
cd /d "%~dp0"
".venv\Scripts\pip.exe" uninstall sageattention -y 2>nul
".venv\Scripts\pip.exe" install git+https://github.com/thu-ml/SageAttention.git --no-build-isolation

echo.
if errorlevel 1 (
    echo BUILD FAILED! See errors above.
) else (
    echo SUCCESS! SageAttention 2.2.0 installed.
)
pause
