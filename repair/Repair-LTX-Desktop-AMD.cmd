@echo off
setlocal
title LTX Desktop - AMD 395 MAX - Repair backend startup
set "APP=%LOCALAPPDATA%\Programs\LTXDesktopAMD\application"
set "PY=%APP%\resources\python\python.exe"
if not exist "%PY%" goto missing
if not exist "%~dp0repair_existing.py" goto extract
if not exist "%~dp0repair_single_gpu.py" goto extract
if not exist "%~dp0diagnose.py" goto extract
"%PY%" -u -X utf8 "%~dp0repair_existing.py"
set "RESULT=%ERRORLEVEL%"
goto done
:missing
echo Could not find LTX Desktop - AMD 395 MAX in this Windows account at:
echo "%APP%"
echo Run this repair under the same account as the installed app.
set "RESULT=1"
goto done
:extract
echo Choose Extract All on the repair ZIP. Run this file from the extracted folder.
set "RESULT=1"
:done
echo.
pause
exit /b %RESULT%
