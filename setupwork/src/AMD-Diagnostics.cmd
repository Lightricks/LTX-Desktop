@echo off
setlocal
"%~dp0resources\python\python.exe" -u -X utf8 "%~dp0diagnose.py"
set "result=%errorlevel%"
pause
exit /b %result%
