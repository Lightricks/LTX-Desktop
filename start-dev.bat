@echo off
cd /d %~dp0

set "PATH=%USERPROFILE%\.local\bin;%PATH%"

echo Starting LTX Desktop...
call pnpm run dev
pause
