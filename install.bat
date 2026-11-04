@echo off
rem Double-click (or run from cmd) to create .venv and install easyEBL.
rem Arguments are passed to install.ps1, e.g.:  install.bat -Recreate -RunTests
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set RC=%ERRORLEVEL%
echo %CMDCMDLINE% | find /i "%~nx0" >nul && pause
exit /b %RC%
