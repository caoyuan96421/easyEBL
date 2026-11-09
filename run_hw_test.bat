@echo off
rem Hardware bring-up for the USB-6363 scan path. Double-click to run.
rem Builds .venv on first use, lists NI devices, then runs scripts\hw_scan_test.py.
rem All output goes to scripts\logs\ so it can be read afterwards.
setlocal
cd /d "%~dp0"
if not exist scripts\logs mkdir scripts\logs
set LOG=scripts\logs\console.txt
echo [%date% %time%] start > %LOG%
if not exist .venv\Scripts\python.exe (
    echo Building .venv - this takes a few minutes on first run...
    echo [%time%] installing >> %LOG%
    powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 > scripts\logs\install.txt 2>&1
    echo [%time%] install exit %ERRORLEVEL% >> %LOG%
)
echo [%time%] NI devices: >> %LOG%
.venv\Scripts\python.exe -c "import nidaqmx.system as s; sysm=s.System.local(); print('driver', sysm.driver_version); [print(d.name, d.product_type) for d in sysm.devices]" >> %LOG% 2>&1
echo [%time%] hw_scan_test %* >> %LOG%
.venv\Scripts\python.exe scripts\hw_scan_test.py --device Dev1 %* >> %LOG% 2>&1
echo [%time%] done exit %ERRORLEVEL% >> %LOG%
echo Finished. See scripts\logs\console.txt
timeout /t 5 >nul
