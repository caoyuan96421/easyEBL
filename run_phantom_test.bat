@echo off
rem Dummy-image test on the USB-6363: a test chart is generated on AO2 (HAL output aux0).
rem Wire AO2 -> AI0+ and AO GND -> AI0- first. The SEM is not involved.
rem   run_phantom_test.bat                 real analogue noise only
rem   run_phantom_test.bat --noise 0.2     add 0.2 V rms synthetic noise (checks sigma/sqrt(samples per pixel))
rem Report and PNG images go to scripts\logs\phantom_<timestamp>\
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist .venv\Scripts\python.exe (
    echo .venv not found - run install.bat first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe scripts\hw_phantom_test.py --device Dev1 %*
pause
