@echo off
rem Manual test of the lost-connection handling on the USB-6363 (Dev1).
rem Live imaging runs; follow the prompts to unplug and re-plug the USB cable.
rem   run_unplug_test.bat                  3 cycles
rem   run_unplug_test.bat --cycles 5 --external   also checks external scan control is restored
rem The log goes to scripts\logs\hw_unplug_<timestamp>.txt
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist .venv\Scripts\python.exe (
    echo .venv not found - run install.bat first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe scripts\hw_unplug_test.py --device Dev1 %*
pause
