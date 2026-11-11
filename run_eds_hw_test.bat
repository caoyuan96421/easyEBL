@echo off
rem EDS hardware test: EDS phantom on the USB-6363 + Amptek DP5 + the EDS backend.
rem Wiring: AO2 -> AI0+ (SE image, RSE)   AO3 -> DP5 analogue input (SDD preamp signal)
rem Close the DP5 GUI / DPPMCA and the SEM / EDS programs first (they hold the devices).
rem The DP5 configuration is saved and restored.  Report: scripts\logs\eds_<timestamp>\report.txt
rem   run_eds_hw_test.bat --quick          shorter acquisitions
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist .venv\Scripts\python.exe (
    echo .venv not found - run install.bat first.
    pause
    exit /b 1
)
if not exist scripts\logs mkdir scripts\logs
.venv\Scripts\python.exe -c "import usb, libusb_package" 2>nul || .venv\Scripts\python.exe -m pip install pyusb libusb-package
.venv\Scripts\python.exe scripts\hw_eds_test.py --device Dev1 %* > scripts\logs\eds_hw_test_console.txt 2>&1
type scripts\logs\eds_hw_test_console.txt
pause
