@echo off
rem SEM imaging + EDS in one program: both windows share the NI USB-6363, so an EDS
rem acquisition pauses the live SEM image and the SEM resumes by itself afterwards.
rem Extra arguments are passed through, e.g.
rem   run_suite.bat --backend sim           simulator (EDS phantom, simulated pulse processor)
rem   run_suite.bat --phantom eds           EDS phantom on the hardware (AO2 -> AI0, AO3 -> DP5)
rem   run_suite.bat --run                   start live SEM scanning immediately
rem Messages and errors are written to scripts\logs\suite_gui.txt
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist .venv\Scripts\python.exe (
    echo .venv not found - run install.bat first.
    pause
    exit /b 1
)
if not exist scripts\logs mkdir scripts\logs
.venv\Scripts\python.exe -c "import usb, libusb_package" 2>nul || .venv\Scripts\python.exe -m pip install pyusb libusb-package
set ARGS=--backend nidaqmx --preset usb6363 --device Dev1
echo %* | find /i "--backend" >nul && set ARGS=--preset usb6363
echo Starting easyEBL SEM + EDS... (log: scripts\logs\suite_gui.txt)
.venv\Scripts\python.exe -c "from easyebl.eds.gui import suite_main; raise SystemExit(suite_main())" %ARGS% %* > scripts\logs\suite_gui.txt 2>&1
if errorlevel 1 (
    echo The program exited with an error:
    type scripts\logs\suite_gui.txt
    pause
)
