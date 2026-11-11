@echo off
rem EDS window (Amptek DP5 + NI USB-6363).  Extra arguments are passed through, e.g.
rem   run_eds.bat --backend sim            simulator + simulated pulse processor, no hardware
rem   run_eds.bat --phantom eds            EDS phantom: SE image on AO2 (-> AI0), SDD signal on AO3 (-> DP5)
rem   run_eds.bat --mca mock               DP5 driver's mock device instead of the USB DP5
rem   run_eds.bat --kv 15 --field-um 120   beam voltage for the analysis, field at the current magnification
rem To use SEM imaging at the same time, start run_suite.bat instead (one process shares the DAQ).
rem Messages and errors are written to scripts\logs\eds_gui.txt
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
echo Starting easyEBL EDS... (log: scripts\logs\eds_gui.txt)
.venv\Scripts\python.exe -m easyebl.eds %ARGS% %* > scripts\logs\eds_gui.txt 2>&1
if errorlevel 1 (
    echo The EDS GUI exited with an error:
    type scripts\logs\eds_gui.txt
    pause
)
