@echo off
rem Live SEM imaging GUI on the NI USB-6363 (preset: AO0/AO1 scan, P0.23 blanker,
rem PFI6/PFI7 relay enables, AI0 video). Extra arguments are passed through, e.g.
rem   run_imaging.bat --field-um 120
rem   run_imaging.bat --backend sim        (simulator, no hardware)
rem Messages and errors are written to scripts\logs\imaging_gui.txt
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo .venv not found - run install.bat first.
    pause
    exit /b 1
)
if not exist scripts\logs mkdir scripts\logs
set ARGS=--backend nidaqmx --preset usb6363 --device Dev1
echo %* | find /i "--backend" >nul && set ARGS=--preset usb6363
echo Starting easyEBL imaging GUI... (log: scripts\logs\imaging_gui.txt)
.venv\Scripts\python.exe -m easyebl.imaging.gui %ARGS% %* > scripts\logs\imaging_gui.txt 2>&1
if errorlevel 1 (
    echo The GUI exited with an error:
    type scripts\logs\imaging_gui.txt
    pause
)
