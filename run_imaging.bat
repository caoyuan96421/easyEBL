@echo off
rem Live SEM imaging GUI on the NI USB-6363 (preset: AO0/AO1 scan, P0.23 blanker,
rem PFI6/PFI7 relay enables, AI0 video). Extra arguments are passed through, e.g.
rem   run_imaging.bat --field-um 120
rem   run_imaging.bat --backend sim        (simulator, no hardware)
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo .venv not found - run install.bat first.
    pause
    exit /b 1
)
set ARGS=--backend nidaqmx --preset usb6363 --device Dev1
echo %* | find /i "--backend" >nul && set ARGS=--preset usb6363
start "" .venv\Scripts\pythonw.exe -m easyebl.imaging.gui %ARGS% %*
