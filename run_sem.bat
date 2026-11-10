@echo off
rem SEM imaging GUI on the NI USB-6363 (AO0/AO1 scan, P0.23 blanker, PFI6/PFI7 relay, AI0 video).
rem Extra arguments are passed through, e.g.
rem   run_sem.bat --field-um 120          full-scale field at the current SEM magnification
rem   run_sem.bat --latency-ms 50         snappier response (smaller output buffer)
rem   run_sem.bat --console               also accept text commands in this window
rem   run_sem.bat --backend sim           simulator, no hardware
rem Messages and errors are written to scripts\logs\sem_gui.txt
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist .venv\Scripts\python.exe (
    echo .venv not found - run install.bat first.
    pause
    exit /b 1
)
if not exist scripts\logs mkdir scripts\logs
set ARGS=--backend nidaqmx --preset usb6363 --device Dev1
echo %* | find /i "--backend" >nul && set ARGS=--preset usb6363
echo %* | find /i "--console" >nul && goto console
echo Starting easyEBL SEM... (log: scripts\logs\sem_gui.txt)
.venv\Scripts\python.exe -m easyebl.imaging.sem %ARGS% %* > scripts\logs\sem_gui.txt 2>&1
if errorlevel 1 (
    echo The SEM GUI exited with an error:
    type scripts\logs\sem_gui.txt
    pause
)
exit /b
:console
.venv\Scripts\python.exe -m easyebl.imaging.sem %ARGS% %*
