@echo off
rem Text console for SEM imaging on the USB-6363 (same engine as the GUI). Examples:
rem   run_sem_cli.bat                                       interactive ('help' lists commands)
rem   run_sem_cli.bat -c "run; wait frames=5; capture shot frames=2 res=2560x1920 speed=Photo1; quit"
rem   run_sem_cli.bat --script my_session.txt
rem   run_sem_cli.bat --external ...                        SEM scans from the DAQ while the console runs
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist .venv\Scripts\python.exe (
    echo .venv not found - run install.bat first.
    pause
    exit /b 1
)
set ARGS=--backend nidaqmx --preset usb6363 --device Dev1
echo %* | find /i "--backend" >nul && set ARGS=--preset usb6363
.venv\Scripts\python.exe -m easyebl.imaging.cli %ARGS% %*
