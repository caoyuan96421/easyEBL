@echo off
rem Robustness tests of the imaging engine on the real USB-6363 (Dev1).
rem Drives AO0/AO1 and the blanker (P0.23), reads AI0; never touches the PFI6/PFI7 scan relay.
rem Options are passed to pytest, e.g.
rem   run_hw_imaging_tests.bat --hw-soak 300        5-minute soak / memory test
rem   run_hw_imaging_tests.bat --hw-loopback        AO0 wired to AI0: image-content checks too
rem The report goes to scripts\logs\hw_imaging_<timestamp>.txt, the console log to scripts\logs\hw_imaging_console.txt
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist .venv\Scripts\python.exe (
    echo .venv not found - run install.bat first.
    pause
    exit /b 1
)
if not exist scripts\logs mkdir scripts\logs
.venv\Scripts\python.exe -m pytest tests\hardware --hw Dev1 -s -v -p no:cacheprovider %* > scripts\logs\hw_imaging_console.txt 2>&1
set RC=%ERRORLEVEL%
type scripts\logs\hw_imaging_console.txt | find /i "passed"
type scripts\logs\hw_imaging_console.txt | find /i "failed"
echo Exit code %RC%. Full log: scripts\logs\hw_imaging_console.txt
pause
exit /b %RC%
