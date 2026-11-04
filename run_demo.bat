@echo off
rem Launch the easyEBL demo GUI from the local .venv.
rem   run_demo.bat                                   simulator
rem   run_demo.bat --backend nidaqmx --device Dev1   hardware
if not exist "%~dp0.venv\Scripts\easyebl-demo.exe" (
    echo .venv not found - run install.bat first.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\easyebl-demo.exe" %*
