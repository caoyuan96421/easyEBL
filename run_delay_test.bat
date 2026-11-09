@echo off
rem AO -> AI delay measurement. Requires a loopback cable AO0 -> AI0.
rem Output: scripts\logs\delay.txt   (extra args go to scripts\measure_delay.py)
cd /d "%~dp0"
if not exist scripts\logs mkdir scripts\logs
.venv\Scripts\python.exe scripts\measure_delay.py --device Dev1 %* > scripts\logs\delay.txt 2>&1
echo Finished. See scripts\logs\delay.txt
timeout /t 5 >nul
