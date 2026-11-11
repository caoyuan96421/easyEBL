@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\diag_dp5_preset.py > scripts\logs\diag_dp5_preset_console.txt 2>&1
