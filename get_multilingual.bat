@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\get_multilingual.ps1" %*
pause
