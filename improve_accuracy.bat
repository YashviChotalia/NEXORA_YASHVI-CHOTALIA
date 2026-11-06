@echo off
cd /d "%~dp0backend"
if not exist ".venv\Scripts\python.exe" ( echo backend\.venv not found: run setup.bat first. & pause & exit /b 1 )
if not exist "..\logs" mkdir "..\logs"
echo Step 1/3: downloading CREMA-D clips (skips what is already there) ...
.venv\Scripts\python.exe eval\train_ser_head_v2.py download > ..\logs\accuracy.log 2>&1
echo Step 2/3: computing embeddings in parallel (about 20-40 min, keep this window open) ...
.venv\Scripts\python.exe eval\train_ser_head_v2.py extract >> ..\logs\accuracy.log 2>&1
echo Step 3/3: training and testing ...
.venv\Scripts\python.exe eval\train_ser_head_v2.py fit >> ..\logs\accuracy.log 2>&1
echo.
echo Done. Log: logs\accuracy.log
pause
