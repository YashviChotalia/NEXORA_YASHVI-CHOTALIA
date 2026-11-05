@echo off
rem Publish the live demo to a Hugging Face Space. Example: deploy_space.bat --space yourname/anvesh --admin judge.admin
"%~dp0backend\.venv\Scripts\python.exe" "%~dp0scripts\deploy_space.py" %*
